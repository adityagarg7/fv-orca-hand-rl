"""
Production reward wrapper — v10 (Finger-First Manipulation)
============================================================

v9 postmortem: agent learned to rotate wrist + use gravity instead of
finger manipulation.  Root cause: r_progress was strategy-agnostic, and
wrist rotation had 16× lower action cost than finger manipulation.

v10 fixes — every change tied to a specific behavioral failure:

  1. Wrist gate on r_progress        → kills wrist exploit
  2. Contact gate on r_progress      → kills gravity exploit
  3. EMA smoothing on r_progress     → makes finger progress detectable above noise
  4. Delta clamping on r_progress    → fixes drop-reward bug (+78 reward for dropping)
  5. Widened r_align σ (0.3→1.0)     → goal gradient at ALL angles, not just last 17°
  6. Increased r_pos (0.5→2.0)       → forces palm-centered manipulation
  7. Increased r_fingers + contact   → finger contact comparable to r_progress magnitude
  8. Finger action bonus             → actively rewards finger joint movement
  9. Asymmetric wrist penalty        → wrist movement 25× more expensive
  10. 10-step success hold           → eliminates flick-through-success
  11. Normalised velocity in success → proper m/s vs rad/s scaling
  12. Gradual drop risk + stronger   → anticipatory drop avoidance

Actuator layout (from MuJoCo model):
  action[0]    = right_wrist (THE exploit joint)
  action[1:13] = finger abd/mcp/pip (pinky, ring, middle, index)
  action[13:17]= thumb cmc/abd/mcp/pip
"""

import gymnasium as gym
import numpy as np


# Fingertip body names on the ORCA right hand.
FINGERTIP_BODIES = (
    "right_T-DP_b7429e50",                   # Thumb distal phalanx
    "right_I-FingerTipAssembly_ec49c16c",     # Index fingertip
    "right_M-FingerTipAssembly_34afb748",     # Middle fingertip
    "right_M-FingerTipAssembly_424a8e75",     # Ring fingertip
    "right_P-FingerTipAssembly_cd219176",     # Pinky fingertip
)

# Joint index for the wrist actuator.
WRIST_ACTION_IDX = 0


class ProductionRewardWrapper(gym.Wrapper):
    """v10 reward: finger-first manipulation with wrist attenuation."""

    def __init__(
        self,
        env,
        # --- Alignment (widened kernel) ---
        align_weight: float = 3.0,       # was 5.0 — reduced because wider σ gives non-zero everywhere
        align_sigma: float = 1.0,        # was 0.3 — 60° scale gives gradient at ALL chapter angles
        # --- Rotation progress (gated + smoothed) ---
        progress_weight: float = 50.0,
        progress_ema_alpha: float = 0.3, # EMA smoothing (5-step effective window)
        progress_clip: float = 0.1,      # clamp Δθ to ±0.1 rad (fixes drop-reward bug)
        wrist_gate_strength: float = 0.8,# how much to penalise wrist-only progress
        contact_threshold: float = 0.020,# 20mm = "finger is touching cube"
        contact_min_fingers: float = 2.0,# need ≥2 fingers for full gate
        # --- Position keeping (palm-centered) ---
        pos_weight: float = 2.0,         # was 0.5 — 4× increase for palm-centered behavior
        pos_sigma: float = 0.03,         # was 0.05 — tighter (3cm) but allows manipulation room
        # --- Fingertip proximity + contact ---
        finger_prox_weight: float = 2.0, # was 0.3 — now comparable to r_progress magnitude
        finger_sigma: float = 0.02,      # metres
        finger_contact_bonus: float = 0.5,# per finger in contact
        finger_active_bonus: float = 0.3,# reward for active finger movement
        # --- Success (hold + normalised velocity + position) ---
        success_bonus: float = 500.0,
        success_hold_steps: int = 10,    # must hold θ<15° for 10 steps (0.2s)
        stability_floor: float = 0.3,
        v_scale: float = 0.05,           # characteristic linear velocity (m/s)
        w_scale: float = 1.5,            # characteristic angular velocity (rad/s)
        palm_success_sigma: float = 0.04,# position factor in success (4cm)
        # --- Alive ---
        alive_bonus: float = 0.1,        # was 0.05 — slight increase
        # --- Drop (gradual + stronger terminal) ---
        drop_penalty: float = 50.0,      # was 10.0 — 5× increase
        drop_post_success_extra: float = 100.0,  # extra penalty for dropping after success
        drop_risk_weight: float = 2.0,   # gradual penalty as cube nears edge
        drop_safe_height: float = 0.12,  # palm surface height (m)
        drop_height: float = 0.08,       # below this = dropped
        # --- Action regularisation (asymmetric) ---
        wrist_rate_weight: float = 0.05, # 25× more expensive than fingers
        finger_rate_weight: float = 0.001,# lighter than v9 (was 0.002)
        action_mag_weight: float = 0.0001,# kept light — doesn't affect jerk
    ):
        super().__init__(env)
        # Store all parameters
        self.align_weight = align_weight
        self.align_sigma = align_sigma
        self.progress_weight = progress_weight
        self.progress_ema_alpha = progress_ema_alpha
        self.progress_clip = progress_clip
        self.wrist_gate_strength = wrist_gate_strength
        self.contact_threshold = contact_threshold
        self.contact_min_fingers = contact_min_fingers
        self.pos_weight = pos_weight
        self.pos_sigma = pos_sigma
        self.finger_prox_weight = finger_prox_weight
        self.finger_sigma = finger_sigma
        self.finger_contact_bonus = finger_contact_bonus
        self.finger_active_bonus = finger_active_bonus
        self.success_bonus = success_bonus
        self.success_hold_steps = success_hold_steps
        self.stability_floor = stability_floor
        self.v_scale = v_scale
        self.w_scale = w_scale
        self.palm_success_sigma = palm_success_sigma
        self.alive_bonus = alive_bonus
        self.drop_penalty = drop_penalty
        self.drop_post_success_extra = drop_post_success_extra
        self.drop_risk_weight = drop_risk_weight
        self.drop_safe_height = drop_safe_height
        self.drop_height = drop_height
        self.wrist_rate_weight = wrist_rate_weight
        self.finger_rate_weight = finger_rate_weight
        self.action_mag_weight = action_mag_weight

        # Internal state
        self._prev_action = None
        self._ema_angle = None
        self._default_cube_pos = None
        self._success_hold_counter = 0
        self._had_success_this_episode = False
        self._total_successes = 0

        # MuJoCo body ID caches
        self._fingertip_body_ids = []
        self._cube_body_id = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _resolve_body_ids(self):
        model = self.unwrapped.model
        self._fingertip_body_ids = [
            model.body(name).id for name in FINGERTIP_BODIES
        ]
        self._cube_body_id = model.body("task_cube").id

    def _fingertip_distances(self) -> list[float]:
        data = self.unwrapped.data
        cube_pos = data.xpos[self._cube_body_id]
        return [
            float(np.linalg.norm(data.xpos[bid] - cube_pos))
            for bid in self._fingertip_body_ids
        ]

    # ------------------------------------------------------------------
    # gym.Wrapper overrides
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        self._default_cube_pos = info["cube_pos"].copy()
        self._ema_angle = info["red_face_up_angle_rad"]
        self._success_hold_counter = 0
        self._had_success_this_episode = False
        self._total_successes = 0

        if not self._fingertip_body_ids:
            self._resolve_body_ids()

        return obs, info

    def step(self, action):
        obs, _orig_reward, base_terminated, truncated, info = self.env.step(action)
        action = np.asarray(action, dtype=np.float32)

        angle_error = info["red_face_up_angle_rad"]
        finger_dists = self._fingertip_distances()
        pos_error = 0.0
        if self._default_cube_pos is not None:
            pos_error = np.linalg.norm(info["cube_pos"] - self._default_cube_pos)

        # ════════════════════════════════════════════════════════════════
        # 1. ALIGNMENT — widened exponential kernel
        # ════════════════════════════════════════════════════════════════
        # σ=1.0 gives meaningful gradient at ALL angles (0.67 at 90°)
        # vs old σ=0.3 which was 0.028 at 90° (dead zone)
        r_align = self.align_weight * np.exp(-angle_error / self.align_sigma)

        # ════════════════════════════════════════════════════════════════
        # 2. PROGRESS — EMA smoothed, clamped, wrist-gated, contact-gated
        # ════════════════════════════════════════════════════════════════
        ema_prev = self._ema_angle
        self._ema_angle = (self.progress_ema_alpha * angle_error
                           + (1 - self.progress_ema_alpha) * self._ema_angle)

        raw_delta = ema_prev - self._ema_angle
        clamped_delta = np.clip(raw_delta, -self.progress_clip, self.progress_clip)

        # Wrist gate: penalise progress earned through wrist rotation
        wrist_effort = np.abs(action[WRIST_ACTION_IDX])
        finger_effort = np.linalg.norm(action[WRIST_ACTION_IDX + 1:])
        wrist_ratio = wrist_effort / (wrist_effort + finger_effort + 1e-8)
        wrist_gate = 1.0 - self.wrist_gate_strength * wrist_ratio

        # Contact gate: only reward progress when fingers touch the cube
        contact_count = sum(1 for d in finger_dists if d < self.contact_threshold)
        contact_gate = min(1.0, contact_count / self.contact_min_fingers)

        r_progress = self.progress_weight * clamped_delta * wrist_gate * contact_gate

        # ════════════════════════════════════════════════════════════════
        # 3. POSITION — keep cube centered in palm
        # ════════════════════════════════════════════════════════════════
        r_pos = self.pos_weight * np.exp(-pos_error / self.pos_sigma)

        # ════════════════════════════════════════════════════════════════
        # 4. FINGERS — proximity + contact count + active movement
        # ════════════════════════════════════════════════════════════════
        # Continuous proximity (how close are fingertips to cube)
        r_fingers_prox = self.finger_prox_weight * np.mean([
            np.exp(-d / self.finger_sigma) for d in finger_dists
        ])

        # Discrete contact bonus (reward having multiple fingers touching)
        r_fingers_contact = self.finger_contact_bonus * contact_count

        # Active finger movement bonus (reward finger joints being used)
        finger_action_mag = np.linalg.norm(action[WRIST_ACTION_IDX + 1:])
        r_fingers_active = self.finger_active_bonus * min(1.0, finger_action_mag / 0.5)

        r_fingers = r_fingers_prox + r_fingers_contact + r_fingers_active

        # ════════════════════════════════════════════════════════════════
        # 5. SUCCESS — 10-step hold + normalised velocity + position
        # ════════════════════════════════════════════════════════════════
        r_success = 0.0
        terminated = False

        if info["is_success"]:
            self._success_hold_counter += 1
        else:
            self._success_hold_counter = 0

        if self._success_hold_counter >= self.success_hold_steps:
            # Cube held in success zone for hold_steps — genuine success
            linear_vel = np.linalg.norm(info["cube_qvel"][:3])
            angular_vel = np.linalg.norm(info["cube_qvel"][3:6])

            # Normalised velocity metric (proper m/s vs rad/s scaling)
            v_norm = (linear_vel / self.v_scale) ** 2
            w_norm = (angular_vel / self.w_scale) ** 2
            vel_metric = np.sqrt(v_norm + w_norm)

            stability = (self.stability_floor
                         + (1 - self.stability_floor) * np.exp(-vel_metric / 1.0))

            # Position factor: cube must be near palm center
            palm_factor = np.exp(-pos_error / self.palm_success_sigma)

            r_success = self.success_bonus * stability * palm_factor
            self._total_successes += 1
            self._had_success_this_episode = True
            terminated = True

        elif self._success_hold_counter > 0:
            # In success zone but haven't held long enough — small encouragement
            r_success = (self.success_bonus * 0.02
                         * self._success_hold_counter)  # ramps up over hold period

        # ════════════════════════════════════════════════════════════════
        # 6. DROP — gradual risk + strong terminal + post-success extra
        # ════════════════════════════════════════════════════════════════
        cube_z = info["cube_pos"][2]
        height_margin = max(0, cube_z - self.drop_height) / (
            self.drop_safe_height - self.drop_height + 1e-8
        )
        height_margin = np.clip(height_margin, 0, 1)

        # Gradual risk penalty (increases as cube nears the edge)
        r_drop_risk = self.drop_risk_weight * (1.0 - height_margin)

        # Terminal drop penalty
        r_drop_terminal = 0.0
        if info["dropped"]:
            r_drop_terminal = self.drop_penalty
            if self._had_success_this_episode:
                r_drop_terminal += self.drop_post_success_extra
            terminated = True

        # ════════════════════════════════════════════════════════════════
        # 7. ALIVE — per-step survival bonus
        # ════════════════════════════════════════════════════════════════
        r_alive = self.alive_bonus if not info["dropped"] else 0.0

        # ════════════════════════════════════════════════════════════════
        # 8. ACTION — asymmetric: wrist expensive, fingers cheap + bonus
        # ════════════════════════════════════════════════════════════════
        # Wrist rate penalty (25× heavier than fingers)
        wrist_delta = action[WRIST_ACTION_IDX] - self._prev_action[WRIST_ACTION_IDX]
        r_wrist_rate = self.wrist_rate_weight * wrist_delta ** 2

        # Finger rate penalty (lighter than v9)
        finger_deltas = action[WRIST_ACTION_IDX + 1:] - self._prev_action[WRIST_ACTION_IDX + 1:]
        r_finger_rate = self.finger_rate_weight * np.sum(finger_deltas ** 2)

        # Action magnitude (kept very light — doesn't affect jerk)
        r_action_mag = self.action_mag_weight * np.sum(action ** 2)

        self._prev_action = action.copy()

        # ════════════════════════════════════════════════════════════════
        # COMBINE
        # ════════════════════════════════════════════════════════════════
        reward = (
            r_align
            + r_progress
            + r_pos
            + r_fingers
            + r_success
            + r_alive
            - r_drop_risk
            - r_drop_terminal
            - r_wrist_rate
            - r_finger_rate
            - r_action_mag
        )

        # ════════════════════════════════════════════════════════════════
        # INFO — full breakdown for W&B logging
        # ════════════════════════════════════════════════════════════════
        info["reward_breakdown"] = {
            "align": float(r_align),
            "progress": float(r_progress),
            "progress_raw_delta": float(raw_delta),
            "progress_wrist_gate": float(wrist_gate),
            "progress_contact_gate": float(contact_gate),
            "pos": float(r_pos),
            "fingers_prox": float(r_fingers_prox),
            "fingers_contact": float(r_fingers_contact),
            "fingers_active": float(r_fingers_active),
            "success": float(r_success),
            "success_hold_count": int(self._success_hold_counter),
            "alive": float(r_alive),
            "drop_risk": float(-r_drop_risk),
            "drop_terminal": float(-r_drop_terminal),
            "wrist_rate": float(-r_wrist_rate),
            "finger_rate": float(-r_finger_rate),
            "action_mag": float(-r_action_mag),
            "total": float(reward),
        }

        info["total_vel"] = float(np.linalg.norm(info["cube_qvel"][:3])
                                  + np.linalg.norm(info["cube_qvel"][3:6]))
        info["total_successes"] = self._total_successes
        info["fingertip_dists"] = finger_dists
        info["contact_count"] = contact_count
        info["wrist_ratio"] = float(wrist_ratio)
        info["is_success"] = (self._success_hold_counter >= self.success_hold_steps)

        return obs, float(reward), terminated, truncated, info
