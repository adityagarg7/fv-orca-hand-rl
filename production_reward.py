"""
Production reward wrapper for OrcaHandRightCubeOrientation — Phase 1.

Fixes three problems with the v1 reward:
  1. Action penalties dominated the alignment signal 20:1 — agent learned
     "don't move" instead of "rotate the cube."  Fixed by reducing action
     penalty weights 10x and boosting alignment weight.
  2. The 15-step stable-grasp hold was an impossible exploration barrier for
     a random policy.  Reduced to 5 steps, and a progressive per-step hold
     bonus now provides immediate signal for each correct-hold frame.
  3. No fingertip contact signal — the agent had no reason to keep fingers
     on the cube.  Added a distance-based fingertip proximity reward.

Changes are backed by:
  - arXiv:2605.21330 (action penalty << task reward)
  - OpenAI Dactyl  (rotation-progress delta as primary dense signal)
  - HORA / IsaacGym ShadowHand (fingertip distance bonuses)
"""

import gymnasium as gym
import numpy as np


# Fingertip body names on the ORCA right hand (found via model introspection).
FINGERTIP_BODIES = (
    "right_T-DP_b7429e50",                   # Thumb distal phalanx
    "right_I-FingerTipAssembly_ec49c16c",     # Index fingertip
    "right_M-FingerTipAssembly_34afb748",     # Middle fingertip
    "right_M-FingerTipAssembly_424a8e75",     # Ring fingertip
    "right_P-FingerTipAssembly_cd219176",     # Pinky fingertip
)


class ProductionRewardWrapper(gym.Wrapper):
    """Wraps OrcaHandRightCubeOrientation with the Phase-1 shaped reward.

    This does NOT modify the original orca_sim source code.  It intercepts
    the reward after each step and replaces it with a properly shaped signal.
    """

    def __init__(
        self,
        env,
        # --- Alignment (primary task signal) ---
        align_weight: float = 5.0,
        align_sigma: float = 0.3,       # radians
        # --- Rotation progress (Dactyl-style delta) ---
        progress_weight: float = 10.0,
        # --- Position keeping (keep cube centred) ---
        pos_weight: float = 0.5,
        pos_sigma: float = 0.03,        # metres
        # --- Fingertip proximity (new — finger-joint tracking) ---
        finger_weight: float = 0.3,
        finger_sigma: float = 0.02,     # metres
        # --- Success bonus (anti-hacking) ---
        success_bonus: float = 100.0,
        # --- Progressive hold bonus ---
        hold_weight: float = 2.0,
        required_hold_steps: int = 5,
        # --- Alive bonus ---
        alive_bonus: float = 0.05,
        # --- Drop penalty ---
        drop_penalty: float = 10.0,
        # --- Action regularisation (10x lighter than v1) ---
        action_rate_weight: float = 0.002,
        action_mag_weight: float = 0.0001,
    ):
        super().__init__(env)
        self.align_weight = align_weight
        self.align_sigma = align_sigma
        self.progress_weight = progress_weight
        self.pos_weight = pos_weight
        self.pos_sigma = pos_sigma
        self.finger_weight = finger_weight
        self.finger_sigma = finger_sigma
        self.success_bonus = success_bonus
        self.hold_weight = hold_weight
        self.required_hold_steps = required_hold_steps
        self.alive_bonus = alive_bonus
        self.drop_penalty = drop_penalty
        self.action_rate_weight = action_rate_weight
        self.action_mag_weight = action_mag_weight

        self._hold_steps = 0
        self._prev_action = None
        self._prev_angle_error = None
        self._default_cube_pos = None

        # Resolve fingertip body IDs (done once at construction).
        self._fingertip_body_ids = []
        self._cube_body_id = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _resolve_body_ids(self):
        """Cache MuJoCo body IDs for fingertips and cube (called once)."""
        model = self.unwrapped.model
        self._fingertip_body_ids = [
            model.body(name).id for name in FINGERTIP_BODIES
        ]
        self._cube_body_id = model.body("task_cube").id

    def _fingertip_distances(self) -> list[float]:
        """Return the Euclidean distance from each fingertip to the cube."""
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
        self._prev_angle_error = info["red_face_up_angle_rad"]
        self._hold_steps = 0

        # Lazy-resolve body IDs on first reset.
        if not self._fingertip_body_ids:
            self._resolve_body_ids()

        return obs, info

    def step(self, action):
        obs, _original_reward, base_terminated, truncated, info = self.env.step(action)

        # ---- 1. Alignment reward (exponential kernel) ----
        angle_error = info["red_face_up_angle_rad"]
        r_align = self.align_weight * np.exp(-angle_error / self.align_sigma)

        # ---- 2. Rotation progress (Dactyl-style delta) ----
        if self._prev_angle_error is not None:
            r_progress = self.progress_weight * (self._prev_angle_error - angle_error)
        else:
            r_progress = 0.0
        self._prev_angle_error = angle_error

        # ---- 3. Position tracking (keep cube centred) ----
        pos_error = 0.0
        if self._default_cube_pos is not None:
            pos_error = np.linalg.norm(info["cube_pos"] - self._default_cube_pos)
            r_pos = self.pos_weight * np.exp(-pos_error / self.pos_sigma)
        else:
            r_pos = 0.0

        # ---- 4. Fingertip proximity (finger-joint tracking) ----
        finger_dists = self._fingertip_distances()
        r_fingers = self.finger_weight * sum(
            np.exp(-d / self.finger_sigma) for d in finger_dists
        ) / len(finger_dists)

        # ---- 5. Stable grasp check & progressive hold ----
        linear_vel = np.linalg.norm(info["cube_qvel"][:3])
        angular_vel = np.linalg.norm(info["cube_qvel"][3:6])
        is_stable = (linear_vel < 0.1) and (angular_vel < 0.5) and (pos_error < 0.05)

        if info["is_success"] and is_stable:
            self._hold_steps += 1
        else:
            self._hold_steps = 0

        # Progressive hold bonus (reward each consecutive hold step).
        r_hold = self.hold_weight * self._hold_steps if self._hold_steps > 0 else 0.0

        # Terminal success bonus — only after the hold duration.
        terminated = False
        r_success = 0.0

        if self._hold_steps >= self.required_hold_steps:
            r_success = self.success_bonus
            terminated = True
        elif info["dropped"]:
            terminated = True

        # ---- 6. Alive bonus ----
        r_alive = self.alive_bonus if not info["dropped"] else 0.0

        # ---- 7. Drop penalty ----
        r_drop = self.drop_penalty if info["dropped"] else 0.0

        # ---- 8. Action regularisation (10x lighter than v1) ----
        action = np.asarray(action, dtype=np.float32)
        r_action_rate = self.action_rate_weight * np.sum((action - self._prev_action) ** 2)
        r_action_mag = self.action_mag_weight * np.sum(action ** 2)
        self._prev_action = action.copy()

        # ---- Combine ----
        reward = (
            r_align
            + r_progress
            + r_pos
            + r_fingers
            + r_hold
            + r_success
            + r_alive
            - r_drop
            - r_action_rate
            - r_action_mag
        )

        # Inject breakdown into info for debugging / W&B logging.
        info["reward_breakdown"] = {
            "align": float(r_align),
            "progress": float(r_progress),
            "pos": float(r_pos),
            "fingers": float(r_fingers),
            "hold": float(r_hold),
            "success": float(r_success),
            "alive": float(r_alive),
            "drop": float(-r_drop),
            "action_rate": float(-r_action_rate),
            "action_mag": float(-r_action_mag),
            "total": float(reward),
        }

        # Debugging info.
        info["hold_steps"] = self._hold_steps
        info["is_stable"] = is_stable
        info["fingertip_dists"] = finger_dists
        info["is_success"] = (self._hold_steps >= self.required_hold_steps)

        return obs, float(reward), terminated, truncated, info
