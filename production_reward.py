"""
Production reward wrapper — v11 (Finger-First, Exploit-Closed)
==============================================================

v10 postmortem (full audit in V10_AUDIT_FINDINGS.md):
  F1  G_contact == 0 always: fingertip body xpos is at the PIP joint, 40-44mm
      proximal of the pad; cube half-extent is 18mm; the 20mm center-distance
      threshold was geometrically unsatisfiable.  r_progress never fired once.
  F2  Success hold ramp (10*counter/step) paid +450 per 9-step cycle vs
      ~150-290 for completing -> farmable boundary-dithering annuity.
  F3  Wrist gate |a0|/(|a0|+||a1:||) was defeated by the 1-vs-16 dimensional
      asymmetry: PPO noise alone kept the gate >= 0.64.
  F4  Standing income (align+pos+fingers+alive ~ 6/step) dominated task income
      (success one-shot ~ 150-290, terminating) -> loitering beat completing.
  F5  EMA-lagged progress x instantaneous gates -> wrist-pulse aliasing kept
      77% of progress credit.

v11 architecture — each mechanism closes a failure class *by construction*:

  1. REAL CONTACTS      data.contact geom-pair iteration (cube body x fingertip
                        bodies).  Closes F1.  No distance proxies anywhere.
  2. PALM-FRAME         Progress attributed via counterfactual decomposition:
     ATTRIBUTION        theta_hold = angle the red face WOULD have if the palm
                        had not moved this step.  Wrist-carry contributes
                        exactly zero credit.  Closes F3/F5 (no action-based
                        gate exists to alias against).
  3. ANTI-RATCHET       Positive progress credit is gated & attenuated;
     INVARIANT          negative progress is ALWAYS charged at full rate.
                        Over any closed cycle in theta, sum(reward) <= 0.
                        No rock-the-cube / wrist-out-finger-back farm exists.
  4. POTENTIAL SHAPING  All dense state terms (align, palm-centering, contact,
                        proximity) enter ONLY through gamma*Phi(s')-Phi(s).
                        Standing income at any fixed state is -(1-gamma)*Phi
                        <= 0.  Closes F4 (loitering pays nothing, ever).
  5. NON-TERMINAL       Success does not end the episode.  While the cube is
     SUCCESS            red-face-up, fingertip-secured, palm-level and still,
                        the agent earns goal_stream_weight/step — the single
                        largest income in the MDP.  Optimal policy = rotate
                        fast, then hold stably.  Closes F2 (flat stream: cycling
                        out of the zone strictly forfeits income) and makes the
                        post-success drop penalty live code (v10's was dead).

Verified physics constants (sources cited, do not change blindly):
  - cube half-extent 0.018 m         scene_right_cube_orientation.xml size attr
  - fingertip pad ~0.040-0.044 m distal of tracked body origin
                                     M-FingerTipAssembly_M-DP-Skin.stl bbox
  - realistic tip-origin-to-cube-center distance at contact: 0.045-0.060 m
  - base env drop termination z < 0.05, goal-or-drop only   task_envs.py:36,239
  - wrist joint "right_wrist", range [-1.134, 0.611] rad    orcahand_right_body.xml:81
  - actions are POSITION TARGETS — never used as a motion/force proxy here
  - friction cone: cube friction 1.2 -> gravity roll needs > ~50 deg palm tilt

Wrapper contract:
  - Overrides reward entirely (base reward discarded).
  - Overrides termination: ONLY drop terminates (verified base env terminates
    on goal|drop only — we intentionally continue past goal for hold training).
  - info["is_success"] = stable success achieved this episode AND cube never
    dropped.  Sticky.  Read by Monitor/curriculum promotion at episode end.
  - Exposes `success_bonus` attribute (CurriculumWrapper syncs it per chapter).
"""

import gymnasium as gym
import numpy as np

# Fingertip BODIES on the ORCA right hand.  Contact is detected by mapping
# contact geoms -> parent body, so every collision geom under these bodies
# counts (the *_IP_IP meshes collide; the *Skin* geoms have contype=0 and
# never appear in data.contact).  NOTE: "424a8e75" IS the ring fingertip —
# the CAD export reused the middle-finger part name; verified by its joint
# (right_r-pip) in orcahand_right_body.xml.
FINGERTIP_BODIES = (
    "right_T-DP_b7429e50",                    # Thumb distal phalanx
    "right_I-FingerTipAssembly_ec49c16c",     # Index fingertip
    "right_M-FingerTipAssembly_34afb748",     # Middle fingertip
    "right_M-FingerTipAssembly_424a8e75",     # Ring fingertip (M- name reuse)
    "right_P-FingerTipAssembly_cd219176",     # Pinky fingertip
)

WRIST_JOINT_NAME = "right_wrist"
CUBE_BODY_NAME = "task_cube"
WRIST_ACTION_IDX = 0
WORLD_UP = np.array([0.0, 0.0, 1.0])


def finger_attributed_progress(theta_prev, theta_now, n_world_now,
                               R_palm_prev, R_palm_now):
    """Counterfactual decomposition of the red-face angle change.

    Returns (d_theta_total, d_theta_finger), both positive-toward-goal:
      d_theta_total  = theta_prev - theta_now
      d_theta_finger = theta_prev - theta_hold, where theta_hold is the angle
                       the red face would make with world-up if the cube had
                       moved relative to the palm exactly as it did, but the
                       palm had stayed at its previous world pose.

    Pure wrist rotation (cube carried rigidly): the cube's palm-relative
    orientation is unchanged -> theta_hold == theta_prev -> finger part = 0.
    Pure finger manipulation (palm stationary): R_palm_now == R_palm_prev
    -> theta_hold == theta_now -> finger part = total.  The two parts sum to
    the total exactly (telescoping identity), so nothing is double-counted.

    Pure function of poses — kept module-level so it can be unit-tested
    without MuJoCo.
    """
    n_rel_now = R_palm_now.T @ n_world_now          # red normal in palm frame
    n_hold = R_palm_prev @ n_rel_now                # replay under frozen palm
    theta_hold = float(np.arccos(np.clip(n_hold @ WORLD_UP, -1.0, 1.0)))
    return theta_prev - theta_now, theta_prev - theta_hold


class ProductionRewardWrapper(gym.Wrapper):
    """v11 reward: potential shaping + attributed progress + hold stream."""

    def __init__(
        self,
        env,
        # --- Discount (MUST match PPO's gamma for exact potential shaping;
        #     Ng et al. 1999 policy-invariance holds only when they agree) ---
        gamma: float = 0.99,
        # --- Potential Phi(s) weights.  These produce NO standing income:
        #     only gamma*Phi(s')-Phi(s) is paid.  Magnitudes are one-time
        #     "energy" released along the trajectory, so they are set to be
        #     tie-breakers (a few units total), not drivers. ---
        pot_align_weight: float = 3.0,    # exp(-theta/1.0); 0.62 at 90deg —
        pot_align_sigma: float = 1.0,     # gradient alive at ALL chapter angles
        pot_pos_weight: float = 1.0,      # palm-centering; potential form means
        pot_pos_sigma: float = 0.05,      # gait excursions are refunded on return
        pot_contact_weight: float = 0.5,  # 0.25 per fingertip up to 2 — one-time
        pot_prox_weight: float = 0.5,     # reach signal for early exploration
        prox_contact_dist: float = 0.045, # tip-ORIGIN-to-center at real contact
                                          # (18mm half-extent + ~27mm of the
                                          #  40-44mm pad offset projected); the
                                          #  v10 value 0.020 was unsatisfiable
        prox_sigma: float = 0.03,
        # --- Progress (the pre-goal driver) ---
        progress_weight: float = 50.0,
        progress_clip: float = 0.1,       # 0.1 rad/step = 286 deg/s cap; a drop
                                          # transient earns <= alpha*50*0.1=1.5
                                          # vs -50 terminal (v9 drop bug dead)
        progress_ema_alpha: float = 0.3,  # ~4.3x delta-noise reduction; applied
                                          # AFTER gating (v10 F5 fix)
        contact_min_fingers: int = 2,     # full credit at >=2 fingertips; 1 tip
                                          # (thumb-topple) earns 50%
        tilt_sigma: float = 0.4,          # rad.  Gravity roll needs >~0.87 rad
                                          # tilt (friction 1.2 -> 50deg cone):
                                          # gate there = exp(-(0.87/0.4)^2)=0.009.
                                          # Legit stabilization at 0.2 rad: 0.78.
        # --- Goal hold stream (the post-goal driver; NON-terminal) ---
        goal_stream_weight: float = 6.0,  # /step in-zone, secured, level, still.
                                          # Must exceed max loiter income (== 0
                                          # by potential construction) — see
                                          # REWARD_REPORT_v11 economics proof
        v_scale: float = 0.05,            # m/s   characteristic linear vel
        w_scale: float = 1.5,             # rad/s characteristic angular vel
        hold_steps: int = 10,             # 0.2 s @50Hz marks is_success; the
                                          # stream pays from the FIRST zone step
                                          # so this is NOT an exploration gate
                                          # (v0 failure mode avoided)
        hold_tilt_max: float = 0.35,      # rad, wrist deviation allowed in hold
        hold_vel_max: float = 1.0,        # vel_metric ceiling to count as held
        success_bonus: float = 100.0,     # ONE-TIME, sticky, on first stable
                                          # hold.  CurriculumWrapper may override
                                          # per chapter; any value is unfarmable
        # --- Drop ---
        drop_penalty: float = 50.0,
        drop_post_success_extra: float = 50.0,  # LIVE in v11 (episodes continue
                                                # past success); main deterrent
                                                # is forfeiting the 6/step stream
        # --- Smoothness regularisers (cosmetic; deterrence lives in the
        #     attribution/gates above, never in action costs) ---
        wrist_rate_weight: float = 0.01,
        finger_rate_weight: float = 0.001,
        # Deleted vs v10 (audit findings, see module docstring):
        #   r_alive (loiter fuel), finger_active_bonus (paid for noise),
        #   action magnitude penalty (0.03% of per-step reward, clutter),
        #   drop risk ramp (constants contradicted env; height drift is covered
        #   continuously by the 3-D position potential), wrist action gate (F3),
        #   counter-scaled hold ramp (F2), terminal success (F4).
    ):
        super().__init__(env)
        self.gamma = gamma
        self.pot_align_weight = pot_align_weight
        self.pot_align_sigma = pot_align_sigma
        self.pot_pos_weight = pot_pos_weight
        self.pot_pos_sigma = pot_pos_sigma
        self.pot_contact_weight = pot_contact_weight
        self.pot_prox_weight = pot_prox_weight
        self.prox_contact_dist = prox_contact_dist
        self.prox_sigma = prox_sigma
        self.progress_weight = progress_weight
        self.progress_clip = progress_clip
        self.progress_ema_alpha = progress_ema_alpha
        self.contact_min_fingers = contact_min_fingers
        self.tilt_sigma = tilt_sigma
        self.goal_stream_weight = goal_stream_weight
        self.v_scale = v_scale
        self.w_scale = w_scale
        self.hold_steps = hold_steps
        self.hold_tilt_max = hold_tilt_max
        self.hold_vel_max = hold_vel_max
        self.success_bonus = success_bonus
        self.drop_penalty = drop_penalty
        self.drop_post_success_extra = drop_post_success_extra
        self.wrist_rate_weight = wrist_rate_weight
        self.finger_rate_weight = finger_rate_weight

        # Per-episode state
        self._prev_action = None
        self._prev_potential = None
        self._prev_theta = None
        self._prev_palm_R = None
        self._default_cube_pos = None
        self._wrist_qpos0 = 0.0
        self._progress_ema = 0.0
        self._hold_counter = 0
        self._achieved = False
        self._episode_successes = 0

        # Model ID caches (resolved lazily; re-resolved if the model changes)
        self._model_ref = None
        self._cube_body_id = None
        self._fingertip_body_ids = frozenset()
        self._geom_bodyid = None
        self._wrist_qpos_adr = None
        self._palm_body_id = None

    # ------------------------------------------------------------------
    # model plumbing
    # ------------------------------------------------------------------

    def _resolve_ids(self):
        model = self.unwrapped.model
        self._cube_body_id = model.body(CUBE_BODY_NAME).id
        self._fingertip_body_ids = frozenset(
            model.body(name).id for name in FINGERTIP_BODIES
        )
        self._geom_bodyid = np.array(model.geom_bodyid)
        wrist = model.joint(WRIST_JOINT_NAME)
        self._wrist_qpos_adr = int(model.jnt_qposadr[wrist.id])
        # Palm frame := the body the wrist joint articulates.  Everything
        # distal of the wrist (the whole hand) is posed relative to it, so
        # cube rotation measured in this frame excludes wrist rotation by
        # construction.  Derived from the model — no body-name guessing.
        self._palm_body_id = int(model.jnt_bodyid[wrist.id])
        self._model_ref = id(model)

    def _ensure_ids(self):
        if self._model_ref != id(self.unwrapped.model):
            self._resolve_ids()

    def _count_fingertip_contacts(self) -> int:
        """Distinct fingertip bodies in ACTUAL MuJoCo contact with the cube.

        Iterates data.contact (typically < 30 entries) and maps each contact
        geom to its parent body.  This is exact collision detection — the v10
        xpos-distance proxy (F1) is dead and must never return.
        """
        data = self.unwrapped.data
        cube = self._cube_body_id
        tips = self._fingertip_body_ids
        touching = set()
        for i in range(data.ncon):
            con = data.contact[i]
            b1 = int(self._geom_bodyid[con.geom1])
            b2 = int(self._geom_bodyid[con.geom2])
            if b1 == cube and b2 in tips:
                touching.add(b2)
            elif b2 == cube and b1 in tips:
                touching.add(b1)
        return len(touching)

    def _fingertip_center_dists(self):
        data = self.unwrapped.data
        cube_pos = data.xpos[self._cube_body_id]
        return [float(np.linalg.norm(data.xpos[b] - cube_pos))
                for b in self._fingertip_body_ids]

    def _palm_R(self) -> np.ndarray:
        return self.unwrapped.data.xmat[self._palm_body_id].reshape(3, 3).copy()

    def _wrist_tilt(self) -> float:
        q = float(self.unwrapped.data.qpos[self._wrist_qpos_adr])
        return q - self._wrist_qpos0

    # ------------------------------------------------------------------
    # potential
    # ------------------------------------------------------------------

    def _potential(self, theta, pos_err, n_contacts, tip_dists) -> float:
        """Phi(s) >= 0, function of state only (never of actions).

        Paid exclusively as gamma*Phi(s')-Phi(s): exact telescoping means the
        integral over any trajectory segment depends only on its endpoints —
        no path through state space farms it, and the make/break contact
        potential refunds itself symmetrically.
        """
        phi_align = self.pot_align_weight * np.exp(-theta / self.pot_align_sigma)
        phi_pos = self.pot_pos_weight * np.exp(-pos_err / self.pot_pos_sigma)
        phi_contact = self.pot_contact_weight * min(
            n_contacts, self.contact_min_fingers) / self.contact_min_fingers
        reach = [np.exp(-max(d - self.prox_contact_dist, 0.0) / self.prox_sigma)
                 for d in tip_dists]
        phi_prox = self.pot_prox_weight * float(np.mean(reach))
        return float(phi_align + phi_pos + phi_contact + phi_prox)

    # ------------------------------------------------------------------
    # gym.Wrapper overrides
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ensure_ids()

        self._prev_action = np.zeros(self.env.action_space.shape,
                                     dtype=np.float32)
        self._default_cube_pos = np.asarray(info["cube_pos"], dtype=np.float64).copy()
        self._wrist_qpos0 = float(
            self.unwrapped.data.qpos[self._wrist_qpos_adr])
        self._prev_theta = float(info["red_face_up_angle_rad"])
        self._prev_palm_R = self._palm_R()
        self._progress_ema = 0.0
        self._hold_counter = 0
        self._achieved = False
        self._episode_successes = 0

        n_c = self._count_fingertip_contacts()
        self._prev_potential = self._potential(
            self._prev_theta, 0.0, n_c, self._fingertip_center_dists())

        info["is_success"] = False
        return obs, info

    def step(self, action):
        # Base env termination is goal|drop only (task_envs.py:239, verified).
        # We deliberately mask the goal half — v11 success is non-terminal —
        # and re-derive the drop half from info["dropped"].
        obs, _base_reward, _base_terminated, truncated, info = self.env.step(action)
        action = np.asarray(action, dtype=np.float32)

        theta = float(info["red_face_up_angle_rad"])
        n_world = np.asarray(info["red_face_world_normal"], dtype=np.float64)
        cube_pos = np.asarray(info["cube_pos"], dtype=np.float64)
        pos_err = float(np.linalg.norm(cube_pos - self._default_cube_pos))
        dropped = bool(info["dropped"])
        palm_R = self._palm_R()
        tilt = self._wrist_tilt()
        n_contacts = self._count_fingertip_contacts()
        tip_dists = self._fingertip_center_dists()

        lin_v = float(np.linalg.norm(info["cube_qvel"][:3]))
        ang_v = float(np.linalg.norm(info["cube_qvel"][3:6]))
        vel_metric = float(np.sqrt((lin_v / self.v_scale) ** 2
                                   + (ang_v / self.w_scale) ** 2))

        # ============================================================
        # 1. POTENTIAL SHAPING — gamma*Phi(s') - Phi(s).  Zero standing
        #    income anywhere; gait excursions refunded on return.
        # ============================================================
        potential = self._potential(theta, pos_err, n_contacts, tip_dists)
        r_potential = self.gamma * potential - self._prev_potential
        self._prev_potential = potential

        # ============================================================
        # 2. PROGRESS — palm-frame attributed, gated credit, full debit
        # ============================================================
        d_total, d_finger = finger_attributed_progress(
            self._prev_theta, theta, n_world, self._prev_palm_R, palm_R)
        self._prev_theta = theta
        self._prev_palm_R = palm_R

        gate_contact = min(1.0, n_contacts / self.contact_min_fingers)
        gate_tilt = float(np.exp(-(tilt / self.tilt_sigma) ** 2))

        if d_total > 0.0:
            # Credit only the finger-attributed share of REAL world progress
            # (share <= 1 caps credit at actual progress — the mixed
            # wrist-away/finger-toward combination cannot mint reward),
            # attenuated by grasp quality and palm levelness.
            share = float(np.clip(d_finger / max(d_total, 1e-8), 0.0, 1.0))
            p_step = (min(d_total, self.progress_clip)
                      * share * gate_contact * gate_tilt)
        else:
            # ANTI-RATCHET INVARIANT: regress is charged ungated, full rate.
            # Over any closed theta-cycle, gated credit <= ungated debit,
            # so oscillation strategies are <= 0 by construction.
            p_step = max(d_total, -self.progress_clip)

        self._progress_ema = (self.progress_ema_alpha * p_step
                              + (1 - self.progress_ema_alpha) * self._progress_ema)
        r_progress = self.progress_weight * self._progress_ema

        # ============================================================
        # 3. GOAL HOLD STREAM — non-terminal success (see docstring #5)
        # ============================================================
        in_zone = theta < np.deg2rad(15.0)
        secured = n_contacts >= 1          # at least one fingertip on the cube:
                                           # rules out toss-catch / palm-only
                                           # gravity rest counting as "held"
        stability = float(np.exp(-vel_metric))
        r_goal = 0.0
        if in_zone and secured and not dropped:
            r_goal = self.goal_stream_weight * stability * gate_tilt

        hold_ok = (in_zone and secured and not dropped
                   and abs(tilt) <= self.hold_tilt_max
                   and vel_metric <= self.hold_vel_max)
        self._hold_counter = self._hold_counter + 1 if hold_ok else 0

        r_bonus = 0.0
        if self._hold_counter >= self.hold_steps and not self._achieved:
            self._achieved = True            # sticky; bonus fires exactly once
            self._episode_successes += 1
            r_bonus = self.success_bonus

        # ============================================================
        # 4. DROP — the only termination this wrapper issues
        # ============================================================
        r_drop = 0.0
        terminated = False
        if dropped:
            r_drop = self.drop_penalty
            if self._achieved:
                r_drop += self.drop_post_success_extra
            terminated = True

        # ============================================================
        # 5. SMOOTHNESS — cosmetic regularisers only
        # ============================================================
        d_a = action - self._prev_action
        r_wrist_rate = self.wrist_rate_weight * float(d_a[WRIST_ACTION_IDX] ** 2)
        r_finger_rate = self.finger_rate_weight * float(
            np.sum(d_a[WRIST_ACTION_IDX + 1:] ** 2))
        self._prev_action = action.copy()

        reward = (r_potential + r_progress + r_goal + r_bonus
                  - r_drop - r_wrist_rate - r_finger_rate)

        # ============================================================
        # LOGGING — every channel, every gate, for W&B.  The v10 lesson:
        # a dead component is invisible unless you look at exactly this.
        # ============================================================
        info["reward_breakdown"] = {
            "potential": float(r_potential),
            "potential_abs": float(potential),
            "progress": float(r_progress),
            "progress_step_raw": float(p_step),
            "progress_d_total": float(d_total),
            "progress_d_finger": float(d_finger),
            "gate_contact": float(gate_contact),
            "gate_tilt": float(gate_tilt),
            "goal_stream": float(r_goal),
            "success_bonus": float(r_bonus),
            "drop": float(-r_drop),
            "wrist_rate": float(-r_wrist_rate),
            "finger_rate": float(-r_finger_rate),
            "total": float(reward),
        }
        info["contact_count"] = int(n_contacts)
        info["wrist_tilt"] = float(tilt)
        info["hold_counter"] = int(self._hold_counter)
        info["vel_metric"] = float(vel_metric)
        info["pos_error"] = float(pos_err)
        info["fingertip_dists"] = tip_dists
        info["episode_successes"] = int(self._episode_successes)
        # Sticky, but a drop revokes it: the promotion gate must count only
        # episodes that end still holding the cube (behavioral requirement 4).
        info["is_success"] = bool(self._achieved and not dropped)

        return obs, float(reward), terminated, truncated, info
