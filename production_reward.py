"""
Production reward wrapper for OrcaHandRightCubeOrientation.

The baseline reward gives only per-step alignment credit, so the agent farms a
partial alignment for the whole episode instead of solving and terminating. This
wrapper adds a large terminal success bonus that dominates farming, an exponential
alignment kernel, position tracking, an alive bonus, a drop penalty, and action
regularisation. The success bonus is granted only after the cube is held correctly
oriented and stable for `required_hold_steps` consecutive steps.
"""

import gymnasium as gym
import numpy as np


class ProductionRewardWrapper(gym.Wrapper):
    """Wraps OrcaHandRightCubeOrientation with a production-grade reward function.

    This does NOT modify the original orca_sim source code. It intercepts the
    reward after each step and replaces it with a properly shaped signal.
    """

    def __init__(
        self,
        env,
        # --- Alignment (primary task signal) ---
        align_weight: float = 2.0,
        align_sigma: float = 0.3,       # radians; controls steepness of exponential
        # --- Success bonus (anti-hacking) ---
        success_bonus: float = 250.0,   # must dominate 200 steps of farming
        # --- Position keeping (keep cube in palm) ---
        pos_weight: float = 0.1,        # small auxiliary signal
        pos_sigma: float = 0.02,         # metres
        # --- Alive bonus (gentle incentive to not drop) ---
        alive_bonus: float = 0.02,
        # --- Drop penalty ---
        drop_penalty: float = 5.0,
        # --- Action regularisation (smooth motions) ---
        action_rate_weight: float = 0.02,    # penalise jerky action changes
        action_mag_weight: float = 0.001,    # penalise large actions
    ):
        super().__init__(env)
        self.align_weight = align_weight
        self.align_sigma = align_sigma
        self.success_bonus = success_bonus
        self.pos_weight = pos_weight
        self.pos_sigma = pos_sigma
        self.alive_bonus = alive_bonus
        self.drop_penalty = drop_penalty
        self.action_rate_weight = action_rate_weight
        self.action_mag_weight = action_mag_weight

        self.required_hold_steps = 15
        self._hold_steps = 0
        self._prev_action = None
        self._default_cube_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        # Store initial cube position as target position
        self._default_cube_pos = info["cube_pos"].copy()
        self._hold_steps = 0
        return obs, info

    def step(self, action):
        obs, _original_reward, base_terminated, truncated, info = self.env.step(action)

        # ---- 1. Alignment reward (exponential kernel) ----
        # angle_error is in [0, π]. At goal, angle_error = 0.
        angle_error = info["red_face_up_angle_rad"]
        r_align = self.align_weight * np.exp(-angle_error / self.align_sigma)

        # ---- 2. Position tracking (keep cube centred) ----
        pos_error = 0.0
        if self._default_cube_pos is not None:
            pos_error = np.linalg.norm(info["cube_pos"] - self._default_cube_pos)
            r_pos = self.pos_weight * np.exp(-pos_error / self.pos_sigma)
        else:
            r_pos = 0.0

        # ---- 3. Stable Grasp Check & Success Bonus ----
        linear_vel = np.linalg.norm(info["cube_qvel"][:3])
        angular_vel = np.linalg.norm(info["cube_qvel"][3:6])
        
        # Cube must be oriented correctly AND stable
        is_stable = (linear_vel < 0.1) and (angular_vel < 0.5) and (pos_error < 0.05)
        
        if info["is_success"] and is_stable:
            self._hold_steps += 1
        else:
            self._hold_steps = 0

        # Override termination and success bonus
        terminated = False
        r_success = 0.0
        
        if self._hold_steps >= self.required_hold_steps:
            r_success = self.success_bonus
            terminated = True
        elif info["dropped"]:
            terminated = True
            
        # Update info for debugging
        info["hold_steps"] = self._hold_steps
        info["is_stable"] = is_stable

        # ---- 4. Alive bonus ----
        r_alive = self.alive_bonus if not info["dropped"] else 0.0

        # ---- 5. Drop penalty ----
        r_drop = self.drop_penalty if info["dropped"] else 0.0

        # ---- 6. Action regularisation ----
        action = np.asarray(action, dtype=np.float32)
        r_action_rate = self.action_rate_weight * np.sum((action - self._prev_action) ** 2)
        r_action_mag = self.action_mag_weight * np.sum(action ** 2)
        self._prev_action = action.copy()

        # ---- Combine ----
        reward = (
            r_align
            + r_pos
            + r_success
            + r_alive
            - r_drop
            - r_action_rate
            - r_action_mag
        )

        # Inject breakdown into info for debugging
        info["reward_breakdown"] = {
            "align": float(r_align),
            "pos": float(r_pos),
            "success": float(r_success),
            "alive": float(r_alive),
            "drop": float(-r_drop),
            "action_rate": float(-r_action_rate),
            "action_mag": float(-r_action_mag),
            "total": float(reward),
        }

        # Override base environment's 'is_success' to reflect true stability
        info["is_success"] = (self._hold_steps >= self.required_hold_steps)

        return obs, float(reward), terminated, truncated, info
