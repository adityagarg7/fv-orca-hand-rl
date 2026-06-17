"""
Action constraints for the ORCA cube-reorientation task.

``WristClampWrapper`` limits the wrist actuator -- the policy's only palm-tilt DOF on
the otherwise fixed-base hand -- to a narrow band around its neutral (palm-up) angle.

Why a hard clamp rather than a reward penalty
---------------------------------------------
The "wrist dump" cheat (flex the wrist forward so the cube tips off the palm and
gravity flips it red-up) is essentially *geometric*: it works at any gravity as long
as the palm can tilt far enough, so a gravity curriculum only slows it. A per-step
*reward penalty* on the wrist froze the policy -- holding everything still to avoid the
penalty beat attempting the hard reorientation. A hard *action* clamp removes the
cheat's mechanism without penalizing anything: the policy still earns full reward for
solving, it simply cannot tilt the palm enough to dump the cube, so the fingers must do
the work. The clamp does not change the action-space dimension (the wrist component is
just clipped before it reaches the env), so the policy/checkpoints are unaffected
otherwise.

The wrist actuator is index 0 of the 17-dim control vector. The joint it drives, and
the neutral palm-up angle, are resolved from the model itself (no hardcoded names), so
the clamp self-calibrates to whatever reset pose the env uses.
"""

import gymnasium as gym
import mujoco
import numpy as np


class WristClampWrapper(gym.Wrapper):
    """Clip the wrist action to ``[neutral - band, neutral + band]`` (radians).

    ``neutral`` is captured from the wrist joint angle at each ``reset`` (the palm-up
    resting pose); ``band`` is the half-width of the allowed wrist range. The clip is
    also bounded by the actuator's own control range. Apply this as the OUTERMOST custom
    wrapper so it clips the policy's action before any inner wrapper/env sees it.
    """

    def __init__(self, env: gym.Env, band: float = 0.15, wrist_actuator_index: int = 0):
        super().__init__(env)
        self.band = float(band)
        self.wrist_index = int(wrist_actuator_index)
        self._neutral = 0.0

        model = env.unwrapped.model
        # Resolve the joint this actuator drives via the transmission table, so we never
        # depend on a hardcoded joint/actuator name. For a joint actuator, trnid[i, 0] is
        # the joint id; jnt_qposadr maps that to the joint's slot in qpos.
        self._wrist_qposadr = None
        if int(model.actuator_trntype[self.wrist_index]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            jid = int(model.actuator_trnid[self.wrist_index, 0])
            if jid >= 0:
                self._wrist_qposadr = int(model.jnt_qposadr[jid])
        if self._wrist_qposadr is None:
            print(f"[action_wrappers] could not resolve the joint for actuator "
                  f"{self.wrist_index}; clamping the wrist around 0.0 rad.")
        # Actuator control range, so the clamp never exceeds the valid command range.
        self._ctrl_low = float(model.actuator_ctrlrange[self.wrist_index, 0])
        self._ctrl_high = float(model.actuator_ctrlrange[self.wrist_index, 1])

    def _wrist_angle(self) -> float:
        if self._wrist_qposadr is None:
            return self._neutral
        return float(self.env.unwrapped.data.qpos[self._wrist_qposadr])

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._neutral = self._wrist_angle()  # palm-up resting angle
        return obs, info

    def step(self, action):
        action = np.array(action, dtype=np.float32)  # copy; don't mutate the caller's array
        lo = max(self._ctrl_low, self._neutral - self.band)
        hi = min(self._ctrl_high, self._neutral + self.band)
        action[self.wrist_index] = np.clip(action[self.wrist_index], lo, hi)
        return self.env.step(action)
