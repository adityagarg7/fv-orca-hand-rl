"""
Gravity curriculum for the ORCA cube-reorientation task.

Why this exists
---------------
A policy trained on the alignment reward learns to flex the actuated wrist forward
and let *gravity* tip the cube off the palm so it flips red-face-up -- a specification
cheat, not the intended in-hand finger manipulation. Penalizing wrist/cube motion
directly (see git history) tends to *freeze* the policy: avoiding the penalty becomes
more valuable than attempting the hard reorientation, so the cube is held motionless
and success collapses.

A more robust fix removes the exploit at its source. Train under **reduced gravity**
first: when gravity is weak the wrist-dump barely works (little gravitational torque
to flip the cube) and the cube does not readily fall, so the only way to reorient it
is with the fingers. Then **anneal gravity up to full** (-9.81) once finger
manipulation is established. This is the gravity curriculum of Chen et al.,
"A System for General In-Hand Object Re-Orientation" (CoRL 2021), adapted to this
PPO / Stable-Baselines3 setup.

* ``GravityCurriculumWrapper`` lets the live MuJoCo gravity be set per sub-env and
  starts each env at the initial (reduced) gravity.
* ``GravityCurriculumCallback`` is **performance-gated**: it holds gravity at the
  current level until the recent success rate *at that level* reaches a threshold,
  then steps the gravity magnitude up (toward ``g_final``) and resets the success
  measurement. This guarantees the policy actually masters each gravity level before
  it is made harder -- unlike a fixed time schedule, which can ramp into full gravity
  (and hand back the wrist-dump cheat) before finger manipulation is learned.

To disable the curriculum, set ``g_start == g_final`` (constant gravity).
"""

from collections import deque

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class GravityCurriculumWrapper(gym.Wrapper):
    """Allow the MuJoCo gravity (z component) to be set at runtime, per env.

    ``gravity`` is the signed z value (negative = downward), e.g. -9.81 for full
    gravity. It lives in the MuJoCo model (``model.opt.gravity``), which persists
    across ``mj_resetData``, so setting it once sticks; we re-apply on reset as a
    safeguard.
    """

    def __init__(self, env: gym.Env, gravity: float = -9.81) -> None:
        super().__init__(env)
        self._gravity = float(gravity)
        self._apply()

    def _apply(self) -> None:
        self.env.unwrapped.model.opt.gravity[2] = self._gravity

    def set_gravity(self, gravity: float) -> float:
        """Set the z gravity (signed). Returns the value set (so env_method reports it)."""
        self._gravity = float(gravity)
        self._apply()
        return self._gravity

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._apply()
        return obs, info


class GravityCurriculumCallback(BaseCallback):
    """Performance-gated gravity curriculum.

    Holds gravity at the current level until the recent success rate at that level
    reaches ``success_threshold`` (measured over the episodes completed since the last
    promotion, requiring at least ``min_episodes`` of them for a reliable estimate),
    then raises the gravity magnitude by ``step`` (clamped at ``g_final``) and clears
    the success buffer so the next level is judged on its own. ``g_start`` / ``g_final``
    / ``step`` are positive magnitudes (m/s^2); the downward z gravity pushed to each
    env is ``-g``. Logs ``curriculum/gravity``, ``curriculum/success_at_level`` and
    ``curriculum/level_episodes``.
    """

    def __init__(self, g_start: float, g_final: float, success_threshold: float = 0.75,
                 step: float = 1.0, min_episodes: int = 50, window: int = 100,
                 verbose: int = 1):
        super().__init__(verbose=verbose)
        self.g_final = float(g_final)
        self.step = float(step)
        self.success_threshold = float(success_threshold)
        self.min_episodes = int(min_episodes)
        self._g = float(g_start)
        self._succ: deque = deque(maxlen=int(window))

    def _set_gravity(self, g: float) -> None:
        self._g = min(self.g_final, float(g))
        self.training_env.env_method("set_gravity", -self._g)

    def _on_training_start(self) -> None:
        # Apply the starting gravity immediately so the first rollout uses it.
        self._set_gravity(self._g)

    def _on_step(self) -> bool:
        # Record success/failure for each episode that finished this step.
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done and "is_success" in info:
                self._succ.append(float(bool(info["is_success"])))

        level_success = safe_mean(list(self._succ)) if len(self._succ) else 0.0

        # Promote only once the policy is competent at the current gravity.
        if (self._g < self.g_final
                and len(self._succ) >= self.min_episodes
                and level_success >= self.success_threshold):
            self._set_gravity(self._g + self.step)
            self._succ.clear()
            if self.verbose:
                print(f"[gravity-curriculum] success {level_success:.0%} at this level "
                      f"-> raising gravity to {self._g:.2f} m/s^2")

        self.logger.record("curriculum/gravity", self._g)
        self.logger.record("curriculum/success_at_level", level_success)
        self.logger.record("curriculum/level_episodes", len(self._succ))
        return True
