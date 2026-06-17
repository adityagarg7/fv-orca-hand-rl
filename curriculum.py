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
* ``GravityCurriculumCallback`` ramps the gravity magnitude from ``g_start`` to
  ``g_final`` over the first ``warmup_steps`` timesteps, holds it at full afterwards,
  and pushes the value to every sub-env via ``VecEnv.env_method``.

To disable the curriculum, set ``g_start == g_final`` (constant gravity).
"""

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


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
    """Anneal gravity magnitude from ``g_start`` to ``g_final`` over ``warmup_steps``.

    ``g_start`` / ``g_final`` are positive magnitudes (m/s^2); the downward z gravity
    pushed to each env is ``-g``. After ``warmup_steps`` total timesteps gravity is
    held at ``-g_final``. The value is broadcast to all sub-envs via ``env_method``
    every ``update_freq`` callback steps and logged as ``curriculum/gravity``.
    """

    def __init__(self, g_start: float, g_final: float, warmup_steps: int,
                 update_freq: int = 20):
        super().__init__()
        self.g_start = float(g_start)
        self.g_final = float(g_final)
        self.warmup_steps = max(1, int(warmup_steps))
        self.update_freq = max(1, int(update_freq))

    def _current_g(self) -> float:
        frac = min(1.0, self.num_timesteps / self.warmup_steps)
        return self.g_start + frac * (self.g_final - self.g_start)

    def _push(self) -> float:
        g = self._current_g()
        self.training_env.env_method("set_gravity", -g)
        self.logger.record("curriculum/gravity", g)
        return g

    def _on_training_start(self) -> None:
        # Apply the starting gravity immediately so the first rollout uses it.
        self._push()

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            self._push()
        return True
