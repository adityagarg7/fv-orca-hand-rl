"""
Gym wrapper that spawns the cube at the curriculum's current difficulty angle.

Wraps OrcaHandRightCubeOrientation → ProductionRewardWrapper → CurriculumWrapper.

The wrapper intercepts reset() to:
  1. Sample a spawn angle from the CurriculumManager
  2. Convert it to a MuJoCo quaternion with RANDOM tilt axis
  3. Pass it as cube_quat to the underlying environment
  4. Sync the per-chapter success_bonus to the reward wrapper
  5. Track episode success for auto-promotion decisions

BUGFIXES from validation:
  - BUG 1: Now samples random tilt axis (not just X) for diverse training
  - BUG 2: Syncs success_bonus from chapter config to ProductionRewardWrapper
"""

import gymnasium as gym
import numpy as np
from curriculum import CurriculumManager
from production_reward import ProductionRewardWrapper


class CurriculumWrapper(gym.Wrapper):
    """Intercepts reset() to control cube spawn angle via curriculum."""

    def __init__(self, env: gym.Env, curriculum: CurriculumManager):
        super().__init__(env)
        self.curriculum = curriculum

        self._episode_steps = 0
        self._current_spawn_angle_deg = 0.0
        self._rng = np.random.default_rng()

    def _find_reward_wrapper(self) -> ProductionRewardWrapper | None:
        """Walk the wrapper chain to find the ProductionRewardWrapper."""
        e = self.env
        while e is not None:
            if isinstance(e, ProductionRewardWrapper):
                return e
            e = getattr(e, "env", None)
        return None

    def _angle_to_random_quat(self, angle_deg: float) -> np.ndarray:
        """Convert a tilt angle to a quaternion with a random tilt axis.

        Samples a random axis in the XY plane (perpendicular to UP=Z),
        then constructs the quaternion for rotation by `angle_deg` around
        that axis.  This ensures the red_face_up_angle equals angle_deg
        regardless of the chosen axis direction.

        Also composes with a random spin around Z (doesn't change tilt)
        to diversify the cube's horizontal orientation.
        """
        theta = np.radians(angle_deg)

        # Random tilt axis in XY plane (perpendicular to Z=UP)
        phi = self._rng.uniform(0, 2 * np.pi)
        ax = np.cos(phi)
        ay = np.sin(phi)

        # Tilt quaternion: rotate by theta around (ax, ay, 0)
        q_tilt = np.array([
            np.cos(theta / 2),
            np.sin(theta / 2) * ax,
            np.sin(theta / 2) * ay,
            0.0,
        ])

        # Random spin around Z (diversifies horizontal orientation)
        psi = self._rng.uniform(0, 2 * np.pi)
        q_spin = np.array([np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)])

        # Compose: tilt first, then spin  (q_spin * q_tilt)
        return self._quat_multiply(q_spin, q_tilt)

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two MuJoCo-convention quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def reset(self, **kwargs):
        # Sample spawn angle from curriculum
        angle_deg = self.curriculum.sample_spawn_angle_deg()
        self._current_spawn_angle_deg = angle_deg

        self._episode_steps = 0

        # Convert to quaternion with RANDOM tilt axis
        cube_quat = self._angle_to_random_quat(angle_deg)

        # Inject into reset options
        options = kwargs.pop("options", {}) or {}
        options["cube_quat"] = cube_quat
        kwargs["options"] = options

        # BUGFIX: sync per-chapter success_bonus to reward wrapper
        reward_wrapper = self._find_reward_wrapper()
        if reward_wrapper is not None:
            reward_wrapper.success_bonus = self.curriculum.current_chapter.success_bonus

        obs, info = self.env.reset(**kwargs)

        # Add curriculum info
        info["curriculum_spawn_angle_deg"] = angle_deg
        info["curriculum_chapter"] = self.curriculum.current_chapter.name

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_steps += 1

        # Track success at END of episode only (not "ever succeeded").
        # The old "any-step" metric was too lenient — the cube would
        # momentarily wobble through the success zone on easy chapters,
        # giving false 100% success rates and premature promotion.
        # Now the agent must HOLD the orientation at episode end.
        episode_success_at_end = info.get("is_success", False)

        # On episode end, record result for curriculum
        if terminated or truncated:
            self.curriculum.record_episode(
                success=episode_success_at_end,
                steps_in_episode=self._episode_steps,
            )

        # Inject curriculum metadata
        info["curriculum_spawn_angle_deg"] = self._current_spawn_angle_deg
        info["curriculum_chapter"] = self.curriculum.current_chapter.name
        info["curriculum_success_rate"] = self.curriculum.rolling_success_rate

        return obs, reward, terminated, truncated, info
