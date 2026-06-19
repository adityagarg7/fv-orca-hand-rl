"""
Gym wrapper that spawns the cube at the curriculum's current difficulty angle.

Wraps OrcaHandRightCubeOrientation → ProductionRewardWrapper → CurriculumWrapper.

The wrapper intercepts reset() to:
  1. Sample a spawn angle from the CurriculumManager
  2. Convert it to a MuJoCo quaternion
  3. Pass it as cube_quat to the underlying environment
  4. Track episode success for auto-promotion decisions

This is the ONLY file that connects the curriculum to the environment.
"""

import gymnasium as gym
import numpy as np
from curriculum import CurriculumManager


class CurriculumWrapper(gym.Wrapper):
    """Intercepts reset() to control cube spawn angle via curriculum."""

    def __init__(self, env: gym.Env, curriculum: CurriculumManager):
        super().__init__(env)
        self.curriculum = curriculum
        self._episode_success = False
        self._episode_steps = 0
        self._current_spawn_angle_deg = 0.0

    def reset(self, **kwargs):
        # Sample spawn angle from curriculum
        angle_deg = self.curriculum.sample_spawn_angle_deg()
        self._current_spawn_angle_deg = angle_deg
        self._episode_success = False
        self._episode_steps = 0

        # Convert angle to MuJoCo quaternion (rotation around X axis)
        theta = np.radians(angle_deg)
        cube_quat = np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])

        # Inject into reset options
        options = kwargs.pop("options", {}) or {}
        options["cube_quat"] = cube_quat
        kwargs["options"] = options

        obs, info = self.env.reset(**kwargs)

        # Add curriculum info
        info["curriculum_spawn_angle_deg"] = angle_deg
        info["curriculum_chapter"] = self.curriculum.current_chapter.name

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_steps += 1

        # Track if this episode ever achieved success
        if info.get("is_success", False):
            self._episode_success = True

        # On episode end, record result for curriculum
        if terminated or truncated:
            self.curriculum.record_episode(
                success=self._episode_success,
                steps_in_episode=self._episode_steps,
            )

        # Inject curriculum metadata
        info["curriculum_spawn_angle_deg"] = self._current_spawn_angle_deg
        info["curriculum_chapter"] = self.curriculum.current_chapter.name
        info["curriculum_success_rate"] = self.curriculum.rolling_success_rate

        return obs, reward, terminated, truncated, info
