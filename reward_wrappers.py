"""
Reward shaping for the ORCA in-hand cube-reorientation task.

Why this exists
---------------
The native ``orca_sim`` reward for ``OrcaHandRightCubeOrientation`` is a dense,
per-step *level* reward::

    reward = 0.5 * (alignment + 1)  +  0.10 * lift_bonus  -  drop_penalty

where ``alignment = dot(red_face_normal, world_up)`` lies in ``[-1, 1]`` (the red
face starts pointing *down*, alignment = -1, so reward is ~0 at reset; the goal is
the red face up within 15 degrees, i.e. ``alignment >= cos(15 deg) ~= 0.966``).

That reward has a pathology: there is **no terminal bonus for success**, and the
environment sets ``terminated = goal_reached or dropped`` -- so *reaching the goal
ends the episode and stops the reward stream*. The return-maximizing policy is
therefore to drive alignment as high as possible **without** crossing the success
threshold (which would terminate) and to loiter there for the full 200-step
horizon. Under gamma=0.99 that loiter farms a discounted return of roughly
``(1 - 0.99**200) / (1 - 0.99) ~= 86`` -- far more than any quick solve, which banks
fewer dense steps and then terminates. The observable symptoms are the classic
ones: episode reward grows with episode *length*, plateaus once episodes hit the
200-step cap, and success rate sits at chance because success is actively
disincentivized.

The fix: potential-based reward shaping (PBRS)
----------------------------------------------
``PotentialShapedReorientationReward`` discards the native reward and rebuilds it
from the ``info`` dict (which already exposes ``red_face_up_alignment``,
``is_success`` and ``dropped`` every step). It uses the potential::

    Phi(s) = 0.5 * (alignment + 1)        # in [0, 1]; 0 at reset, 1 at the goal

and the PBRS shaping term ``F = gamma * Phi(s') - Phi(s)`` (Ng, Harada & Russell,
1999). Because the shaping telescopes, the *total* dense reward on any trajectory
from start to a given state depends only on the endpoint potential, not on how
long the agent dwells there -- so loitering accrues ~0 and the ~86-point farm
collapses to <= ``align_coeff``. On top of the shaping we add:

* a large **terminal success bonus** on the ``terminated`` step where the goal is
  reached (SB3 does not bootstrap a terminated step, so the bonus is credited in
  full), which makes "solve now" strictly beat "loiter";
* a small constant **time penalty** every step, so faster solves are preferred and
  stalling is net-negative;
* a **drop penalty** on the ``terminated`` step where the cube is dropped.

Keep the coefficient ordering ``success_bonus > drop_penalty >
horizon * time_penalty > align_coeff`` when tuning.

Anti-cheat: discouraging the "wrist dump"
-----------------------------------------
The ORCA hand's base is fixed, but the wrist (``right_wrist``) is a policy-controlled
actuator (the first of 17). A policy trained on alignment alone learns to flex the
wrist forward and let the cube tip off the palm so *gravity* flips it red-face-up --
high success, but not the intended in-hand finger manipulation. Two further per-step
penalties make that shortcut costly while leaving honest in-hand reorientation almost
untouched:

* ``wrist_penalty * |wrist_angle - wrist_neutral|`` -- attacks the *cause*. The
  neutral (palm-up) wrist angle is captured at ``reset()``; deviating from it costs
  reward, so the policy is pushed to keep the palm flat.
* ``slide_penalty * max(0, ||cube_pos - cube_home|| - slide_deadzone)`` -- attacks the
  *effect*. The cube's reset position is captured at ``reset()``; the cube center
  drifting away from it (beyond a small deadzone) costs reward. This penalizes
  *translation* (sliding/falling off the palm), not *rotation* -- a cube reorienting
  in place keeps its center roughly fixed, so the intended skill is barely penalized.
"""

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


def alignment_potential(info: dict) -> float:
    """Potential ``Phi = 0.5 * (alignment + 1)`` in ``[0, 1]`` from an env info dict.

    Defaults to the reset state (alignment = -1, i.e. red face down, Phi = 0) if the
    alignment key is missing, so the wrapper is robust to an info dict that omits it.
    """
    return 0.5 * (float(info.get("red_face_up_alignment", -1.0)) + 1.0)


class PotentialShapedReorientationReward(gym.Wrapper):
    """Replace the native cube-reorientation reward with potential-based shaping.

    The wrapped env's own reward is discarded; the reward returned here is computed
    from the ``info`` dict plus the wrist joint angle (read from the MuJoCo model)::

        r = align_coeff * (gamma * Phi(s') - Phi(s))     # progress; loitering -> ~0
            - time_penalty                                # per-step cost
            - wrist_penalty * |wrist_angle - wrist_neutral|
            - slide_penalty * max(0, ||cube_pos - cube_home|| - slide_deadzone)
            + success_bonus   if (terminated and info["is_success"])
            - drop_penalty    if (terminated and info["dropped"])

    ``gamma`` should match the training discount so the PBRS invariance is exact.
    Set ``wrist_penalty`` / ``slide_penalty`` to 0.0 to recover the pure alignment
    reward (e.g. the v1 behaviour). Each parallel env must hold its own wrapper
    instance (its own ``_prev_phi`` / neutral references); constructing the wrapper
    inside ``make_env`` -- once per sub-env -- guarantees this.
    """

    def __init__(
        self,
        env: gym.Env,
        align_coeff: float = 1.0,
        success_bonus: float = 10.0,
        time_penalty: float = 0.01,
        drop_penalty: float = 5.0,
        wrist_penalty: float = 0.5,
        slide_penalty: float = 5.0,
        slide_deadzone: float = 0.03,
        wrist_joint_name: str = "right_wrist",
        gamma: float = 0.99,
    ) -> None:
        super().__init__(env)
        self.align_coeff = align_coeff
        self.success_bonus = success_bonus
        self.time_penalty = time_penalty
        self.drop_penalty = drop_penalty
        self.wrist_penalty = wrist_penalty
        self.slide_penalty = slide_penalty
        self.slide_deadzone = slide_deadzone
        self.gamma = gamma
        self._prev_phi = 0.0
        self._wrist_neutral = 0.0
        self._cube_home = None

        # Resolve the wrist joint's qpos address once (robust to observation layout).
        # If the joint isn't found, disable the wrist penalty rather than crash.
        model = env.unwrapped.model
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wrist_joint_name)
        if jid < 0:
            self._wrist_qposadr = None
            if self.wrist_penalty:
                print(f"[reward_wrappers] wrist joint '{wrist_joint_name}' not found; "
                      "disabling wrist penalty.")
                self.wrist_penalty = 0.0
        else:
            self._wrist_qposadr = int(model.jnt_qposadr[jid])

    def _wrist_angle(self) -> float:
        if self._wrist_qposadr is None:
            return self._wrist_neutral
        return float(self.env.unwrapped.data.qpos[self._wrist_qposadr])

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Seed the previous potential from the actual start state so the first
        # step's shaping term is correct (start is usually Phi ~= 0, but read it).
        self._prev_phi = alignment_potential(info)
        # Capture the neutral references from the reset (palm-up, cube seated) pose.
        self._wrist_neutral = self._wrist_angle()
        self._cube_home = np.array(info["cube_pos"], dtype=float) if "cube_pos" in info else None
        return obs, info

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        phi = alignment_potential(info)
        shaping = self.gamma * phi - self._prev_phi
        self._prev_phi = phi

        # Anti-cheat penalties: keep the wrist near neutral and the cube near home.
        wrist_dev = abs(self._wrist_angle() - self._wrist_neutral)
        if self._cube_home is not None and "cube_pos" in info:
            cube_slide = float(np.linalg.norm(np.asarray(info["cube_pos"], dtype=float) - self._cube_home))
        else:
            cube_slide = 0.0

        reward = (
            self.align_coeff * shaping
            - self.time_penalty
            - self.wrist_penalty * wrist_dev
            - self.slide_penalty * max(0.0, cube_slide - self.slide_deadzone)
        )

        # Success and drop are mutually exclusive and both imply terminated.
        if terminated and info.get("is_success", False):
            reward += self.success_bonus
        if terminated and info.get("dropped", False):
            reward -= self.drop_penalty

        # Keep the native reward and diagnostics visible for debugging / comparison.
        info["base_reward"] = base_reward
        info["shaped_reward"] = reward
        info["phi"] = phi
        info["wrist_dev"] = wrist_dev
        info["cube_slide"] = cube_slide
        return obs, reward, terminated, truncated, info
