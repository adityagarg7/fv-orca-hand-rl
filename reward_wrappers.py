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
"""

from typing import Any

import gymnasium as gym


def alignment_potential(info: dict) -> float:
    """Potential ``Phi = 0.5 * (alignment + 1)`` in ``[0, 1]`` from an env info dict.

    Defaults to the reset state (alignment = -1, i.e. red face down, Phi = 0) if the
    alignment key is missing, so the wrapper is robust to an info dict that omits it.
    """
    return 0.5 * (float(info.get("red_face_up_alignment", -1.0)) + 1.0)


class PotentialShapedReorientationReward(gym.Wrapper):
    """Replace the native cube-reorientation reward with potential-based shaping.

    The wrapped env's own reward is discarded; the reward returned here is computed
    purely from the ``info`` dict::

        r = align_coeff * (gamma * Phi(s') - Phi(s))     # progress; loitering -> ~0
            - time_penalty                                # per-step cost
            + success_bonus   if (terminated and info["is_success"])
            - drop_penalty    if (terminated and info["dropped"])

    ``gamma`` should match the training discount so the PBRS invariance is exact.
    Each parallel env must hold its own wrapper instance (its own ``_prev_phi``);
    constructing the wrapper inside ``make_env`` -- once per sub-env -- guarantees
    this.
    """

    def __init__(
        self,
        env: gym.Env,
        align_coeff: float = 1.0,
        success_bonus: float = 10.0,
        time_penalty: float = 0.01,
        drop_penalty: float = 5.0,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(env)
        self.align_coeff = align_coeff
        self.success_bonus = success_bonus
        self.time_penalty = time_penalty
        self.drop_penalty = drop_penalty
        self.gamma = gamma
        self._prev_phi = 0.0

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Seed the previous potential from the actual start state so the first
        # step's shaping term is correct (start is usually Phi ~= 0, but read it).
        self._prev_phi = alignment_potential(info)
        return obs, info

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        phi = alignment_potential(info)
        shaping = self.gamma * phi - self._prev_phi
        self._prev_phi = phi

        reward = self.align_coeff * shaping - self.time_penalty

        # Success and drop are mutually exclusive and both imply terminated.
        if terminated and info.get("is_success", False):
            reward += self.success_bonus
        if terminated and info.get("dropped", False):
            reward -= self.drop_penalty

        # Keep the native reward visible for debugging / comparison.
        info["base_reward"] = base_reward
        info["shaped_reward"] = reward
        info["phi"] = phi
        return obs, reward, terminated, truncated, info
