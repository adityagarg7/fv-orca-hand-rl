# CLAUDE.md

Operational notes for working in this repo with Claude Code. See `README.md` for
full setup and team-facing docs; this file only captures the non-obvious gotchas.

## What this is
RL training for the ORCA dexterous hand. First task: in-hand cube reorientation,
PPO (Stable-Baselines3) trained directly on the reward built into the `orca_sim`
environment (`OrcaHandRightCubeOrientation._get_reward`).

## Environment — use pixi
- The env is pixi-managed. Run Python through it: `pixi run python ...` (or
  `pixi shell` first). Bare `python` won't see the dependencies.
- `orca_sim` is installed **editable from a sibling clone (`../orca_sim`)** and is
  **not** in `pixi.lock`. `pixi install` alone won't make `import orca_sim` work —
  `pixi run setup-orca` is required (clones the sibling + `pip install -e`).
- Defined tasks: `setup-orca`, `login`, `smoke`, `train`, `render`.

## Running training
- `pixi run smoke` runs a 20k-step training and **creates a real W&B run** in the
  `fourvectors` entity. It is not a free local test — it has an external side effect.
- `--entity` defaults to `fourvectors`; no need to pass it.
- `--upload-model` is **off by default** (keeps smoke runs fast); pass it only on
  runs whose model you want kept as a W&B artifact.
- Training is **CPU by design** (MLP policy on low-dim obs). `--device cuda` is
  rarely wanted and often slower.

## Key files
- The env's native reward lives in `orca_sim` itself
  (`OrcaHandRightCubeOrientation._get_reward`: alignment + lift bonus − drop
  penalty). Training does **not** use it directly: `reward_wrappers.py` defines
  `PotentialShapedReorientationReward`, a gymnasium wrapper that discards the native
  reward and rebuilds it from the env `info` dict using potential-based shaping
  (Φ = alignment) + a terminal success bonus + a per-step time penalty + a drop
  penalty. This fixes the "loiter to farm dense reward" pathology of the native
  reward (success terminates the episode, so the native reward rewards holding just
  below the success threshold for the full horizon instead of solving). Coefficients
  are CLI flags on `train.py` (`--align-coeff/--success-bonus/--time-penalty/--drop-penalty`).
  To reshape further, edit the wrapper here; the deeper env mechanics still live in
  the sibling `../orca_sim` clone.
- `curriculum.py` — **gravity curriculum** (`GravityCurriculumWrapper` +
  `GravityCurriculumCallback`). The wrist is an actuated DOF; an alignment-only policy
  learns to flex it forward and let *gravity* dump the cube off the palm so it flips
  red-up (a cheat). Rather than penalize the wrist/cube directly (which freezes the
  policy), training starts under **reduced gravity** (the dump barely works and the
  cube can't fall, so fingers must do the reorientation) and the callback is
  **performance-gated**: it raises gravity only once the success rate at the current
  level clears `--gravity-success-threshold`, so it can't ramp into the cheat before
  finger manipulation is learned. Flags: `--gravity-start/--gravity-final/
  --gravity-success-threshold/--gravity-step/--gravity-min-episodes` (set start ==
  final to disable). Gravity is set live on MuJoCo via `model.opt.gravity`.
- `action_wrappers.py` — **wrist clamp** (`WristClampWrapper`). The wrist-dump cheat is
  *geometric* (it works at any gravity if the palm can tilt far enough), so the gravity
  curriculum only slows it and a reward penalty on the wrist froze the policy. This
  wrapper instead hard-clips the wrist actuator (index 0 of the action vector) to a
  narrow band around its neutral palm-up angle — captured from the wrist joint angle at
  each `reset()` — so the policy physically can't flex the wrist far enough to dump the
  cube, without any penalty. It is the **outermost** wrapper in `make_env` (clips the
  action before any inner wrapper/env), doesn't change the action-space dimension, and is
  applied in `render_policy.py` too so a clamp-trained policy is evaluated under the same
  constraint. Flag: `--wrist-band` (half-width in radians, default 0.15). The gravity
  curriculum is kept alongside as a genuine learning aid (easier early reorientation),
  no longer relied on as the anti-cheat.
- `train.py` — PPO entrypoint, all knobs are CLI flags.
- `render_policy.py` — load a model and watch it in the MuJoCo viewer
  (`pixi run render <model.zip>`; macOS needs `mjpython`).

## Conventions
- Code lives in Git; everything a run produces lives in W&B. Models, checkpoints,
  and TensorBoard/W&B logs are gitignored — never commit them.
- Routine changes can go straight to `main`; use a feature branch + PR for
  experimental work (new reward shapes, sweeps, new tasks).
- `requirements.txt` is the Colab / plain-pip dep list; keep it in sync with
  `pixi.toml`.
