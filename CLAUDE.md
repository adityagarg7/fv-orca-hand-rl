# CLAUDE.md

Operational notes for working in this repo with Claude Code. See `README.md` for
full setup and team-facing docs; this file only captures the non-obvious gotchas.

## What this is
RL training for the ORCA dexterous hand. First task: in-hand cube reorientation,
PPO (Stable-Baselines3) + a shaped reward that prevents reward hacking.

## Environment — use pixi
- The env is pixi-managed. Run Python through it: `pixi run python ...` (or
  `pixi shell` first). Bare `python` won't see the dependencies.
- `orca_sim` is installed **editable from a sibling clone (`../orca_sim`)** and is
  **not** in `pixi.lock`. `pixi install` alone won't make `import orca_sim` work —
  `pixi run setup-orca` is required (clones the sibling + `pip install -e`).
- Defined tasks: `setup-orca`, `login`, `curriculum`, `curriculum-cpu`,
  `curriculum-smoke`, `curriculum-resume`, `render`.

## Running training
- `pixi run curriculum` is the full 8-chapter run; `pixi run curriculum-smoke` is
  the quick 50k-step sanity check. Either **creates a real W&B run** in the
  `fourvectors` entity — not a free local test; it has an external side effect.
- `--entity` defaults to `fourvectors`; no need to pass it.
- `--upload-model` is **off by default** (keeps smoke runs fast); pass it only on
  runs whose model you want kept as a W&B artifact.
- Training is **CPU by design** (MLP policy on low-dim obs). `--device cuda` is
  rarely wanted and often slower.

## Key files
- `production_reward.py` — the v11 shaped reward (potential-based shaping,
  palm-frame finger-attributed progress, real MuJoCo contact detection, and a
  non-terminal goal-hold stream). This is the heart of the project and the
  anti-reward-hacking logic; reward changes are high-stakes — change deliberately.
- `train_curriculum.py` — the curriculum PPO entrypoint (8 chapters, auto-promotion,
  adaptive entropy/clip, VecNormalize). All knobs are CLI flags. The old single-phase
  `train.py` was removed — use this.
- `render_policy.py` — load a model and watch it in the MuJoCo viewer
  (`pixi run render <model.zip>`; macOS needs `mjpython`).

## Conventions
- Code lives in Git; everything a run produces lives in W&B. Models, checkpoints,
  and TensorBoard/W&B logs are gitignored — never commit them.
- Routine changes can go straight to `main`; use a feature branch + PR for
  experimental work (new reward shapes, sweeps, new tasks).
- `requirements.txt` is the Colab / plain-pip dep list; keep it in sync with
  `pixi.toml`.
