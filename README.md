# ORCA Hand — Reinforcement Learning

Reinforcement-learning training and research for the ORCA dexterous hand at Four Vectors.

This repository is the shared home for RL work on the ORCA hand. The first task
implemented here is **in-hand cube reorientation**, trained with PPO directly on
the reward built into the `orca_sim` environment. The repository is structured to
host additional manipulation tasks as the work expands.

---

## How the project is organized

| Artifact | Where it lives |
| --- | --- |
| Code (training scripts, notebooks) | this Git repository |
| Training metrics and curves | Weights & Biases (auto-synced from TensorBoard) |
| Trained models and checkpoints | W&B artifacts (never committed to Git) |
| The `orca_sim` environment | installed editable from its own clone (see setup), not vendored here |

Guiding principle: **code lives in Git; everything a run produces lives in W&B.**
This keeps the repository small and reviewable, and removes any need to pass model
or log files around by hand.

---

## Repository contents

- `train.py` — PPO training entrypoint with W&B logging. Run length and other
  hyperparameters are command-line flags, so no file edits are needed for quick
  experiments.
- `render_policy.py` — load a trained model and inspect its behaviour in the
  MuJoCo viewer.
- `colab_train.ipynb` — run training on Google Colab (clones this repo, logs to W&B).
- `requirements.txt`, `.gitignore`.

---

## Getting started

### Prerequisites
- Access to this private repository (request from a repository admin).
- Membership in the Four Vectors Weights & Biases team (request from a W&B admin).
- [pixi](https://pixi.sh) for environment management. Install it once:
  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```
  pixi reads `pixi.toml`, pins everything in `pixi.lock`, and builds an isolated
  Python 3.11 environment — no conda or manual venv needed.

### 1. Clone the repository
```bash
git clone https://github.com/adityagarg7/fv-orca-hand-rl.git
cd fv-orca-hand-rl
```

### 2. Build the environment
```bash
pixi install        # creates the env from pixi.toml + pixi.lock
```

Then install the `orca_sim` environment editable from a clone. Its published
package omits some MuJoCo assets, so an editable source install is required; the
task below clones it as a sibling folder (outside this repo, so it isn't
committed) and installs it editable:
```bash
pixi run setup-orca
```

### 3. Authenticate with Weights & Biases
Generate a personal API key at https://wandb.ai/authorize, then log in once
(the key is stored in `~/.netrc`, not in the project):
```bash
pixi run login      # or just: wandb login
```

---

## Usage

### Training locally
```bash
pixi run smoke                                                            # quick sanity check (20k steps)
pixi run train --timesteps 20000000 --run-name prod-20M --upload-model    # full run, keep the model
```
`pixi run <task>` runs inside the project environment; extra flags pass straight
through to `train.py`. (Equivalently, activate the env with `pixi shell` and call
`python train.py ...` directly.) Runs log to the `fourvectors` W&B entity by
default — override with `--entity <team>` if needed. Metrics stream to W&B in
real time. Pass `--upload-model` to also save the trained
model to W&B as a versioned artifact (`model-<run_id>`); it is off by default so
quick test runs stay fast. For reference, the cube-reorientation reward reaches
roughly 53% success by ~500k steps, so the full 20M run is rarely necessary. Pass
`--project <name>` to log a different task to its own W&B project, or `--entity
<team>` to target the shared team workspace.

### Training on Colab
Open `colab_train.ipynb`. Credentials are read from Colab Secrets (a GitHub
access token and a W&B API key) — see the notebook's first cell. Colab is best
suited to short experiments; long runs should be done locally, as free Colab
sessions disconnect.

### Comparing runs
Open the project in the W&B workspace and select any runs to overlay their
success-rate and reward curves.

### Retrieving a trained model
Any team member can pull a model trained by someone else:
```python
import wandb
api = wandb.Api()
art = api.artifact("<entity>/<project>/model-<run_id>:latest", type="model")
path = art.download()   # the .zip is downloaded to `path`
```

### Rendering a policy
```bash
pixi run render path/to/model.zip                   # Linux
pixi run mjpython render_policy.py path/to/model.zip # macOS (the viewer needs mjpython)
```

---

## Contributing

Always pull before starting work:
```bash
git pull
```
Routine changes can go directly on `main`. For anything experimental — a new
reward shape, a hyperparameter sweep, or a new task — use a feature branch and
open a pull request for review:
```bash
git checkout -b experiment/<short-description>
# commit your changes
git push -u origin experiment/<short-description>
# then open a pull request on GitHub
```
Keep commits focused and write clear, descriptive messages.

### Extending to new tasks
The current training entrypoint targets the cube-reorientation environment. New
manipulation tasks follow the same pattern — an `orca_sim` environment (with its
own reward) plus a training entrypoint — and log to their own W&B project. Add
them here as the scope
of the work grows.

---

## Conventions and notes
- `train.py` runs on CPU by default, which is appropriate for the MLP policy on
  low-dimensional observations; a GPU provides little benefit and is often slower.
  Use `--device cuda` only for deliberate experiments.
- Trained models, checkpoints, and TensorBoard/W&B logs are intentionally excluded
  from version control (see `.gitignore`).
- For reproducible runs, pin a specific `orca_sim` version by checking out a known
  commit in your `orca_sim` clone (`git checkout <commit>`) before `pip install -e`.
