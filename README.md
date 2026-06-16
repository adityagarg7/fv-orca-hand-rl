# ORCA Hand — Reinforcement Learning

Reinforcement-learning training and research for the ORCA dexterous hand at Four Vectors.

This repository is the shared home for RL work on the ORCA hand. The first task
implemented here is **in-hand cube reorientation**, trained with PPO and a shaped
"production" reward that eliminates the flick-and-pray reward-hacking exploit. The
repository is structured to host additional manipulation tasks as the work expands.

---

## How the project is organized

| Artifact | Where it lives |
| --- | --- |
| Code (training scripts, reward wrappers, notebooks) | this Git repository |
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
- `production_reward.py` — the reward wrapper for the cube-reorientation task
  (exponential alignment kernel, terminal success bonus, stable-grasp hold
  constraint, action regularisation).
- `render_policy.py` — load a trained model and inspect its behaviour in the
  MuJoCo viewer.
- `colab_train.ipynb` — run training on Google Colab (clones this repo, logs to W&B).
- `requirements.txt`, `.gitignore`.

---

## Getting started

### Prerequisites
- Access to this private repository (request from a repository admin).
- Membership in the Four Vectors Weights & Biases team (request from a W&B admin).
- Python 3.11. Conda is recommended, matching the `orca_sim` documentation.

### 1. Clone the repository
```bash
git clone https://github.com/adityagarg7/fv-orca-hand-rl.git
cd fv-orca-hand-rl
```

### 2. Create the environment and install dependencies
```bash
conda create -n orca python=3.11 -y
conda activate orca
pip install -r requirements.txt
```

Then install the `orca_sim` environment editable from a clone. Its published
package omits some MuJoCo assets, so an editable source install is required; keep
the clone outside this repo (e.g. as a sibling folder) so it isn't committed:
```bash
git clone https://github.com/orcahand/orca_sim.git ../orca_sim
pip install -e ../orca_sim
```

### 3. Authenticate with Weights & Biases
Generate a personal API key at https://wandb.ai/authorize, then log in once
(the key is stored in `~/.netrc`):
```bash
wandb login
```

### 4. Set your commit identity
Make sure commits are attributed to your own GitHub account. Using the no-reply
address from your GitHub email settings keeps your real address private:
```bash
git config user.name "Your Name"
git config user.email "ID+username@users.noreply.github.com"
```

---

## Usage

### Training locally
```bash
python train.py --timesteps 20000 --run-name smoke-test     # quick sanity check
python train.py --timesteps 20000000 --run-name prod-20M    # full run
```
Metrics stream to W&B in real time, and the final model is logged as a versioned
artifact named `model-<run_id>`. For reference, the cube-reorientation reward
reaches roughly 53% success by ~500k steps, so the full 20M run is rarely
necessary. Pass `--project <name>` to log a different task to its own W&B project,
or `--entity <team>` to target the shared team workspace.

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
python render_policy.py path/to/model.zip      # Linux
mjpython render_policy.py path/to/model.zip    # macOS
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
manipulation tasks follow the same pattern — a task-specific reward wrapper plus a
training entrypoint — and log to their own W&B project. Add them here as the scope
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
