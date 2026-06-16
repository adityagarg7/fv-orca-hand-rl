# ORCA Hand — In-Hand Cube Reorientation (Four Vectors)

PPO training for in-hand cube reorientation on the ORCA hand, using a shaped
"production" reward that fixes the flick-and-pray reward-hacking exploit.

This repo is set up so two people can collaborate cleanly:

| What | Where it lives |
| --- | --- |
| Code (scripts, reward function, notebook) | **this Git repo** |
| Training metrics / curves | **Weights & Biases** (auto-synced) |
| Trained models & checkpoints | **W&B artifacts** (not committed to git) |
| The `orca_sim` environment | a **dependency** (installed via `requirements.txt`), not vendored here |

The rule of thumb: **code goes in git, everything a run produces goes in W&B.**
Git stays small and diffable; nobody emails zip files around.

---

## Repository contents

- `train.py` — PPO training with W&B logging. Step count and other knobs are
  CLI flags, so you never edit the file for a quick test.
- `production_reward.py` — the reward wrapper (exponential alignment kernel,
  success bonus, stable-grasp hold constraint, action regularisation).
- `render_policy.py` — load a model and watch it in the MuJoCo viewer.
- `colab_train.ipynb` — run training on Colab (clones this repo, logs to W&B).
- `requirements.txt`, `.gitignore`.

---

## One-time setup

### 1. Create the private repo and add your collaborator
On GitHub: **New repository → Private**, owned by the Four Vectors org (or your
account). Then **Settings → Collaborators** and invite Siddhant. Push this
scaffold to it:

```bash
cd four-vectors-orca
git init
git add -A
git commit -m "Initial scaffold: W&B training, production reward, Colab"
git branch -M main
git remote add origin https://github.com/<owner>/<repo>.git
git push -u origin main
```

### 2. Set up Weights & Biases
- Create an account at https://wandb.ai (free), and a **Team** so you and
  Siddhant share one workspace.
- Grab your API key from https://wandb.ai/authorize.

### 3. Local environment (your own machine)
Use conda (matches the `orca_sim` docs) or a venv with Python 3.11:

```bash
conda create -n orca python=3.11 -y
conda activate orca
pip install -r requirements.txt   # installs orca_sim from source too
wandb login                       # paste your API key once
```

---

## Daily workflow

### Git loop
```bash
git pull                          # always start by pulling the latest
# ... edit reward / scripts ...
git add -A
git commit -m "Tweak success bonus to 300"
git push
```
For anything experimental (e.g. trying a new reward shape), work on a **branch**
and open a Pull Request so you don't clobber each other:
```bash
git checkout -b experiment/higher-success-bonus
# ...commit, push...
git push -u origin experiment/higher-success-bonus
# then open a PR on GitHub
```

### Train (locally)
```bash
python train.py --timesteps 20000 --run-name smoke-test     # quick sanity check
python train.py --timesteps 20000000 --run-name prod-20M    # full run
```
Metrics stream to W&B live; the final model is logged as `model-<run_id>`.
Note the reward already reaches ~53% success around 500k steps, so you rarely
need the full 20M.

### Train (on Colab)
Open `colab_train.ipynb`. It uses **Colab Secrets** for credentials — see the
notebook's first cell. Full 20M runs are better done locally (free Colab
disconnects); Colab is best for quick experiments.

### Compare runs
Open your project at https://wandb.ai — select any runs to overlay their
success-rate / reward curves. (This replaces the old `plot_comparison.py`.)

### Pull a trained model
Anyone on the team can fetch a model another person trained:
```python
import wandb
api = wandb.Api()
art = api.artifact("<entity>/<project>/model-<run_id>:latest", type="model")
path = art.download()   # the .zip lands in `path`
```

### Render a policy
```bash
python render_policy.py path/to/model.zip      # Linux
mjpython render_policy.py path/to/model.zip     # macOS only
```

---

## Notes
- `train.py` uses `device="cpu"` by default — correct for this MLP policy on
  low-dim observations (the GPU doesn't help and is often slower). Pass
  `--device cuda` only if you want to experiment.
- Models, checkpoints, and TensorBoard/W&B logs are gitignored on purpose.
- For fully reproducible runs, pin a specific `orca_sim` commit in
  `requirements.txt` (see the comment in that file).
