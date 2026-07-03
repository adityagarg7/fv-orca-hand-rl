# ORCA Hand — Reinforcement Learning (v7 — Workstation Edition)

Reinforcement-learning training and research for the ORCA dexterous hand at Four Vectors.

The primary task is **in-hand cube reorientation**: the agent must reorient a cube
held in the ORCA dexterous hand from any starting angle (up to 180°) to the upright
goal orientation, using only finger torques — no external support.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [File Reference](#3-file-reference)
4. [Environment Setup](#4-environment-setup)
5. [Running Training (Workstation)](#5-running-training-workstation)
6. [Curriculum Design](#6-curriculum-design)
7. [Reward Function](#7-reward-function)
8. [Monitoring with W&B](#8-monitoring-with-wb)
9. [Resuming After Crash](#9-resuming-after-crash)
10. [Hardware Recommendations](#10-hardware-recommendations)
11. [Contributing](#11-contributing)

---

## 1. Project Overview

### What this project trains
A PPO policy that controls the 16 joints of the ORCA right hand to reorient a cube
through up to 180° of rotation. The policy receives a 60-dimensional observation
(joint positions, joint velocities, cube position, cube orientation as quaternion,
fingertip positions) and outputs 16 continuous torque commands.

### Why this is hard
- The cube has 6 degrees of freedom. The hand has 16 joints. The contact dynamics
  are non-smooth and highly non-linear.
- At angles beyond 90°, the required motor skill changes discontinuously — from
  "nudging" (small torques near goal) to "rolling" (coordinated lateral forces to
  push the cube past the equator). This phase transition is the primary training
  bottleneck.
- With only 4 parallel environments (as in early versions), the agent sees ~24
  distinct cube starting angles per PPO rollout — not enough to learn a general
  policy. At 64 parallel environments, each rollout covers ~640 distinct angles,
  which is sufficient.

### Version history

| Version | Key Change | Result |
|---------|-----------|--------|
| v0 | Hold-timer success gate | Impossible exploration — 0% success |
| v1 | Removed hold timer | "Perfect drop" loophole — 100% false positive |
| v2 | Fixed success definition | Entropy collapse (std=1.15 jitter) |
| v3 | Fixed entropy | Frozen learning (clip_fraction=0) |
| v4 | Adaptive entropy + clip restart | Ch1 solved, Ch2 flatlined at 0% |
| v5 | Larger [256,256] network | Ch1 faster, Ch2 still 0% (overfit to Ch1) |
| v6 | 8 finer chapters, 16 DummyVecEnv | Ch2 improvement, still limited by 2 CPU cores |
| **v7** | **64 SubprocVecEnv, [512,256] net, 150M steps** | **Current — workstation target** |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  SubprocVecEnv (64 independent OS processes)                        │
│                                                                     │
│  Process 0: OrcaSim → ProductionRewardWrapper → CurriculumWrapper   │
│  Process 1: OrcaSim → ProductionRewardWrapper → CurriculumWrapper   │
│  ...                                                                │
│  Process 63: OrcaSim → ProductionRewardWrapper → CurriculumWrapper  │
│                                                                     │
│  Each process: MuJoCo physics engine running on CPU                 │
│  Each process: ~100–150 MB RAM                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              observations (64 × 60 floats)
              actions (64 × 16 floats)
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  PPO (Stable-Baselines3)                                            │
│                                                                     │
│  Policy network: MLP [512 → 256 → 16 actions]   ← CUDA GPU         │
│  Value network:  MLP [512 → 256 → 1 value]      ← CUDA GPU         │
│                                                                     │
│  n_steps = 4096 per env                                             │
│  Total rollout = 4096 × 64 = 262,144 steps                         │
│  Batch size = 1024 (PPO mini-batch)                                 │
└─────────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  CurriculumCallback (main process)                                  │
│                                                                     │
│  • Monitors rolling success rate (SB3 ep_success_buffer)            │
│  • Promotes to next chapter at 80% success                          │
│  • Broadcasts chapter change to all 64 subprocesses (env_method)    │
│  • Adaptive entropy: adjusts ent_coef to keep std in [0.75, 1.05]  │
│  • Clip warm-restart: widens to 0.3 at promotion, decays to 0.2    │
│  • Saves checkpoint every 500k steps + at every promotion           │
└─────────────────────────────────────────────────────────────────────┘
```

### Data flow on promotion
When the agent hits 80% success rate in the current chapter:
1. `CurriculumCallback.should_promote()` returns True
2. Main process `CurriculumManager.promote()` advances `_chapter_idx`
3. `env_method("set_chapter_idx", new_idx)` broadcasts to all 64 subprocesses
4. Each subprocess's `CurriculumWrapper.set_chapter_idx()` updates its local
   `CurriculumManager` so spawn angles are sampled from the new chapter's band
5. PPO hyperparameters (lr, ent_coef, batch_size) are updated in the main process

---

## 3. File Reference

| File | Purpose |
|------|---------|
| `train_curriculum.py` | **Main training entry point.** All configuration via CLI flags. Handles SubprocVecEnv creation, PPO model, callbacks, W&B logging, and crash recovery. |
| `curriculum.py` | Defines the 8-chapter `CHAPTERS` list and the `CurriculumManager` class that tracks success rate, handles promotion, and serialises/deserialises state for crash recovery. |
| `curriculum_wrapper.py` | Gym wrapper that intercepts `reset()` to sample cube spawn angles from the current chapter and intercepts `step()` to record episode success. Also exposes `set_chapter_idx()` for SubprocVecEnv broadcasting. |
| `production_reward.py` | The shaped reward function. 8 components: alignment, rotation progress (Dactyl-style delta), position keeping, fingertip proximity, success bonus (stability-scaled), alive bonus, drop penalty, action regularisation. |
| `render_policy.py` | Load a trained `.zip` model and render it in the MuJoCo viewer. |
| `train.py` | Legacy single-phase training script (no curriculum). Useful for quick isolated experiments. |
| `pixi.toml` | Environment and task definitions. Run `pixi run <task>` instead of activating an env manually. |

---

## 4. Environment Setup

### Prerequisites

| Requirement | Notes |
|------------|-------|
| Linux (Ubuntu 22.04+) or macOS | Windows is not supported (MuJoCo subprocess issues) |
| Python 3.11 | Managed automatically by pixi |
| NVIDIA GPU + CUDA 12.x | Required for `--device cuda`. Use `--device cpu` otherwise. |
| [pixi](https://pixi.sh) | Conda-like env manager. Install once. |
| W&B account (fourvectors team) | For metric logging and model storage |
| GitHub access to this repo | Private repo |

### Step 1 — Install pixi (once per machine)
```bash
curl -fsSL https://pixi.sh/install.sh | bash
# then open a new terminal or source ~/.bashrc
```

### Step 2 — Clone the repository
```bash
git clone https://github.com/adityagarg7/fv-orca-hand-rl.git
cd fv-orca-hand-rl
git checkout siddhant/phase1-reward     # active development branch
```

### Step 3 — Install Python environment
```bash
pixi install
```
This creates an isolated Python 3.11 environment with all dependencies pinned in
`pixi.lock`. No manual conda or venv needed.

### Step 4 — Install orca_sim (editable)
`orca_sim` is the MuJoCo simulation environment for the ORCA hand. It must be
installed as an editable source install (the published PyPI package omits MuJoCo
assets):
```bash
pixi run setup-orca
```
This clones `orca_sim` as a sibling folder (`../orca_sim`) and installs it editable.
Safe to re-run if already done.

### Step 5 — Authenticate with W&B
```bash
pixi run login
# Paste your API key from https://wandb.ai/authorize
```
The key is stored in `~/.netrc`. Never committed to Git.

### Step 6 — Verify CUDA is detected
```bash
pixi shell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Should print `True  NVIDIA GeForce RTX ...`. If False, check CUDA drivers.

### Step 7 — Run the smoke test
A quick 50k-step sanity check (no GPU needed, 2 envs, DummyVecEnv):
```bash
pixi run curriculum-smoke
```
If it runs without errors and logs to W&B, your setup is correct.

---

## 5. Running Training (Workstation)

### Primary command (64 envs, CUDA GPU)
```bash
pixi run curriculum
```
Equivalent to:
```bash
python train_curriculum.py \
    --n-envs 64 \
    --device cuda \
    --subproc \
    --timesteps 150_000_000 \
    --save-dir ./checkpoints \
    --upload-model
```

### CPU-only (no GPU, 48 envs)
```bash
pixi run curriculum-cpu
```

### Resume after crash
```bash
pixi run curriculum-resume
```
The script auto-detects `./checkpoints/curriculum_state.json` and
`./checkpoints/latest_model.zip` and resumes seamlessly. No manual intervention needed.

### Custom run with a specific name
```bash
python train_curriculum.py \
    --n-envs 64 \
    --device cuda \
    --subproc \
    --timesteps 150_000_000 \
    --save-dir ./checkpoints \
    --run-name v7-my-experiment \
    --upload-model
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | `150_000_000` | Maximum total steps (curriculum completion is the real stop) |
| `--n-envs` | `64` | Parallel environments. Sweet spot: 3–4 × CPU cores. |
| `--device` | `cuda` | Torch device (`cuda` or `cpu`) |
| `--subproc` | `True` | Use SubprocVecEnv (multiprocessing). Default ON. |
| `--no-subproc` | — | Force DummyVecEnv (single-process, debugging only) |
| `--net-arch` | `512 256` | Hidden layer sizes for actor + critic |
| `--n-steps` | `4096` | Steps per env per rollout |
| `--gamma` | `0.99` | Discount factor |
| `--gae-lambda` | `0.95` | GAE lambda |
| `--max-grad-norm` | `0.5` | Gradient clipping |
| `--save-dir` | `./checkpoints` | Checkpoint directory |
| `--save-every` | `500_000` | Periodic checkpoint interval (steps) |
| `--project` | `orca-cube-reorientation` | W&B project name |
| `--entity` | `fourvectors` | W&B team name |
| `--run-name` | auto | W&B run name |
| `--upload-model` | off | Upload final model to W&B as versioned artifact |

---

## 6. Curriculum Design

The curriculum has **8 chapters** of progressively harder cube starting angles.
The agent must achieve **80% success rate** on a 200-episode rolling window before
promoting to the next chapter.

| Chapter | Name | Angle Range | Episodes | Key Skill |
|---------|------|------------|----------|-----------|
| Ch1 | `ch1_small_tilt` | 16°–50° | ≤300 | Basic contact, nudging |
| Ch2a | `ch2a_moderate_tilt` | 40°–62° | ≤350 | Controlled pushing |
| Ch2b | `ch2b_medium_tilt` | 55°–78° | ≤400 | Initiating roll |
| Ch2c | `ch2c_steep_tilt` | 68°–93° | ≤450 | Full rolling skill |
| Ch3a | `ch3a_side_roll` | 82°–115° | ≤500 | Multi-finger coordination |
| Ch3b | `ch3b_deep_roll` | 105°–135° | ≤550 | Pushing past equator |
| Ch4 | `ch4_near_flip` | 122°–163° | ≤650 | Re-catching at apex |
| Ch5 | `ch5_full_flip` | 150°–180° | ≤800 | Full 180° reorientation |

**Why overlapping bands?** Each chapter starts 10–15° before the previous one ends.
This prevents catastrophic forgetting — the agent must still succeed at angles it
has already mastered when it enters the new chapter.

**Why split Ch2 into 3?** The original Ch2 (40°–90°) had a 50° band with a
phase transition at ~75° where the required motor skill changes from "nudging" to
"rolling". With a single 50° band, the agent memorised nudging patterns and
couldn't generalise to rolling. Splitting into 20° sub-chapters eliminates this cliff.

---

## 7. Reward Function

The reward is a sum of 8 components at every step:

```
r = r_align + r_progress + r_pos + r_fingers + r_success + r_alive - r_drop - r_action
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Alignment** | `5.0 × exp(-θ / 0.3)` | Dense orientation signal. Peaks at goal (θ=0). |
| **Progress** | `10.0 × (θ_prev - θ_now)` | Dactyl-style delta. Rewards any rotation toward goal. |
| **Position** | `0.5 × exp(-d_pos / 0.05)` | Keeps cube near palm centre. |
| **Fingertips** | `0.3 × mean(exp(-d_tip / 0.02))` | All 5 fingertips near cube surface. |
| **Success bonus** | `bonus × stability × palm_rest` | Immediate on `is_success=True`. Scaled 30%–100% by cube velocity (stable grasps earn more). |
| **Alive** | `+0.05` | Small per-step incentive to keep going. |
| **Drop penalty** | `-10.0` | Episode-ending if cube falls. |
| **Action regularisation** | `-(rate² + mag²) × weights` | Suppresses jittery joint movements. |

**Key design choice:** There is **no hold timer** and **no velocity gate** on success.
The agent gets the success bonus the instant `is_success=True`. Stability is
incentivised *after* success via the `stability × palm_rest_factor` multiplier,
not as a gate before it. This eliminates the impossible-exploration problem that
caused v0–v1 to achieve 0% success.

---

## 8. Monitoring with W&B

All metrics stream to [wandb.ai/fourvectors/orca-cube-reorientation](https://wandb.ai/fourvectors/orca-cube-reorientation) in real time.

### Key metrics to watch

| Metric | What it means | Healthy range |
|--------|--------------|---------------|
| `rollout/success_rate` | Current rolling success rate | Climbing toward 0.80 |
| `curriculum/rolling_success_rate` | Same as above (curriculum view) | Climbing toward 0.80 |
| `curriculum/chapter_idx` | Current chapter (0–7) | Should increase over time |
| `adaptive/action_std` | Policy action standard deviation | `[0.75, 1.05]` |
| `adaptive/ent_coef` | Current entropy coefficient | Decreasing slowly |
| `adaptive/clip_range` | PPO clip range | Spikes to 0.3 at promotion, decays to 0.2 |
| `train/explained_variance` | How well the value function predicts returns | `> 0.6` is good |
| `train/clip_fraction` | Fraction of clipped PPO updates | `[0.02, 0.08]` is healthy |
| `meta/steps_per_hour` | Training throughput | Higher = faster. 64 envs: ~200k–400k/hour |

### Red flags

| Symptom | Cause | Fix |
|---------|-------|-----|
| `success_rate` flat at 0% for >500k steps | Policy not exploring | Increase `ent_coef` manually or wait for adaptive entropy |
| `action_std` > 1.2 | Entropy too high (jitter) | Adaptive entropy should auto-correct |
| `clip_fraction` = 0 | LR too low or clip_range too tight | Check if clip warm-restart fired after promotion |
| `explained_variance` < 0.2 | Value function capacity bottleneck | Increase `--net-arch` |
| `success_rate` oscillates 60–75% but never reaches 80% | Chapter too hard | Check if Ch2 split is active in curriculum.py |

---

## 9. Resuming After Crash

The training script saves two files after every checkpoint:
- `./checkpoints/latest_model.zip` — the PPO policy weights
- `./checkpoints/curriculum_state.json` — the curriculum manager state (chapter index,
  success history, total steps)

To resume, run the **exact same command** as the original run:
```bash
pixi run curriculum-resume
```
The script detects the checkpoint files and prints `📂 Resumed curriculum at ch2b_medium_tilt`.

At every promotion, a versioned snapshot is also saved:
```
./checkpoints/model_ch2a_moderate_tilt_5000000.zip
./checkpoints/model_ch2b_medium_tilt_12000000.zip
...
```
These let you roll back to any chapter if something goes wrong.

---

## 10. Hardware Recommendations

### Recommended workstation spec

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8-core (for 32 envs) | 12-core Ryzen 9 9900X / 9950X (for 48–64 envs) |
| RAM | 32 GB | **64 GB** (100–150 MB per subprocess × 64 = ~9 GB for envs) |
| GPU | RTX 3080 (10 GB VRAM) | **RTX 4090 / 5090** (PPO gradient updates) |
| Storage | 100 GB SSD | 500 GB NVMe SSD (checkpoints, W&B cache) |
| OS | Ubuntu 22.04 | Ubuntu 22.04 (best MuJoCo subprocess support) |

### n_envs sweet spot

| CPU cores | Recommended n_envs | RAM needed |
|-----------|-------------------|------------|
| 8 | 24–32 | 16 GB |
| 12 | 36–48 | 24 GB |
| 16 | 48–64 | 32 GB |
| 24+ | 64–96 | 48–64 GB |

Rule of thumb: 3–4 environments per physical core, up to your RAM limit.
Every environment uses ~150 MB RAM for MuJoCo + Python overhead.

### GPU usage
The GPU is used **only** for PPO's neural network:
- Forward pass (inference): ~2ms per rollout step
- Backward pass (gradient descent): ~50ms per PPO update

The GPU sits idle 80–90% of the time waiting for CPU physics. This is normal.
A faster GPU does not significantly speed up this training. More CPU cores / more
environments have a larger impact.

---

## 11. Contributing

### Git workflow
```bash
git pull                          # always pull before starting
git checkout -b experiment/my-fix # feature branch for experiments
# ... make changes ...
git commit -m "short description"
git push -u origin experiment/my-fix
# open pull request on GitHub
```

### Adding a new task
1. Create a new reward wrapper (`task_name_reward.py`) modelled on `production_reward.py`.
2. Create a new training script or add `--task` flag to `train_curriculum.py`.
3. Add a new curriculum (`task_name_curriculum.py`) modelled on `curriculum.py`.
4. Log to a new W&B project: `--project orca-<task-name>`.

### Changing reward weights
All reward weights are constructor arguments to `ProductionRewardWrapper`. Pass
them as keyword arguments in `make_env()` inside `train_curriculum.py`. Never
hardcode magic numbers inside the wrapper — always expose them as parameters.

### Running unit tests
```bash
pixi run curriculum-smoke    # 50k steps, verifies the full stack works end-to-end
```

---

## Appendix: Metrics Reference

### Training metrics (SB3)

| Metric | Description |
|--------|-------------|
| `rollout/ep_rew_mean` | Mean episode reward (rolling average) |
| `rollout/ep_len_mean` | Mean episode length |
| `rollout/success_rate` | Fraction of episodes where `is_success=True` at end |
| `train/loss` | Total PPO loss |
| `train/value_loss` | Critic MSE loss |
| `train/policy_gradient_loss` | Actor loss |
| `train/entropy_loss` | Entropy regularisation term |
| `train/approx_kl` | KL divergence between old and new policy |
| `train/clip_fraction` | Fraction of updates that were clipped by PPO |
| `train/explained_variance` | R² of value function predictions |
| `train/std` | Mean action standard deviation |

### Curriculum metrics

| Metric | Description |
|--------|-------------|
| `curriculum/chapter_idx` | Current chapter (0-indexed, 0–7) |
| `curriculum/rolling_success_rate` | Rolling success rate over last 200 episodes |
| `curriculum/chapter_steps` | Steps completed in current chapter |
| `curriculum/total_steps` | Total steps across all chapters |
| `curriculum/promotion_event` | 1 at every promotion event |
| `curriculum/success_rate_at_promotion` | SR at the moment of promotion |

### Adaptive mechanism metrics

| Metric | Description |
|--------|-------------|
| `adaptive/action_std` | Current policy action std (target: [0.75, 1.05]) |
| `adaptive/ent_coef` | Current entropy coefficient |
| `adaptive/clip_range` | Current clip range (spikes at promotion) |
| `meta/wall_time_hours` | Total elapsed wall clock time |
| `meta/steps_per_hour` | Training throughput |
