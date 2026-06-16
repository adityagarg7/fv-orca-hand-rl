"""
Four Vectors -- ORCA Hand in-hand cube reorientation (PPO).

Trains a PPO policy with the production reward wrapper and logs everything to
Weights & Biases:
  * metrics are auto-synced from Stable-Baselines3's TensorBoard output, and
  * the trained model is saved as a versioned W&B artifact.

The script runs identically on a local machine and on Colab. Step count and a
few other knobs are command-line flags, so you never need to edit this file to
do a quick test run.

Examples
--------
Quick smoke test (a few minutes, just to confirm the pipeline works):
    python train.py --timesteps 20000 --run-name smoke-test

Full run:
    python train.py --timesteps 20000000 --run-name prod-20M

Resume from a local checkpoint:
    python train.py --timesteps 20000000 --resume checkpoints/<run_id>/ppo_production_2000000_steps.zip
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

import wandb
from wandb.integration.sb3 import WandbCallback

from orca_sim import OrcaHandRightCubeOrientation
from production_reward import ProductionRewardWrapper


class ProgressLogger(BaseCallback):
    """Prints a one-line reward breakdown every `log_freq` steps."""

    def __init__(self, log_freq=200_000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            for info in self.locals.get("infos", []):
                if "reward_breakdown" in info:
                    bd = info["reward_breakdown"]
                    parts = " | ".join(f"{k}={v:.2f}" for k, v in bd.items())
                    print(f"  [step {self.num_timesteps:>10,d}] {parts}")
                    break
        return True


def make_production_env():
    return ProductionRewardWrapper(OrcaHandRightCubeOrientation(render_mode=None))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train PPO on ORCA cube reorientation with Weights & Biases logging."
    )
    p.add_argument("--timesteps", type=int, default=20_000_000,
                   help="Total training timesteps (default: 20M). Use e.g. 20000 for a smoke test.")
    p.add_argument("--n-envs", type=int, default=4,
                   help="Number of parallel environments (default: 4).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Torch device. CPU is recommended for this MLP policy (default: cpu).")
    p.add_argument("--project", default="orca-cube-reorientation",
                   help="W&B project name.")
    p.add_argument("--entity", default=None,
                   help="W&B entity (your team/org). Defaults to your personal entity.")
    p.add_argument("--run-name", default=None,
                   help="Optional human-readable name for this run.")
    p.add_argument("--save-freq", type=int, default=500_000,
                   help="Checkpoint cadence in total timesteps (default: 500k).")
    p.add_argument("--resume", default=None,
                   help="Path to a .zip checkpoint to resume training from.")
    return p.parse_args()


def main():
    args = parse_args()

    config = {
        "algo": "PPO",
        "policy": "MlpPolicy",
        "total_timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "env": "OrcaHandRightCubeOrientation",
        "reward": "ProductionRewardWrapper",
    }

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config=config,
        sync_tensorboard=True,   # upload SB3's TensorBoard metrics to W&B automatically
        save_code=True,          # snapshot the training code alongside the run
    )

    print("=" * 60)
    print("  ORCA Cube Reorientation -- PPO + Weights & Biases")
    print(f"  Run:    {run.name}  ({run.id})")
    print(f"  Steps:  {args.timesteps:,}   Envs: {args.n_envs}   Device: {args.device}")
    print(f"  W&B:    {run.url}")
    print("=" * 60)

    env = make_vec_env(make_production_env, n_envs=args.n_envs)
    tb_dir = f"tb_production/{run.id}"

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = PPO.load(args.resume, env=env, device=args.device, tensorboard_log=tb_dir)
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1,
            n_steps=config["n_steps"], batch_size=config["batch_size"],
            n_epochs=config["n_epochs"], learning_rate=config["learning_rate"],
            gamma=config["gamma"], gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"], ent_coef=config["ent_coef"],
            device=args.device, tensorboard_log=tb_dir,
        )

    # CheckpointCallback and WandbCallback count per-env steps, so convert the
    # total-timestep cadence into per-env steps.
    per_env_save_freq = max(1, args.save_freq // args.n_envs)

    callbacks = [
        ProgressLogger(log_freq=200_000),
        # Local checkpoints -- handy for fast in-session resume.
        CheckpointCallback(
            save_freq=per_env_save_freq,
            save_path=f"checkpoints/{run.id}",
            name_prefix="ppo_production",
        ),
        # Periodic + final model upload to W&B (survives a Colab disconnect).
        WandbCallback(
            model_save_path=f"models/{run.id}",
            model_save_freq=per_env_save_freq,
            verbose=2,
        ),
    ]

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name="ppo_production",
        reset_num_timesteps=not bool(args.resume),
    )

    # Final save: local file + a cleanly-named, versioned W&B artifact that a
    # teammate can pull by name (model-<run_id>:latest).
    final_path = "ppo_orca_production_final.zip"
    model.save(final_path)
    artifact = wandb.Artifact(
        f"model-{run.id}", type="model",
        metadata={"timesteps": args.timesteps, "env": config["env"]},
    )
    artifact.add_file(final_path)
    run.log_artifact(artifact)
    print(f"\nTraining complete. Saved {final_path} and logged it to W&B as model-{run.id}.")

    env.close()
    run.finish()


if __name__ == "__main__":
    main()
