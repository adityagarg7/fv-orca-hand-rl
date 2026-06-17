"""
PPO training for ORCA in-hand cube reorientation, logging to Weights & Biases.

    python train.py --timesteps 20000 --run-name smoke-test                # quick test
    python train.py --timesteps 20000000 --run-name prod --upload-model     # full run, keep model
    python train.py --resume checkpoints/<id>/ppo_production_2000000_steps.zip --upload-model
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import wandb

from orca_sim import OrcaHandRightCubeOrientation
from production_reward import ProductionRewardWrapper


class ProgressLogger(BaseCallback):
    """Print the reward breakdown every `log_freq` steps."""

    def __init__(self, log_freq=200_000):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            for info in self.locals.get("infos", []):
                if "reward_breakdown" in info:
                    parts = " | ".join(f"{k}={v:.2f}" for k, v in info["reward_breakdown"].items())
                    print(f"  [step {self.num_timesteps:>10,d}] {parts}")
                    break
        return True


def make_env():
    return ProductionRewardWrapper(OrcaHandRightCubeOrientation(render_mode=None))


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on ORCA cube reorientation with W&B logging.")
    p.add_argument("--timesteps", type=int, default=20_000_000, help="Total timesteps (~20000 for a smoke test).")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="CPU is best for this MLP policy.")
    p.add_argument("--project", default="orca-cube-reorientation", help="W&B project.")
    p.add_argument("--entity", default="fourvectors", help="W&B entity (team/org).")
    p.add_argument("--run-name", default=None)
    p.add_argument("--save-freq", type=int, default=100_000, help="Local checkpoint cadence in timesteps.")
    p.add_argument("--resume", default=None, help="Checkpoint .zip to resume from.")
    p.add_argument("--upload-model", action="store_true",
                   help="Upload the trained model to W&B. Off by default so test runs stay fast.")
    return p.parse_args()


def main():
    args = parse_args()

    config = dict(
        algo="PPO", policy="MlpPolicy", env="OrcaHandRightCubeOrientation",
        reward="ProductionRewardWrapper", total_timesteps=args.timesteps, n_envs=args.n_envs,
        n_steps=2048, batch_size=256, n_epochs=10, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    )

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name,
                     config=config, sync_tensorboard=True, save_code=True)
    print(f"Run {run.name} ({run.id}) | {args.timesteps:,} steps | {args.n_envs} envs | "
          f"{args.device} | upload_model={args.upload_model}\n{run.url}")

    env = make_vec_env(make_env, n_envs=args.n_envs)
    tb_dir = f"tb_production/{run.id}"

    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device, tensorboard_log=tb_dir)
    else:
        model = PPO("MlpPolicy", env, verbose=1, device=args.device, tensorboard_log=tb_dir,
                    n_steps=config["n_steps"], batch_size=config["batch_size"], n_epochs=config["n_epochs"],
                    learning_rate=config["learning_rate"], gamma=config["gamma"],
                    gae_lambda=config["gae_lambda"], clip_range=config["clip_range"], ent_coef=config["ent_coef"])

    # save_freq is per-env, so divide the total-step cadence by n_envs.
    checkpoints = CheckpointCallback(save_freq=max(1, args.save_freq // args.n_envs),
                                     save_path=f"checkpoints/{run.id}", name_prefix="ppo_production")

    model.learn(total_timesteps=args.timesteps, callback=[ProgressLogger(), checkpoints],
                tb_log_name="ppo_production", reset_num_timesteps=not args.resume)

    model.save("ppo_orca_production_final.zip")
    if args.upload_model:
        artifact = wandb.Artifact(f"model-{run.id}", type="model", metadata={"timesteps": args.timesteps})
        artifact.add_file("ppo_orca_production_final.zip")
        run.log_artifact(artifact)
        print(f"Uploaded model-{run.id} to W&B.")

    env.close()
    run.finish()


if __name__ == "__main__":
    main()
