"""
PPO training for ORCA in-hand cube reorientation, logging to Weights & Biases.

    python train.py --timesteps 20000 --run-name smoke-test                # quick test
    python train.py --timesteps 20000000 --run-name prod --upload-model     # full run, keep model
    python train.py --resume checkpoints/<id>/ppo_production_2000000_steps.zip --upload-model
"""

import argparse
import functools

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import wandb

from orca_sim import OrcaHandRightCubeOrientation
from reward_wrappers import PotentialShapedReorientationReward
from curriculum import GravityCurriculumWrapper, GravityCurriculumCallback


class ProgressLogger(BaseCallback):
    """Print the cube-orientation state every `log_freq` steps."""

    def __init__(self, log_freq=200_000):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            for info in self.locals.get("infos", []):
                if "red_face_up_angle_rad" in info:
                    print(
                        f"  [step {self.num_timesteps:>10,d}] "
                        f"angle_rad={info['red_face_up_angle_rad']:.2f} | "
                        f"alignment={info['red_face_up_alignment']:+.2f} | "
                        f"success={info['is_success']} | dropped={info['dropped']}"
                    )
                    break
        return True


def make_env(align_coeff, success_bonus, time_penalty, drop_penalty, gamma, gravity_start):
    """Build a single wrapped env. Called once per sub-env by make_vec_env, so each
    sub-env gets its own reward wrapper instance (and its own potential tracking).
    The env starts at the (reduced) curriculum gravity; the callback ramps it up."""
    env = OrcaHandRightCubeOrientation(render_mode=None)
    env = PotentialShapedReorientationReward(
        env,
        align_coeff=align_coeff,
        success_bonus=success_bonus,
        time_penalty=time_penalty,
        drop_penalty=drop_penalty,
        gamma=gamma,
    )
    return GravityCurriculumWrapper(env, gravity=-abs(gravity_start))


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on ORCA cube reorientation with W&B logging.")
    p.add_argument("--timesteps", type=int, default=20_000_000, help="Total timesteps (~20000 for a smoke test).")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="CPU is best for this MLP policy.")
    p.add_argument("--project", default="orca-cube-reorientation", help="W&B project.")
    p.add_argument("--entity", default="fourvectors", help="W&B entity (team/org).")
    p.add_argument("--run-name", default=None)
    p.add_argument("--save-freq", type=int, default=500_000, help="Local checkpoint cadence in timesteps.")
    p.add_argument("--resume", default=None, help="Checkpoint .zip to resume from.")
    p.add_argument("--upload-model", action="store_true",
                   help="Upload the trained model to W&B. Off by default so test runs stay fast.")
    # Reward-shaping knobs (potential-based shaping; see reward_wrappers.py).
    p.add_argument("--align-coeff", type=float, default=1.0,
                   help="Scale on the potential-based alignment-progress reward.")
    p.add_argument("--success-bonus", type=float, default=10.0,
                   help="Terminal reward added when the cube reaches the goal orientation.")
    p.add_argument("--time-penalty", type=float, default=0.01,
                   help="Constant per-step penalty; encourages faster solves.")
    p.add_argument("--drop-penalty", type=float, default=5.0,
                   help="Terminal penalty when the cube is dropped.")
    # Gravity curriculum (see curriculum.py): train under reduced gravity first so the
    # wrist-dump cheat can't work, then raise gravity only once the policy is competent
    # at the current level. Set --gravity-start equal to --gravity-final to disable.
    p.add_argument("--gravity-start", type=float, default=2.0,
                   help="Initial gravity magnitude (m/s^2); reduced so the cube can't be dumped.")
    p.add_argument("--gravity-final", type=float, default=9.81,
                   help="Final (full) gravity magnitude (m/s^2).")
    p.add_argument("--gravity-success-threshold", type=float, default=0.75,
                   help="Success rate at the current gravity required before raising it.")
    p.add_argument("--gravity-step", type=float, default=1.0,
                   help="Gravity magnitude increment (m/s^2) per promotion.")
    p.add_argument("--gravity-min-episodes", type=int, default=50,
                   help="Min episodes at the current gravity before a promotion can trigger.")
    return p.parse_args()


def main():
    args = parse_args()

    config = dict(
        algo="PPO", policy="MlpPolicy", env="OrcaHandRightCubeOrientation",
        reward="potential_shaped_v1",
        align_coeff=args.align_coeff, success_bonus=args.success_bonus,
        time_penalty=args.time_penalty, drop_penalty=args.drop_penalty,
        gravity_start=args.gravity_start, gravity_final=args.gravity_final,
        gravity_success_threshold=args.gravity_success_threshold,
        gravity_step=args.gravity_step, gravity_min_episodes=args.gravity_min_episodes,
        total_timesteps=args.timesteps, n_envs=args.n_envs,
        n_steps=2048, batch_size=256, n_epochs=10, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    )

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name,
                     config=config, sync_tensorboard=True, save_code=True)
    print(f"Run {run.name} ({run.id}) | {args.timesteps:,} steps | {args.n_envs} envs | "
          f"{args.device} | upload_model={args.upload_model}\n{run.url}")

    # Bind the reward-shaping coefficients (and the training gamma, so PBRS uses the
    # same discount) into a zero-arg factory that make_vec_env calls per sub-env.
    env_fn = functools.partial(
        make_env,
        align_coeff=args.align_coeff,
        success_bonus=args.success_bonus,
        time_penalty=args.time_penalty,
        drop_penalty=args.drop_penalty,
        gamma=config["gamma"],
        gravity_start=args.gravity_start,
    )
    env = make_vec_env(env_fn, n_envs=args.n_envs)
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

    # Performance-gated gravity curriculum: start at reduced gravity (so the wrist-dump
    # cheat can't work) and raise it only once the policy is competent at the current
    # level, so it can't ramp into the cheat before learning finger manipulation.
    gravity = GravityCurriculumCallback(
        g_start=args.gravity_start, g_final=args.gravity_final,
        success_threshold=args.gravity_success_threshold,
        step=args.gravity_step, min_episodes=args.gravity_min_episodes)

    model.learn(total_timesteps=args.timesteps, callback=[ProgressLogger(), checkpoints, gravity],
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
