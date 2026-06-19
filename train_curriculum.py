"""
Crash-proof curriculum training for ORCA cube reorientation.

Designed to survive Colab/Kaggle runtime crashes:
  - Saves model checkpoint + curriculum state every SAVE_EVERY steps
  - On restart, auto-detects and resumes from latest checkpoint
  - All state saved to a single directory (mount Google Drive / Kaggle output)

Usage (Colab):
    !python train_curriculum.py --save-dir /content/drive/MyDrive/orca_checkpoints

Usage (Kaggle):
    !python train_curriculum.py --save-dir /kaggle/working/orca_checkpoints

Usage (Local):
    python train_curriculum.py --save-dir ./checkpoints

The script will auto-resume if it finds existing checkpoints in --save-dir.
"""

import argparse
import glob
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import wandb

from orca_sim import OrcaHandRightCubeOrientation
from production_reward import ProductionRewardWrapper
from curriculum import CurriculumManager, CHAPTERS
from curriculum_wrapper import CurriculumWrapper


# ── Shared curriculum manager (accessible by all parallel envs) ──────
# We use a single global instance so all 4 vec-envs share the same
# curriculum state.  This is safe because SB3's SubprocVecEnv copies
# the env at creation time, but DummyVecEnv (our default) shares memory.
_CURRICULUM_MANAGER: CurriculumManager | None = None


def get_curriculum() -> CurriculumManager:
    global _CURRICULUM_MANAGER
    if _CURRICULUM_MANAGER is None:
        _CURRICULUM_MANAGER = CurriculumManager()
    return _CURRICULUM_MANAGER


def make_env():
    """Create the full env stack: OrcaSim → RewardWrapper → CurriculumWrapper."""
    base = OrcaHandRightCubeOrientation(render_mode=None)
    rewarded = ProductionRewardWrapper(base)
    return CurriculumWrapper(rewarded, get_curriculum())


# ── Callbacks ────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """Handles auto-promotion, checkpointing, and W&B logging."""

    def __init__(self, curriculum: CurriculumManager, save_dir: str,
                 save_every: int = 100_000, log_freq: int = 8_192):
        super().__init__()
        self.curriculum = curriculum
        self.save_dir = save_dir
        self.save_every = save_every
        self.log_freq = log_freq
        self._last_save_step = 0
        self._last_promotion_check = 0

    def _on_step(self) -> bool:
        # Track steps in curriculum
        self.curriculum.record_steps(self.training_env.num_envs)

        # ── Periodic logging ────────────────────────────────────────
        if self.n_calls % (self.log_freq // self.training_env.num_envs) == 0:
            status = self.curriculum.status_dict()
            if wandb.run is not None:
                wandb.log(status, step=self.num_timesteps)

            # Print compact status
            ch = self.curriculum.current_chapter
            print(f"  📊 Ch{self.curriculum.current_chapter_idx + 1} "
                  f"[{ch.name}] "
                  f"sr={self.curriculum.rolling_success_rate:.1%} "
                  f"(need {ch.promotion_threshold:.0%}) "
                  f"| steps={self.curriculum.chapter_steps:,}"
                  f"/{self.curriculum.total_steps:,}")

        # ── Final chapter completion check (MUST be before promotion) ──
        # This is separate from should_promote() because should_promote()
        # returns False for the final chapter (nothing to promote to).
        if self.curriculum.is_final_chapter:
            target = self.curriculum.current_chapter.promotion_threshold
            if (len(self.curriculum._success_history) >= self.curriculum.ROLLING_WINDOW
                    and self.curriculum.rolling_success_rate >= target):
                print(f"\n🏆 CURRICULUM COMPLETE! Final success rate: "
                      f"{self.curriculum.rolling_success_rate:.1%}")
                if wandb.run is not None:
                    wandb.log({
                        "curriculum/completed": 1,
                        "curriculum/final_success_rate": self.curriculum.rolling_success_rate,
                    }, step=self.num_timesteps)
                self._save_checkpoint("final")
                return False  # Stop training

        # ── Auto-promotion check (chapters 1-4 only) ────────────────
        if self.curriculum.should_promote():
            # Log promotion event to W&B
            if wandb.run is not None:
                wandb.log({
                    "curriculum/promotion_event": 1,
                    "curriculum/promoted_from": self.curriculum.current_chapter_idx,
                    "curriculum/promoted_at_step": self.num_timesteps,
                    "curriculum/success_rate_at_promotion": self.curriculum.rolling_success_rate,
                }, step=self.num_timesteps)

            self.curriculum.promote()

            # Save checkpoint at promotion
            self._save_checkpoint("promotion")

            # Update PPO hyperparameters for new chapter
            new_ch = self.curriculum.current_chapter
            self.model.learning_rate = new_ch.lr
            self.model.n_epochs = new_ch.n_epochs
            self.model.batch_size = new_ch.batch_size
            self.model.ent_coef = new_ch.ent_coef
            print(f"  ⚙️  Updated hyperparams: lr={new_ch.lr}, "
                  f"epochs={new_ch.n_epochs}, batch={new_ch.batch_size}, "
                  f"ent_coef={new_ch.ent_coef}")

        # ── Periodic checkpoint ─────────────────────────────────────
        if (self.num_timesteps - self._last_save_step) >= self.save_every:
            self._save_checkpoint("periodic")
            self._last_save_step = self.num_timesteps

        return True

    def _save_checkpoint(self, reason: str):
        """Save model + curriculum state for crash recovery."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Save PPO model
        model_path = os.path.join(self.save_dir, "latest_model.zip")
        self.model.save(model_path)

        # Save curriculum state
        curriculum_path = os.path.join(self.save_dir, "curriculum_state.json")
        self.curriculum.save_state(curriculum_path)

        # Save a versioned copy at promotions
        if reason == "promotion":
            ch_name = self.curriculum.current_chapter.name
            step = self.curriculum.total_steps
            versioned = os.path.join(self.save_dir,
                                     f"model_{ch_name}_{step}.zip")
            self.model.save(versioned)

        print(f"  💾 Checkpoint saved ({reason}): {self.save_dir}")


class ProgressLogger(BaseCallback):
    """Print reward breakdown periodically."""

    def __init__(self, log_freq=200_000):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            for info in self.locals.get("infos", []):
                if "reward_breakdown" in info:
                    parts = " | ".join(f"{k}={v:.2f}"
                                       for k, v in info["reward_breakdown"].items())
                    print(f"  [step {self.num_timesteps:>10,d}] {parts}")
                    break
        return True


# ── Main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Curriculum training for ORCA cube reorientation.")
    p.add_argument("--timesteps", type=int, default=10_000_000,
                   help="Max total timesteps across all chapters.")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--save-dir", default="./checkpoints_curriculum",
                   help="Directory for checkpoints (use Google Drive path on Colab).")
    p.add_argument("--save-every", type=int, default=100_000,
                   help="Save checkpoint every N steps.")
    p.add_argument("--project", default="orca-cube-reorientation")
    p.add_argument("--entity", default="fourvectors")
    p.add_argument("--run-name", default=None)
    p.add_argument("--upload-model", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Initialize curriculum ────────────────────────────────────────
    curriculum = get_curriculum()

    # ── Check for existing checkpoint (crash recovery) ───────────────
    curriculum_state_path = os.path.join(args.save_dir, "curriculum_state.json")
    model_path = os.path.join(args.save_dir, "latest_model.zip")
    resuming = curriculum.load_state(curriculum_state_path)

    # ── W&B init ─────────────────────────────────────────────────────
    ch = curriculum.current_chapter
    config = dict(
        algo="PPO", policy="MlpPolicy", env="OrcaHandRightCubeOrientation",
        reward="ProductionRewardWrapper-v1.1",
        curriculum="5-chapter-progressive",
        starting_chapter=ch.name,
        total_timesteps=args.timesteps, n_envs=args.n_envs,
        n_steps=2048, batch_size=ch.batch_size, n_epochs=ch.n_epochs,
        learning_rate=ch.lr, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=ch.ent_coef, max_grad_norm=0.5,
        resumed=resuming,
    )

    run_name = args.run_name or f"curriculum-{ch.name}"
    run = wandb.init(project=args.project, entity=args.entity,
                     name=run_name, config=config,
                     sync_tensorboard=True, save_code=True,
                     resume="allow" if resuming else None)

    print(f"\n{'='*60}")
    print(f"  ORCA Curriculum Training")
    print(f"  Run: {run.name} ({run.id})")
    print(f"  Chapter: {ch.name} (angles {ch.angle_min_deg}°–{ch.angle_max_deg}°)")
    print(f"  Target: {ch.promotion_threshold:.0%} success rate")
    print(f"  Device: {args.device} | Envs: {args.n_envs}")
    print(f"  Checkpoints: {args.save_dir}")
    print(f"  Resumed: {resuming}")
    print(f"  W&B: {run.url}")
    print(f"{'='*60}\n")

    # ── Create vectorized environment ────────────────────────────────
    env = make_vec_env(make_env, n_envs=args.n_envs)
    tb_dir = os.path.join(args.save_dir, "tensorboard_logs")  # fixed name for resume continuity

    # ── Create or load PPO model ─────────────────────────────────────
    if resuming and os.path.exists(model_path):
        print(f"📂 Resuming PPO model from {model_path}")
        model = PPO.load(model_path, env=env, device=args.device,
                         tensorboard_log=tb_dir)
        # Apply current chapter's hyperparameters
        model.learning_rate = ch.lr
        model.n_epochs = ch.n_epochs
        model.batch_size = ch.batch_size
        model.ent_coef = ch.ent_coef
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1,
            device=args.device, tensorboard_log=tb_dir,
            n_steps=2048, batch_size=ch.batch_size,
            n_epochs=ch.n_epochs, learning_rate=ch.lr,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=ch.ent_coef, max_grad_norm=0.5,
        )

    # ── Remaining timesteps ──────────────────────────────────────────
    remaining = max(0, args.timesteps - curriculum.total_steps)
    if remaining == 0:
        print("⚠️  Already reached total timestep budget. Exiting.")
        return

    print(f"  Training for {remaining:,} more steps "
          f"(already completed {curriculum.total_steps:,})\n")

    # ── Train! ───────────────────────────────────────────────────────
    callbacks = [
        CurriculumCallback(curriculum, args.save_dir,
                           save_every=args.save_every),
        ProgressLogger(),
    ]

    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        tb_log_name="curriculum",
        reset_num_timesteps=not resuming,
    )

    # ── Final save ───────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, "ppo_curriculum_final.zip")
    model.save(final_path)
    curriculum.save_state(curriculum_state_path)

    if args.upload_model and wandb.run is not None:
        artifact = wandb.Artifact(f"model-curriculum-{run.id}", type="model",
                                  metadata={"chapters_completed": curriculum.current_chapter_idx + 1,
                                            "final_success_rate": curriculum.rolling_success_rate})
        artifact.add_file(final_path)
        run.log_artifact(artifact)
        print(f"\n📤 Uploaded model to W&B.")

    print(f"\n✅ Training complete!")
    print(f"   Final chapter: {curriculum.current_chapter.name}")
    print(f"   Final success rate: {curriculum.rolling_success_rate:.1%}")
    print(f"   Total steps: {curriculum.total_steps:,}")

    env.close()
    run.finish()


if __name__ == "__main__":
    main()
