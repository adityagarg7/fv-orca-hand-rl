"""
ORCA Hand — Curriculum RL Training  (v7 — Desktop/Workstation Edition)
=======================================================================
Industry-grade training script for the ORCA dexterous hand cube reorientation
task.  Designed to fully utilise a high-end workstation (12+ CPU cores,
64 GB RAM, RTX-class GPU).

Architecture:
  ┌─────────────────────────────────────────────┐
  │  SubprocVecEnv (64 processes, 64 CPU cores)  │   ← Physics on CPU
  │  Each process: OrcaSim → RewardWrapper → CW  │
  └────────────────────┬────────────────────────┘
                       │  observations / actions
  ┌────────────────────▼────────────────────────┐
  │  PPO (MlpPolicy [512, 256])                 │   ← Inference + gradients on GPU
  │  n_steps = 4096 per env → 262k per rollout  │
  └─────────────────────────────────────────────┘

Key design decisions (backed by research):
  1. SubprocVecEnv @ 64 envs  — each env in its own OS process, true CPU
     parallelism (DexPBT, OpenAI Dactyl style).
  2. [512, 256] network — enough capacity to represent complex finger-cube
     contact policies without overfitting to a single chapter.
  3. 8-chapter finer curriculum — Ch2 split into 3 sub-chapters (20° bands)
     eliminates the "phase transition cliff" between nudging and rolling.
  4. Adaptive entropy (AE-PPO) — auto-adjusts ent_coef to keep action std
     in [0.75, 1.05] sweet spot.  Wider std → reduce ent_coef; collapsed
     std → increase ent_coef.
  5. Adaptive clip warm-restart — widens clip_range to 0.3 at each chapter
     promotion, decays to 0.2 over 300k steps to help the policy distribution
     shift without catastrophic forgetting.
  6. 150M step budget — sufficient for all 8 chapters on this hardware.
     Actual run time: ~6–10 hours on a 12-core workstation.

Usage (workstation with CUDA GPU):
    python train_curriculum.py \\
        --save-dir ./checkpoints \\
        --n-envs 64 \\
        --device cuda \\
        --subproc \\
        --timesteps 150_000_000 \\
        --run-name v7-64envs-desktop

Usage (CPU-only, 12 cores):
    python train_curriculum.py \\
        --save-dir ./checkpoints \\
        --n-envs 48 \\
        --device cpu \\
        --subproc

Crash recovery:
    The script auto-detects an existing checkpoint in --save-dir and resumes
    seamlessly.  Re-run the exact same command after a crash.
"""

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

from orca_sim import OrcaHandRightCubeOrientation
from production_reward import ProductionRewardWrapper
from curriculum import CurriculumManager, CHAPTERS
from curriculum_wrapper import CurriculumWrapper


# ── Environment factory (SubprocVecEnv-compatible) ────────────────────────────
# Each subprocess spawns its own independent Python interpreter and MuJoCo
# instance.  The CurriculumManager in each subprocess is a "follower" — it
# only handles spawn-angle sampling.  The "leader" CurriculumManager lives
# in the main process and is the sole authority for promotion decisions.
# Chapter changes are broadcast to all followers via env_method().

def make_env(start_chapter_idx: int = 0):
    """Factory closure: returns a callable that creates one full env stack."""
    def _init():
        cm = CurriculumManager()
        cm._chapter_idx = start_chapter_idx   # sync to current chapter
        base = OrcaHandRightCubeOrientation(render_mode=None)
        rewarded = ProductionRewardWrapper(base)
        env = CurriculumWrapper(rewarded, cm)
        # info_keywords=("is_success",) tells Monitor to forward is_success
        # into episode stats, which SB3 uses for rollout/success_rate logging.
        return Monitor(env, info_keywords=("is_success",))
    return _init


# ── Callbacks ─────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """Master callback: handles promotion, checkpointing, adaptive hyperparams,
    W&B reward-component logging, and SubprocVecEnv chapter broadcasting.

    Research-backed mechanisms:
      • AE-PPO adaptive entropy (arXiv:2209.07886): keeps action std in target band.
      • Clip-range warm restart (DexPBT / PL-CGS): widens clip at chapter boundary
        to allow larger policy updates when the task distribution shifts.
      • Reward component logging: individual terms tracked in W&B for debugging
        reward engineering decisions.
    """

    # ── Adaptive clip range ──────────────────────────────────────────────────
    CLIP_WARMUP_STEPS = 300_000    # steps to decay clip_range after promotion
    CLIP_RANGE_HIGH   = 0.30       # widened clip at promotion (allows bigger updates)
    CLIP_RANGE_LOW    = 0.20       # steady-state clip range

    # ── Adaptive entropy (AE-PPO) ─────────────────────────────────────────────
    STD_HIGH        = 1.05         # collapse ent_coef if std exceeds this
    STD_LOW         = 0.75         # boost  ent_coef if std falls below this
    ENT_COEF_MIN    = 0.005        # entropy floor (raised 100×: prevents exploration collapse)
    ENT_COEF_MAX    = 0.02         # entropy ceiling (prevents jitter)
    ENT_ADJUST_RATE = 0.92         # multiplicative adjustment per log interval

    # ── Success tracking ───────────────────────────────────────────────────
    # We maintain our OWN rolling success buffer in the callback (main
    # process), because:
    #   - SubprocVecEnv episodes complete in isolated OS processes, so the
    #     main-process CurriculumManager's _success_history stays empty.
    #   - SB3's ep_success_buffer is an internal attribute whose name and
    #     availability varies across versions.  Relying on it is fragile.
    #
    # Instead, we scrape is_success directly from self.locals["infos"],
    # which SB3 populates from the SubprocVecEnv return values on every
    # step.  When an episode ends, SB3 stashes the terminal info in
    # info["terminal_info"] or info["episode"], depending on the VecEnv
    # auto-reset behavior.
    SUCCESS_WINDOW = 200   # rolling window size (matches CurriculumManager)

    def __init__(
        self,
        curriculum: CurriculumManager,
        save_dir: str,
        save_every: int = 200_000,
        log_freq: int = 16_384,    # steps between status prints (scales with n_envs)
    ):
        super().__init__(verbose=1)
        self.curriculum      = curriculum
        self.save_dir        = save_dir
        self.save_every      = save_every
        self.log_freq        = log_freq
        self._last_save_step = 0
        self._clip_warmup_remaining = 0
        self._run_start_time = time.time()
        self._reward_accum   = {}   # for rolling reward component averages
        # Our own success tracking buffer (main process, immune to SubprocVecEnv isolation)
        self._success_buffer: deque[bool] = deque(maxlen=self.SUCCESS_WINDOW)
        # v9: Sustained promotion gate — must pass threshold for N consecutive
        # log-interval checks.  Eliminates false promotions from statistical
        # noise (the v8 bug where Ch2a promoted at ~72% true rate).
        self.PROMOTE_SUSTAIN_COUNT = 3
        self._promote_streak = 0

    # ── Per-step logic ───────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        n_envs = self.training_env.num_envs

        # Track steps in curriculum leader
        self.curriculum.record_steps(n_envs)

        # ── Scrape is_success from info dicts (SubprocVecEnv-safe) ──────────
        # self.locals["infos"] is a list of info dicts, one per env.
        # When an episode ends in a VecEnv, the env auto-resets and SB3
        # stores the terminal step's info in either:
        #   - info["terminal_info"] (SB3 ≥ 2.1 with new VecEnv API), or
        #   - info["terminal_observation"] existing + info itself having is_success
        # We check both paths for robustness.
        infos = self.locals.get("infos", [])
        for info in infos:
            # Check if this env just finished an episode
            # SB3 VecEnv sets "episode" key (from Monitor) when episode ends
            if "episode" in info:
                # Episode just ended — check is_success
                # Try terminal_info first (newer SB3), then fall back to info itself
                terminal_info = info.get("terminal_info", info)
                success = bool(terminal_info.get("is_success", False))
                self._success_buffer.append(success)

        # Push our tracked success rate to the curriculum leader
        if len(self._success_buffer) > 0:
            sr = sum(self._success_buffer) / len(self._success_buffer)
            self.curriculum.override_success_rate(sr, len(self._success_buffer))

        # ── Adaptive clip range decay ─────────────────────────────────────
        if self._clip_warmup_remaining > 0:
            self._clip_warmup_remaining -= n_envs
            progress = max(0.0, self._clip_warmup_remaining / self.CLIP_WARMUP_STEPS)
            new_clip = self.CLIP_RANGE_LOW + (self.CLIP_RANGE_HIGH - self.CLIP_RANGE_LOW) * progress
            self.model.clip_range = lambda _: new_clip

        # ── Periodic logging (adaptive entropy on slow timescale) ─────────
        log_interval_steps = self.log_freq // n_envs
        if self.n_calls % max(1, log_interval_steps) == 0:
            status = self.curriculum.status_dict()

            # Adaptive entropy: read current action std from policy
            current_std = self._get_action_std()
            if current_std is not None:
                if current_std > self.STD_HIGH:
                    self.model.ent_coef = max(
                        self.ENT_COEF_MIN,
                        self.model.ent_coef * self.ENT_ADJUST_RATE
                    )
                elif current_std < self.STD_LOW:
                    self.model.ent_coef = min(
                        self.ENT_COEF_MAX,
                        self.model.ent_coef / self.ENT_ADJUST_RATE
                    )
                status["adaptive/action_std"] = current_std
                status["adaptive/ent_coef"]   = self.model.ent_coef

            # Current clip range
            clip_val = (self.model.clip_range(1.0)
                        if callable(self.model.clip_range)
                        else self.model.clip_range)
            status["adaptive/clip_range"] = clip_val

            # Training throughput
            elapsed = time.time() - self._run_start_time
            status["meta/wall_time_hours"] = elapsed / 3600.0
            status["meta/steps_per_hour"]  = (
                self.num_timesteps / elapsed * 3600.0 if elapsed > 0 else 0
            )

            if wandb.run is not None:
                wandb.log(status, step=self.num_timesteps)

            # Compact console status
            ch      = self.curriculum.current_chapter
            std_str = f" std={current_std:.3f}" if current_std else ""
            hours   = elapsed / 3600.0
            print(
                f"  📊 Ch{self.curriculum.current_chapter_idx + 1}"
                f"[{ch.name}]"
                f"  sr={self.curriculum.rolling_success_rate:.1%}"
                f" (need {ch.promotion_threshold:.0%})"
                f" | buf={len(self._success_buffer)}/{self.SUCCESS_WINDOW}"
                f" | steps={self.curriculum.total_steps:,}"
                f" ch_steps={self.curriculum.chapter_steps:,}"
                f"{std_str}"
                f"  ent={self.model.ent_coef:.5f}"
                f"  ⏱ {hours:.1f}h"
            )

        # ── Promotion & completion checks (LOG INTERVAL ONLY) ──────────────
        # v9 FIX: Promotion is checked ONLY at log intervals, NOT every step.
        # In v8, the check ran 4096× per log interval, giving thousands of
        # chances to hit a statistical spike.  This caused false promotions
        # at ~72-75% true rate.  Now we check once per log interval and
        # require PROMOTE_SUSTAIN_COUNT (3) consecutive passes.
        if self.n_calls % max(1, log_interval_steps) == 0:

            # ── Final chapter completion check ────────────────────────────
            if self.curriculum.is_final_chapter:
                target = self.curriculum.current_chapter.promotion_threshold
                if (len(self._success_buffer) >= self.SUCCESS_WINDOW
                        and self.curriculum.rolling_success_rate >= target):
                    self._promote_streak += 1
                    if self._promote_streak >= self.PROMOTE_SUSTAIN_COUNT:
                        print(f"\n🏆 CURRICULUM COMPLETE! "
                              f"Final success rate: {self.curriculum.rolling_success_rate:.1%}"
                              f" (sustained {self.PROMOTE_SUSTAIN_COUNT} checks)")
                        if wandb.run is not None:
                            wandb.log({
                                "curriculum/completed": 1,
                                "curriculum/final_success_rate": self.curriculum.rolling_success_rate,
                            }, step=self.num_timesteps)
                        self._save_checkpoint("final")
                        return False   # stop training
                else:
                    self._promote_streak = 0

            # ── Auto-promotion (chapters 1–7) ─────────────────────────────
            elif self.curriculum.should_promote():
                self._promote_streak += 1
                sr = self.curriculum.rolling_success_rate
                print(f"  🔒 Promotion gate: {self._promote_streak}/{self.PROMOTE_SUSTAIN_COUNT}"
                      f" consecutive passes (sr={sr:.1%})")

                if self._promote_streak >= self.PROMOTE_SUSTAIN_COUNT:
                    if wandb.run is not None:
                        wandb.log({
                            "curriculum/promotion_event":         1,
                            "curriculum/promoted_from":           self.curriculum.current_chapter_idx,
                            "curriculum/promoted_at_step":        self.num_timesteps,
                            "curriculum/success_rate_at_promotion": sr,
                        }, step=self.num_timesteps)

                    self.curriculum.promote()
                    self._save_checkpoint("promotion")
                    self._promote_streak = 0  # reset for next chapter

                    # ── Clip-range warm restart ───────────────────────────
                    self._clip_warmup_remaining = self.CLIP_WARMUP_STEPS
                    self.model.clip_range = lambda _: self.CLIP_RANGE_HIGH
                    print(f"  🔧 Clip warm-restart: {self.CLIP_RANGE_HIGH} → "
                          f"{self.CLIP_RANGE_LOW} over {self.CLIP_WARMUP_STEPS:,} steps")

                    # ── Apply new chapter's PPO hyperparameters ───────────
                    new_ch = self.curriculum.current_chapter
                    self.model.learning_rate = new_ch.lr
                    self.model.n_epochs      = new_ch.n_epochs
                    self.model.batch_size    = new_ch.batch_size
                    self.model.ent_coef      = new_ch.ent_coef
                    print(f"  ⚙️  PPO hyperparams → lr={new_ch.lr}  "
                          f"epochs={new_ch.n_epochs}  batch={new_ch.batch_size}  "
                          f"ent_coef={new_ch.ent_coef}")

                    # ── Broadcast chapter change to all subprocess envs ───
                    new_idx = self.curriculum.current_chapter_idx
                    self.training_env.env_method("set_chapter_idx", new_idx)
                    print(f"  📡 Chapter {new_idx} broadcast to "
                          f"{self.training_env.num_envs} subprocesses")

                    # Clear our local success buffer for fresh start
                    self._success_buffer.clear()

            else:
                # Below threshold — reset streak
                self._promote_streak = 0

        # ── Periodic checkpoint ───────────────────────────────────────────
        if (self.num_timesteps - self._last_save_step) >= self.save_every:
            self._save_checkpoint("periodic")
            self._last_save_step = self.num_timesteps

        return True

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def _save_checkpoint(self, reason: str):
        os.makedirs(self.save_dir, exist_ok=True)
        model_path      = os.path.join(self.save_dir, "latest_model.zip")
        curriculum_path = os.path.join(self.save_dir, "curriculum_state.json")

        self.model.save(model_path)
        self.curriculum.save_state(curriculum_path)

        # Versioned snapshot at every promotion
        if reason == "promotion":
            ch_name    = self.curriculum.current_chapter.name
            step       = self.curriculum.total_steps
            versioned  = os.path.join(self.save_dir, f"model_{ch_name}_{step}.zip")
            self.model.save(versioned)

        print(f"  💾 Checkpoint ({reason}) → {self.save_dir}")

    def _get_action_std(self) -> float | None:
        try:
            return float(self.model.policy.log_std.exp().mean().item())
        except AttributeError:
            return None


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ORCA Curriculum RL Training (v7 — Workstation Edition)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Core training ────────────────────────────────────────────────────────
    p.add_argument("--timesteps", type=int, default=150_000_000,
                   help="Maximum total environment steps. The curriculum's own "
                        "80%% completion check is the real stop condition.")
    p.add_argument("--n-envs", type=int, default=64,
                   help="Number of parallel environments. "
                        "Sweet spot: 2–4 × physical CPU cores. "
                        "12-core machine: use 32–48.")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                   help="Torch device. Use 'cuda' if an NVIDIA GPU is available.")
    p.add_argument("--subproc", action="store_true", default=True,
                   help="Use SubprocVecEnv (true multiprocessing). Default ON. "
                        "Pass --no-subproc for DummyVecEnv (debugging only).")
    p.add_argument("--no-subproc", dest="subproc", action="store_false",
                   help="Force DummyVecEnv (single-process, for debugging).")

    # ── Network ──────────────────────────────────────────────────────────────
    p.add_argument("--net-arch", type=int, nargs="+", default=[512, 256],
                   help="Hidden layer sizes for actor and critic networks.")

    # ── PPO core hyperparameters ─────────────────────────────────────────────
    p.add_argument("--n-steps", type=int, default=4096,
                   help="Steps collected per env per rollout. "
                        "Total rollout = n_steps × n_envs.")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # ── Checkpointing & logging ──────────────────────────────────────────────
    p.add_argument("--save-dir", default="./checkpoints",
                   help="Directory for model checkpoints and curriculum state.")
    p.add_argument("--save-every", type=int, default=500_000,
                   help="Save a periodic checkpoint every N total steps.")
    p.add_argument("--project", default="orca-cube-reorientation",
                   help="W&B project name.")
    p.add_argument("--entity", default="fourvectors",
                   help="W&B entity (team) name.")
    p.add_argument("--run-name", default=None,
                   help="W&B run name. Auto-generated if not set.")
    p.add_argument("--upload-model", action="store_true",
                   help="Upload final model to W&B as a versioned artifact.")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Leader CurriculumManager (main process) ──────────────────────────────
    curriculum = CurriculumManager()

    # ── Crash recovery ───────────────────────────────────────────────────────
    curriculum_state_path = os.path.join(args.save_dir, "curriculum_state.json")
    model_path            = os.path.join(args.save_dir, "latest_model.zip")
    resuming              = curriculum.load_state(curriculum_state_path)

    # ── W&B initialisation ───────────────────────────────────────────────────
    ch     = curriculum.current_chapter
    config = dict(
        version           = "v7",
        algo              = "PPO",
        policy            = "MlpPolicy",
        env               = "OrcaHandRightCubeOrientation",
        reward            = "ProductionRewardWrapper-v1.2-relaxed-pos",
        curriculum        = "8-chapter-finer-progressive",
        net_arch          = args.net_arch,
        n_envs            = args.n_envs,
        vec_env_type      = "SubprocVecEnv" if args.subproc else "DummyVecEnv",
        n_steps           = args.n_steps,
        total_steps_budget= args.timesteps,
        # Per-chapter hyperparams logged at start (will update on promotion)
        starting_chapter  = ch.name,
        batch_size        = ch.batch_size,
        n_epochs          = ch.n_epochs,
        learning_rate     = ch.lr,
        ent_coef          = ch.ent_coef,
        gamma             = args.gamma,
        gae_lambda        = args.gae_lambda,
        clip_range        = 0.2,
        max_grad_norm     = args.max_grad_norm,
        max_episode_steps = ch.max_episode_steps,
        pos_sigma         = 0.05,
        adaptive_entropy  = True,
        adaptive_clip     = True,
        resumed           = resuming,
    )

    run_name = args.run_name or f"v7-{ch.name}-{args.n_envs}envs"
    run = wandb.init(
        project = args.project,
        entity  = args.entity,
        name    = run_name,
        config  = config,
        save_code = True,
        resume  = "allow" if resuming else None,
    )

    # ── Print header ─────────────────────────────────────────────────────────
    total_rollout = args.n_steps * args.n_envs
    vec_type = "SubprocVecEnv" if args.subproc else "DummyVecEnv"
    print(f"\n{'='*70}")
    print(f"  ORCA Curriculum Training  v7  —  Workstation Edition")
    print(f"  Run    : {run.name}  ({run.id})")
    print(f"  Chapter: {ch.name}  ({ch.angle_min_deg}°–{ch.angle_max_deg}°)")
    print(f"  Target : {ch.promotion_threshold:.0%} success rate")
    print(f"  Device : {args.device.upper()}  |  Envs: {args.n_envs}  ({vec_type})")
    print(f"  Network: {args.net_arch}")
    print(f"  Rollout: {total_rollout:,} steps/update "
          f"({args.n_steps} steps × {args.n_envs} envs)")
    print(f"  Budget : {args.timesteps:,} steps total")
    print(f"  Saves  : {args.save_dir}  (every {args.save_every:,} steps)")
    print(f"  Resumed: {resuming}")
    print(f"  W&B    : {run.url}")
    print(f"{'='*70}\n")

    # ── Create vectorised environment ─────────────────────────────────────────
    start_chapter = curriculum.current_chapter_idx

    if args.subproc:
        print(f"  🚀 Spawning {args.n_envs} subprocesses (SubprocVecEnv)…")
        env_fns = [make_env(start_chapter) for _ in range(args.n_envs)]
        # forkserver is not available on Windows; use spawn instead.
        start_method = "spawn" if sys.platform == "win32" else "forkserver"
        env = SubprocVecEnv(env_fns, start_method=start_method)
        print(f"  ✅ {args.n_envs} environments ready.\n")
    else:
        print(f"  🔄 DummyVecEnv (single-process debugging mode)…")
        env_fns = [make_env(start_chapter) for _ in range(args.n_envs)]
        env = DummyVecEnv(env_fns)
        print(f"  ✅ {args.n_envs} environments ready.\n")

    # ── Create or load PPO model ──────────────────────────────────────────────
    if resuming and os.path.exists(model_path):
        print(f"📂 Resuming PPO model from {model_path}")
        model = PPO.load(model_path, env=env, device=args.device)
        # Reapply chapter hyperparams (may have been updated since last save)
        model.learning_rate = ch.lr
        model.n_epochs      = ch.n_epochs
        model.batch_size    = ch.batch_size
        model.ent_coef      = ch.ent_coef
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose         = 1,
            device          = args.device,
            policy_kwargs   = dict(net_arch=args.net_arch),
            n_steps         = args.n_steps,
            batch_size      = ch.batch_size,
            n_epochs        = ch.n_epochs,
            learning_rate   = ch.lr,
            gamma           = args.gamma,
            gae_lambda      = args.gae_lambda,
            clip_range      = 0.2,
            ent_coef        = ch.ent_coef,
            max_grad_norm   = args.max_grad_norm,
        )

    # ── Remaining budget ──────────────────────────────────────────────────────
    remaining = max(0, args.timesteps - curriculum.total_steps)
    if remaining == 0:
        print("⚠️  Already reached timestep budget. Exiting.")
        env.close()
        run.finish()
        return

    print(f"  Training for {remaining:,} remaining steps "
          f"(completed: {curriculum.total_steps:,})\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        WandbCallback(
            model_save_path = None,   # handled by CurriculumCallback
            verbose         = 2,      # log ALL SB3 metrics to W&B
        ),
        CurriculumCallback(
            curriculum  = curriculum,
            save_dir    = args.save_dir,
            save_every  = args.save_every,
            log_freq    = max(16_384, args.n_steps * args.n_envs),
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps    = remaining,
        callback           = callbacks,
        reset_num_timesteps = not resuming,
    )

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, "ppo_curriculum_final.zip")
    model.save(final_path)
    curriculum.save_state(curriculum_state_path)
    print(f"\n✅ Training complete!")
    print(f"   Final chapter    : {curriculum.current_chapter.name}")
    print(f"   Final success rate: {curriculum.rolling_success_rate:.1%}")
    print(f"   Total steps      : {curriculum.total_steps:,}")
    print(f"   Model saved      : {final_path}")

    if args.upload_model and wandb.run is not None:
        artifact = wandb.Artifact(
            f"model-curriculum-{run.id}",
            type = "model",
            metadata = {
                "version"              : "v7",
                "chapters_completed"   : curriculum.current_chapter_idx + 1,
                "final_success_rate"   : curriculum.rolling_success_rate,
                "total_steps"          : curriculum.total_steps,
            },
        )
        artifact.add_file(final_path)
        run.log_artifact(artifact)
        print(f"📤 Model uploaded to W&B as artifact.")

    env.close()
    run.finish()


if __name__ == "__main__":
    main()
