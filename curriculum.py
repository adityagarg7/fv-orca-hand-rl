"""
Curriculum Manager for ORCA cube reorientation — 5-chapter progressive difficulty.

Backed by:
  - OpenAI Dactyl ADR: promote when agent masters current difficulty
  - HORA (CoRL 2022): progressive complexity on a single axis
  - PL-CGS: goals at the boundary of current competence

Usage:
    cm = CurriculumManager()
    cm.current_chapter   # → 0
    cm.sample_spawn_angle_deg()  # → e.g. 22.5 (within Chapter 1's 10°–30° band)
    cm.record_episode(success=True)
    cm.should_promote()  # → True when rolling success_rate >= threshold
"""

import json
import os
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


@dataclass
class ChapterConfig:
    """Configuration for one curriculum chapter."""
    name: str
    angle_min_deg: float        # min spawn angle from goal
    angle_max_deg: float        # max spawn angle from goal
    promotion_threshold: float  # success rate to advance
    lr: float                   # PPO learning rate
    n_epochs: int               # PPO update epochs
    batch_size: int             # PPO minibatch size
    ent_coef: float             # entropy coefficient
    max_episode_steps: int = 200  # episode length (scaled per chapter)
    success_bonus: float = 100.0  # can increase for harder chapters


# ── 8-chapter progressive difficulty (v9 — Production Fix) ──────────────────
# Changes from v8:
#   1. Overlaps widened from 2–5° to 8° (Dactyl/HORA standard: ~30% familiar
#      angles at each transition).  v8's 2–5° caused severe cold-start drops
#      that wasted millions of steps re-establishing a baseline.
#   2. Ch2b learning rate halved (3e-4 → 1.5e-4) based on diagnostic evidence:
#      clip_fraction was 28% (should be <20%), meaning the optimizer was being
#      throttled by PPO's conservative constraint.
#   3. Promotion logic moved to log-interval-only checks with 3-consecutive
#      sustained gate (see train_curriculum.py).
# v11 changes:
#   - Episode lengths +10: success needs a 10-step stable hold, so a goal
#     reached in the final <10 steps must still be able to qualify.
#   - success_bonus flattened to 100 for ALL chapters: under the v11 reward
#     the per-step goal-hold stream is the real payout (holding dominates
#     everything), and the bonus is only a one-time discovery spike.
# Retained from v8:
#   - Episode lengths scaled per chapter
#   - 80% promotion threshold for ALL chapters (no lowering)
CHAPTERS = [
    # ── Chapter 1: Small tilt  (16°–50°) ──────────────────────────────
    # Cube nearly upright. Task: learn basic finger contact + nudging.
    ChapterConfig(
        name="ch1_small_tilt",
        angle_min_deg=16,   angle_max_deg=50,
        promotion_threshold=0.80,
        lr=3e-4, n_epochs=10, batch_size=1024, ent_coef=0.003,
        max_episode_steps=210, success_bonus=100.0,
    ),
    # ── Chapter 2a: Moderate tilt  (42°–62°) ────────────────────────
    # Bridge chapter: nudging transitions to controlled pushing.
    # Overlap with Ch1: 8° (42–50°) → 40% of band is familiar.
    ChapterConfig(
        name="ch2a_moderate_tilt",
        angle_min_deg=42,   angle_max_deg=62,
        promotion_threshold=0.80,
        lr=3e-4, n_epochs=10, batch_size=1024, ent_coef=0.002,
        max_episode_steps=310, success_bonus=100.0,
    ),
    # ── Chapter 2b: Medium tilt  (54°–78°) ──────────────────────────
    # Agent must apply lateral force to initiate rolling motion.
    # Overlap with Ch2a: 8° (54–62°) → 33% of band is familiar.
    # LR reduced to 1.5e-4 (clip_fraction was 28% at 3e-4).
    ChapterConfig(
        name="ch2b_medium_tilt",
        angle_min_deg=54,   angle_max_deg=78,
        promotion_threshold=0.80,
        lr=1.5e-4, n_epochs=10, batch_size=1024, ent_coef=0.002,
        max_episode_steps=360, success_bonus=100.0,
    ),
    # ── Chapter 2c: Steep tilt  (70°–93°) ───────────────────────────
    # Full rolling skill required. Cube approaching the side-flat position.
    # Overlap with Ch2b: 8° (70–78°) → 35% of band is familiar.
    ChapterConfig(
        name="ch2c_steep_tilt",
        angle_min_deg=70,   angle_max_deg=93,
        promotion_threshold=0.80,
        lr=1.5e-4, n_epochs=10, batch_size=1024, ent_coef=0.0015,
        max_episode_steps=410, success_bonus=100.0,
    ),
    # ── Chapter 3a: Side roll  (85°–115°) ───────────────────────────
    # Cube mostly on its side. Coordinated multi-finger rolling.
    # Overlap with Ch2c: 8° (85–93°) → 27% of band is familiar.
    ChapterConfig(
        name="ch3a_side_roll",
        angle_min_deg=85,   angle_max_deg=115,
        promotion_threshold=0.80,
        lr=2e-4, n_epochs=10, batch_size=1024, ent_coef=0.001,
        max_episode_steps=460, success_bonus=100.0,
    ),
    # ── Chapter 3b: Deep roll  (107°–135°) ──────────────────────────
    # Agent must push the cube past the equator (>90°). Hardest transition.
    # Overlap with Ch3a: 8° (107–115°) → 29% of band is familiar.
    ChapterConfig(
        name="ch3b_deep_roll",
        angle_min_deg=107,  angle_max_deg=135,
        promotion_threshold=0.80,
        lr=2e-4, n_epochs=10, batch_size=1024, ent_coef=0.001,
        max_episode_steps=510, success_bonus=100.0,
    ),
    # ── Chapter 4: Near flip  (127°–163°) ───────────────────────────
    # Cube nearly upside down. Agent must learn to re-catch at the apex.
    # Overlap with Ch3b: 8° (127–135°) → 22% of band is familiar.
    ChapterConfig(
        name="ch4_near_flip",
        angle_min_deg=127,  angle_max_deg=163,
        promotion_threshold=0.80,
        lr=1e-4, n_epochs=10, batch_size=1024, ent_coef=0.0008,
        max_episode_steps=610, success_bonus=100.0,
    ),
    # ── Chapter 5: Full flip  (155°–180°) ───────────────────────────
    # Full 180° reorientation. Graduation chapter.
    # Overlap with Ch4: 8° (155–163°) → 32% of band is familiar.
    ChapterConfig(
        name="ch5_full_flip",
        angle_min_deg=155,  angle_max_deg=180,
        promotion_threshold=0.80,
        lr=1e-4, n_epochs=10, batch_size=1024, ent_coef=0.0005,
        max_episode_steps=760, success_bonus=100.0,
    ),
]


class CurriculumManager:
    """Tracks training progress and auto-promotes through chapters.

    The manager is fully serializable: call `save_state()` and
    `load_state()` to persist across Colab/Kaggle session crashes.
    """

    ROLLING_WINDOW = 200           # episodes for success rate calc
    MIN_STEPS_BEFORE_PROMOTE = 100_000   # min steps before allowing promotion
    # NO force-promotion. Agent stays on a chapter until it genuinely
    # hits the threshold. Promoting below the mark destroys learning.
    # Stamped into curriculum_state.json; load_state refuses to resume across
    # a version change (blocks silently resuming v9/v10 checkpoints).
    STATE_VERSION = "v11"

    def __init__(self, chapters: list[ChapterConfig] | None = None):
        self.chapters = chapters or CHAPTERS
        self._chapter_idx = 0
        self._success_history: deque[bool] = deque(maxlen=self.ROLLING_WINDOW)
        self._chapter_steps = 0
        self._total_steps = 0
        self._chapter_episodes = 0
        self._total_episodes = 0
        self._promotion_log: list[dict] = []
        self._rng = np.random.default_rng()

    # ── Properties ────────────────────────────────────────────────────

    @property
    def current_chapter_idx(self) -> int:
        return self._chapter_idx

    @property
    def current_chapter(self) -> ChapterConfig:
        return self.chapters[self._chapter_idx]

    @property
    def is_final_chapter(self) -> bool:
        return self._chapter_idx >= len(self.chapters) - 1

    @property
    def rolling_success_rate(self) -> float:
        if hasattr(self, "_forced_success_rate"):
            return self._forced_success_rate
        if len(self._success_history) == 0:
            return 0.0
        return sum(self._success_history) / len(self._success_history)

    @property
    def chapter_steps(self) -> int:
        return self._chapter_steps

    @property
    def total_steps(self) -> int:
        return self._total_steps

    # ── Core API ──────────────────────────────────────────────────────

    def sample_spawn_angle_deg(self) -> float:
        """Sample a random spawn angle within the current chapter's band."""
        ch = self.current_chapter
        return float(self._rng.uniform(ch.angle_min_deg, ch.angle_max_deg))

    def record_episode(self, success: bool, steps_in_episode: int = 1):
        """Record whether an episode was successful."""
        self._success_history.append(success)
        self._chapter_episodes += 1
        self._total_episodes += 1

    def override_success_rate(self, sr: float, buffer_len: int = 0):
        """Force the rolling success rate to exactly match an external value (SB3 buffer)."""
        self._forced_success_rate = sr
        self._forced_success_buffer_len = buffer_len

    def record_steps(self, n_steps: int):
        """Record training steps taken."""
        self._chapter_steps += n_steps
        self._total_steps += n_steps

    def should_promote(self) -> bool:
        """Check if the agent should advance to the next chapter."""
        if self.is_final_chapter:
            return False

        # Must have enough data
        current_len = getattr(self, "_forced_success_buffer_len", len(self._success_history))
        if current_len < self.ROLLING_WINDOW:
            return False

        # Must have trained long enough
        if self._chapter_steps < self.MIN_STEPS_BEFORE_PROMOTE:
            return False

        # Check success rate — this is the ONLY way to promote
        if self.rolling_success_rate >= self.current_chapter.promotion_threshold:
            return True

        return False

    def promote(self):
        """Advance to the next chapter."""
        old = self.current_chapter
        self._promotion_log.append({
            "from_chapter": old.name,
            "from_idx": self._chapter_idx,
            "success_rate": self.rolling_success_rate,
            "chapter_steps": self._chapter_steps,
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
        })
        print(f"\n🎓 PROMOTION: {old.name} → {self.chapters[self._chapter_idx + 1].name}")
        print(f"   Success rate: {self.rolling_success_rate:.1%} "
              f"(threshold: {old.promotion_threshold:.0%})")
        print(f"   Steps in chapter: {self._chapter_steps:,}")
        print(f"   Total steps: {self._total_steps:,}\n")

        self._chapter_idx += 1
        self._chapter_steps = 0
        self._chapter_episodes = 0
        self._success_history.clear()

        # CRITICAL: Clear forced success rate from the callback override.
        # Without this, the stale 0.98 from Ch1 would persist and trigger
        # instant promotion through Ch2→Ch3→...→Ch8 after just 100k steps
        # each (MIN_STEPS_BEFORE_PROMOTE), without actually learning.
        if hasattr(self, "_forced_success_rate"):
            del self._forced_success_rate
        if hasattr(self, "_forced_success_buffer_len"):
            del self._forced_success_buffer_len

    # ── Persistence (crash-proof) ─────────────────────────────────────

    def save_state(self, path: str):
        """Save full curriculum state to JSON for crash recovery."""
        state = {
            "state_version": self.STATE_VERSION,
            "chapter_idx": self._chapter_idx,
            "success_history": list(self._success_history),
            "chapter_steps": self._chapter_steps,
            "total_steps": self._total_steps,
            "chapter_episodes": self._chapter_episodes,
            "total_episodes": self._total_episodes,
            "promotion_log": self._promotion_log,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> bool:
        """Load curriculum state from JSON. Returns True if loaded.

        Refuses to resume across reward/curriculum versions.  A v9/v10
        curriculum_state.json (no state_version, and chapter_idx up to 7 with
        the wrist-exploit model saved alongside as latest_model.zip) is NOT a
        valid v11 resume: it would silently continue the wrong policy at the
        wrong chapter under an incompatible reward scale.  Hard-stop instead
        of corrupting the run — use a fresh --save-dir or delete the stale
        files deliberately.
        """
        if not os.path.exists(path):
            return False
        with open(path) as f:
            state = json.load(f)
        saved_version = state.get("state_version")
        if saved_version != self.STATE_VERSION:
            raise RuntimeError(
                f"Refusing to resume from '{path}': it is "
                f"'{saved_version or 'pre-v11 (v9/v10)'}' but this code is "
                f"'{self.STATE_VERSION}'. Resuming would silently continue an "
                f"incompatible policy/chapter. Point --save-dir at a fresh, empty "
                f"directory (e.g. ./checkpoints_v11) or delete the stale "
                f"curriculum_state.json + latest_model.zip + vecnormalize.pkl there."
            )
        self._chapter_idx = state["chapter_idx"]
        self._success_history = deque(state["success_history"],
                                      maxlen=self.ROLLING_WINDOW)
        self._chapter_steps = state["chapter_steps"]
        self._total_steps = state["total_steps"]
        self._chapter_episodes = state["chapter_episodes"]
        self._total_episodes = state["total_episodes"]
        self._promotion_log = state["promotion_log"]
        print(f"📂 Resumed curriculum at {self.current_chapter.name} "
              f"(step {self._total_steps:,}, "
              f"success_rate={self.rolling_success_rate:.1%})")
        return True

    def status_dict(self) -> dict:
        """Return a dict of metrics for W&B logging."""
        ch = self.current_chapter
        return {
            "curriculum/chapter_idx": self._chapter_idx,
            "curriculum/chapter_name": ch.name,
            "curriculum/angle_min_deg": ch.angle_min_deg,
            "curriculum/angle_max_deg": ch.angle_max_deg,
            "curriculum/rolling_success_rate": self.rolling_success_rate,
            "curriculum/promotion_threshold": ch.promotion_threshold,
            "curriculum/chapter_steps": self._chapter_steps,
            "curriculum/total_steps": self._total_steps,
            "curriculum/lr": ch.lr,
        }
