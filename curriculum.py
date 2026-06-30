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


# ── 8-chapter progressive difficulty (v8 — Production Fix) ──────────────────
# Changes from v7:
#   1. Overlaps narrowed from 10–13° to 2–5° to eliminate the mathematical
#      success-rate ceiling caused by the agent coasting on already-mastered
#      angles.  (Ch2a's old 40–62° range meant 45% of episodes were easy.)
#   2. Success bonuses increased 5–10× so solving dominates per-step farming.
#      At v7 weights, farming earned 377pts/episode vs 45pts for solving.
#   3. Episode lengths shortened to reduce the farming window.  Shorter episodes
#      = less farming revenue = success bonus is relatively more valuable.
#   4. LR schedule unchanged (3e-4 early, 1e-4 late).
CHAPTERS = [
    # ── Chapter 1: Small tilt  (16°–50°) ──────────────────────────────
    # Cube nearly upright. Task: learn basic finger contact + nudging.
    ChapterConfig(
        name="ch1_small_tilt",
        angle_min_deg=16,   angle_max_deg=50,
        promotion_threshold=0.80,
        lr=3e-4, n_epochs=10, batch_size=1024, ent_coef=0.003,
        max_episode_steps=200, success_bonus=500.0,
    ),
    # ── Chapter 2a: Moderate tilt  (48°–62°) ────────────────────────
    # Bridge chapter: nudging transitions to controlled pushing.
    # Overlap with Ch1: only 2° (48–50°), down from 10° in v7.
    ChapterConfig(
        name="ch2a_moderate_tilt",
        angle_min_deg=48,   angle_max_deg=62,
        promotion_threshold=0.80,
        lr=3e-4, n_epochs=10, batch_size=1024, ent_coef=0.002,
        max_episode_steps=300, success_bonus=600.0,
    ),
    # ── Chapter 2b: Medium tilt  (58°–78°) ──────────────────────────
    # Agent must apply lateral force to initiate rolling motion.
    # Overlap with Ch2a: 4° (58–62°).
    ChapterConfig(
        name="ch2b_medium_tilt",
        angle_min_deg=58,   angle_max_deg=78,
        promotion_threshold=0.80,
        lr=3e-4, n_epochs=10, batch_size=1024, ent_coef=0.002,
        max_episode_steps=350, success_bonus=700.0,
    ),
    # ── Chapter 2c: Steep tilt  (73°–93°) ───────────────────────────
    # Full rolling skill required. Cube approaching the side-flat position.
    # Overlap with Ch2b: 5° (73–78°).
    ChapterConfig(
        name="ch2c_steep_tilt",
        angle_min_deg=73,   angle_max_deg=93,
        promotion_threshold=0.80,
        lr=3e-4, n_epochs=10, batch_size=1024, ent_coef=0.0015,
        max_episode_steps=400, success_bonus=800.0,
    ),
    # ── Chapter 3a: Side roll  (88°–115°) ───────────────────────────
    # Cube mostly on its side. Coordinated multi-finger rolling.
    # Overlap with Ch2c: 5° (88–93°).
    ChapterConfig(
        name="ch3a_side_roll",
        angle_min_deg=88,   angle_max_deg=115,
        promotion_threshold=0.80,
        lr=2e-4, n_epochs=10, batch_size=1024, ent_coef=0.001,
        max_episode_steps=450, success_bonus=1000.0,
    ),
    # ── Chapter 3b: Deep roll  (110°–135°) ──────────────────────────
    # Agent must push the cube past the equator (>90°). Hardest transition.
    # Overlap with Ch3a: 5° (110–115°).
    ChapterConfig(
        name="ch3b_deep_roll",
        angle_min_deg=110,  angle_max_deg=135,
        promotion_threshold=0.80,
        lr=2e-4, n_epochs=10, batch_size=1024, ent_coef=0.001,
        max_episode_steps=500, success_bonus=1200.0,
    ),
    # ── Chapter 4: Near flip  (130°–163°) ───────────────────────────
    # Cube nearly upside down. Agent must learn to re-catch at the apex.
    # Overlap with Ch3b: 5° (130–135°).
    ChapterConfig(
        name="ch4_near_flip",
        angle_min_deg=130,  angle_max_deg=163,
        promotion_threshold=0.80,
        lr=1e-4, n_epochs=10, batch_size=1024, ent_coef=0.0008,
        max_episode_steps=600, success_bonus=1500.0,
    ),
    # ── Chapter 5: Full flip  (158°–180°) ───────────────────────────
    # Full 180° reorientation. Graduation chapter.
    # Overlap with Ch4: 5° (158–163°).
    ChapterConfig(
        name="ch5_full_flip",
        angle_min_deg=158,  angle_max_deg=180,
        promotion_threshold=0.80,
        lr=1e-4, n_epochs=10, batch_size=1024, ent_coef=0.0005,
        max_episode_steps=750, success_bonus=2000.0,
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
        """Load curriculum state from JSON. Returns True if loaded."""
        if not os.path.exists(path):
            return False
        with open(path) as f:
            state = json.load(f)
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
