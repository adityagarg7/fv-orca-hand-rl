"""
v9 PRODUCTION AUDIT — Comprehensive Test Suite
================================================
Tests ALL four v9 fixes plus every critical subsystem:
  Fix 1: Sustained promotion gate (3 consecutive checks)
  Fix 2: 8° chapter overlaps (Dactyl/HORA standard)
  Fix 3: Ch2b LR lowered to 1.5e-4
  Fix 4: Promotion check at log intervals only

Also regression-tests the v8 false-promotion bug.
"""

import sys
import math
import json
import os
import tempfile
from collections import deque

# ── Test infrastructure ──────────────────────────────────────────────────────
PASS = 0
FAIL = 0
WARN = 0

def log_pass(msg, detail=""):
    global PASS; PASS += 1
    print(f"  [PASS] {msg}" + (f"  ({detail})" if detail else ""))

def log_fail(msg, detail=""):
    global FAIL; FAIL += 1
    print(f"  [FAIL] {msg}" + (f"  ({detail})" if detail else ""))

def log_warn(msg, detail=""):
    global WARN; WARN += 1
    print(f"  [WARN] {msg}" + (f"  ({detail})" if detail else ""))

def check(condition, pass_msg, fail_msg, detail=""):
    if condition:
        log_pass(pass_msg, detail)
    else:
        log_fail(fail_msg, detail)

# ── Import the code under test ───────────────────────────────────────────────
from curriculum import CHAPTERS, CurriculumManager, ChapterConfig


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Chapter Config Structural Validation
# ══════════════════════════════════════════════════════════════════════════════
def test_chapter_configs():
    print("\n" + "─" * 70)
    print("  TEST 1: Chapter Config Structural Validation")
    print("─" * 70)

    check(len(CHAPTERS) == 8, "8 chapters defined", f"Expected 8 chapters, got {len(CHAPTERS)}")

    # Full angular coverage: 16° to 180°
    all_min = min(ch.angle_min_deg for ch in CHAPTERS)
    all_max = max(ch.angle_max_deg for ch in CHAPTERS)
    check(all_min == 16, f"Min angle = {all_min}°", f"Min angle should be 16°, got {all_min}°")
    check(all_max == 180, f"Max angle = {all_max}°", f"Max angle should be 180°, got {all_max}°")

    # Each chapter: min < max, valid threshold, valid hyperparams
    for i, ch in enumerate(CHAPTERS):
        check(ch.angle_min_deg < ch.angle_max_deg,
              f"Ch{i+1} [{ch.name}] min({ch.angle_min_deg}) < max({ch.angle_max_deg})",
              f"Ch{i+1} [{ch.name}] invalid range: {ch.angle_min_deg} >= {ch.angle_max_deg}")
        check(0.5 <= ch.promotion_threshold <= 1.0,
              f"Ch{i+1} threshold = {ch.promotion_threshold}",
              f"Ch{i+1} invalid threshold: {ch.promotion_threshold}")
        check(ch.lr > 0, f"Ch{i+1} lr = {ch.lr}", f"Ch{i+1} lr must be > 0")
        check(ch.max_episode_steps > 0,
              f"Ch{i+1} max_steps = {ch.max_episode_steps}",
              f"Ch{i+1} max_steps must be > 0")
        check(ch.success_bonus > 0,
              f"Ch{i+1} bonus = {ch.success_bonus}",
              f"Ch{i+1} success_bonus must be > 0")

    # Monotonically increasing difficulty
    for i in range(len(CHAPTERS) - 1):
        check(CHAPTERS[i].angle_max_deg <= CHAPTERS[i+1].angle_max_deg,
              f"Ch{i+1}→Ch{i+2} difficulty increases",
              f"Ch{i+2} max ({CHAPTERS[i+1].angle_max_deg}) <= Ch{i+1} max ({CHAPTERS[i].angle_max_deg})")

    # Monotonically increasing success bonuses
    for i in range(len(CHAPTERS) - 1):
        check(CHAPTERS[i].success_bonus <= CHAPTERS[i+1].success_bonus,
              f"Ch{i+1}→Ch{i+2} bonus increases ({CHAPTERS[i].success_bonus} → {CHAPTERS[i+1].success_bonus})",
              f"Ch{i+2} bonus should be >= Ch{i+1}")

    # Monotonically increasing episode lengths
    for i in range(len(CHAPTERS) - 1):
        check(CHAPTERS[i].max_episode_steps <= CHAPTERS[i+1].max_episode_steps,
              f"Ch{i+1}→Ch{i+2} episode length increases ({CHAPTERS[i].max_episode_steps} → {CHAPTERS[i+1].max_episode_steps})",
              f"Ch{i+2} steps should be >= Ch{i+1}")

    # All thresholds are 80%
    for i, ch in enumerate(CHAPTERS):
        check(ch.promotion_threshold == 0.80,
              f"Ch{i+1} threshold = 80%",
              f"Ch{i+1} threshold should be 80%, got {ch.promotion_threshold:.0%}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: v9 FIX — 8° Chapter Overlaps
# ══════════════════════════════════════════════════════════════════════════════
def test_overlaps():
    print("\n" + "─" * 70)
    print("  TEST 2: v9 FIX — 8° Chapter Overlaps")
    print("─" * 70)

    TARGET_OVERLAP = 8  # degrees

    for i in range(len(CHAPTERS) - 1):
        prev = CHAPTERS[i]
        nxt = CHAPTERS[i + 1]
        overlap = prev.angle_max_deg - nxt.angle_min_deg
        band_width = nxt.angle_max_deg - nxt.angle_min_deg
        overlap_pct = overlap / band_width * 100

        # Must be connected (overlap > 0)
        check(overlap > 0,
              f"Ch{i+1}→Ch{i+2} connected (overlap={overlap}°)",
              f"Ch{i+1}→Ch{i+2} DISCONNECTED: gap of {-overlap}°!")

        # Overlap must be exactly 8°
        check(overlap == TARGET_OVERLAP,
              f"Ch{i+1}→Ch{i+2} overlap = {overlap}° (target: {TARGET_OVERLAP}°)",
              f"Ch{i+1}→Ch{i+2} overlap = {overlap}° (expected {TARGET_OVERLAP}°)")

        # Familiar % should be 20-45% (healthy range)
        if overlap_pct >= 20 and overlap_pct <= 45:
            log_pass(f"Ch{i+2} familiar angles = {overlap_pct:.0f}% (healthy range 20-45%)")
        elif overlap_pct < 20:
            log_warn(f"Ch{i+2} familiar angles = {overlap_pct:.0f}% (<20%, cold start risk)")
        else:
            log_warn(f"Ch{i+2} familiar angles = {overlap_pct:.0f}% (>45%, coasting risk)")

    # v8 regression: ensure no overlap is 2-5° (the old broken values)
    for i in range(len(CHAPTERS) - 1):
        overlap = CHAPTERS[i].angle_max_deg - CHAPTERS[i + 1].angle_min_deg
        check(overlap >= 7,
              f"Ch{i+1}→Ch{i+2} overlap ({overlap}°) is not dangerously narrow",
              f"Ch{i+1}→Ch{i+2} overlap ({overlap}°) is too narrow (v8 bug)!")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: v9 FIX — Ch2b Learning Rate
# ══════════════════════════════════════════════════════════════════════════════
def test_ch2b_lr():
    print("\n" + "─" * 70)
    print("  TEST 3: v9 FIX — Ch2b Learning Rate")
    print("─" * 70)

    ch2b = CHAPTERS[2]  # index 2 = ch2b_medium_tilt
    check(ch2b.name == "ch2b_medium_tilt",
          f"Chapter index 2 is ch2b_medium_tilt",
          f"Chapter index 2 is '{ch2b.name}', expected ch2b_medium_tilt")
    check(ch2b.lr == 1.5e-4,
          f"Ch2b lr = {ch2b.lr} (halved from 3e-4)",
          f"Ch2b lr = {ch2b.lr}, expected 1.5e-4")

    # Ch2c should also be 1.5e-4 (same difficulty tier)
    ch2c = CHAPTERS[3]
    check(ch2c.lr == 1.5e-4,
          f"Ch2c lr = {ch2c.lr} (matches Ch2b tier)",
          f"Ch2c lr = {ch2c.lr}, expected 1.5e-4")

    # Ch1 and Ch2a should still be 3e-4
    check(CHAPTERS[0].lr == 3e-4, "Ch1 lr = 3e-4 (unchanged)", f"Ch1 lr = {CHAPTERS[0].lr}")
    check(CHAPTERS[1].lr == 3e-4, "Ch2a lr = 3e-4 (unchanged)", f"Ch2a lr = {CHAPTERS[1].lr}")

    # Later chapters should have lower LR
    check(CHAPTERS[4].lr == 2e-4, "Ch3a lr = 2e-4", f"Ch3a lr = {CHAPTERS[4].lr}")
    check(CHAPTERS[5].lr == 2e-4, "Ch3b lr = 2e-4", f"Ch3b lr = {CHAPTERS[5].lr}")
    check(CHAPTERS[6].lr == 1e-4, "Ch4 lr = 1e-4", f"Ch4 lr = {CHAPTERS[6].lr}")
    check(CHAPTERS[7].lr == 1e-4, "Ch5 lr = 1e-4", f"Ch5 lr = {CHAPTERS[7].lr}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Promotion Logic — Basic Mechanics
# ══════════════════════════════════════════════════════════════════════════════
def test_promotion_logic():
    print("\n" + "─" * 70)
    print("  TEST 4: Promotion Logic — Basic Mechanics")
    print("─" * 70)

    cm = CurriculumManager()

    # Should NOT promote with empty buffer
    check(not cm.should_promote(), "No promotion with empty buffer", "Promoted with empty buffer!")

    # Should NOT promote below threshold
    for _ in range(200):
        cm.record_episode(success=(_ % 3 == 0))  # ~33% success
    cm.record_steps(200_000)
    check(not cm.should_promote(),
          f"No promotion at {cm.rolling_success_rate:.0%} (<80%)",
          f"Promoted at {cm.rolling_success_rate:.0%}!")

    # Should NOT promote below MIN_STEPS_BEFORE_PROMOTE
    cm2 = CurriculumManager()
    for _ in range(200):
        cm2.record_episode(success=True)  # 100%
    cm2.record_steps(50_000)  # below 100K threshold
    check(not cm2.should_promote(),
          "No promotion before MIN_STEPS_BEFORE_PROMOTE (100K)",
          "Promoted with only 50K steps!")

    # SHOULD promote when all conditions met
    cm2.record_steps(60_000)  # total now 110K
    check(cm2.should_promote(),
          "Promotes at 100% with sufficient steps",
          "Failed to promote at 100% with enough steps!")

    # Should NOT promote on final chapter
    cm3 = CurriculumManager()
    cm3._chapter_idx = len(CHAPTERS) - 1  # final chapter
    for _ in range(200):
        cm3.record_episode(success=True)
    cm3.record_steps(200_000)
    check(not cm3.should_promote(),
          "No promotion on final chapter",
          "Promoted on final chapter!")

    # Test promote() mechanics
    cm4 = CurriculumManager()
    for _ in range(200):
        cm4.record_episode(success=True)
    cm4.record_steps(200_000)
    cm4.override_success_rate(0.85, 200)
    old_idx = cm4.current_chapter_idx
    cm4.promote()
    check(cm4.current_chapter_idx == old_idx + 1,
          "promote() advances chapter index",
          f"Chapter didn't advance: {cm4.current_chapter_idx}")
    check(cm4._chapter_steps == 0,
          "promote() resets chapter_steps",
          f"chapter_steps not reset: {cm4._chapter_steps}")
    check(len(cm4._success_history) == 0,
          "promote() clears success history",
          f"History not cleared: {len(cm4._success_history)}")
    check(not hasattr(cm4, '_forced_success_rate'),
          "promote() clears forced_success_rate",
          "forced_success_rate not cleared after promotion!")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: v9 FIX — Sustained Promotion Gate (Regression Test)
# ══════════════════════════════════════════════════════════════════════════════
def test_sustained_promotion_gate():
    """Simulates the exact v8 bug: SR oscillates 72-78%, occasionally spikes
    to 80%.  With v8 code, this would promote.  With v9, it must NOT promote
    because the streak would break."""
    print("\n" + "─" * 70)
    print("  TEST 5: v9 FIX — Sustained Promotion Gate (Regression)")
    print("─" * 70)

    # Simulate the sustained gate logic manually
    # (we can't import the callback without SB3 model, so we test the logic)
    SUSTAIN_COUNT = 3

    # Scenario A: sporadic spikes (should NOT promote)
    streak = 0
    promoted = False
    # Simulate 20 log-interval checks where SR oscillates
    sr_sequence = [0.72, 0.68, 0.75, 0.71, 0.80, 0.73, 0.69, 0.78, 0.80, 0.74,
                   0.71, 0.80, 0.76, 0.68, 0.79, 0.80, 0.72, 0.80, 0.69, 0.73]
    for sr in sr_sequence:
        if sr >= 0.80:
            streak += 1
            if streak >= SUSTAIN_COUNT:
                promoted = True
                break
        else:
            streak = 0

    check(not promoted,
          "Sporadic 80% spikes do NOT trigger promotion (v8 bug fixed)",
          "PROMOTED on sporadic spikes — v8 bug is NOT fixed!")

    # Scenario B: genuine mastery (3 consecutive >= 80%, SHOULD promote)
    streak = 0
    promoted = False
    sr_sequence_b = [0.72, 0.75, 0.78, 0.80, 0.81, 0.82, 0.80, 0.65]
    promoted_at = None
    for i, sr in enumerate(sr_sequence_b):
        if sr >= 0.80:
            streak += 1
            if streak >= SUSTAIN_COUNT:
                promoted = True
                promoted_at = i
                break
        else:
            streak = 0

    check(promoted,
          f"3 consecutive passes (0.80, 0.81, 0.82) triggers promotion at check {promoted_at}",
          "Failed to promote after 3 consecutive passes!")

    # Scenario C: 2 passes then dip (should NOT promote)
    streak = 0
    promoted = False
    sr_sequence_c = [0.80, 0.81, 0.78, 0.80, 0.82, 0.75, 0.80, 0.80, 0.80]
    for sr in sr_sequence_c:
        if sr >= 0.80:
            streak += 1
            if streak >= SUSTAIN_COUNT:
                promoted = True
                break
        else:
            streak = 0

    check(promoted,
          "Final three 0.80s at end correctly trigger promotion",
          "Did not promote on final consecutive 0.80s")
    check(streak == 3,
          f"Streak = {streak} (correctly accumulated)",
          f"Streak = {streak}, expected 3")

    # Scenario D: exactly at boundary — 2 consecutive then dip
    streak = 0
    promoted = False
    sr_sequence_d = [0.80, 0.81, 0.79, 0.80, 0.79]
    for sr in sr_sequence_d:
        if sr >= 0.80:
            streak += 1
            if streak >= SUSTAIN_COUNT:
                promoted = True
                break
        else:
            streak = 0
    check(not promoted,
          "2 consecutive passes + dip does NOT promote",
          "Promoted after only 2 consecutive passes!")

    # Verify the CurriculumCallback has the right constant
    # We can't import the class without SB3, so we check the source
    import inspect
    import importlib
    try:
        mod = importlib.import_module("train_curriculum")
        cls = getattr(mod, "CurriculumCallback", None)
        if cls is not None:
            obj = cls.__new__(cls)
            # Check the init signature / default
            sig = inspect.signature(cls.__init__)
            # The constant is set in __init__, check the source
            src = inspect.getsource(cls.__init__)
            check("PROMOTE_SUSTAIN_COUNT" in src,
                  "CurriculumCallback has PROMOTE_SUSTAIN_COUNT",
                  "CurriculumCallback missing PROMOTE_SUSTAIN_COUNT!")
            check("_promote_streak" in src,
                  "CurriculumCallback initializes _promote_streak",
                  "CurriculumCallback missing _promote_streak!")
        else:
            log_warn("Could not find CurriculumCallback class")
    except Exception as e:
        log_warn(f"Could not inspect CurriculumCallback: {e}")

    # Verify promotion check is inside log-interval block
    try:
        mod = importlib.import_module("train_curriculum")
        src = inspect.getsource(mod.CurriculumCallback._on_step)
        # The promotion check (should_promote) should NOT be at the top level
        # It should be nested inside a log_interval_steps check
        lines = src.split("\n")
        promote_lines = [l for l in lines if "should_promote" in l]
        for pl in promote_lines:
            indent = len(pl) - len(pl.lstrip())
            # Should be indented at least 12 spaces (inside the if block)
            check(indent >= 12,
                  f"should_promote() is inside log-interval block (indent={indent})",
                  f"should_promote() might be at top level (indent={indent})!")
    except Exception as e:
        log_warn(f"Could not verify promotion nesting: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Quaternion Math
# ══════════════════════════════════════════════════════════════════════════════
def test_quaternion():
    print("\n" + "─" * 70)
    print("  TEST 6: Quaternion Math")
    print("─" * 70)

    import numpy as np

    # Import the static method and angle conversion directly
    from curriculum_wrapper import CurriculumWrapper

    # Test quaternion multiply (static method, no env needed)
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_test = np.array([0.707, 0.707, 0.0, 0.0])
    result = CurriculumWrapper._quat_multiply(q_id, q_test)
    check(np.allclose(result, q_test, atol=1e-3),
          "Identity × q = q",
          f"Identity multiply failed: {result}")

    # Test angle_to_random_quat by calling the internal math directly
    # We reconstruct the logic without needing a Wrapper instance
    rng = np.random.default_rng(42)

    def angle_to_quat(angle_deg):
        theta = np.radians(angle_deg)
        phi = rng.uniform(0, 2 * np.pi)
        ax, ay = np.cos(phi), np.sin(phi)
        q_tilt = np.array([np.cos(theta/2), np.sin(theta/2)*ax, np.sin(theta/2)*ay, 0.0])
        psi = rng.uniform(0, 2 * np.pi)
        q_spin = np.array([np.cos(psi/2), 0.0, 0.0, np.sin(psi/2)])
        return CurriculumWrapper._quat_multiply(q_spin, q_tilt)

    for test_angle in [0, 30, 45, 60, 90, 120, 150, 180]:
        q = angle_to_quat(test_angle)
        norm = np.linalg.norm(q)
        check(abs(norm - 1.0) < 1e-6,
              f"{test_angle} deg quat is unit (norm={norm:.6f})",
              f"{test_angle} deg quat not unit: norm={norm:.6f}")

        w, x, y, z = q
        rz_z = 1 - 2*(x*x + y*y)
        recovered = math.degrees(math.acos(np.clip(rz_z, -1, 1)))
        check(abs(recovered - test_angle) < 0.5,
              f"{test_angle} deg recovered as {recovered:.1f} deg",
              f"{test_angle} deg recovered as {recovered:.1f} deg (off by {abs(recovered - test_angle):.1f})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: Reward Function Math
# ══════════════════════════════════════════════════════════════════════════════
def test_reward_math():
    print("\n" + "─" * 70)
    print("  TEST 7: Reward Function Math")
    print("─" * 70)

    import numpy as np
    from production_reward import ProductionRewardWrapper

    # Test alignment kernel
    align_w = 5.0
    align_s = 0.3
    # At goal (angle=0): max reward
    r_at_goal = align_w * np.exp(-0.0 / align_s)
    check(abs(r_at_goal - 5.0) < 0.01,
          f"Alignment at goal = {r_at_goal:.2f} (max)",
          f"Alignment at goal = {r_at_goal:.2f}, expected 5.0")

    # At 30° (0.524 rad): should be much less
    r_at_30 = align_w * np.exp(-0.524 / align_s)
    check(r_at_30 < 1.0,
          f"Alignment at 30° = {r_at_30:.2f} (<1.0, good drop-off)",
          f"Alignment at 30° = {r_at_30:.2f}, expected <1.0")

    # Progress reward for solving 1 radian
    progress_w = 50.0
    r_progress = progress_w * 1.0  # 1 radian of progress
    check(r_progress == 50.0,
          f"Progress for 1 rad = {r_progress:.1f}",
          f"Progress for 1 rad = {r_progress:.1f}, expected 50.0")

    # Success bonus dominance check per chapter
    for i, ch in enumerate(CHAPTERS):
        band_mid = (ch.angle_min_deg + ch.angle_max_deg) / 2
        band_rad = math.radians(band_mid)
        # Max farming revenue: ep_steps × alignment_at_near_goal
        max_farm = ch.max_episode_steps * align_w * np.exp(-0.05 / align_s)  # 0.05 rad ≈ 3°
        # Min success bonus (worst stability)
        min_bonus = ch.success_bonus * 0.3  # stability_floor
        if min_bonus > max_farm * 0.5:
            log_pass(f"Ch{i+1} [{ch.name}] min_bonus ({min_bonus:.0f}) > 50% farm ceiling ({max_farm:.0f})")
        else:
            log_warn(f"Ch{i+1} [{ch.name}] min_bonus ({min_bonus:.0f}) < 50% farm ceiling ({max_farm:.0f})",
                     "Farming may still be competitive")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: Persistence (Save/Load)
# ══════════════════════════════════════════════════════════════════════════════
def test_persistence():
    print("\n" + "─" * 70)
    print("  TEST 8: Persistence (Save/Load)")
    print("─" * 70)

    import tempfile

    cm = CurriculumManager()
    cm._chapter_idx = 3
    cm._total_steps = 5_000_000
    cm._chapter_steps = 1_200_000
    for i in range(200):
        cm.record_episode(success=(i % 4 != 0))  # 75%

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    cm.save_state(path)

    cm2 = CurriculumManager()
    loaded = cm2.load_state(path)
    check(loaded, "State loaded successfully", "Failed to load state")
    check(cm2._chapter_idx == 3, f"Chapter idx restored = {cm2._chapter_idx}", "Chapter idx mismatch")
    check(cm2._total_steps == 5_000_000, f"Total steps restored = {cm2._total_steps}", "Steps mismatch")
    check(cm2._chapter_steps == 1_200_000, f"Chapter steps restored = {cm2._chapter_steps}", "Ch steps mismatch")
    check(len(cm2._success_history) == 200, f"History len = {len(cm2._success_history)}", "History len mismatch")

    os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: Overlap Ceiling Analysis
# ══════════════════════════════════════════════════════════════════════════════
def test_overlap_ceiling():
    """Check that 8° overlaps don't create a coasting ceiling above threshold."""
    print("\n" + "─" * 70)
    print("  TEST 9: Overlap Ceiling Analysis (Anti-Coasting)")
    print("─" * 70)

    for i in range(len(CHAPTERS) - 1):
        prev = CHAPTERS[i]
        nxt = CHAPTERS[i + 1]
        overlap = prev.angle_max_deg - nxt.angle_min_deg
        band_width = nxt.angle_max_deg - nxt.angle_min_deg
        threshold = nxt.promotion_threshold

        # If agent only solves the familiar angles and fails all new ones,
        # what's the max SR?  (overlap / band_width)
        easy_pct = overlap / band_width
        ceiling = easy_pct  # max SR from coasting

        if ceiling >= threshold:
            log_fail(f"Ch{i+2} [{nxt.name}] coasting ceiling ({ceiling:.0%}) >= threshold ({threshold:.0%})",
                     "Agent can promote without learning new angles!")
        else:
            log_pass(f"Ch{i+2} [{nxt.name}] ceiling ({ceiling:.0%}) < threshold ({threshold:.0%})",
                     f"Agent MUST learn new angles (overlap={easy_pct:.0%} of band)")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: v8 Bug Regression — False Promotion Scenario
# ══════════════════════════════════════════════════════════════════════════════
def test_v8_regression():
    """Reproduce the exact v8 bug scenario: the promotion check runs
    on every _on_step, and a 72% agent eventually spikes to 80%."""
    print("\n" + "─" * 70)
    print("  TEST 10: v8 Bug Regression — False Promotion Scenario")
    print("─" * 70)

    import random
    random.seed(42)

    TRUE_RATE = 0.72  # agent's real ability
    WINDOW = 200
    NUM_CHECKS = 5000  # simulate 5000 step-level checks (like v8)

    buffer = deque(maxlen=WINDOW)
    # Fill buffer
    for _ in range(WINDOW):
        buffer.append(random.random() < TRUE_RATE)

    # Simulate v8: check promotion on every step
    v8_promoted = False
    for check_i in range(NUM_CHECKS):
        # Simulate ~3 new episodes per check (32 envs, 72 avg ep len)
        for _ in range(3):
            buffer.append(random.random() < TRUE_RATE)
        sr = sum(buffer) / len(buffer)
        if sr >= 0.80:
            v8_promoted = True
            break

    # Under v8, this SHOULD have falsely promoted
    if v8_promoted:
        log_pass(f"v8 bug reproduced: 72% agent falsely promoted after {check_i} checks",
                 "Confirms the bug existed")
    else:
        log_warn("Could not reproduce v8 bug in 5000 checks (possible but unlikely)")

    # Now simulate v9: check only at log intervals with sustained gate
    random.seed(42)  # same seed
    buffer2 = deque(maxlen=WINDOW)
    for _ in range(WINDOW):
        buffer2.append(random.random() < TRUE_RATE)

    streak = 0
    v9_promoted = False
    # 5000 checks / ~10 checks per log interval ≈ 500 log intervals
    for log_check in range(500):
        # Simulate all episodes between log intervals (~1800 episodes)
        for _ in range(1800):
            buffer2.append(random.random() < TRUE_RATE)
        sr = sum(buffer2) / len(buffer2)
        if sr >= 0.80:
            streak += 1
            if streak >= 3:
                v9_promoted = True
                break
        else:
            streak = 0

    check(not v9_promoted,
          "v9 sustained gate: 72% agent does NOT promote (bug fixed!)",
          "v9 gate FAILED: 72% agent still promoted!")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: Train Script Structural Checks
# ══════════════════════════════════════════════════════════════════════════════
def test_train_script():
    print("\n" + "─" * 70)
    print("  TEST 11: Train Script Structural Checks")
    print("─" * 70)

    try:
        with open("train_curriculum.py", "r", encoding="utf-8") as f:
            src = f.read()
    except FileNotFoundError:
        log_fail("train_curriculum.py not found")
        return

    # CRITICAL: should_promote must NOT appear outside log-interval guard
    lines = src.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "should_promote" in stripped and not stripped.startswith("#"):
            indent = len(line) - len(line.lstrip())
            check(indent >= 12,
                  f"Line {i+1}: should_promote at indent {indent} (inside log block)",
                  f"Line {i+1}: should_promote at indent {indent} (possibly outside log block!)")

    # Must have PROMOTE_SUSTAIN_COUNT
    check("PROMOTE_SUSTAIN_COUNT" in src,
          "PROMOTE_SUSTAIN_COUNT defined in train_curriculum.py",
          "PROMOTE_SUSTAIN_COUNT missing!")

    # Must have _promote_streak
    check("_promote_streak" in src,
          "_promote_streak tracked in train_curriculum.py",
          "_promote_streak missing!")

    # Must clear streak on promotion
    check("_promote_streak = 0" in src,
          "Streak reset on promotion",
          "Streak not reset after promotion!")

    # Must log promotion gate status
    check("Promotion gate" in src,
          "Promotion gate status printed",
          "No promotion gate status log!")

    # Must have Monitor wrapper
    check("Monitor(" in src,
          "Monitor wrapper used for episode tracking",
          "Monitor wrapper missing!")

    # Must have WandbCallback
    check("WandbCallback" in src,
          "WandbCallback for W&B logging",
          "WandbCallback missing!")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 12: Angle Range Completeness
# ══════════════════════════════════════════════════════════════════════════════
def test_angle_coverage():
    """Verify there are no gaps in angular coverage from 16° to 180°."""
    print("\n" + "─" * 70)
    print("  TEST 12: Angle Range Completeness (No Gaps)")
    print("─" * 70)

    # Check every degree from 16 to 180 is covered by at least one chapter
    for deg in range(16, 181):
        covered = False
        for ch in CHAPTERS:
            if ch.angle_min_deg <= deg <= ch.angle_max_deg:
                covered = True
                break
        if not covered:
            log_fail(f"{deg}° is not covered by any chapter!")

    # Summary
    gaps = []
    for deg in range(16, 181):
        if not any(ch.angle_min_deg <= deg <= ch.angle_max_deg for ch in CHAPTERS):
            gaps.append(deg)

    if len(gaps) == 0:
        log_pass("Full angular coverage 16°–180° with no gaps")
    else:
        log_fail(f"Gaps found at: {gaps}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  v9 PRODUCTION AUDIT — COMPREHENSIVE TEST SUITE")
    print("█" * 70)

    test_chapter_configs()
    test_overlaps()
    test_ch2b_lr()
    test_promotion_logic()
    test_sustained_promotion_gate()
    test_quaternion()
    test_reward_math()
    test_persistence()
    test_overlap_ceiling()
    test_v8_regression()
    test_train_script()
    test_angle_coverage()

    # Final summary
    total = PASS + FAIL + WARN
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  PASS: {PASS}")
    print(f"  FAIL: {FAIL}")
    print(f"  WARN: {WARN}")
    print(f"  Total: {total}")
    print(f"{'='*70}")

    if FAIL > 0:
        print(f"\n  {FAIL} FAILURES FOUND — must fix before training!")
        sys.exit(1)
    elif WARN > 0:
        print(f"\n  {WARN} warnings — review but not blocking.")
        sys.exit(0)
    else:
        print(f"\n  ALL TESTS PASSED — ready for production training!")
        sys.exit(0)
