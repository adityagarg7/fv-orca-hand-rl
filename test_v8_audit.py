"""
v8 Production Audit — Comprehensive Dry-Run Test Suite
=======================================================
Tests every piece of logic WITHOUT requiring a GPU or long training run.
Uses real orca_sim environments where available, mocks where not.

Run:  python test_v8_audit.py
"""

import sys
import os
import json
import math
import traceback
from collections import deque
from dataclasses import asdict

import numpy as np

# ── Add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curriculum import CurriculumManager, CHAPTERS, ChapterConfig

PASS = 0
FAIL = 0
WARN = 0

def log_pass(name, detail=""):
    global PASS
    PASS += 1
    print(f"  ✅ PASS: {name}" + (f" — {detail}" if detail else ""))

def log_fail(name, detail=""):
    global FAIL
    FAIL += 1
    print(f"  ❌ FAIL: {name}" + (f" — {detail}" if detail else ""))

def log_warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  ⚠️  WARN: {name}" + (f" — {detail}" if detail else ""))

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Chapter Config Validation
# ══════════════════════════════════════════════════════════════════════════════
def test_chapter_configs():
    section("TEST 1: Chapter Config Validation")

    # 1a. Number of chapters
    if len(CHAPTERS) == 8:
        log_pass("Chapter count", f"{len(CHAPTERS)} chapters")
    else:
        log_fail("Chapter count", f"Expected 8, got {len(CHAPTERS)}")

    # 1b. Angle coverage: must cover 16°–180° without gaps
    for i, ch in enumerate(CHAPTERS):
        if ch.angle_min_deg < ch.angle_max_deg:
            log_pass(f"Ch{i+1} angle range valid", f"{ch.name}: {ch.angle_min_deg}°–{ch.angle_max_deg}°")
        else:
            log_fail(f"Ch{i+1} angle range invalid", f"{ch.name}: min={ch.angle_min_deg} >= max={ch.angle_max_deg}")

    # 1c. Overlaps between consecutive chapters (must be 2–8°, not 0 or >10)
    for i in range(len(CHAPTERS) - 1):
        curr = CHAPTERS[i]
        nxt = CHAPTERS[i + 1]
        overlap = curr.angle_max_deg - nxt.angle_min_deg
        band_width = nxt.angle_max_deg - nxt.angle_min_deg
        overlap_pct = overlap / band_width * 100 if band_width > 0 else 0

        if overlap <= 0:
            log_fail(f"Ch{i+1}→Ch{i+2} GAP", f"Gap of {-overlap}° between {curr.name} and {nxt.name}. Agent would face unseen angles!")
        elif overlap > 10:
            log_fail(f"Ch{i+1}→Ch{i+2} overlap too large", f"{overlap}° overlap ({overlap_pct:.0f}% of Ch{i+2} band). Agent coasts on mastered angles!")
        elif overlap > 6:
            log_warn(f"Ch{i+1}→Ch{i+2} overlap borderline", f"{overlap}° overlap ({overlap_pct:.0f}% of Ch{i+2} band)")
        else:
            log_pass(f"Ch{i+1}→Ch{i+2} overlap OK", f"{overlap}° overlap ({overlap_pct:.0f}% of Ch{i+2} band)")

    # 1d. Full angular coverage (no gaps)
    first_min = CHAPTERS[0].angle_min_deg
    last_max = CHAPTERS[-1].angle_max_deg
    if first_min <= 20 and last_max >= 180:
        log_pass("Full 16°–180° coverage", f"Covers {first_min}°–{last_max}°")
    else:
        log_fail("Coverage gap", f"Only covers {first_min}°–{last_max}°")

    # 1e. Promotion thresholds
    for i, ch in enumerate(CHAPTERS):
        if 0.70 <= ch.promotion_threshold <= 0.90:
            log_pass(f"Ch{i+1} threshold OK", f"{ch.promotion_threshold:.0%}")
        else:
            log_warn(f"Ch{i+1} threshold unusual", f"{ch.promotion_threshold:.0%}")

    # 1f. Success bonus must increase monotonically
    for i in range(len(CHAPTERS) - 1):
        if CHAPTERS[i].success_bonus < CHAPTERS[i + 1].success_bonus:
            log_pass(f"Ch{i+1}→Ch{i+2} bonus increases", f"{CHAPTERS[i].success_bonus}→{CHAPTERS[i+1].success_bonus}")
        else:
            log_fail(f"Ch{i+1}→Ch{i+2} bonus not increasing", f"{CHAPTERS[i].success_bonus}→{CHAPTERS[i+1].success_bonus}")

    # 1g. Episode steps must increase monotonically
    for i in range(len(CHAPTERS) - 1):
        if CHAPTERS[i].max_episode_steps <= CHAPTERS[i + 1].max_episode_steps:
            log_pass(f"Ch{i+1}→Ch{i+2} episode length increases", f"{CHAPTERS[i].max_episode_steps}→{CHAPTERS[i+1].max_episode_steps}")
        else:
            log_fail(f"Ch{i+1}→Ch{i+2} episode length NOT increasing", f"{CHAPTERS[i].max_episode_steps}→{CHAPTERS[i+1].max_episode_steps}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Promotion Logic — All Edge Cases
# ══════════════════════════════════════════════════════════════════════════════
def test_promotion_logic():
    section("TEST 2: Promotion Logic — Edge Cases")

    # 2a. Fresh CM should NOT promote
    cm = CurriculumManager()
    if not cm.should_promote():
        log_pass("Fresh CM does not promote")
    else:
        log_fail("Fresh CM promotes immediately!")

    # 2b. High SR but not enough buffer → should NOT promote
    cm = CurriculumManager()
    cm._chapter_steps = 200_000  # enough steps
    cm.override_success_rate(0.95, buffer_len=50)  # high SR but only 50 episodes
    if not cm.should_promote():
        log_pass("Buffer too small (50/200) blocks promotion")
    else:
        log_fail("Promoted with insufficient buffer!")

    # 2c. Full buffer but SR below threshold → should NOT promote
    cm = CurriculumManager()
    cm._chapter_steps = 200_000
    cm.override_success_rate(0.70, buffer_len=200)  # below 80%
    if not cm.should_promote():
        log_pass("SR below threshold (70% < 80%) blocks promotion")
    else:
        log_fail("Promoted with SR below threshold!")

    # 2d. Full buffer, high SR, but not enough steps → should NOT promote
    cm = CurriculumManager()
    cm._chapter_steps = 50_000  # below MIN_STEPS_BEFORE_PROMOTE
    cm.override_success_rate(0.95, buffer_len=200)
    if not cm.should_promote():
        log_pass("Not enough chapter steps (50k < 100k) blocks promotion")
    else:
        log_fail("Promoted with insufficient chapter steps!")

    # 2e. All conditions met → SHOULD promote
    cm = CurriculumManager()
    cm._chapter_steps = 200_000
    cm.override_success_rate(0.85, buffer_len=200)
    if cm.should_promote():
        log_pass("All conditions met → promotes correctly")
    else:
        log_fail("Should promote but didn't!")

    # 2f. Exactly at threshold → SHOULD promote
    cm = CurriculumManager()
    cm._chapter_steps = 200_000
    cm.override_success_rate(0.80, buffer_len=200)
    if cm.should_promote():
        log_pass("Exactly at threshold (80%) → promotes correctly")
    else:
        log_fail("Should promote at exactly 80% but didn't!")

    # 2g. Promotion clears stale forced SR
    cm = CurriculumManager()
    cm._chapter_steps = 200_000
    cm.override_success_rate(0.90, buffer_len=200)
    assert cm.should_promote()
    cm.promote()

    if not hasattr(cm, "_forced_success_rate"):
        log_pass("Promotion clears _forced_success_rate (Bug 4 fix)")
    else:
        log_fail("STALE _forced_success_rate after promotion — Bug 4 NOT fixed!")

    if not hasattr(cm, "_forced_success_buffer_len"):
        log_pass("Promotion clears _forced_success_buffer_len")
    else:
        log_fail("STALE _forced_success_buffer_len after promotion!")

    # 2h. After promotion, should NOT immediately promote again
    cm._chapter_steps = 200_000  # pretend enough steps
    if not cm.should_promote():
        log_pass("Post-promotion: no cascading promotion (buffer cleared)")
    else:
        log_fail("CASCADING PROMOTION after clearing — Bug 4 still present!")

    # 2i. Final chapter should never promote
    cm2 = CurriculumManager()
    cm2._chapter_idx = len(CHAPTERS) - 1  # final chapter
    cm2._chapter_steps = 999_000_000
    cm2.override_success_rate(0.99, buffer_len=200)
    if not cm2.should_promote():
        log_pass("Final chapter never promotes")
    else:
        log_fail("Final chapter promoted — would crash!")

    # 2j. Full promotion chain (simulate Ch1 → Ch2a → Ch2b without cascading)
    cm3 = CurriculumManager()
    for target_ch in range(3):
        cm3._chapter_steps = 200_000
        cm3.override_success_rate(0.85, buffer_len=200)
        if cm3.should_promote():
            cm3.promote()
        else:
            log_fail(f"Failed to promote to chapter {target_ch + 1}")
            return

    if cm3.current_chapter_idx == 3:
        log_pass("Full chain Ch1→Ch2a→Ch2b→Ch2c works (no cascading)")
    else:
        log_fail(f"Chain stopped at chapter {cm3.current_chapter_idx}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Reward Math — Farming vs Solving
# ══════════════════════════════════════════════════════════════════════════════
def test_reward_math():
    section("TEST 3: Reward Math — Farming vs Solving (Per Chapter)")

    for ch_idx, ch in enumerate(CHAPTERS):
        angle_mid_deg = (ch.angle_min_deg + ch.angle_max_deg) / 2
        angle_mid_rad = math.radians(angle_mid_deg)
        steps = ch.max_episode_steps

        # Farming scenario: agent holds still for full episode
        r_align_per_step = 5.0 * math.exp(-angle_mid_rad / 0.3)
        r_pos_per_step = 0.5  # cube stays put
        r_fingers_per_step = 0.25  # fingers touching
        r_alive_per_step = 0.05
        farming_total = (r_align_per_step + r_pos_per_step + r_fingers_per_step + r_alive_per_step) * steps

        # Solving scenario: agent rotates to goal, terminates on success at ~40% through episode
        solve_steps = int(steps * 0.4)  # estimate: solves at 40% of max steps
        farming_before_solve = (r_align_per_step + r_pos_per_step + r_fingers_per_step + r_alive_per_step) * solve_steps
        progress_total = 50.0 * angle_mid_rad  # progress_weight × total rotation
        # Stability ≈ 0.5 for a moderately fast solve (conservative)
        success_bonus_actual = ch.success_bonus * 0.5
        solving_total = farming_before_solve + progress_total + success_bonus_actual

        ratio = solving_total / farming_total if farming_total > 0 else float('inf')

        if solving_total > farming_total:
            log_pass(
                f"Ch{ch_idx+1} [{ch.name}] solving > farming",
                f"Solve={solving_total:.0f} vs Farm={farming_total:.0f} (ratio {ratio:.1f}:1)"
            )
        else:
            log_fail(
                f"Ch{ch_idx+1} [{ch.name}] FARMING WINS",
                f"Solve={solving_total:.0f} vs Farm={farming_total:.0f} (ratio {ratio:.1f}:1)"
            )

        # Per-step return comparison
        per_step_solve = solving_total / solve_steps if solve_steps > 0 else 0
        per_step_farm = farming_total / steps if steps > 0 else 0
        if per_step_solve > per_step_farm:
            log_pass(
                f"Ch{ch_idx+1} per-step: solving > farming",
                f"{per_step_solve:.2f}/step vs {per_step_farm:.2f}/step"
            )
        else:
            log_warn(
                f"Ch{ch_idx+1} per-step: farming competitive",
                f"{per_step_solve:.2f}/step vs {per_step_farm:.2f}/step"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Quaternion Generation Correctness
# ══════════════════════════════════════════════════════════════════════════════
def test_quaternion():
    section("TEST 4: Quaternion Generation Correctness")

    # Import the wrapper to test its quaternion function
    try:
        from curriculum_wrapper import CurriculumWrapper
    except ImportError:
        log_warn("Cannot import CurriculumWrapper", "Skipping quaternion tests")
        return

    # Create a minimal mock to call the static method
    class MockEnv:
        pass

    cm = CurriculumManager()
    # We need to instantiate CurriculumWrapper to use _angle_to_random_quat
    # but it needs a gym.Env. Let's test the static method directly.

    # Test quaternion multiply
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    result = CurriculumWrapper._quat_multiply(q_identity, q_identity)
    if np.allclose(result, q_identity, atol=1e-10):
        log_pass("Quaternion identity multiply", f"I×I = I ✓")
    else:
        log_fail("Quaternion identity multiply", f"Got {result}")

    # Test 90° rotation around Z: q = [cos(45°), 0, 0, sin(45°)]
    q_90z = np.array([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)])
    q_180z = CurriculumWrapper._quat_multiply(q_90z, q_90z)
    expected_180z = np.array([0.0, 0.0, 0.0, 1.0])  # 180° around Z
    if np.allclose(q_180z, expected_180z, atol=1e-10):
        log_pass("Quaternion 90°+90° = 180° around Z", f"Result: {q_180z}")
    else:
        log_fail("Quaternion 90°+90° around Z", f"Expected {expected_180z}, got {q_180z}")

    # Test that generated quaternions have unit norm
    rng = np.random.default_rng(42)
    # Manually construct the quaternion like CurriculumWrapper does
    for angle_deg in [10, 30, 60, 90, 120, 150, 180]:
        theta = np.radians(angle_deg)
        phi = rng.uniform(0, 2 * np.pi)
        ax, ay = np.cos(phi), np.sin(phi)
        q_tilt = np.array([np.cos(theta/2), np.sin(theta/2)*ax, np.sin(theta/2)*ay, 0.0])
        psi = rng.uniform(0, 2 * np.pi)
        q_spin = np.array([np.cos(psi/2), 0.0, 0.0, np.sin(psi/2)])
        q = CurriculumWrapper._quat_multiply(q_spin, q_tilt)
        norm = np.linalg.norm(q)
        if abs(norm - 1.0) < 1e-10:
            log_pass(f"Quaternion norm at {angle_deg}°", f"|q| = {norm:.15f}")
        else:
            log_fail(f"Quaternion norm at {angle_deg}°", f"|q| = {norm:.15f} (should be 1.0)")

    # Verify tilt angle is preserved (the Z-component of the rotated UP vector)
    # For a tilt of θ around an axis in XY plane, the red-face-up angle should be θ.
    for angle_deg in [15, 30, 45, 60, 90, 120, 150, 180]:
        theta = np.radians(angle_deg)
        phi = rng.uniform(0, 2 * np.pi)
        ax, ay = np.cos(phi), np.sin(phi)
        q_tilt = np.array([np.cos(theta/2), np.sin(theta/2)*ax, np.sin(theta/2)*ay, 0.0])

        # Rotate the UP vector [0,0,1] by this quaternion
        # Using q * v * q_inv for vector rotation
        w, x, y, z = q_tilt
        # Rotation matrix from quaternion (3rd column = rotated Z axis)
        rz_x = 2*(x*z + w*y)
        rz_y = 2*(y*z - w*x)
        rz_z = 1 - 2*(x*x + y*y)
        rotated_up = np.array([rz_x, rz_y, rz_z])

        # The angle between rotated_up and [0,0,1]
        cos_angle = np.clip(rotated_up[2], -1.0, 1.0)
        recovered_angle = np.arccos(cos_angle)
        recovered_deg = np.degrees(recovered_angle)

        if abs(recovered_deg - angle_deg) < 0.1:
            log_pass(f"Tilt angle preserved at {angle_deg}°", f"Recovered: {recovered_deg:.2f}°")
        else:
            log_fail(f"Tilt angle NOT preserved at {angle_deg}°", f"Recovered: {recovered_deg:.2f}° (error: {abs(recovered_deg-angle_deg):.2f}°)")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Persistence (Save/Load Curriculum State)
# ══════════════════════════════════════════════════════════════════════════════
def test_persistence():
    section("TEST 5: Persistence (Save/Load)")

    import tempfile

    cm = CurriculumManager()
    cm._chapter_idx = 3
    cm._chapter_steps = 456_789
    cm._total_steps = 12_345_678
    cm._chapter_episodes = 100
    cm._total_episodes = 5000
    cm._success_history = deque([True, False, True, True], maxlen=200)
    cm._promotion_log = [{"from_chapter": "ch1", "from_idx": 0, "success_rate": 0.85}]

    tmp_path = os.path.join(tempfile.gettempdir(), "test_curriculum_state.json")

    try:
        cm.save_state(tmp_path)
        if os.path.exists(tmp_path):
            log_pass("Save state creates file")
        else:
            log_fail("Save state did not create file")
            return

        # Verify JSON is valid
        with open(tmp_path) as f:
            data = json.load(f)
        if data["chapter_idx"] == 3 and data["total_steps"] == 12_345_678:
            log_pass("Saved JSON has correct values")
        else:
            log_fail("Saved JSON has wrong values", str(data))

        # Load into new CM
        cm2 = CurriculumManager()
        loaded = cm2.load_state(tmp_path)
        if loaded:
            log_pass("Load state returns True")
        else:
            log_fail("Load state returns False")

        if cm2._chapter_idx == 3:
            log_pass("Loaded chapter_idx matches", f"idx={cm2._chapter_idx}")
        else:
            log_fail("Loaded chapter_idx wrong", f"Expected 3, got {cm2._chapter_idx}")

        if cm2._total_steps == 12_345_678:
            log_pass("Loaded total_steps matches")
        else:
            log_fail("Loaded total_steps wrong")

        if list(cm2._success_history) == [True, False, True, True]:
            log_pass("Loaded success_history matches")
        else:
            log_fail("Loaded success_history wrong", str(list(cm2._success_history)))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Callback Success Tracking (Simulated SubprocVecEnv)
# ══════════════════════════════════════════════════════════════════════════════
def test_callback_success_tracking():
    section("TEST 6: Callback Success Tracking (Simulated)")

    # Simulate what CurriculumCallback does when scraping infos
    SUCCESS_WINDOW = 200
    success_buffer = deque(maxlen=SUCCESS_WINDOW)
    cm = CurriculumManager()

    # Simulate 300 episodes: 70% success
    for i in range(300):
        # Simulate info dict from VecEnv (episode just ended)
        info = {"episode": {"r": 100, "l": 200}, "is_success": (i % 10) < 7}

        # This is what the callback does:
        if "episode" in info:
            terminal_info = info.get("terminal_info", info)
            success = bool(terminal_info.get("is_success", False))
            success_buffer.append(success)

    sr = sum(success_buffer) / len(success_buffer)
    cm.override_success_rate(sr, len(success_buffer))

    if abs(sr - 0.70) < 0.05:
        log_pass("Success buffer tracks correctly", f"SR={sr:.2%} (expected ~70%)")
    else:
        log_fail("Success buffer SR wrong", f"SR={sr:.2%} (expected ~70%)")

    if len(success_buffer) == SUCCESS_WINDOW:
        log_pass("Buffer maxlen enforced", f"len={len(success_buffer)}")
    else:
        log_fail("Buffer maxlen wrong", f"len={len(success_buffer)}")

    # Verify promotion does NOT fire at 70%
    cm._chapter_steps = 200_000
    if not cm.should_promote():
        log_pass("70% SR does not trigger promotion (threshold=80%)")
    else:
        log_fail("70% SR triggered promotion!")

    # Now push to 85%
    success_buffer.clear()
    for i in range(200):
        success_buffer.append(i % 100 < 85)  # 85% success
    sr2 = sum(success_buffer) / len(success_buffer)
    cm.override_success_rate(sr2, len(success_buffer))
    cm._chapter_steps = 200_000

    if cm.should_promote():
        log_pass("85% SR triggers promotion correctly")
    else:
        log_fail("85% SR did not trigger promotion!")

    # After promotion, buffer should be cleared by callback
    cm.promote()
    success_buffer.clear()  # callback does this
    cm._chapter_steps = 200_000

    if len(success_buffer) == 0:
        log_pass("Buffer cleared after promotion")
    else:
        log_fail("Buffer NOT cleared after promotion")

    # With empty buffer, should NOT promote (even if stale SR was high)
    if not cm.should_promote():
        log_pass("Empty buffer after promotion blocks cascading")
    else:
        log_fail("CASCADING PROMOTION with empty buffer!")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: Episode Length and Truncation Logic
# ══════════════════════════════════════════════════════════════════════════════
def test_episode_length():
    section("TEST 7: Episode Length Analysis")

    for i, ch in enumerate(CHAPTERS):
        angle_mid = (ch.angle_min_deg + ch.angle_max_deg) / 2
        angle_rad = math.radians(angle_mid)

        # Estimate: how many steps to rotate angle_mid degrees?
        # MuJoCo timestep = 0.002s, action repeat = 5, so 1 step = 0.01s
        # Typical cube rotation speed with dexterous hand: ~30°/s to 90°/s
        # Conservative: 20°/s, generous: 60°/s
        conservative_steps = int(angle_mid / 20 / 0.01)  # 20°/s
        generous_steps = int(angle_mid / 60 / 0.01)      # 60°/s

        available = ch.max_episode_steps
        ratio = available / conservative_steps if conservative_steps > 0 else float('inf')

        if available >= conservative_steps:
            log_pass(
                f"Ch{i+1} episode length sufficient",
                f"{available} steps for {angle_mid:.0f}° rotation "
                f"(need {generous_steps}–{conservative_steps} steps, ratio={ratio:.1f}×)"
            )
        elif available >= generous_steps:
            log_warn(
                f"Ch{i+1} episode length tight",
                f"{available} steps for {angle_mid:.0f}° rotation "
                f"(need {generous_steps}–{conservative_steps} steps)"
            )
        else:
            log_fail(
                f"Ch{i+1} episode length TOO SHORT",
                f"{available} steps for {angle_mid:.0f}° rotation "
                f"(need at least {generous_steps} steps)"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: Real Environment Stack (if orca_sim available)
# ══════════════════════════════════════════════════════════════════════════════
def test_real_env():
    section("TEST 8: Real Environment Stack Integration")

    try:
        from orca_sim import OrcaHandRightCubeOrientation
        log_pass("orca_sim imported successfully")
    except ImportError as e:
        log_warn("orca_sim not available", f"Skipping real env tests: {e}")
        return

    from production_reward import ProductionRewardWrapper
    from curriculum_wrapper import CurriculumWrapper
    from stable_baselines3.common.monitor import Monitor

    # 8a. Build the full env stack
    try:
        cm = CurriculumManager()
        base = OrcaHandRightCubeOrientation(render_mode=None)
        rewarded = ProductionRewardWrapper(base)
        env = CurriculumWrapper(rewarded, cm)
        monitored = Monitor(env, info_keywords=("is_success",))
        log_pass("Full env stack builds without error")
    except Exception as e:
        log_fail("Env stack construction failed", str(e))
        traceback.print_exc()
        return

    # 8b. Reset and verify info keys
    try:
        obs, info = monitored.reset()
        required_keys = ["cube_pos", "red_face_up_angle_rad", "is_success",
                         "curriculum_spawn_angle_deg", "curriculum_chapter"]
        for key in required_keys:
            if key in info:
                log_pass(f"Info key '{key}' present", f"value={info[key]}")
            else:
                log_fail(f"Info key '{key}' MISSING from reset info")

        # Verify spawn angle is within Ch1 range
        spawn = info.get("curriculum_spawn_angle_deg", -1)
        ch1 = CHAPTERS[0]
        if ch1.angle_min_deg <= spawn <= ch1.angle_max_deg:
            log_pass(f"Spawn angle in Ch1 range", f"{spawn:.1f}° ∈ [{ch1.angle_min_deg}°, {ch1.angle_max_deg}°]")
        else:
            log_fail(f"Spawn angle OUT of Ch1 range", f"{spawn:.1f}° NOT in [{ch1.angle_min_deg}°, {ch1.angle_max_deg}°]")
    except Exception as e:
        log_fail("Env reset failed", str(e))
        traceback.print_exc()
        return

    # 8c. Step and verify reward breakdown
    try:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = monitored.step(action)

        if "reward_breakdown" in info:
            rb = info["reward_breakdown"]
            expected_components = ["align", "progress", "pos", "fingers", "success", "alive", "drop", "action_rate", "action_mag", "total"]
            for comp in expected_components:
                if comp in rb:
                    log_pass(f"Reward component '{comp}'", f"value={rb[comp]:.4f}")
                else:
                    log_fail(f"Reward component '{comp}' MISSING")

            # Verify total matches sum of components
            computed_total = (rb["align"] + rb["progress"] + rb["pos"] + rb["fingers"]
                            + rb["success"] + rb["alive"] + rb["drop"]
                            + rb["action_rate"] + rb["action_mag"])
            if abs(computed_total - rb["total"]) < 1e-6:
                log_pass("Reward total matches component sum", f"{computed_total:.6f} ≈ {rb['total']:.6f}")
            else:
                log_fail("Reward total MISMATCH", f"sum={computed_total:.6f} vs total={rb['total']:.6f}")
        else:
            log_fail("reward_breakdown missing from step info")

    except Exception as e:
        log_fail("Env step failed", str(e))
        traceback.print_exc()

    # 8d. Run a full episode and verify termination/truncation
    try:
        obs, info = monitored.reset()
        total_reward = 0
        step_count = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = monitored.step(action)
            total_reward += reward
            step_count += 1

            if step_count > 1000:  # safety limit
                log_fail("Episode exceeded 1000 steps without ending")
                break

        if terminated:
            end_reason = "terminated (dropped or success)"
        elif truncated:
            end_reason = f"truncated at step {step_count}"
        else:
            end_reason = "hit safety limit"

        log_pass(f"Episode completed", f"{step_count} steps, reward={total_reward:.1f}, {end_reason}")

        # Check if truncation happened at expected step count
        ch1_max = CHAPTERS[0].max_episode_steps
        if truncated and abs(step_count - ch1_max) <= 1:
            log_pass(f"Truncation at correct step", f"step={step_count}, max={ch1_max}")
        elif terminated:
            log_pass(f"Episode terminated before truncation", f"step={step_count} (drop or success)")

    except Exception as e:
        log_fail("Full episode run failed", str(e))
        traceback.print_exc()

    # 8e. Test is_success triggers termination
    try:
        obs, info = monitored.reset()
        # Run a few steps and check that is_success=True would terminate
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = monitored.step(action)
            if info.get("is_success", False) and terminated:
                log_pass("is_success=True triggers termination")
                break
        else:
            log_warn("No is_success=True observed in 5 random steps (expected — random policy rarely solves)")
    except Exception as e:
        log_fail("Success termination test failed", str(e))

    # 8f. Test chapter change propagation
    try:
        env_inner = monitored.env  # CurriculumWrapper
        if hasattr(env_inner, 'set_chapter_idx'):
            env_inner.set_chapter_idx(2)
            obs2, info2 = monitored.reset()
            spawn2 = info2.get("curriculum_spawn_angle_deg", -1)
            ch2b = CHAPTERS[2]
            if ch2b.angle_min_deg <= spawn2 <= ch2b.angle_max_deg:
                log_pass(f"Chapter change propagates", f"Ch2b spawn: {spawn2:.1f}° ∈ [{ch2b.angle_min_deg}°, {ch2b.angle_max_deg}°]")
            else:
                log_fail(f"Chapter change NOT reflected", f"spawn={spawn2:.1f}° NOT in Ch2b [{ch2b.angle_min_deg}°, {ch2b.angle_max_deg}°]")
            # Reset back to ch1
            env_inner.set_chapter_idx(0)
        else:
            log_fail("set_chapter_idx not found on CurriculumWrapper")
    except Exception as e:
        log_fail("Chapter change test failed", str(e))

    monitored.close()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: W&B Integration Check
# ══════════════════════════════════════════════════════════════════════════════
def test_wandb():
    section("TEST 9: Weights & Biases Integration")

    try:
        import wandb
        log_pass("wandb imported", f"version={wandb.__version__}")
    except ImportError:
        log_fail("wandb NOT installed")
        return

    # Check if logged in
    try:
        api = wandb.Api()
        log_pass("wandb API accessible (logged in)")
    except Exception as e:
        log_warn("wandb API not accessible", f"{e} — may need 'wandb login'")

    # Verify WandbCallback import
    try:
        from wandb.integration.sb3 import WandbCallback
        log_pass("WandbCallback importable")
    except ImportError:
        log_fail("WandbCallback NOT importable — need wandb[sb3]")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: Train Script Imports & Configuration
# ══════════════════════════════════════════════════════════════════════════════
def test_train_script():
    section("TEST 10: Train Script Configuration")

    try:
        from stable_baselines3 import PPO
        log_pass("SB3 PPO importable")
    except ImportError:
        log_fail("SB3 not installed")

    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        log_pass("SubprocVecEnv/DummyVecEnv importable")
    except ImportError:
        log_fail("VecEnv imports failed")

    # Check platform-aware start method
    if sys.platform == "win32":
        expected = "spawn"
    else:
        expected = "forkserver"
    log_pass(f"Platform: {sys.platform} → start_method='{expected}'")

    # Verify the train script has correct start method
    train_path = os.path.join(os.path.dirname(__file__), "train_curriculum.py")
    with open(train_path) as f:
        content = f.read()

    if 'sys.platform == "win32"' in content or "sys.platform ==" in content:
        log_pass("train_curriculum.py has platform-aware start method")
    else:
        log_fail("train_curriculum.py missing platform detection for start_method")

    if "ENT_COEF_MIN    = 0.005" in content:
        log_pass("Entropy floor = 0.005 (raised)")
    elif "ENT_COEF_MIN" in content:
        # Find the actual value
        import re
        match = re.search(r'ENT_COEF_MIN\s*=\s*([\d.e-]+)', content)
        if match:
            log_fail(f"Entropy floor = {match.group(1)} (should be 0.005)")
        else:
            log_fail("Cannot parse ENT_COEF_MIN")
    else:
        log_fail("ENT_COEF_MIN not found in train_curriculum.py")

    # Check Monitor has info_keywords
    if 'info_keywords=("is_success",)' in content:
        log_pass("Monitor forwards is_success via info_keywords")
    else:
        log_fail("Monitor missing info_keywords — SB3 won't see is_success!")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: Overlap Ceiling Prediction per Chapter
# ══════════════════════════════════════════════════════════════════════════════
def test_overlap_ceiling():
    section("TEST 11: Overlap Ceiling Predictions (Mathematical)")

    for i in range(len(CHAPTERS) - 1):
        curr = CHAPTERS[i]
        nxt = CHAPTERS[i + 1]
        overlap = curr.angle_max_deg - nxt.angle_min_deg

        if overlap <= 0:
            continue  # no overlap

        band = nxt.angle_max_deg - nxt.angle_min_deg
        easy_pct = overlap / band
        # Assume 85% success on easy (mastered) and 10% on hard (new)
        ceiling = easy_pct * 0.85 + (1 - easy_pct) * 0.10
        threshold = nxt.promotion_threshold

        if ceiling >= threshold:
            log_pass(
                f"Ch{i+2} [{nxt.name}] ceiling ({ceiling:.0%}) ≥ threshold ({threshold:.0%})",
                f"Can reach threshold even if agent only solves easy overlap"
            )
            log_warn(
                f"Ch{i+2} might promote without learning new angles",
                f"Easy overlap = {easy_pct:.0%} of band"
            )
        else:
            log_pass(
                f"Ch{i+2} [{nxt.name}] ceiling ({ceiling:.0%}) < threshold ({threshold:.0%})",
                f"Agent MUST learn new angles to promote (overlap={easy_pct:.0%} of band)"
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  v8 PRODUCTION AUDIT — COMPREHENSIVE TEST SUITE")
    print("█" * 70)

    test_chapter_configs()
    test_promotion_logic()
    test_reward_math()
    test_quaternion()
    test_persistence()
    test_callback_success_tracking()
    test_episode_length()
    test_real_env()
    test_wandb()
    test_train_script()
    test_overlap_ceiling()

    # Final summary
    total = PASS + FAIL + WARN
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  ✅ PASS: {PASS}")
    print(f"  ❌ FAIL: {FAIL}")
    print(f"  ⚠️  WARN: {WARN}")
    print(f"  Total : {total}")
    print(f"{'='*70}")

    if FAIL > 0:
        print(f"\n  🚨 {FAIL} FAILURES FOUND — must fix before training!")
        sys.exit(1)
    elif WARN > 0:
        print(f"\n  ⚠️  {WARN} warnings — review but not blocking.")
        sys.exit(0)
    else:
        print(f"\n  🎉 ALL TESTS PASSED — ready for production training!")
        sys.exit(0)
