# ORCA Hand — Reinforcement Learning (v11 — Exploit-Closed Edition)

Reinforcement-learning training for the **ORCA dexterous hand** at Four Vectors.

**Task:** In-hand cube reorientation — rotate a 36mm cube held in the ORCA right hand
from any starting angle (up to 180°) to the red-face-up goal orientation, using
coordinated finger manipulation. No wrist flicking. No gravity exploitation.
No dropping after success. Human-like, palm-centered, finger-first dexterous control.

---

## Table of Contents

1. [Version History: v0 → v11](#1-version-history-v0--v11)
2. [System Architecture](#2-system-architecture)
3. [Reward Function — Complete Mathematical Specification](#3-reward-function--complete-mathematical-specification)
4. [v9 → v10 → v11: What Changed and Why](#4-v9--v10--v11-what-changed-and-why)
5. [Curriculum Design](#5-curriculum-design)
6. [File Reference](#6-file-reference)
7. [Environment Setup](#7-environment-setup)
8. [Running Training](#8-running-training)
9. [Monitoring & Debugging](#9-monitoring--debugging)
10. [Hardware Recommendations](#10-hardware-recommendations)

---

## 1. Version History: v0 → v11

### The Journey (Every Bug, Every Fix)

| Version | Problem | Root Cause | Fix |
|---------|---------|-----------|-----|
| **v0** | 0% success | Hold-timer gate: impossible to hold cube still with a noisy random policy | Removed hold timer |
| **v1** | 100% false positive | "Perfect drop" — cube dropped face-up counted as success | Fixed: success = angular check only |
| **v2** | Entropy collapse (std=1.15) | No entropy management → policy collapsed to jitter | Added adaptive entropy (AE-PPO) |
| **v3** | Frozen learning (clip_fraction=0) | LR too low after entropy fix | Clip warm-restart at promotion |
| **v4** | Ch2 flatline at 0% | 40°–90° band has phase transition at 75° (nudge → roll) | Split Ch2 into 3 sub-chapters |
| **v5** | Capacity bottleneck | [256,256] network too small for dexterous policy | Upgraded to [512,256] |
| **v6** | Throughput bottleneck | DummyVecEnv = single process on 2 Colab cores | SubprocVecEnv with 64 processes |
| **v7** | Promotion never triggers | 3 cascading bugs: SB3 `ep_success_buffer` missing, Monitor keywords, SubprocVecEnv info isolation | Custom success tracking + `info_keywords` |
| **v8** | Instant cascading promotions | Stale `_forced_success_rate` not cleared after `promote()` | Clear stale attributes on promotion |
| **v9** | **Wrist exploitation** | r_progress strategy-agnostic; wrist 16× cheaper than fingers; drop-reward bug (+147); r_align dead above 60° | Completed all chapters in 263M steps but with wrong strategy |
| **v10** | **5 critical failures** (see §4) | G_contact ≡ 0 (geometry bug); farmable success ramp; action-based wrist gate bypassable; standing income > task income; EMA pulse exploit | External audit found v10 would reproduce v9's failure mode |
| **v11** | — | All v10 audit findings fixed by construction: real contacts, palm-frame attribution, potential shaping, non-terminal success, anti-ratchet invariant | **Current version** |

### v11 Design Principles

v11 is built on 5 **structural guarantees**, each closing a class of exploit:

| Mechanism | What it closes | How |
|-----------|---------------|-----|
| **Real contacts** (`data.contact`) | Impossible-threshold distance proxy (v10 F1) | Iterate MuJoCo collision pairs — exact, ungameable, 0 false negatives |
| **Palm-frame attribution** | Wrist exploit + EMA pulse aliasing (v10 F3/F5) | Counterfactual decomposition: measure how much θ changed due to finger motion vs wrist carry. Wrist contribution = 0 by construction. |
| **Potential-based shaping** (Ng et al. 1999) | Standing income → loiter equilibrium (v10 F4) | All dense state terms enter as γΦ(s')−Φ(s). At any fixed state: reward = −(1−γ)Φ ≤ 0. No loitering income, ever. |
| **Non-terminal success** | Farmable ramp + termination forfeiture (v10 F2/F4) | Episode continues past goal. Agent earns +6/step while holding — the single largest income. Succeeding > loitering. |
| **Anti-ratchet invariant** | Rock-the-cube / oscillation farming | Positive progress gated & attenuated; negative progress charged at full rate. Over any closed θ-cycle: Σ reward ≤ 0. |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  SubprocVecEnv (64 independent OS processes)                         │
│                                                                      │
│  Process k:                                                          │
│    OrcaHandRightCubeOrientation(max_episode_steps=1_000_000)         │
│      ↳ MuJoCo physics at 50Hz, 51-dim obs, 17-dim action            │
│    ProductionRewardWrapper(gamma=0.99)                               │
│      ↳ v11 reward: potential + attributed progress + goal stream     │
│      ↳ Real contact detection via data.contact                       │
│      ↳ Palm-frame counterfactual attribution                         │
│      ↳ Overrides termination: ONLY drop terminates (success → hold)  │
│    CurriculumWrapper(cm)                                             │
│      ↳ Spawns cube at chapter-specific angle range                   │
│      ↳ Truncates at chapter-specific episode length                  │
│      ↳ Records is_success at episode end for promotion gate          │
│    Monitor(info_keywords=("is_success",))                            │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
              observations (64 × 51), actions (64 × 17)
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│  PPO (Stable-Baselines3)                                             │
│  Policy: MLP [512 → 256 → 17]   ← CUDA GPU                         │
│  Value:  MLP [512 → 256 → 1]    ← CUDA GPU                         │
│  n_steps = 4096 per env → 262,144 steps per rollout                 │
│  gamma = 0.99, GAE λ = 0.95, batch = 1024                           │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│  CurriculumCallback                                                  │
│  • Promotes at 80% rolling success (200 episodes, 3× sustained)      │
│  • AE-PPO: keeps action std in [0.75, 1.05]                         │
│  • Clip warm-restart: 0.3 at promotion, decays to 0.2               │
│  • Broadcasts chapter changes to all 64 subprocesses                 │
│  • Logs every reward component to W&B                                │
└──────────────────────────────────────────────────────────────────────┘
```

### v11 Critical Bugfix: Silent 200-Step Episode Cap

**All v7–v10 training runs were silently capped at 200 steps per episode.**

The base env `OrcaHandRightCubeOrientation` defaults to `max_episode_steps=200` and
truncates itself in `_get_truncated()`. The curriculum's per-chapter episode lengths
(210–760 steps) were never applied — the base env truncated first.

v11 fix: `OrcaHandRightCubeOrientation(max_episode_steps=1_000_000)` disables base
truncation. `CurriculumWrapper` is now the single source of episode length.

---

## 3. Reward Function — Complete Mathematical Specification

### 3.1 Overview

The v11 reward at each timestep t is:

```
R(t) = r_potential(t) + r_progress(t) + r_goal(t) + r_bonus(t)
       − r_drop(t) − r_wrist_rate(t) − r_finger_rate(t)
```

### 3.2 Component 1: Potential-Based Shaping

**Theory:** Ng, Harada & Russell (1999) proved that adding a reward of the form
`F(s, s') = γ·Φ(s') − Φ(s)` to any MDP preserves the optimal policy. The key
property: at any fixed state s where s' = s, F = (γ−1)·Φ(s) < 0. **There is no
positive standing income at any state.** Loitering is impossible.

**Implementation:**

```
Φ(s) = Φ_align + Φ_pos + Φ_contact + Φ_prox
```

| Component | Formula | Range | Purpose |
|-----------|---------|-------|---------|
| Φ_align | `3.0 · exp(−θ / 1.0)` | [0, 3.0] | Orientation proximity to goal |
| Φ_pos | `1.0 · exp(−d_pos / 0.05)` | [0, 1.0] | Palm-centering (distance from default cube position) |
| Φ_contact | `0.5 · min(N_c, 2) / 2` | [0, 0.5] | Fingertip contact reward (real MuJoCo contacts) |
| Φ_prox | `0.5 · mean(exp(−max(d_i − 0.045, 0) / 0.03))` | [0, 0.5] | Fingertip proximity reach signal |

**Total potential range:** [0, 5.0]

**Shaping reward:**

```
r_potential(t) = γ · Φ(s_t) − Φ(s_{t-1})     where γ = 0.99
```

**Mathematical properties:**

1. **No loiter income:** At a fixed state: `r_potential = (0.99 − 1) · Φ = −0.01 · Φ ≤ 0`
2. **Gait excursion refund:** If the cube is displaced during finger gaiting and then returned, the potential exactly cancels: `Σ F(s, s') = γ·Φ(s_final) − Φ(s_initial)`. If endpoints match, net = `(γ−1)·Φ ≈ 0`.
3. **Policy invariance:** The optimal policy is unchanged by the addition of F.

#### Why σ = 1.0 for Φ_align

The previous versions used σ = 0.3, which made `exp(−θ/0.3)` vanish above 60°:

| θ | σ = 0.3 (v9) | σ = 1.0 (v11) | Improvement |
|---|-------------|---------------|-------------|
| 30° (0.52 rad) | 0.87 | 0.59 | — |
| 60° (1.05 rad) | 0.03 | 0.35 | 12× |
| 90° (π/2 rad) | 0.005 | 0.21 | 42× |
| 120° (2.09 rad) | 0.0009 | 0.12 | 133× |
| 150° (2.62 rad) | 0.0002 | 0.07 | 350× |

With σ = 1.0, the value function can learn "being at 90° is better than 120°" for **all 8 chapters**.

#### Why Φ_prox uses d − 0.045

**The v10 fatal bug:** v10 used `exp(−d/0.02)` with a 20mm contact threshold on
`xpos[fingertip_body]`. But fingertip body origins are at the PIP joint, **40–44mm
behind the pad surface** (verified from STL meshes). The cube half-extent is 18mm.
Minimum achievable `‖xpos_tip − xpos_cube_center‖ ≈ 26mm` (knuckle pressed into face),
realistic grasps measure 45–60mm. The 20mm threshold was **geometrically unsatisfiable**.
`contact_count ≡ 0` always.

v11 fix: `exp(−max(d − 0.045, 0) / 0.03)` measures **distance beyond the expected contact
point** (45mm from body origin to cube center at true contact). At real contact: `d ≈ 0.045`
→ `exp(0) = 1.0`. At 60mm: `exp(−15/30) = 0.61`. The proximity signal is now calibrated
to real physics.

#### Why Φ_contact uses data.contact

```python
def _count_fingertip_contacts(self):
    data = self.unwrapped.data
    cube = self._cube_body_id
    tips = self._fingertip_body_ids
    touching = set()
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = int(self._geom_bodyid[con.geom1])
        b2 = int(self._geom_bodyid[con.geom2])
        if b1 == cube and b2 in tips:
            touching.add(b2)
        elif b2 == cube and b1 in tips:
            touching.add(b1)
    return len(touching)
```

This iterates MuJoCo's actual collision detection array. The fingertip skin geoms have
`contype="0" conaffinity="0"` (never collide); actual contacts happen on the `*_IP_IP`
collision meshes. Since we map `geom → parent body`, all collision geoms under a fingertip
body are automatically included. **Exact, free (~30 contacts max per step), ungameable.**

---

### 3.3 Component 2: Finger-Attributed Progress

**The core innovation of v11.** Progress is measured in the **palm frame**, making it
invariant to wrist rotation by construction. No action-based gate needed. No aliasing.

#### 3.3.1 Counterfactual Decomposition

Define:
- `n_world` = red face normal in world frame (from env info)
- `R_palm` = palm body rotation matrix (from `data.xmat`)
- `θ = arccos(n_world · ẑ)` = angle between red face and world up

The **total progress** is simply `Δθ_total = θ_{t-1} − θ_t` (positive = toward goal).

The **finger-attributed progress** asks: "what would θ be if the palm had NOT moved?"

```python
def finger_attributed_progress(theta_prev, theta_now, n_world_now,
                               R_palm_prev, R_palm_now):
    n_rel_now = R_palm_now.T @ n_world_now          # red normal in current palm frame
    n_hold = R_palm_prev @ n_rel_now                 # replay under frozen palm
    theta_hold = arccos(clip(n_hold · ẑ, -1, 1))    # angle if palm hadn't moved
    return theta_prev - theta_now, theta_prev - theta_hold
```

**Mathematical proof of wrist invariance:**

Let the cube's orientation in the palm frame be `R_rel`. The world-frame red normal is
`n_world = R_palm · R_rel · n_red_body`.

- **Pure wrist rotation** (R_rel unchanged): `R_palm` changes but `R_rel` stays the same.
  `n_rel_now = R_palm_now^T · R_palm_now · R_rel · n_body = R_rel · n_body = n_rel_prev`.
  Therefore `n_hold = R_palm_prev · n_rel_prev = n_world_prev` → `θ_hold = θ_prev` →
  **finger progress = 0**. ∎

- **Pure finger manipulation** (R_palm unchanged): `R_palm_now = R_palm_prev`, so
  `n_hold = R_palm_prev · R_palm_prev^T · n_world_now = n_world_now` → `θ_hold = θ_now` →
  **finger progress = total progress**. ∎

- **Mixed motion**: the two parts telescope exactly:
  `d_finger + d_wrist = (θ_prev − θ_hold) + (θ_hold − θ_now) = θ_prev − θ_now = d_total`. ∎

#### 3.3.2 Credit/Debit Asymmetry (Anti-Ratchet)

```python
if d_total > 0:  # progress toward goal
    share = clip(d_finger / max(d_total, ε), 0, 1)
    p_step = min(d_total, 0.1) × share × gate_contact × gate_tilt
else:            # regress away from goal
    p_step = max(d_total, -0.1)  # UNGATED, FULL RATE
```

**Why asymmetric:** If positive progress is gated (reduced by share, contact, tilt) but
negative progress is full-rate, then over any closed cycle where θ returns to its starting
value: `Σ gated_credit ≤ Σ full_debit → net ≤ 0`.

**Anti-ratchet proof:**

For a cycle of N steps where `Σ d_total_i = 0`:
```
Σ reward_i = Σ_{d>0} (d_i × share_i × g_c × g_t)  +  Σ_{d<0} d_i
           ≤ Σ_{d>0} d_i  +  Σ_{d<0} d_i      (since share×g_c×g_t ≤ 1)
           = Σ d_i = 0
```

No oscillation strategy profits. Rock-the-cube → net negative. ∎

**Verified by test:** `test_v11_math.py` runs 1000 adversarial random closed cycles
with random gate values. All ≤ 0.

#### 3.3.3 Contact Gate

```python
gate_contact = min(1.0, N_contacts / 2)   # 0 fingers → 0; 1 → 0.5; 2+ → 1.0
```

Uses **real MuJoCo contacts** (not distance proxy). Gravity-only rotation (no finger
contact) earns exactly 0 progress credit.

#### 3.3.4 Tilt Gate

```python
gate_tilt = exp(−(tilt / 0.4)²)
```

Where `tilt` = current wrist angle deviation from reset position.

- At 0° tilt (palm level): gate = 1.0
- At 23° tilt (0.4 rad): gate = 0.37
- At 50° tilt (0.87 rad, gravity roll threshold): gate = 0.009

Gravity roll requires tilting the palm beyond the friction cone (~50° for friction=1.2).
The tilt gate reduces progress credit to <1% at that angle. **Gravity exploitation is
non-viable.**

#### 3.3.5 EMA Smoothing (Post-Gating)

```python
progress_ema = α · p_step + (1−α) · progress_ema_prev    # α = 0.3
r_progress = 50.0 × progress_ema
```

**Critical v10 fix:** EMA is applied AFTER gating. In v10, the EMA was applied to raw
Δθ, then gates were applied to the EMA output. A wrist pulse (1 step active, 4 steps
quiet) delivered 30% of credit at the low gate and 70% at gate ≈ 1.0 → effective
gate = 0.77 instead of 0.2. v11 gates the raw delta FIRST, then smooths the gated
result.

#### 3.3.6 Drop Transient Bound

With clip = 0.1 rad and EMA α = 0.3:

```
Drop transient: θ jumps by π rad → clamped to −0.1
First-step EMA: 0.3 × (−0.1) = −0.03
r_progress = 50 × (−0.03) = −1.5/step
```

Compare to drop penalty of −50. **Net = −51.5. The v9 drop-reward bug (+147 net) is dead.**

---

### 3.4 Component 3: Goal Hold Stream (Non-Terminal Success)

**The breakthrough insight from the v10 audit:** if success terminates the episode,
the agent forfeits all remaining standing income. If standing income exceeds the success
bonus, the optimal policy is to **never succeed**. v10 tried to fix this with higher
success bonuses; the audit showed the economics were still wrong.

v11 solution: **success does not terminate.** The episode continues, and the agent earns
a per-step goal stream — the single largest income in the MDP.

```python
r_goal = 0  # default
if in_zone AND secured AND not_dropped:
    r_goal = 6.0 × stability × gate_tilt
```

Where:
- `in_zone = θ < 15°` (red face nearly up)
- `secured = N_contacts ≥ 1` (at least one real fingertip contact — rules out gravity rest)
- `stability = exp(−vel_metric)` where `vel_metric = √((v/0.05)² + (ω/1.5)²)`
- `gate_tilt = exp(−(tilt/0.4)²)` — palm must be level

**Maximum r_goal:** 6.0/step (cube still, palm level, finger contact).

**Why 6.0?** The maximum standing income from potential shaping at any fixed state is
`−(1−γ)·Φ_max = −0.01 × 5.0 = −0.05/step` (NEGATIVE). The goal stream at 6.0/step
is **strictly dominant** — no loitering strategy outside the goal zone can compete.

**Success economics proof:**

```
Loitering near goal (θ ≈ 20°, cube held, not in zone):
  r_potential ≈ −0.01 × Φ ≈ −0.04/step    (slightly negative)
  r_progress ≈ 0                            (not moving)
  r_goal = 0                                (not in zone)
  TOTAL ≈ −0.04/step

Holding in goal zone (θ < 15°, cube held, in zone):
  r_potential ≈ −0.01 × Φ ≈ −0.05/step    (slightly negative)
  r_goal = 6.0 × S × G_tilt ≈ 4–6/step    (dominant!)
  TOTAL ≈ +4 to +6/step
```

**Completing strictly dominates loitering by ≈6/step.** ∎

### One-Time Success Bonus

```python
if hold_counter ≥ 10 AND not previously_achieved:
    r_bonus = 100.0    # fires exactly once, sticky
    achieved = True
```

The bonus is a one-time discovery incentive. It fires when the cube has been held in the
success zone for 10 consecutive steps (0.2s at 50Hz) with:
- θ < 15° (in zone)
- N_contacts ≥ 1 (finger secured)
- |tilt| ≤ 0.35 rad (palm approximately level)
- vel_metric ≤ 1.0 (cube approximately still)

**Not farmable:** `achieved` is sticky. Fires once per episode, period.

**Not an exploration barrier:** The goal stream starts paying from the **first step**
in the zone (not after 10 steps). The 10-step hold only gates the one-time bonus and
the `is_success` flag for curriculum promotion.

---

### 3.5 Component 4: Drop Penalty

```python
if dropped:
    r_drop = 50.0
    if achieved:  # post-success drop
        r_drop += 50.0    # LIVE in v11 (episodes continue past success)
    terminated = True      # ONLY termination condition in v11
```

**Why post-success penalty works in v11 but was dead code in v10:**
In v10, success immediately terminated the episode → no step exists after success → the
post-success branch never executes. In v11, success is non-terminal → the agent plays
out the rest of the episode → a post-success drop IS reachable → the extra −50 fires.

**Total drop cost:**
- Immediate: −50 (or −100 post-success)
- Forgone stream: if the agent was earning 4–6/step in the goal zone, dropping forfeits
  all remaining goal stream income. Over 200 remaining steps: 200 × 5 = 1000 forfeited.
- **Effective drop penalty ≈ −1050 to −1100**, massively deterrent.

---

### 3.6 Component 5: Smoothness Regularisers

```python
r_wrist_rate = 0.01 × (Δa_wrist)²     # wrist smoothness
r_finger_rate = 0.001 × Σ(Δa_finger)²  # finger smoothness
```

**These are cosmetic only.** The actual deterrence against wrist exploitation comes from
the palm-frame attribution (wrist credit = 0 by construction) and the tilt gate. The
rate penalties just produce smoother joint trajectories.

**Why not stronger?** Full-range wrist swing over 25 steps: `Σ 0.01 × (0.08)² = 0.0016`.
This is negligible and intentionally so. Quadratic rate penalties on [-1, 1] actions
are inherently microscopic; relying on them for deterrence was v10's mistake.

---

### 3.7 Deleted Components (vs v10)

| Removed | Why |
|---------|-----|
| `r_alive` (0.1/step) | Pure loiter fuel; adds to standing income that must be dominated by goal stream |
| `finger_active_bonus` (0.3/step) | L2 norm of 16-dim action saturated by PPO noise (σ≥0.125 → ‖a‖/0.5 > 1 always) → unconditional +0.3 for doing nothing |
| `action_magnitude` (0.0001·‖a‖²) | 0.03% of per-step reward; penalises holding grasps (static a ≠ 0); zero behavioral effect |
| `drop_risk_ramp` (−2·(1−h)) | Constants contradicted env (0.08 vs 0.05); height drift is covered by Φ_pos potential |
| Wrist action gate | |a₀|/(|a₀|+‖a_{1:}‖) defeated by 1-vs-16 dimensional asymmetry; noise floor kept gate ≥ 0.64 |
| Counter-scaled hold ramp | Farmable annuity: 450 per 9-step cycle > 290 terminal payout |
| Terminal success | Forfeiting standing income made succeeding a net loss |

---

### 3.8 Complete Reward Formula (Reference)

```
R(t) = [γ·Φ(s_t) − Φ(s_{t-1})]                                          potential shaping
     + 50 · ema(gated_progress)                                           attributed progress
     + 6.0 · stability · gate_tilt · 𝟙(in_zone ∧ secured ∧ ¬dropped)    goal hold stream
     + 100 · 𝟙(hold≥10 ∧ ¬achieved)                                      one-time bonus
     − 50 · 𝟙(dropped) − 50 · 𝟙(dropped ∧ achieved)                      drop penalty
     − 0.01 · (Δa_wrist)² − 0.001 · ‖Δa_fingers‖²                       smoothness
```

Where:
```
Φ(s) = 3·exp(−θ/1.0) + 1·exp(−d_pos/0.05) + 0.5·min(N_c/2, 1)
       + 0.5·mean(exp(−max(d_i−0.045, 0)/0.03))

gated_progress = { min(d_total, 0.1) · share · g_contact · g_tilt    if d_total > 0
                 { max(d_total, −0.1)                                  if d_total ≤ 0

share = clip(d_finger / d_total, 0, 1)
g_contact = min(1, N_contacts/2)
g_tilt = exp(−(tilt/0.4)²)
stability = exp(−√((v/0.05)² + (ω/1.5)²))
vel_metric = √((v_lin/0.05)² + (v_ang/1.5)²)
```

### 3.9 Symbol Table

| Symbol | Unit | Source | Description |
|--------|------|--------|-------------|
| θ | rad | `info["red_face_up_angle_rad"]` | Angle between red face normal and world up |
| d_pos | m | `‖cube_pos − default_pos‖` | Displacement from reset position |
| N_c | count | `data.contact` geom iteration | Distinct fingertip bodies touching cube |
| d_i | m | `‖xpos[tip_body] − xpos[cube]‖` | Body-origin-to-center distance (NOT pad distance) |
| tilt | rad | `qpos[wrist] − qpos0[wrist]` | Wrist deviation from reset |
| v_lin | m/s | `‖cube_qvel[:3]‖` | Cube linear velocity |
| v_ang | rad/s | `‖cube_qvel[3:6]‖` | Cube angular velocity |
| share | [0,1] | `d_finger / d_total` | Fraction of progress attributed to fingers |
| γ | — | 0.99 (must match PPO) | Discount factor |

---

## 4. v9 → v10 → v11: What Changed and Why

### v9: What It Learned (Wrong)

v9 trained for 263M steps across chapters 1–4. The agent learned:
1. Rotate the wrist (action[0]) to reduce θ by ~65°
2. Tilt the hand to let gravity roll the cube
3. Topple the cube forward, then grab with the thumb
4. Fingers are passive — they don't actively manipulate

**Why:** Under v9's reward, wrist rotation earned the same `r_progress` as finger
manipulation but cost 16× less in action penalty. The agent correctly optimised the
reward — just not the behavior we wanted.

### v10: What the Audit Found

An external RL engineer reviewed v10 and assigned a **10% confidence score** that it
would produce finger-first manipulation. Five critical failures:

| # | Failure | Impact |
|---|---------|--------|
| F1 | `G_contact ≡ 0` always (20mm threshold vs 26mm minimum geometry) | Entire progress channel dead; centerpiece anti-exploit mechanism inert |
| F2 | Success hold ramp farmable (450 cycling > 290 completing) | Boundary dithering at θ ≈ 15° |
| F3 | Wrist gate bypassable (1-vs-16 L2 norm asymmetry, noise floor = 1.2) | PPO exploration noise keeps gate ≥ 0.64 |
| F4 | Standing income (6/step) > task income (440 one-shot, terminates) | Loiter equilibrium: succeeding is a net loss |
| F5 | EMA lag × instantaneous gate = wrist pulse aliasing (77% credit kept) | 1-step wrist pulse, 4-step quiet → most credit passes ungated |

### v11: How Each Failure Was Fixed

| v10 Failure | v11 Fix | Mechanism |
|-------------|---------|-----------|
| F1: distance proxy impossible | `data.contact` geom-pair iteration | Exact MuJoCo collision detection |
| F2: farmable ramp | Flat goal stream (+6/step while in zone) | Cycling OUT forfeits income |
| F3: action-based gate bypassable | Palm-frame counterfactual attribution | Wrist credit = 0 by linear algebra, not by a gate |
| F4: standing income > task income | Potential-based shaping (zero standing income) | Ng et al. 1999: γΦ(s')−Φ(s) → no loiter |
| F5: EMA pulse aliasing | Gate BEFORE smooth (not after) | Gated delta enters EMA; ungated delta never reaches it |

---

## 5. Curriculum Design

8 chapters of progressively harder cube starting angles. Agent must achieve **80%
success rate** over a 200-episode rolling window before promoting.

| Ch | Name | Angle Range | Overlap | Episode Steps | Key Skill |
|----|------|-------------|---------|---------------|-----------|
| 1 | `ch1_small_tilt` | 16°–50° | — | 210 | Basic contact, nudging |
| 2a | `ch2a_moderate_tilt` | 42°–62° | 8° with Ch1 | 310 | Controlled pushing |
| 2b | `ch2b_medium_tilt` | 54°–78° | 8° with Ch2a | 360 | Initiating roll |
| 2c | `ch2c_steep_tilt` | 70°–93° | 8° with Ch2b | 410 | Full rolling skill |
| 3a | `ch3a_side_roll` | 85°–115° | 8° with Ch2c | 460 | Multi-finger coordination |
| 3b | `ch3b_deep_roll` | 107°–135° | 8° with Ch3a | 510 | Pushing past equator |
| 4 | `ch4_near_flip` | 127°–163° | 8° with Ch3b | 610 | Re-catching at apex |
| 5 | `ch5_full_flip` | 155°–180° | 8° with Ch4 | 760 | Full 180° reorientation |

### v11 Changes to Curriculum

1. **Episode lengths +10:** Success requires a 10-step stable hold. Chapters need
   enough steps for the hold to complete even when the goal is reached late.
2. **Success bonus flattened to 100:** Under v11, the per-step goal stream is the real
   payout (6/step × remaining steps >> 100). The bonus is just a discovery incentive.
3. **is_success semantics changed:** v11's `is_success` = "achieved stable hold AND
   cube never dropped this episode" (sticky but revocable on drop). This is stricter
   than v9's instant angular check — expect longer chapter durations.

---

## 6. File Reference

| File | Purpose |
|------|---------|
| `production_reward.py` | **v11 reward function.** Potential shaping + palm-frame attribution + goal stream. 470 lines. |
| `curriculum.py` | 8-chapter `CHAPTERS` list + `CurriculumManager` class (tracks SR, handles promotion, serialises state). |
| `curriculum_wrapper.py` | Gym wrapper: samples spawn angles, syncs `success_bonus`, manages episode length, records success. |
| `train_curriculum.py` | **Main training entry point.** PPO setup, SubprocVecEnv, callbacks, W&B logging, crash recovery. |
| `test_v11_math.py` | Unit tests: wrist-carry zero credit, finger-only full credit, anti-ratchet invariant (1000 random cycles), drop transient bound. |
| `render_policy.py` | Load a trained `.zip` model and render in MuJoCo viewer. |
| `REWARD_REPORT.md` | Concise v9→v10 mathematical comparison (historical reference). |
| `checkpoints/` | v9 trained model snapshots (Ch1–Ch4 promotions + final). |

---

## 7. Environment Setup

### Prerequisites

| Requirement | Notes |
|------------|-------|
| Linux (Ubuntu 22.04+) or macOS | Windows not supported (MuJoCo subprocess issues) |
| Python 3.11 | Managed by pixi |
| NVIDIA GPU + CUDA 12.x | For `--device cuda` |
| [pixi](https://pixi.sh) | Package manager |
| W&B account (fourvectors team) | Metric logging |

### Quick Start
```bash
git clone https://github.com/adityagarg7/fv-orca-hand-rl.git
cd fv-orca-hand-rl
git checkout main
pixi install
pixi run setup-orca        # install orca_sim (MuJoCo env)
pixi run login             # authenticate W&B
python test_v11_math.py    # verify reward math (no MuJoCo needed)
```

---

## 8. Running Training

### v11 Training (fresh start — do NOT resume from v9/v10 models)
```bash
python train_curriculum.py \
    --n-envs 64 \
    --device cuda \
    --subproc \
    --timesteps 150_000_000 \
    --save-dir ./checkpoints_v11 \
    --run-name v11-finger-first
```

### CPU-only
```bash
python train_curriculum.py \
    --n-envs 48 \
    --device cpu \
    --subproc \
    --save-dir ./checkpoints_v11
```

### Resume after crash
```bash
python train_curriculum.py \
    --n-envs 64 --device cuda --subproc \
    --save-dir ./checkpoints_v11    # auto-detects curriculum_state.json + latest_model.zip
```

### All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | `150_000_000` | Maximum total steps |
| `--n-envs` | `64` | Parallel environments (3–4× CPU cores) |
| `--device` | `cuda` | Torch device |
| `--subproc` / `--no-subproc` | `True` | SubprocVecEnv vs DummyVecEnv |
| `--net-arch` | `512 256` | Policy/value network hidden layers |
| `--n-steps` | `4096` | Steps per env per rollout |
| `--gamma` | `0.99` | Discount factor (**must match reward wrapper**) |
| `--save-dir` | `./checkpoints` | Checkpoint directory |
| `--run-name` | auto | W&B run name |

---

## 9. Monitoring & Debugging

### Key W&B Metrics

| Metric | Healthy Range | What It Means |
|--------|--------------|---------------|
| `rollout/success_rate` | Climbing → 0.80 | Rolling success rate |
| `curriculum/chapter_idx` | 0 → 7 over training | Current chapter |
| `reward/gate_contact` | 0.3–1.0 during manipulation | **If always 0: contact detection is broken** |
| `reward/progress` | Non-zero during active rotation | **If always 0: progress channel is dead** |
| `reward/goal_stream` | Positive when in success zone | Goal hold stream is paying |
| `reward/progress_d_finger` | ≈ `d_total` for finger manipulation | Palm-frame attribution working |
| `adaptive/action_std` | [0.75, 1.05] | Policy exploration level |
| `train/clip_fraction` | [0.02, 0.08] | PPO update efficiency |

### v11 Validation Checklist (Run at Step 1000)

1. ☐ Histogram `contact_count` — should be 0–5, NOT always 0
2. ☐ Check `r_progress` is non-zero when fingers are active
3. ☐ Verify `d_finger ≈ d_total` when wrist is stationary
4. ☐ Verify `d_finger ≈ 0` when only wrist moves
5. ☐ Check `goal_stream > 0` when cube is in success zone
6. ☐ Verify `gate_tilt < 0.01` when palm is tilted >50°
7. ☐ Confirm episode length > 200 (base env truncation disabled)
8. ☐ Check `potential` is non-zero and varies with state
9. ☐ Verify drop penalty fires correctly (r_drop < 0)
10. ☐ Run `python test_v11_math.py` — all tests pass

### Red Flags

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `contact_count` always 0 | Fingertip body names don't match model | Check FINGERTIP_BODIES against XML |
| `r_progress` always 0 | `gate_contact` always 0 | Same as above — verify contacts |
| `goal_stream` always 0 | Agent never reaches θ < 15° | Normal in early chapters — check curriculum |
| `episode_len` capped at 200 | Base env `max_episode_steps` not overridden | Ensure `max_episode_steps=1_000_000` in `make_env` |
| Success rate oscillates near 80% | Stricter v11 `is_success` (10-step hold) | Normal — expect longer chapter durations |

---

## 10. Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8-core (32 envs) | 12-core Ryzen 9 (64 envs) |
| RAM | 32 GB | 64 GB |
| GPU | RTX 3080 | RTX 4090/5090 |
| Storage | 100 GB SSD | 500 GB NVMe |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

GPU is used only for PPO neural network forward/backward passes (~10–20% utilisation).
More CPU cores → more parallel environments → faster training.

---

## Appendix A: Actuator/Joint Map

```
Index  Joint              Role            Control Range (rad)
──────────────────────────────────────────────────────────────
[0]    right_wrist        WRIST           [-1.134, 0.611]  (65° total range)
[1]    right_p-abd        Pinky abd       [-0.52,  0.52]
[2]    right_p-mcp        Pinky mcp       [-0.44,  1.75]
[3]    right_p-pip        Pinky pip       [-0.26,  1.87]
[4]    right_r-abd        Ring abd        [-0.47,  0.47]
[5]    right_r-mcp        Ring mcp        [-0.44,  1.75]
[6]    right_r-pip        Ring pip        [-0.26,  1.87]
[7]    right_m-abd        Middle abd      [-0.47,  0.47]
[8]    right_m-mcp        Middle mcp      [-0.44,  1.75]
[9]    right_m-pip        Middle pip      [-0.26,  1.87]
[10]   right_i-abd        Index abd       [-0.44,  0.52]
[11]   right_i-mcp        Index mcp       [-0.44,  1.75]
[12]   right_i-pip        Index pip       [-0.26,  1.87]
[13]   right_t-cmc        Thumb cmc       [-0.79,  0.58]
[14]   right_t-abd        Thumb abd       [-0.31,  0.96]
[15]   right_t-mcp        Thumb mcp       [-0.44,  1.75]
[16]   right_t-pip        Thumb pip       [-0.26,  1.87]
```

## Appendix B: Verified Physics Constants

| Constant | Value | Source |
|----------|-------|--------|
| Cube half-extent | 0.018 m (36mm total) | `scene_right_cube_orientation.xml` size attr |
| Fingertip pad offset | 40–44mm from body origin | `M-FingerTipAssembly_M-DP-Skin.stl` bbox |
| Tip-to-center at contact | ~45–60mm | Measured from realistic grasps |
| Drop termination height | z < 0.05 m | `task_envs.py:36` (`_cube_dropped`) |
| Cube friction | 1.2 | Scene XML condim/friction |
| Gravity roll threshold | ~50° palm tilt | `arctan(1/μ) = arctan(1/1.2) ≈ 40°` + margin |
| Sim timestep | 50 Hz (0.02s) | MuJoCo model |
| Skin geom collision | `contype="0" conaffinity="0"` | Never collide |
| Contact geoms | `*_IP_IP` meshes | Actual collision bodies |
