# Reward Function Report: v9 → v10

## v9 Result (263M steps, all 8 chapters)

**Learned strategy:** Wrist rotation → gravity roll → thumb grab.
**Why:** Under v9 reward, wrist and finger strategies earned equal `r_progress` but wrist cost 16× less in action penalty. The agent correctly optimised the reward — just not the behavior we wanted.

---

## v9 Reward (what was)

```
R_v9 = 5·exp(-θ/0.3)                           r_align     ∈ [0, 5] (dead above 60°)
     + 50·(θ_{t-1} − θ_t)                      r_progress  (unbounded, noisy, strategy-agnostic)
     + 0.5·exp(-d_pos/0.05)                     r_pos       ∈ [0, 0.5]
     + 0.3·mean(exp(-d_i/0.02))                 r_fingers   ∈ [0, 0.3]
     + 500·[0.3 + 0.7·exp(-(|v|+|ω|)/0.5)]     r_success   (instant, mixed units)
     + 0.05                                     r_alive
     − 10·𝟙(drop)                               r_drop      (too weak vs drop-reward bug)
     − 0.002·‖Δa‖²                              r_action    (symmetric — wrist = fingers)
     − 0.0001·‖a‖²
```

### Why wrist won (the math)

For Δθ = 1 rad of rotation:

| | Wrist (1 joint, 50 steps) | Fingers (16 joints, 200 steps) |
|---|---|---|
| `r_progress` | 50 × 1.0 = **50** | 50 × 1.0 = **50** |
| `r_action_rate` | 0.002 × 1 × (Δa)² × 50 ≈ **0.0001** | 0.002 × 16 × (Δa)² × 200 ≈ **0.02** |
| **Net** | **50.0** | **49.98** |

Same reward, but wrist is faster → higher reward/time → PPO prefers it.

### Three critical bugs

1. **Drop-reward:** cube drops, θ jumps ±π → `r_progress = 50·π ≈ 157` vs `r_drop = −10`. **Net +147 for dropping.**
2. **Noise:** at 50Hz, noise σ ≈ 0.01 rad → `r_progress_noise = 50·0.014 = 0.71/step` vs signal `0.10/step`. **SNR = 0.14.**
3. **Dead align:** `5·exp(-π/2 / 0.3) = 0.028` at 90°. **6 of 8 chapters get zero alignment gradient.**

---

## v10 Reward (what changed)

```
R_v10 = 3·exp(-θ/1.0)                                                    r_align
      + 50·clip(Δθ_ema, ±0.1)·G_wrist·G_contact                          r_progress
      + 2·exp(-d_pos/0.03)                                               r_pos
      + 2·mean(exp(-d_i/0.02)) + 0.5·N_contact + 0.3·min(1, |a_f|/0.5)  r_fingers
      + 500·S·P·𝟙(hold ≥ 10)                                             r_success
      + 0.1                                                              r_alive
      − 2·(1−h_margin) − 50·𝟙(drop) − 100·𝟙(drop ∧ had_success)        r_drop
      − 0.05·(Δa_wrist)² − 0.001·‖Δa_fingers‖² − 0.0001·‖a‖²           r_action
```

Where:

| Symbol | Definition | Purpose |
|--------|-----------|---------|
| `Δθ_ema` | `α·θ_t + (1−α)·ema_{t-1}`, α=0.3 | 5-step noise smoothing |
| `G_wrist` | `1 − 0.8·\|a₀\|/(\|a₀\| + ‖a_{1:}‖ + ε)` | Wrist action → 80% penalty |
| `G_contact` | `min(1, N_contact/2)` | Need ≥2 fingers on cube |
| `N_contact` | `Σ 𝟙(d_i < 20mm)` | Finger count touching cube |
| `S` (stability) | `0.3 + 0.7·exp(−√((v/0.05)²+(ω/1.5)²))` | Normalised velocity |
| `P` (palm) | `exp(−d_pos/0.04)` | Cube must be on palm |
| `h_margin` | `clip((z − 0.08)/0.04, 0, 1)` | Drop risk proximity |

---

## Change-by-Change: Cause → Effect

| # | Change | Cause (v9 problem) | Effect (v10 fix) |
|---|--------|-------------------|------------------|
| 1 | `σ_align`: 0.3 → **1.0** | 0.028 at 90° (dead zone) | 0.67 at 90° (22× improvement) |
| 2 | `progress` += EMA α=0.3 | SNR = 0.14 (noise > signal) | SNR ≈ 1.0 (signal ≈ noise) |
| 3 | `progress` += clip ±0.1 | Drop gives +157 reward | Drop gives +5 max |
| 4 | `progress` × `G_wrist` | Wrist = fingers reward | Wrist gets 20% of reward |
| 5 | `progress` × `G_contact` | Gravity gets full reward | Gravity gets 0% (no contact) |
| 6 | `pos`: 0.5 → **2.0**, σ: 0.05→0.03 | Cube displacement ignored | 4× stronger centering force |
| 7 | `fingers`: 0.3 → **2.0** + contact | 167× weaker than progress | Comparable to progress |
| 8 | `+0.3·finger_active` | No reward for finger use | Active fingers earn +0.3/step |
| 9 | `wrist_rate`: 0.002 → **0.05** | Same cost as 1 finger | 25× more expensive |
| 10 | `success` hold 10 steps | Flick-through earns 500 | Must hold 0.2s for bonus |
| 11 | `vel_metric` normalised | \|v\|+\|ω\| mixes m/s, rad/s | Proper √((v/v₀)²+(ω/ω₀)²) |
| 12 | `drop`: 10 → **50** + risk | Weaker than drop-reward bug | 5× stronger + gradual |

---

## Expected Episode Budget (Ch2b, 66° → success)

| Component | v9 Wrist | v9 Finger | v10 Wrist | v10 Finger |
|-----------|---------|-----------|-----------|------------|
| r_progress | 51.5 | 51.5 | **5.2** | **51.5** |
| r_align | 356 | 233 | 200 | 160 |
| r_pos | 41 | 102 | 60 | **190** |
| r_fingers | 6 | 31 | 30 | **120** |
| r_success | 221 | 342 | 100 | **300** |
| r_action | −0.0 | −0.02 | **−25** | −1 |
| **TOTAL** | **683** | **773** | **~370** | **~870** |

**v10 finger advantage: 2.35×** (was 1.13× in v9).
