"""Unit tests for v11 attribution math + anti-ratchet invariant (no MuJoCo)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from production_reward import finger_attributed_progress, WORLD_UP


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def angle_up(n):
    return float(np.arccos(np.clip(n @ WORLD_UP, -1, 1)))


def make_state(theta):
    """Red normal at angle theta from world up (tilted about x)."""
    return rot_x(theta) @ WORLD_UP


def credit_debit(d_total, d_finger, gate_contact, gate_tilt, clip=0.1):
    """Mirror of the wrapper's progress rule."""
    if d_total > 0:
        share = float(np.clip(d_finger / max(d_total, 1e-8), 0, 1))
        return min(d_total, clip) * share * gate_contact * gate_tilt
    return max(d_total, -clip)


# --- Test 1: pure wrist carry -> zero finger credit -----------------------
theta0 = np.deg2rad(60)
n0 = make_state(theta0)
R_palm0 = np.eye(3)
d_wrist_rot = rot_x(-np.deg2rad(10))          # rotates cube toward up
n1 = d_wrist_rot @ n0
R_palm1 = d_wrist_rot @ R_palm0               # palm carried identically
d_tot, d_fing = finger_attributed_progress(theta0, angle_up(n1), n1, R_palm0, R_palm1)
assert abs(d_tot - np.deg2rad(10)) < 1e-9, d_tot
assert abs(d_fing) < 1e-9, f"wrist carry leaked finger credit: {d_fing}"
print("T1 wrist-carry: d_total=%.4f d_finger=%.2e  OK" % (d_tot, d_fing))

# --- Test 2: pure finger manipulation -> full credit ----------------------
n1 = rot_x(-np.deg2rad(10)) @ n0              # cube rotates, palm fixed
d_tot, d_fing = finger_attributed_progress(theta0, angle_up(n1), n1, R_palm0, R_palm0)
assert abs(d_fing - d_tot) < 1e-9
print("T2 finger-only: d_total=%.4f d_finger=%.4f  OK" % (d_tot, d_fing))

# --- Test 3: mixed motion, decomposition telescopes exactly ---------------
A = rot_x(-np.deg2rad(4))                     # wrist part
B = rot_x(-np.deg2rad(7))                     # finger part (palm frame)
R_palm1 = A @ R_palm0
n1 = R_palm1 @ (B @ (R_palm0.T @ n0))         # cube: carried by A, plus B in palm
theta1 = angle_up(n1)
d_tot, d_fing = finger_attributed_progress(theta0, theta1, n1, R_palm0, R_palm1)
d_wrist = d_tot - d_fing
assert abs(d_tot - (theta0 - theta1)) < 1e-9
assert abs(d_fing - np.deg2rad(7)) < 1e-6, d_fing   # finger part == B's rotation
print("T3 mixed: total=%.4f finger=%.4f wrist=%.4f (sum exact)  OK"
      % (d_tot, d_fing, d_wrist))

# --- Test 4: anti-ratchet — closed cycles never profit --------------------
# (a) wrist toward then wrist back
r = credit_debit(+0.10, 0.0, 1.0, 1.0) + credit_debit(-0.10, 0.0, 1.0, 1.0)
assert r <= 1e-12, r
# (b) finger toward WITH contact, finger back WITHOUT contact (v10-style farm)
r = credit_debit(+0.05, +0.05, 1.0, 1.0) + credit_debit(-0.05, -0.05, 0.0, 1.0)
assert r <= 1e-12, r
# (c) finger-toward+wrist-away-more, then wrist-toward+finger-away (my farm)
r = credit_debit(-0.01, +0.03, 1.0, 1.0) + credit_debit(+0.01, -0.03, 1.0, 1.0)
assert r <= 1e-12, r
# (d) 1000 random closed cycles with adversarial gates
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(1000):
    legs = rng.normal(0, 0.03, size=8)
    legs[-1] -= legs.sum()                    # closed: total d_theta == 0
    total = sum(
        credit_debit(d, rng.uniform(-0.05, 0.08) + d, rng.uniform(0, 1),
                     rng.uniform(0, 1))
        for d in legs)
    worst = max(worst, total)
assert worst <= 1e-12, worst
print("T4 anti-ratchet: cycles a-c and 1000 adversarial random cycles all <= 0  OK")

# --- Test 5: drop transient bounded ----------------------------------------
p = credit_debit(-np.pi, -np.pi, 1.0, 1.0)    # theta jumps by pi on a drop
assert p == -0.1
print("T5 drop transient: p_step capped at %.2f -> EMA first-step reward %.2f  OK"
      % (p, 50 * 0.3 * p))

print("\nALL MATH TESTS PASSED")
