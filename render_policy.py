"""
Render a trained policy slowly in the MuJoCo viewer for visual inspection.
Each step pauses 0.15s so you can clearly see the cube being reoriented.

Usage:
  # Linux:
  python render_policy.py path/to/model.zip
  # macOS (the viewer needs mjpython on Mac):
  mjpython render_policy.py path/to/model.zip

If no path is given, it defaults to ppo_orca_production_final.zip in the
current directory. To grab a model trained on Colab, pull it from W&B first
(see README, "Pulling a trained model").
"""

import os
import sys
import time

from stable_baselines3 import PPO
from orca_sim import OrcaHandRightCubeOrientation
from production_reward import ProductionRewardWrapper

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "ppo_orca_production_final.zip"

if not os.path.exists(MODEL_PATH):
    sys.exit(
        f"Model not found: {MODEL_PATH}\n"
        "Pass a path explicitly, e.g.  python render_policy.py runs/my_model.zip"
    )

print(f"Loading model: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

# Create the env WITH rendering enabled.
env = ProductionRewardWrapper(OrcaHandRightCubeOrientation(render_mode="human"))

NUM_EPISODES = 5
STEP_DELAY = 0.15  # 150 ms per step -- slow enough to watch clearly
results = []

print(f"\nRunning {NUM_EPISODES} episodes at {STEP_DELAY}s per step (slow, for inspection).\n")

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    total_reward = 0.0

    input(f">>> Press ENTER to start Episode {ep} <<<")
    print(f"  Episode {ep} running... watch the viewer!")

    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 5 == 0:
            align = info["red_face_up_alignment"]
            angle_deg = info["red_face_up_angle_rad"] * 180 / 3.14159
            print(f"    step {step:3d} | alignment={align:+.3f} | angle_from_goal={angle_deg:.1f} deg | reward={reward:.2f}")

        time.sleep(STEP_DELAY)

        if terminated or truncated:
            if info["is_success"]:
                result = "SUCCESS"
            elif info["dropped"]:
                result = "DROPPED"
            else:
                result = "TIMEOUT"
            print(f"  >>> Episode {ep}: {result} at step {step}, total_reward={total_reward:.1f} <<<")
            results.append((result, step, total_reward))
            break

    time.sleep(1.0)

print("\n" + "=" * 50)
print("  SUMMARY")
print("=" * 50)
successes = sum(1 for r, _, _ in results if r == "SUCCESS")
drops = sum(1 for r, _, _ in results if r == "DROPPED")
timeouts = sum(1 for r, _, _ in results if r == "TIMEOUT")
avg_steps = sum(s for _, s, _ in results) / len(results) if results else 0
print(f"  Success: {successes}/{NUM_EPISODES} ({100*successes/max(1,NUM_EPISODES):.0f}%)")
print(f"  Drops:   {drops}/{NUM_EPISODES}")
print(f"  Timeout: {timeouts}/{NUM_EPISODES}")
print(f"  Avg steps to solve: {avg_steps:.1f}")

env.close()
