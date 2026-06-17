"""
Run a trained policy in the MuJoCo viewer, in one of two modes:

  interactive (default) -- pauses before each episode for an ENTER press and steps
      slowly (0.15s/step) so you can watch the cube being reoriented at your pace.
  non-interactive       -- same viewer, but no ENTER prompts and no slow stepping;
      episodes run back-to-back at full speed.

Both modes open the viewer, so both need a display (and, on macOS, mjpython).

Usage:
  # Linux:
  python render_policy.py path/to/model.zip                 # interactive (default)
  python render_policy.py path/to/model.zip --non-interactive
  python render_policy.py path/to/model.zip --episodes 20 --non-interactive
  python render_policy.py --random --non-interactive          # untrained baseline
  # macOS (the viewer needs mjpython on Mac):
  mjpython render_policy.py path/to/model.zip

If no path is given, it defaults to ppo_orca_production_final.zip in the current
directory. To grab a model trained on Colab, pull it from W&B first (see README,
"Pulling a trained model").
"""

import argparse
import os
import time

from stable_baselines3 import PPO
from orca_sim import OrcaHandRightCubeOrientation


def parse_args():
    p = argparse.ArgumentParser(description="Render a trained ORCA cube-reorientation policy in the MuJoCo viewer.")
    p.add_argument("model", nargs="?", default="ppo_orca_production_final.zip",
                   help="Path to the trained model .zip (default: ppo_orca_production_final.zip). Ignored with --random.")
    p.add_argument("--random", action="store_true",
                   help="Run a random policy (uniform action sampling) instead of loading a model.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--interactive", dest="interactive", action="store_true",
                      help="Pause for ENTER before each episode and step slowly (default).")
    mode.add_argument("--non-interactive", dest="interactive", action="store_false",
                      help="No ENTER prompts and no slow stepping; episodes run at full speed.")
    p.set_defaults(interactive=True)
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    p.add_argument("--step-delay", type=float, default=None,
                   help="Seconds to pause per step (default: 0.15 interactive, 0.0 non-interactive).")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.random and not os.path.exists(args.model):
        raise SystemExit(
            f"Model not found: {args.model}\n"
            "Pass a path explicitly, e.g.  python render_policy.py runs/my_model.zip\n"
            "Or run an untrained baseline with --random."
        )

    step_delay = args.step_delay if args.step_delay is not None else (0.15 if args.interactive else 0.0)

    if args.random:
        print("Running a RANDOM policy (no model loaded).")
        model = None
    else:
        print(f"Loading model: {args.model}")
        model = PPO.load(args.model)

    # Create the env WITH rendering enabled (viewer is shown in both modes).
    env = OrcaHandRightCubeOrientation(render_mode="human")

    mode_label = "interactive" if args.interactive else "non-interactive"
    print(f"\nRunning {args.episodes} episodes in {mode_label} mode at {step_delay}s per step.\n")

    results = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0.0

        if args.interactive:
            input(f">>> Press ENTER to start Episode {ep} <<<")
        print(f"  Episode {ep} running... watch the viewer!")

        for step in range(200):
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 5 == 0:
                align = info["red_face_up_alignment"]
                angle_deg = info["red_face_up_angle_rad"] * 180 / 3.14159
                print(f"    step {step:3d} | alignment={align:+.3f} | angle_from_goal={angle_deg:.1f} deg | reward={reward:.2f}")

            if step_delay:
                time.sleep(step_delay)

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

        if args.interactive:
            time.sleep(1.0)

    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    successes = sum(1 for r, _, _ in results if r == "SUCCESS")
    drops = sum(1 for r, _, _ in results if r == "DROPPED")
    timeouts = sum(1 for r, _, _ in results if r == "TIMEOUT")
    avg_steps = sum(s for _, s, _ in results) / len(results) if results else 0
    print(f"  Success: {successes}/{args.episodes} ({100*successes/max(1,args.episodes):.0f}%)")
    print(f"  Drops:   {drops}/{args.episodes}")
    print(f"  Timeout: {timeouts}/{args.episodes}")
    print(f"  Avg steps to solve: {avg_steps:.1f}")

    env.close()


if __name__ == "__main__":
    main()
