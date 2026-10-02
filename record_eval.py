#!/usr/bin/env python3
"""Record evaluation episodes as a video for review."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import envs  # noqa: F401
from envs.reach_env import PandaReachEnv
from stable_baselines3 import PPO
import numpy as np

model = PPO.load("models/best/best_model.zip", device="cpu")
env = PandaReachEnv(render_mode="rgb_array", max_episode_steps=100)

os.makedirs("eval_frames", exist_ok=True)

frame_idx = 0
for ep in range(5):
    obs, info = env.reset()
    done, truncated = False, False
    total_reward = 0
    steps = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Save every 2nd frame to keep count manageable
        if steps % 2 == 0 or done or truncated:
            frame = env.render()
            if frame is not None:
                import imageio
                imageio.imwrite(f"eval_frames/frame_{frame_idx:04d}_ep{ep+1}_step{steps:03d}.png", frame)
                frame_idx += 1

    status = "SUCCESS" if info.get("is_success", False) else "FAIL"
    print(f"  Episode {ep+1}/5  |  Reward: {total_reward:>7.2f}  |  Steps: {steps:>3}  |  {status}")

    # Save first frame of next episode to show reset
    obs, info = env.reset()
    frame = env.render()
    if frame is not None:
        import imageio
        imageio.imwrite(f"eval_frames/frame_{frame_idx:04d}_ep{ep+1}_reset.png", frame)
        frame_idx += 1

env.close()
print(f"\n  Saved {frame_idx} frames to eval_frames/")
