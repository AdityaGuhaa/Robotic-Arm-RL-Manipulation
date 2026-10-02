#!/usr/bin/env python3
"""Visual evaluation of the v2 Panda Reach agent with VecNormalize."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import envs  # noqa: F401
from envs.reach_env import PandaReachEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

# Load model
model = PPO.load("models/best/best_model.zip", device="cpu")

# Create env wrapped with saved VecNormalize stats
env = DummyVecEnv([lambda: PandaReachEnv(render_mode="human", max_episode_steps=100)])
env = VecNormalize.load("models/vec_normalize.pkl", env)
env.training = False    # don't update stats during eval
env.norm_reward = False  # don't normalise rewards

print("\n  MuJoCo viewer launched — watch the improved Panda arm!\n")

for ep in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        total_reward += reward[0]
        steps += 1
        time.sleep(0.05)  # slow for viewing

    info = infos[0]
    status = "✓" if info.get("is_success", False) else "✗"
    dist = info.get("distance", -1)
    orient = info.get("orientation_error", -1)
    print(
        f"  Episode {ep+1:>2}/10  |  "
        f"Reward: {total_reward:>7.2f}  |  "
        f"Steps: {steps:>3}  |  "
        f"Dist: {dist:.4f}  |  "
        f"Orient: {orient:.4f}  |  {status}"
    )
    time.sleep(1.0)

try:
    env.close()
except Exception:
    pass

print("\n  Done!\n")
