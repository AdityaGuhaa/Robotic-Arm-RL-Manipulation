#!/usr/bin/env python3
"""Record a demo GIF of the trained v2 Panda Reach agent."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MUJOCO_GL"] = "egl"

import envs  # noqa: F401
from envs.reach_env import PandaReachEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import imageio

model = PPO.load("models/best/best_model.zip", device="cpu")

# Use raw env for rendering (no VecNormalize wrapper for the render env)
render_env = PandaReachEnv(render_mode="rgb_array", max_episode_steps=100)

# Separate normalized env for predictions
vec_env = DummyVecEnv([lambda: PandaReachEnv(max_episode_steps=100)])
vec_env = VecNormalize.load("models/vec_normalize.pkl", vec_env)
vec_env.training = False
vec_env.norm_reward = False

frames = []
n_episodes = 5

for ep in range(n_episodes):
    # Reset both envs with same state
    obs_raw, info = render_env.reset()
    vec_obs = vec_env.reset()

    # Sync the vec_env state with render_env by copying qpos/qvel/ctrl/mocap
    raw_env = vec_env.venv.envs[0]
    raw_env.data.qpos[:] = render_env.data.qpos[:]
    raw_env.data.qvel[:] = render_env.data.qvel[:]
    raw_env.data.ctrl[:] = render_env.data.ctrl[:]
    raw_env.data.mocap_pos[:] = render_env.data.mocap_pos[:]
    import mujoco
    mujoco.mj_forward(raw_env.model, raw_env.data)
    vec_obs = vec_env.normalize_obs(render_env._get_obs())
    vec_obs = np.expand_dims(vec_obs, 0)

    done = False
    steps = 0

    # Capture initial frame
    frame = render_env.render()
    if frame is not None:
        frames.append(frame)

    while not done and steps < 100:
        action, _ = model.predict(vec_obs, deterministic=True)
        
        # Step both envs
        obs_raw, reward, terminated, truncated, info = render_env.step(action[0])
        done = terminated or truncated
        steps += 1

        # Normalize obs for next prediction
        vec_obs = vec_env.normalize_obs(obs_raw)
        vec_obs = np.expand_dims(vec_obs, 0)

        frame = render_env.render()
        if frame is not None:
            frames.append(frame)

    # Add a few pause frames between episodes
    if frame is not None:
        for _ in range(8):
            frames.append(frame)

    status = "✓" if info.get("is_success", False) else "✗"
    print(f"  Episode {ep+1}/{n_episodes}  |  Steps: {steps:>3}  |  {status}")

render_env.close()
vec_env.close()

# Save as GIF
os.makedirs("assets", exist_ok=True)
gif_path = "assets/panda_reach_demo.gif"
imageio.mimsave(gif_path, frames, fps=20, loop=0)
print(f"\n  Saved GIF: {gif_path}  ({len(frames)} frames)")

# Also save a high-res hero image (first frame of episode 1)
hero_path = "assets/panda_reach_hero.png"
imageio.imwrite(hero_path, frames[0])
print(f"  Saved hero image: {hero_path}")
