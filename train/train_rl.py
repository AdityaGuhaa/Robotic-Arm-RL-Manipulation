#!/usr/bin/env python3
# Copyright (c) 2026 Aditya Guha. All rights reserved.
# This source code is proprietary and may not be used, copied, modified,
# or distributed without prior written permission. See LICENSE for details.

"""
Train a PPO agent on the Panda Reach task using Stable-Baselines3.

Usage:
    python train/train_rl.py
    python train/train_rl.py --timesteps 1000000 --seed 0
    python train/train_rl.py --timesteps 200000 --eval-freq 5000 --verbose

TensorBoard:
    tensorboard --logdir logs/
"""

import argparse
import os
import sys

# Ensure project root is on the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Trigger Gymnasium registration
import envs  # noqa: F401
from envs.reach_env import PandaReachEnv


def parse_args():
    p = argparse.ArgumentParser(
        description="Train PPO on Panda Reach",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--timesteps", type=int, default=500_000,
                    help="Total training timesteps")
    p.add_argument("--n-envs", type=int, default=4,
                    help="Number of parallel environments to run")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    p.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate")
    p.add_argument("--batch-size", type=int, default=64,
                    help="Minibatch size")
    p.add_argument("--n-steps", type=int, default=2048,
                    help="Rollout buffer length (per update)")
    p.add_argument("--n-epochs", type=int, default=10,
                    help="Number of SGD epochs per update")
    p.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor")
    p.add_argument("--gae-lambda", type=float, default=0.95,
                    help="GAE lambda")
    p.add_argument("--clip-range", type=float, default=0.2,
                    help="PPO clip range")
    p.add_argument("--net-arch", type=int, nargs="+", default=[256, 256],
                    help="Hidden layer sizes for policy & value networks")
    p.add_argument("--eval-freq", type=int, default=10_000,
                    help="Evaluate every N timesteps")
    p.add_argument("--eval-episodes", type=int, default=10,
                    help="Episodes per evaluation")
    p.add_argument("--checkpoint-freq", type=int, default=50_000,
                    help="Save a checkpoint every N timesteps")
    p.add_argument("--log-dir", type=str, default="logs",
                    help="TensorBoard / eval log directory")
    p.add_argument("--model-dir", type=str, default="models",
                    help="Directory for saved models")
    p.add_argument("--verbose", action="store_true",
                    help="Verbose SB3 output")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ── Environments ─────────────────────────────────────────────
    train_env = make_vec_env(
        "PandaReach-v0",
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )
    eval_env = Monitor(PandaReachEnv())

    # ── Callbacks ────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.model_dir, "best"),
        log_path=os.path.join(args.log_dir, "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(args.model_dir, "checkpoints"),
        name_prefix="panda_reach",
        verbose=1,
    )

    callbacks = CallbackList([eval_cb, checkpoint_cb])

    # ── PPO Agent ────────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        policy_kwargs={"net_arch": list(args.net_arch)},
        verbose=1 if args.verbose else 0,
        seed=args.seed,
        tensorboard_log=args.log_dir,
    )

    # ── Train ────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  Panda Reach — PPO Training")
    print("=" * 62)
    print(f"  Timesteps     : {args.timesteps:>12,}")
    print(f"  Parallel Envs : {args.n_envs:>12}")
    print(f"  Learning rate : {args.lr:>12g}")
    print(f"  Batch size    : {args.batch_size:>12,}")
    print(f"  Net arch      : {args.net_arch}")
    print(f"  Seed          : {args.seed}")
    print(f"  Log dir       : {args.log_dir}")
    print(f"  Model dir     : {args.model_dir}")
    print("-" * 62)
    print(f"  TensorBoard   : tensorboard --logdir {args.log_dir}")
    print("=" * 62)
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # ── Save final model ─────────────────────────────────────────
    final_path = os.path.join(args.model_dir, "panda_reach_final")
    model.save(final_path)
    print(f"\n  Final model saved to: {final_path}.zip")

    train_env.close()
    eval_env.close()

    print("  Training complete ✓\n")


if __name__ == "__main__":
    main()
