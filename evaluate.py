#!/usr/bin/env python3
# Copyright (c) 2026 Aditya Guha. All rights reserved.
# This source code is proprietary and may not be used, copied, modified,
# or distributed without prior written permission. See LICENSE for details.

"""
Evaluation script for the Robotic Arm RL Manipulation project.

Loads a trained RL agent and evaluates it in the pick-and-place
environment, reporting metrics and optionally recording videos.

Usage:
    python evaluate.py --model-path models/ppo_pick_place.zip
    python evaluate.py --model-path models/ppo_pick_place.zip --render
    python evaluate.py --model-path models/ppo_pick_place.zip --record --n-episodes 20
    python evaluate.py --model-path models/ppo_pick_place.zip --output results/eval.json --verbose
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Trigger Gymnasium registration
import envs  # noqa: F401


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL agent on the Panda pick-and-place task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the trained SB3 model checkpoint (.zip)",
    )
    parser.add_argument(
        "--env", type=str, default="PandaReach-v0",
        help="Gymnasium environment ID (e.g. PandaReach-v0)",
    )
    parser.add_argument(
        "--algo", type=str, default="PPO",
        choices=["PPO", "SAC", "TD3", "A2C", "DDPG", "HER", "TQC", "RecurrentPPO"],
        help="RL algorithm used during training",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=100,
        help="Number of evaluation episodes to run",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render the MuJoCo viewer during evaluation (interactive)",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record evaluation episodes as video files",
    )
    parser.add_argument(
        "--record-dir", type=str, default="eval_videos",
        help="Directory to save recorded videos",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic (greedy) actions during evaluation",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic actions (overrides --deterministic)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save evaluation results as JSON",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-episode results",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Algorithm loader
# ---------------------------------------------------------------------------
def load_model(algo_name: str, model_path: str):
    """Load a trained SB3 model by algorithm name."""
    algo_name = algo_name.upper()

    # Standard SB3 algorithms
    if algo_name == "PPO":
        from stable_baselines3 import PPO as AlgoClass
    elif algo_name == "SAC":
        from stable_baselines3 import SAC as AlgoClass
    elif algo_name == "TD3":
        from stable_baselines3 import TD3 as AlgoClass
    elif algo_name == "A2C":
        from stable_baselines3 import A2C as AlgoClass
    elif algo_name == "DDPG":
        from stable_baselines3 import DDPG as AlgoClass
    # sb3-contrib algorithms
    elif algo_name == "TQC":
        from sb3_contrib import TQC as AlgoClass
    elif algo_name == "RECURRENTPPO":
        from sb3_contrib import RecurrentPPO as AlgoClass
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    model = AlgoClass.load(model_path)
    print(f"  Algorithm : {algo_name}")
    print(f"  Policy    : {model.policy.__class__.__name__}")
    return model


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def create_env(env_id="PandaReach-v0", render=False, record=False, record_dir="eval_videos"):
    """
    Create the evaluation environment.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID (e.g. 'PandaReach-v0').
    render : bool
        If True, launch the interactive MuJoCo viewer.
    record : bool
        If True, wrap the environment to record episode videos.
    record_dir : str
        Directory in which to save recorded videos.

    Returns
    -------
    gymnasium.Env
    """
    import gymnasium

    render_mode = None
    if render:
        render_mode = "human"
    elif record:
        render_mode = "rgb_array"

    env = gymnasium.make(env_id, render_mode=render_mode)

    if record:
        from gymnasium.wrappers import RecordVideo

        os.makedirs(record_dir, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=record_dir,
            episode_trigger=lambda ep_id: True,  # record every episode
            name_prefix="eval",
        )
        print(f"  Recording videos to: {os.path.abspath(record_dir)}")

    return env


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
def evaluate(model, env, n_episodes: int = 100,
             deterministic: bool = True, verbose: bool = False):
    """
    Run evaluation episodes and collect metrics.

    Returns
    -------
    results : dict
        Summary statistics.
    episode_rewards : list[float]
    episode_lengths : list[int]
    successes : list[float]
    """
    episode_rewards = []
    episode_lengths = []
    successes = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            
            # Add delay when rendering so it plays at real-time speed (~50 Hz)
            if env.render_mode == "human":
                time.sleep(0.02)
                
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # The environment should set info["is_success"] on terminal steps
        success = info.get("is_success", False)
        successes.append(float(success))

        if verbose:
            status = "\033[92m✓\033[0m" if success else "\033[91m✗\033[0m"
            print(
                f"  Episode {ep + 1:>4d}/{n_episodes}  |  "
                f"Reward: {total_reward:>9.2f}  |  "
                f"Steps: {steps:>4d}  |  "
                f"Success: {status}"
            )

    # Summary statistics
    results = {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "median_reward": float(np.median(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "success_rate": float(np.mean(successes)),
        "total_successes": int(np.sum(successes)),
    }

    return results, episode_rewards, episode_lengths, successes


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------
def print_results(results: dict):
    """Print a formatted summary table."""
    print()
    print("=" * 62)
    print("   EVALUATION RESULTS")
    print("=" * 62)
    print(f"   Episodes evaluated : {results['n_episodes']}")
    sr = results['success_rate'] * 100
    print(
        f"   Success rate       : {sr:.1f}%  "
        f"({results['total_successes']}/{results['n_episodes']})"
    )
    print("-" * 62)
    print(
        f"   Mean reward        : {results['mean_reward']:.2f}"
        f"  ± {results['std_reward']:.2f}"
    )
    print(f"   Median reward      : {results['median_reward']:.2f}")
    print(
        f"   Min / Max reward   : {results['min_reward']:.2f}"
        f"  /  {results['max_reward']:.2f}"
    )
    print(
        f"   Mean ep. length    : {results['mean_length']:.1f}"
        f"  ± {results['std_length']:.1f}"
    )
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    deterministic = not args.stochastic  # --stochastic overrides default

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"\033[91mError:\033[0m Model not found at '{args.model_path}'")
        sys.exit(1)

    print()
    print("─" * 62)
    print("  Panda Pick-and-Place — RL Evaluation")
    print("─" * 62)
    print(f"  Model     : {args.model_path}")

    # Load model
    model = load_model(args.algo, args.model_path)

    # Create environment
    env = create_env(
        env_id=args.env,
        render=args.render,
        record=args.record,
        record_dir=args.record_dir,
    )

    # Seed
    if args.seed is not None:
        env.reset(seed=args.seed)
        np.random.seed(args.seed)

    print(f"  Episodes  : {args.n_episodes}")
    print(f"  Policy    : {'deterministic' if deterministic else 'stochastic'}")
    print(f"  Seed      : {args.seed}")
    print("─" * 62)
    print()

    # Run evaluation
    t_start = time.time()
    results, rewards, lengths, successes = evaluate(
        model, env,
        n_episodes=args.n_episodes,
        deterministic=deterministic,
        verbose=args.verbose,
    )
    elapsed = time.time() - t_start

    # Attach metadata
    results["model_path"] = os.path.abspath(args.model_path)
    results["algorithm"] = args.algo
    results["deterministic"] = deterministic
    results["seed"] = args.seed
    results["wall_time_seconds"] = round(elapsed, 2)
    results["timestamp"] = datetime.now().isoformat()

    # Print summary
    print_results(results)
    print(f"  Wall time: {elapsed:.1f}s "
          f"({elapsed / args.n_episodes:.2f}s per episode)")

    # Optionally save to JSON
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Include per-episode data
        full_results = {
            **results,
            "per_episode": {
                "rewards": rewards,
                "lengths": lengths,
                "successes": successes,
            },
        }
        with open(args.output, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"  Results saved to: {os.path.abspath(args.output)}")

    print()
    env.close()


if __name__ == "__main__":
    main()
