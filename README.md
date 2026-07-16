<div align="center">
  <img width="100%" alt="Franka Emika Panda Simulation" src="https://github.com/user-attachments/assets/ac091b06-14e7-4536-b164-b2f83f6aedf2" />
  <h1>Robotic Arm RL Manipulation</h1>
  <p><em>A representational showcase of reinforcement learning-based robotic manipulation using the Franka Emika Panda arm in MuJoCo.</em></p>
</div>

---

## 📖 Overview

This repository contains a state-of-the-art robotics simulation project focused on teaching a 7-DOF robotic arm complex manipulation tasks using deep reinforcement learning. By leveraging the high-performance **MuJoCo** physics engine and **Stable-Baselines3**, this project demonstrates a modular, scalable pipeline for continuous control in robotics.

The project systematically progresses from fundamental robot kinematics and teleoperation to autonomous, policy-driven behaviors like reaching, grasping, and pick-and-place operations.

> [!NOTE]
> **🚧 Current Status: Model Training in Progress**
> The reinforcement learning agents (PPO) are currently undergoing active training. Evaluation metrics and final trained model weights will be updated once the training cycles are complete and convergence is achieved.

## ✨ Key Features

- **High-Fidelity Simulation**: Utilizes MuJoCo for fast, accurate physics simulation of the Franka Emika Panda robotic arm.
- **Custom Gymnasium Environments**: Implements fully custom, OpenAI Gym-compliant (`gymnasium`) environments for granular control over states, rewards, and episodes.
- **Deep Reinforcement Learning**: Integrates `Stable-Baselines3` to train robust Proximal Policy Optimization (PPO) agents for continuous action spaces.
- **Interactive Teleoperation**: Includes a real-time keyboard teleoperation script to manually control end-effector kinematics, facilitating debugging and intuition-building for reward shaping.
- **Modular Architecture**: Clean separation of concerns between environment definitions, training logic, evaluation, and simulation assets.

## 🏗️ Project Architecture

```text
Robotic-Arm-RL-Manipulation/
├── envs/                     # Custom Gymnasium environments
│   ├── reach_env.py          # Reaching task environment
│   └── pick_place_env.py     # Pick-and-place task environment (WIP)
├── franka_emika_panda/       # MuJoCo assets (MJCF XMLs, meshes, textures)
├── scripts/                  # Utilities and debugging tools
│   ├── teleop_robot.py       # Interactive keyboard teleoperation
│   └── test_robot.py         # Basic simulation loop testing
├── train/                    # RL Training pipeline
│   └── train_rl.py           # PPO training script using SB3
├── evaluate.py               # Model evaluation and rendering script
├── environment.yml           # Conda environment definition
└── requirements.txt          # Python dependencies
```

## ⚙️ System Requirements & Prerequisites

The codebase is engineered and tested under the following environment:

- **Operating System**: macOS (Apple Silicon / Intel compatible)
- **Python**: Version 3.10
- **Core Dependencies**:
  - `mujoco >= 3.0.0`
  - `gymnasium[mujoco] == 0.29.1`
  - `stable-baselines3 == 2.4.1`
  - `torch >= 2.2.0`
  - `numpy == 1.26.4`

A comprehensive dependency list is maintained in both `requirements.txt` and `environment.yml` for reproducible environment setups (e.g., via `conda`).

## 🚀 Execution Pipeline

Although this codebase is proprietary and not licensed for external use, the internal execution pipeline operates as follows:

1. **Environment Setup**: 
   Dependencies are managed via `conda` and `pip` (MuJoCo, Gymnasium, Stable-Baselines3).
2. **Teleoperation & Verification**: 
   `python scripts/teleop_robot.py` allows manual verification of physics, collision boundaries, and joint limits.
3. **Training Phase**: 
   `python train/train_rl.py` initializes the PPO agent and begins interaction with the custom `PandaReach-v0` environment. *(Currently in progress)*
4. **Evaluation Phase**: 
   `python evaluate.py` loads the best-performing model checkpoints to visualize the learned policy and calculate success rates.

---

## ⚠️ License & Legal Notice

**© 2026 Aditya Guha. All rights reserved.**

This project is provided **for viewing and representational purposes only**. No permission is granted to use, copy, modify, distribute, or create derivative works from any part of this repository — including source code, documentation, models, and any other materials — for **any purpose**, whether commercial, academic, personal, or otherwise.

Unauthorized use may result in legal action under applicable intellectual property laws.

See the full [LICENSE](./LICENSE) file for details.
