<div align="center">
  <img width="100%" alt="Franka Emika Panda Simulation" src="assets/panda_reach_demo.gif" />
  <h1>Robotic Arm RL Manipulation</h1>
  <p><em>A state-of-the-art showcase of reinforcement learning-based robotic manipulation using the Franka Emika Panda arm in MuJoCo.</em></p>
</div>

---

## 📖 Overview

This repository contains a high-fidelity robotics simulation project focused on teaching a 7-DOF robotic arm complex manipulation tasks using deep reinforcement learning. By leveraging the **MuJoCo** physics engine and **Stable-Baselines3**, this project demonstrates a modular, scalable pipeline for continuous control in robotics.

The project is currently focused on the **Reaching Task** but systematically builds the foundation for autonomous, policy-driven behaviors like grasping and pick-and-place operations.

> [!NOTE]
> **✅ Current Status: Model Training Complete (v2)**
> The PPO agent has been successfully trained on the reaching task with a **100% success rate**. The agent utilizes orientation penalties, action smoothness penalties, and curriculum learning for natural and robust trajectories.

---

## ✨ Key Features & Architecture

- **High-Fidelity Simulation**: Utilizes MuJoCo for fast, accurate physics simulation of the Franka Emika Panda robotic arm.
- **Custom Gymnasium Environments**: Fully custom, OpenAI Gym-compliant (`gymnasium`) environments providing granular control over states, multi-component dense rewards, and episodes.
- **Deep Reinforcement Learning**: Integrates `Stable-Baselines3` to train robust Proximal Policy Optimization (PPO) agents for continuous action spaces.
- **Curriculum Learning**: Progressively scales task difficulty by expanding the target workspace volume dynamically during early training episodes.
- **Modular Architecture**: Clean separation of concerns between environment definitions, training logic, evaluation, and simulation assets.

### Directory Structure
```text
Robotic-Arm-RL-Manipulation/
├── envs/                     # Custom Gymnasium environments
│   ├── reach_env.py          # Reaching task environment (v2)
│   └── __init__.py           # Gym environment registrations
├── franka_emika_panda/       # MuJoCo assets (MJCF XMLs, meshes, textures)
├── scripts/                  # Utilities and debugging tools
│   └── teleop_robot.py       # Interactive keyboard teleoperation
├── train/                    # RL Training pipeline
│   └── train_rl.py           # PPO training script using SB3
├── evaluate.py               # Model evaluation and rendering script
├── environment.yml           # Conda environment definition
└── requirements.txt          # Python dependencies
```

---

## 🚀 Version History & Technical Progression

### Version 1: The Baseline Reach (Deprecated)
The initial prototype established the foundational RL pipeline and `PandaReach-v0` environment.

- **State Space (20-dim)**: 7 joint positions, 7 joint velocities, 3 end-effector (EE) Cartesian coordinates, 3 target Cartesian coordinates.
- **Action Space (7-dim)**: Normalized delta joint-position commands.
- **Reward Function**: Dense Cartesian distance penalty (`-distance`) + sparse success bonus (`+10`).
- **Results**: Achieved a ~90% success rate across 500k timesteps using a standard `[256, 256]` MLP policy.
- **Limitations**: The agent learned to reach the target but exhibited jerky, abrupt joint-level movements. It also arrived at the target with uncontrolled gripper orientation (often sideways or upside-down), which is problematic for future grasping tasks. Some boundary targets resulted in unreachable failure cases.

### Version 2: The Enhanced Agent (Latest)
A complete overhaul of the environment and training pipeline addressing all v1 limitations. The agent now moves fluidly and aligns perfectly for grasping.

- **State Space Expanded (23-dim)**: Introduced the EE z-axis directional vector (3-dim) to allow the agent to perceive its orientation relative to the target.
- **Multi-Component Reward Shaping**:
  - **Orientation Penalty**: Penalizes the EE z-axis deviating from the downward vector (`-Z`), forcing the gripper to approach the target from above.
  - **Smoothness Penalty**: Penalizes large changes between consecutive actions (`||action - prev_action||`), resulting in fluid, natural trajectories.
  - **Orientation Bonus**: The success bonus is scaled by the final orientation quality.
- **Curriculum Learning**: The target spawn volume starts strictly constrained to the center of the workspace and dynamically expands over the first 300 episodes, ensuring the agent learns baseline reaching before attempting edge cases.
- **Tightened Workspace Bounds**: Eliminated mechanically unreachable edge cases.
- **Aggressive Training Pipeline (`train_rl.py`)**:
  - Scaled up to **2,000,000 timesteps** across **16 parallel environments**.
  - **Deepened Network**: `[512, 256, 128]` hidden layers with `Tanh` activations.
  - **Hyperparameters**: Applied linear learning rate decay (`3e-4 → 0`), large rollout buffers (`4096` steps/env), large batch sizes (`2048`), and an entropy coefficient (`0.005`) for exploration.
  - **VecNormalize**: Implemented running standard normalization for observations and rewards.
- **Results**: **100% success rate**. Trajectory length reduced by 2x (averaging ~6-8 steps). Highly accurate value function (`0.981` explained variance).

---

## ⚙️ System Requirements & Prerequisites

The codebase is engineered and tested under the following environment:

- **Operating System**: Linux / macOS
- **Python**: Version 3.10
- **Core Dependencies**:
  - `mujoco >= 3.0.0`
  - `gymnasium[mujoco] == 0.29.1`
  - `stable-baselines3 == 2.4.0`
  - `torch >= 2.2.0`
  - `numpy == 1.26.4`

A comprehensive dependency list is maintained in both `requirements.txt` and `environment.yml` for reproducible environment setups via `conda`.

---

## 💻 Execution & Usage

### 1. Environment Setup
Create the isolated conda environment and install dependencies:
```bash
conda env create -f environment.yml
conda activate robotics
```

### 2. Manual Teleoperation
To manually verify physics, collision boundaries, and joint limits via keyboard:
```bash
python scripts/teleop_robot.py
```

### 3. Model Training
Train a new PPO agent from scratch using the v2 pipeline. Model checkpoints and normalizers will be saved to `models/`.
```bash
python train/train_rl.py --timesteps 2000000 --n-envs 16 --device auto --normalize
```

### 4. Evaluation & Visualization
Load the pre-trained `best_model.zip` and evaluate it using the custom evaluation script. The script automatically loads `VecNormalize` statistics.

**Console evaluation (100 episodes):**
```bash
python evaluate.py --model-path models/best/best_model.zip --n-episodes 100
```

**Visual evaluation (Interactive MuJoCo rendering):**
```bash
python evaluate.py --model-path models/best/best_model.zip --render --n-episodes 10
```
*(Note for Wayland users on Linux: You may need to prepend `MUJOCO_GL=glx` if the viewer fails to initialize).*

---

## 🔮 Roadmap

- [x] Integrate Franka Panda MuJoCo assets.
- [x] Build keyboard teleoperation script.
- [x] Design custom `Gymnasium` reaching environment.
- [x] Train baseline PPO reaching policy.
- [x] Overhaul environment with orientation, smoothness penalties, and curriculum learning.
- [x] Scale up training pipeline (VecNormalize, deeper MLPs, parallel envs).
- [ ] Implement `PickAndPlace-v0` environment (adding block manipulation).
- [ ] Add domain randomization (friction, mass, sensor noise) for sim-to-real transferability.

---

## ⚠️ License & Legal Notice

**© 2026 Aditya Guha. All rights reserved.**

This project is provided **for viewing and representational purposes only**. No permission is granted to use, copy, modify, distribute, or create derivative works from any part of this repository — including source code, documentation, models, and any other materials — for **any purpose**, whether commercial, academic, personal, or otherwise.

Unauthorized use may result in legal action under applicable intellectual property laws.

See the full [LICENSE](./LICENSE) file for details.
