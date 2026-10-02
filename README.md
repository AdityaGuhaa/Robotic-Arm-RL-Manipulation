<div align="center">
  <img width="100%" alt="Franka Emika Panda Simulation" src="assets/panda_reach_demo.gif" />
  <h1>Robotic Arm RL Manipulation</h1>
  <p><em>A high-fidelity simulation showcasing continuous control policies for robotic manipulation using deep reinforcement learning.</em></p>
</div>

---

## Overview

This repository hosts a state-of-the-art robotics simulation framework dedicated to developing autonomous manipulation policies for the 7-DOF Franka Emika Panda robotic arm. Built upon the **MuJoCo** physics engine and **Stable-Baselines3**, the project provides a modular, easily extensible pipeline for training continuous control policies in complex environments.

The current implementation focuses on precision end-effector reaching with constrained orientation, serving as the kinematic and policy foundation for advanced manipulation tasks such as pick-and-place operations, multi-object interactions, and dynamic obstacle avoidance.

> [!NOTE]
> **Current Status: Model Training Complete (v2)**
> The Proximal Policy Optimization (PPO) agent has been successfully trained on the enhanced reaching task, achieving a deterministic 100% success rate across benchmark evaluations. The converged policy demonstrates fluid, natural trajectories and precise grasping-ready orientation through advanced reward shaping and curriculum learning.

---

## Architecture and Core Components

The framework is constructed with strict separation of concerns, ensuring high performance during simulation and flexibility during policy iteration.

- **High-Fidelity Physics Engine**: Employs MuJoCo for rapid, computationally efficient forward kinematics and dynamics simulation, essential for generating millions of transitions for reinforcement learning.
- **Custom Gymnasium Interface**: Implements highly optimized, custom OpenAI Gym-compliant (`gymnasium`) environments. These define granular state boundaries, multi-component dense reward structures, and precise episode termination conditions.
- **Deep Reinforcement Learning Pipeline**: Integrates `Stable-Baselines3` to train highly robust, continuous-action PPO agents.
- **Adaptive Curriculum Learning**: Dynamically scales the difficulty of the task boundary conditions based on agent progression, mitigating early-stage divergence and sparse-reward plateauing.
- **Normalization Pipelines**: Utilizes observation and reward normalization wrappers (`VecNormalize`) to stabilize the learning updates across diverse numerical scales inherent in robotic state spaces.

### Directory Structure
```text
Robotic-Arm-RL-Manipulation/
├── envs/                     # Custom Gymnasium environments
│   ├── reach_env.py          # Primary kinematic reaching environment
│   └── __init__.py           # Submodule registration
├── franka_emika_panda/       # MuJoCo assets (MJCF XML models, meshes, physical properties)
├── scripts/                  # Utility and diagnostic scripts
│   └── teleop_robot.py       # Interactive keyboard teleoperation for debugging
├── train/                    # RL Training pipeline
│   └── train_rl.py           # Configurable PPO training execution script
├── evaluate.py               # Deterministic and stochastic model evaluation logic
├── environment.yml           # Conda dependency specification
└── requirements.txt          # Python dependency specification
```

---

## Technical Progression and Iterations

### Version 1: Baseline Kinematic Reaching (Deprecated)
The initial prototype established the foundational environment (`PandaReach-v0`) and verified the simulation-to-learning integration.

- **State Space (20-dim)**: 
  - 7x Joint positions ($q$)
  - 7x Joint velocities ($\dot{q}$)
  - 3x End-effector Cartesian coordinates ($x_{ee}$)
  - 3x Target Cartesian coordinates ($x_{target}$)
- **Action Space (7-dim)**: Delta joint-position commands normalized to $[-1, 1]$, scaled by $0.05$ rad, and applied to actuator control inputs.
- **Reward Function**: Dense Euclidean distance penalty: $R = -||x_{ee} - x_{target}||_2 + \text{SuccessBonus}$.
- **Performance**: Achieved a ~90% success rate across 500k timesteps using a standard `[256, 256]` MLP policy.
- **Identified Limitations**: The learned policy reached Cartesian targets but exhibited highly erratic joint velocities and unnatural trajectories. Crucially, the end-effector arrived at targets with arbitrary spatial orientations (e.g., inverted or lateral), rendering it mechanically unviable for sequential grasping tasks.

### Version 2: The Enhanced Control Policy (Latest)
A comprehensive architectural revision of the environment and training hyperparameters, explicitly addressing mechanical viability, trajectory smoothness, and robust convergence.

- **Expanded State Space (23-dim)**: Introduced the end-effector's local Z-axis directional vector (3-dim). This enables the policy network to continuously perceive its spatial orientation relative to the gravitational axis.
- **Multi-Component Reward Shaping**:
  - **Position Penalty**: Base Euclidean distance minimization.
  - **Orientation Penalty**: Evaluates the cosine similarity between the end-effector's Z-axis and the desired downward vector ($-Z$). The penalty minimizes deviation, forcing a top-down approach essential for gripping operations.
  - **Action Smoothness Penalty**: Penalizes the $L_2$ norm of the delta between consecutive actions ($||a_t - a_{t-1}||_2$). This drastically attenuates high-frequency oscillation and mechanical jitter.
  - **Dynamic Success Bonus**: The scalar terminal bonus ($+10.0$) is proportionally scaled by the final orientation quality, heavily incentivizing mechanically sound final states.
- **Curriculum Learning Implementation**: Target spawning is initially constrained to a minimal bounding box at the workspace origin. The bounding box dynamically expands over a 300-episode warmup phase to encompass the full reachable volume, effectively bootstrapping the policy gradient.
- **Training Pipeline Optimizations (`train_rl.py`)**:
  - **Compute Scale**: Training volume increased to **2,000,000 timesteps** distributed across **16 parallel environments** (`SubprocVecEnv`).
  - **Network Architecture**: Expanded capacity utilizing a `[512, 256, 128]` MLP structure with `Tanh` non-linearities for bounded output stability.
  - **Hyperparameters**: Implemented linear learning rate decay ($3\times10^{-4} \to 0$), increased rollout buffers (4096 steps per environment), larger batch sizes (2048), and an entropy coefficient ($0.005$) to prevent premature deterministic convergence.
  - **State Normalization**: Deployed running mean/variance tracking (`VecNormalize`) for both observations and rewards.
- **Final Metrics**: The trained model achieved a strict **100% success rate**. Trajectory efficiency improved by 50% (averaging 6-8 timesteps per episode). Value function accuracy stabilized exceptionally well (Explained Variance: `0.981`).

---

## System Requirements and Prerequisites

The execution pipeline has been engineered and validated under the following configuration:

- **Operating System**: Linux / macOS
- **Python**: Version 3.10
- **Core Dependencies**:
  - `mujoco >= 3.0.0`
  - `gymnasium[mujoco] == 0.29.1`
  - `stable-baselines3 == 2.4.0`
  - `torch >= 2.2.0`
  - `numpy == 1.26.4`

A deterministic environment definition is provided for `conda` to ensure reproducible execution.

---

## Execution and Usage

### 1. Environment Setup
Initialize the isolated execution environment:
```bash
conda env create -f environment.yml
conda activate robotics
```

### 2. Manual Diagnostics
To manually inspect collision bounds, joint limits, and inverse kinematics behavior:
```bash
python scripts/teleop_robot.py
```

### 3. Model Training
Initiate the PPO training sequence. Model checkpoints, optimal policies, and normalizer statistics will be persisted to the `models/` directory.
```bash
python train/train_rl.py --timesteps 2000000 --n-envs 16 --device auto --normalize
```

### 4. Evaluation and Visualization
Execute the custom evaluation script to parse the optimal policy and corresponding normalization statistics. 

**Headless Batch Evaluation (Quantitative Metrics):**
```bash
python evaluate.py --model-path models/best/best_model.zip --n-episodes 100
```

**Interactive Visual Evaluation (Qualitative Assessment):**
```bash
python evaluate.py --model-path models/best/best_model.zip --render --n-episodes 10
```
*(Note for Linux users utilizing Wayland display servers: Prepend `MUJOCO_GL=glx` to the execution command if the OpenGL context fails to initialize).*

---

## Development Roadmap

- [x] Integrate Franka Emika Panda MuJoCo definitions and assets.
- [x] Construct diagnostic keyboard teleoperation utilities.
- [x] Develop custom `Gymnasium` reaching environment (`PandaReach-v0`).
- [x] Establish and validate baseline PPO policy training.
- [x] Refactor environment for kinematic viability (orientation/smoothness penalties, curriculum learning).
- [x] Optimize distributed training pipeline (Normalization, deep MLPs, parallel execution).
- [ ] Implement `PickAndPlace-v0` environment (incorporating object manipulation and contact physics).
- [ ] Integrate domain randomization (friction coefficients, mass variance, actuator noise) to facilitate sim-to-real policy transfer.

---

## License and Legal Notice

**© 2026 Aditya Guha. All rights reserved.**

This repository and its contents are provided **for viewing and representational purposes only**. No permission is granted to use, copy, modify, distribute, or create derivative works from any part of this repository—including source code, documentation, models, and any other associated materials—for **any purpose**, whether commercial, academic, personal, or otherwise.

Unauthorized use may result in legal action under applicable intellectual property laws.

See the full [LICENSE](./LICENSE) file for comprehensive legal details.
