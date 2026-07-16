# Copyright (c) 2026 Aditya Guha. All rights reserved.
# This source code is proprietary and may not be used, copied, modified,
# or distributed without prior written permission. See LICENSE for details.

"""
PandaReachEnv — Gymnasium environment for the Franka Panda reach task.

The agent must move the robot's end-effector to a randomly placed target
position in 3-D space.

Observation (20-dim):
    joint_pos (7) + joint_vel (7) + ee_pos (3) + target_pos (3)

Action (7-dim):
    Normalised delta joint-position commands for the 7 arm joints,
    scaled by ``action_scale`` rad and added to the current actuator ctrl.

Reward:
    Dense: −distance(EE, target) + bonus on success.

Success:
    End-effector within ``success_threshold`` metres of the target.
"""

import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


class PandaReachEnv(gym.Env):
    """Franka Panda reach-to-target environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 100,
        action_scale: float = 0.05,
        success_threshold: float = 0.05,
        n_substeps: int = 20,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.success_threshold = success_threshold
        self.n_substeps = n_substeps

        # ── Load MuJoCo model ────────────────────────────────────
        xml_path = os.path.join(
            os.path.dirname(__file__),
            "..", "franka_emika_panda", "reach_scene.xml",
        )
        xml_path = os.path.abspath(xml_path)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ── Cache IDs ────────────────────────────────────────────
        self._hand_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        # Home keyframe index
        self._home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )

        # ── Arm joints ───────────────────────────────────────────
        self.n_arm_joints = 7
        self.n_actions = 7

        # ── Spaces ───────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32,
        )
        #  joint_pos(7) + joint_vel(7) + ee_pos(3) + target_pos(3) = 20
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64,
        )

        # ── Target workspace (reachable volume around the arm) ──
        self._target_low = np.array([0.3, -0.3, 0.2])
        self._target_high = np.array([0.7, 0.3, 0.6])

        # ── Internal state ───────────────────────────────────────
        self._step_count = 0
        self._viewer = None
        self._renderer = None

    # ==================================================================
    # Gym API
    # ==================================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset to home keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        # Randomise target position
        target_pos = self.np_random.uniform(self._target_low, self._target_high)
        self.data.mocap_pos[0] = target_pos

        # Forward kinematics so that xpos is up-to-date
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Apply delta joint-position commands to arm actuators
        for i in range(self.n_arm_joints):
            lo, hi = self.model.actuator_ctrlrange[i]
            self.data.ctrl[i] = np.clip(
                self.data.ctrl[i] + self.action_scale * action[i], lo, hi,
            )

        # Keep gripper closed (not part of this task)
        self.data.ctrl[7] = 255.0

        # Substep the simulation
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Observation & info
        obs = self._get_obs()
        info = self._get_info()

        # Reward
        distance = info["distance"]
        reward = -distance
        success = distance < self.success_threshold
        if success:
            reward += 10.0

        info["is_success"] = success

        terminated = success
        truncated = self._step_count >= self.max_episode_steps

        # Render (human mode syncs the viewer each step)
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data,
                )
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model, height=480, width=640,
                )
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ==================================================================
    # Helpers
    # ==================================================================

    def _get_obs(self) -> np.ndarray:
        """Build the 20-dim observation vector."""
        joint_pos = self.data.qpos[: self.n_arm_joints].copy()
        joint_vel = self.data.qvel[: self.n_arm_joints].copy()
        ee_pos = self.data.xpos[self._hand_body_id].copy()
        target_pos = self.data.mocap_pos[0].copy()
        return np.concatenate([joint_pos, joint_vel, ee_pos, target_pos])

    def _get_info(self) -> dict:
        ee_pos = self.data.xpos[self._hand_body_id]
        target_pos = self.data.mocap_pos[0]
        distance = float(np.linalg.norm(ee_pos - target_pos))
        return {"distance": distance}
