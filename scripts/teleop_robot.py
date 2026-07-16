# Copyright (c) 2026 Aditya Guha. All rights reserved.
# This source code is proprietary and may not be used, copied, modified,
# or distributed without prior written permission. See LICENSE for details.

"""
Interactive keyboard control for the Franka Emika Panda robot.

Controls:
    1-7        Select arm joint
    ↑ / ↓      Move selected joint (increase / decrease)
    ← / →      Select previous / next joint
    G          Toggle gripper open / close
    R          Reset to home position
    Q / ESC    Quit
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# ── Load model ───────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

# ── State ────────────────────────────────────────────────────────
selected_joint = 0          # 0–6 for the 7 arm joints
joint_step = 0.05           # radians per key press
gripper_open = True

JOINT_NAMES = [f"Joint {i+1}" for i in range(7)]

# GLFW key codes
KEY_1, KEY_7 = 49, 55
KEY_UP, KEY_DOWN = 265, 264
KEY_LEFT, KEY_RIGHT = 263, 262
KEY_G = 71
KEY_R = 82
KEY_Q = 81
KEY_ESC = 256


def print_status():
    """Print current joint values."""
    parts = []
    for i in range(7):
        marker = "▶" if i == selected_joint else " "
        val = data.ctrl[i]
        parts.append(f"  {marker} J{i+1}: {val:>7.3f}")
    grip = "OPEN" if gripper_open else "CLOSED"
    print("\033[2J\033[H")  # clear terminal
    print("╔══════════════════════════════════════╗")
    print("║   Panda Interactive Keyboard Control ║")
    print("╠══════════════════════════════════════╣")
    print("║  1-7: Select joint  ↑↓: Move joint  ║")
    print("║  ←→:  Cycle joint   G:  Gripper      ║")
    print("║  R:   Reset         Q:  Quit         ║")
    print("╠══════════════════════════════════════╣")
    for line in parts:
        print(f"║{line:>36s}  ║")
    print(f"║  {'  Gripper: ' + grip:>36s}  ║")
    print("╚══════════════════════════════════════╝")


def key_callback(keycode):
    global selected_joint, gripper_open

    # ── Select joint by number (1-7) ──
    if KEY_1 <= keycode <= KEY_7:
        selected_joint = keycode - KEY_1
        print_status()

    # ── Cycle joint with ← / → ──
    elif keycode == KEY_LEFT:
        selected_joint = (selected_joint - 1) % 7
        print_status()
    elif keycode == KEY_RIGHT:
        selected_joint = (selected_joint + 1) % 7
        print_status()

    # ── Move selected joint with ↑ / ↓ ──
    elif keycode == KEY_UP:
        lo, hi = model.actuator_ctrlrange[selected_joint]
        data.ctrl[selected_joint] = np.clip(
            data.ctrl[selected_joint] + joint_step, lo, hi
        )
        print_status()
    elif keycode == KEY_DOWN:
        lo, hi = model.actuator_ctrlrange[selected_joint]
        data.ctrl[selected_joint] = np.clip(
            data.ctrl[selected_joint] - joint_step, lo, hi
        )
        print_status()

    # ── Toggle gripper ──
    elif keycode == KEY_G:
        gripper_open = not gripper_open
        data.ctrl[7] = 0.0 if gripper_open else 255.0
        print_status()

    # ── Reset ──
    elif keycode == KEY_R:
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = 0
        gripper_open = True
        print_status()


# ── Launch viewer ────────────────────────────────────────────────
with mujoco.viewer.launch_passive(
    model, data, key_callback=key_callback
) as viewer:
    print_status()
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)  # ~500 Hz, smooth without maxing CPU
