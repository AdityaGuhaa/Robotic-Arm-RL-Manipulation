# Copyright (c) 2026 Aditya Guha. All rights reserved.
# This source code is proprietary and may not be used, copied, modified,
# or distributed without prior written permission. See LICENSE for details.

"""
Gymnasium environment registration for the Panda manipulation tasks.

Importing this module registers all custom environments with Gymnasium.
"""

from gymnasium.envs.registration import register

register(
    id="PandaReach-v0",
    entry_point="envs.reach_env:PandaReachEnv",
)
