# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.managers.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import EventTermCfg as EventTerm



from adr_isaaclab.assets.adr import KINOVA_BIMANUAL_CFG

# @configclass
# class EventCfg:
#     """Configuration for randomization."""

#     add_base_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="base"),
#             "mass_distribution_params": (0.0, 5.0),
#             "operation": "add",
#         },
#     )

@configclass
class AdrIsaaclabEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    physics_dt = 1.0 / 200.0
    policy_dt = physics_dt * decimation
    episode_length_s = 20.0
    # - spaces definition
    action_space = 14  # 7 dof per arm
    observation_space = 4
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        gravity=(0.0, 0.0, 0.0),
    )

    # robot(s)
    robot_cfg: ArticulationCfg = KINOVA_BIMANUAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # -- Action Parameters --
    action_scale = 0.5

    # -- Command parameters --
    default_ee_pos_left_offset = (0.5, -0.5, 0.0)
    default_ee_pos_right_offset = (0.5, 0.5, 0.0)

    target_pos_x_range = [-1.5, 1.5]
    target_pos_y_range = [-1.5, 1.5]
    target_pos_z_range = [0.5, 1.5]

    target_roll_range = [-3.14, 3.14]
    target_pitch_range = [-3.14, 3.14]
    target_yaw_range = [-3.14, 3.14]

    # -- Reward parameters --
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
