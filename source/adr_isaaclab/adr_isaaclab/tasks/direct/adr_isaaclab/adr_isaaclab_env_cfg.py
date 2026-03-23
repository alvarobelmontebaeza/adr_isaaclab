# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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
    action_space = 14 + 3 + 3  # 7 dof per arm + thruster commands + torque commands
    observation_space = 66
    state_space = 0
    debug_vis = False

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

    # -- Target satellite configuration --
    target_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/alvaro/adr_isaaclab/source/adr_isaaclab/adr_isaaclab/assets/data/target_satellite.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.44581, 0.00206, -0.03),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # -- Action Parameters --
    arm_action_scale = 0.0      # 0.015
    thruster_scale = 10.0
    torque_scale = 5.0
    thruster_time_constant = 0.05

    # -- Command parameters --
    default_ee_pos_left_offset = (0.45, 0.2345, 0.0)
    default_ee_pos_right_offset = (0.45, -0.2345, 0.0)
    default_ee_rot_left_offset = (-1.57, -3.14, -3.14)
    default_ee_rot_right_offset = (-1.57, 3.14, 0.0)

    # Capture velocity ranges
    max_curriculum_factor = 0.7  # Reach max vel at 70% of the total training steps
    max_target_lin_vel = 0.0
    max_target_ang_vel = 0.05

    # -- Reward parameters --
    # - reward scales
    # Task rewards
    lin_rew_sigma = 0.05    # 0.1
    ang_rew_sigma = 0.1    # 0.25
    target_linear_vel_rew_scale = 5.0
    target_angular_vel_rew_scale = 5.0
    base_linear_velocity_rew_scale = 5.0
    base_angular_velocity_rew_scale = 5.0
    # Regularization rewards
    # Fine-grained tracking rewards
    target_lin_vel_fine_grained_rew_scale = -0.5
    target_ang_vel_fine_grained_rew_scale = -0.5
    base_lin_vel_fine_grained_rew_scale = -0.5
    base_ang_vel_fine_grained_rew_scale = -0.5
    # Acceleration penalties
    target_lin_acc_rew_scale = -0.05
    target_ang_acc_rew_scale = -0.05
    base_lin_acc_rew_scale = -0.05
    base_ang_acc_rew_scale = -0.05
    # Joint-level rewards
    arm_deviation_rew_scale = -0.1  # -1.0
    joint_velocity_rew_scale = -1e-3
    joint_torque_rew_scale = -1e-5
    # Action rate rewards
    arm_action_rate_rew_scale = -5e-2  # -5e-2
    thruster_action_rate_rew_scale = -0.5    # -5e-1
    torque_action_rate_rew_scale = -0.5  # -5e-1
    # Fuel consumption reward
    fuel_consumption_rew_scale = -0.0  # -1e-2
    collision_rew_scale = -100.0