# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, subtract_frame_transforms, quat_error_magnitude, quat_from_euler_xyz, quat_conjugate

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG


from .adr_isaaclab_env_cfg import AdrIsaaclabEnvCfg


class AdrIsaaclabEnv(DirectRLEnv):
    cfg: AdrIsaaclabEnvCfg

    def __init__(self, cfg: AdrIsaaclabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action buffers
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        # Action scales
        self._arm_action_scale = torch.tensor(self.cfg.arm_action_scale, device=self.device)
        self._thruster_scale = torch.tensor(self.cfg.thruster_scale, device=self.device)
        self._torque_scale = torch.tensor(self.cfg.torque_scale, device=self.device)
        self._thruster_dynamics_alpha = self.step_dt / (self.step_dt + self.cfg.thruster_time_constant)

        # Action: Target joint positions + thruster commands + torque commands
        self._target_joint_pos = torch.zeros((self.num_envs, 14), device=self.device)
        self._desired_thruster_forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._applied_thruster_forces = torch.zeros_like(self._desired_thruster_forces)
        self._desired_reaction_torques = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._applied_reaction_torques = torch.zeros_like(self._desired_reaction_torques)

        # Buffers for end-effector poses
        self.ee_pose_left_w = torch.zeros((self.num_envs, 7), device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self.ee_pose_right_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.ee_pose_left_b = torch.zeros((self.num_envs, 7), device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self.ee_pose_right_b = torch.zeros((self.num_envs, 7), device=self.device)

        # -- Commands --
        self._default_body_rotation = torch.zeros((self.num_envs, 4), device=self.device) # default base rotation as a quaternion
        self._default_body_rotation[:, 0] = 1.0 # set default rotation to identity quaternion (w=1, x=0, y=0, z=0)
        
        self._default_ee_pos_left_offset = torch.tensor(self.cfg.default_ee_pos_left_offset, device=self.device)
        self._default_ee_pos_right_offset = torch.tensor(self.cfg.default_ee_pos_right_offset, device=self.device)
        # Target poses for each arm in base frame
        self._target_ee_pose_left_b = torch.zeros((self.num_envs, 7), device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self._target_ee_pose_right_b = torch.zeros((self.num_envs, 7), device=self.device)
        # Target poses for each arm in world frame
        self._target_ee_pose_left_w = torch.zeros((self.num_envs, 7), device=self.device)
        self._target_ee_pose_right_w = torch.zeros((self.num_envs, 7), device=self.device)

        # -- Curriculum learning parameters --
        self.global_step_counter = 0
        self.max_learning_epochs = 2000
        self.max_curriculum_steps = int(self.cfg.max_curriculum_factor * self.max_learning_epochs * self.num_envs * 24)
        self.curriculum_factor = 0.0
        self.max_lin_vel = 0.0
        self.max_ang_vel = 0.0


        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "target_linear_vel",
                "target_angular_vel",
                "base_linear_velocity",
                "base_angular_velocity",
                "target_linear_vel_fine_grained",
                "target_angular_vel_fine_grained",
                "base_linear_vel_fine_grained",
                "base_angular_vel_fine_grained",
                "target_lin_acc",
                "target_ang_acc",
                "base_lin_acc",
                "base_ang_acc",
                "arm_deviation",
                "joint_velocity",
                "joint_torque",
                "arm_action_rate",
                "thruster_action_rate",
                "torque_action_rate",
                "fuel_consumption",
                "collision",
            ]
        }

        # Get indices of the bodies and joints of interest
        self._base_id, _ = self._contact_sensor.find_bodies(".*base")
        self._left_ee_id, _ = self._robot.find_bodies(".*gen3n7_left_end_effector_link")
        self._right_ee_id, _ = self._robot.find_bodies(".*gen3n7_right_end_effector_link")
        self._undesired_collision_body_ids, _ = self._contact_sensor.find_bodies([".*base", ".*link"])
        self._target_base_id, _ = self._target.find_bodies(".*Target")
        # Arm joint indices
        self._left_arm_joint_ids, _ = self._robot.find_joints(".*left_joint_.*")
        self._right_arm_joint_ids, _ = self._robot.find_joints(".*right_joint_.*")
        self._arm_joint_ids = torch.cat((torch.tensor(self._left_arm_joint_ids), torch.tensor(self._right_arm_joint_ids)), dim=0)
        # Gripper joint indices
        self._right_gripper_joint_id, _ = self._robot.find_joints("finger_joint") # TODO: Check that joint names are correct
        self._left_gripper_joint_id, _ = self._robot.find_joints("finger_joint_0")

        # Add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        # Add robot and sensors to the scene
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._target = RigidObject(self.cfg.target_cfg)
        self.scene.rigid_objects["target"] = self._target

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        # add lights
        light_cfg = sim_utils.DistantLightCfg(angle=30, intensity=1000.0, color=(0.75, 0.75, 0.75)) #sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Update step counter for curriculum learning
        self.global_step_counter += self.num_envs

        # Retrieve raw actions and apply scaling and processing to obtain target joint positions, thruster forces, and reaction torques
        self._actions = actions.clone()
        # Parse actions to be: [target_joint_pos, thruster_commands, torque_commands]
        self.arm_actions = self._actions[:, :14]
        self.thruster_actions = self._actions[:, 14:14 + 3]
        self.torque_actions = self._actions[:, 14 + 3:]
        # apply scaling and add to default joint positions to obtain processed actions as target joint positions
        # Residual Joint Control
        self._target_joint_pos = self.arm_actions * self._arm_action_scale + self._robot.data.default_joint_pos[:, self._arm_joint_ids]
        # Delta joint control
        # self._target_joint_pos = self.arm_actions * self._arm_action_scale + self._robot.data.joint_pos[:, self._arm_joint_ids]
        # Scale thruster and torque actions
        self._desired_thruster_forces[:, 0, :] = self.thruster_actions.clamp(-1.0, 1.0) * self._thruster_scale
        self._desired_reaction_torques[:, 0, :] = self.torque_actions.clamp(-1.0, 1.0) * self._torque_scale
        # Apply first order dynamics
        prev_thruster_forces = self._applied_thruster_forces.clone()
        prev_reaction_torques = self._applied_reaction_torques.clone()
        self._applied_thruster_forces = prev_thruster_forces * (1 - self._thruster_dynamics_alpha) + self._thruster_dynamics_alpha * self._desired_thruster_forces
        self._applied_reaction_torques = prev_reaction_torques * (1 - self._thruster_dynamics_alpha) + self._thruster_dynamics_alpha * self._desired_reaction_torques
        # Inject target motion at the beginning of each episode after reset
        self.update_target_motion()

    def _apply_action(self) -> None:
        # Arm joint targets
        self._robot.set_joint_position_target(self._target_joint_pos, joint_ids=self._arm_joint_ids)
        # Close grippers
        self._robot.set_joint_position_target(1.0, joint_ids=[self._left_gripper_joint_id])
        self._robot.set_joint_position_target(1.0, joint_ids=[self._right_gripper_joint_id])
        # Thrust and reaction torques
        self._robot.set_external_force_and_torque(
            forces=self._applied_thruster_forces,
            torques=self._applied_reaction_torques,
            body_ids=self._base_id
        )
        self._robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # Get current robot rotation in world frame
        base_rot_w = self._robot.data.root_link_pose_w[:, 3:] # [num_envs, 4] quaternion (w, x, y, z)
        # Obtain conjugate of base rotation to transform target velocities from world frame to body frame
        base_rot_w_conj = quat_conjugate(base_rot_w)    

        # Target twist
        target_linear_vel_w = self._target.data.root_link_vel_w[:, :3]
        target_angular_vel_w = self._target.data.root_link_vel_w[:, 3:]

        # Transform target velocities to body frame
        target_lin_vel_b = quat_apply(base_rot_w_conj, target_linear_vel_w)
        target_ang_vel_b = quat_apply(base_rot_w_conj, target_angular_vel_w)
        # Base twist
        base_linear_vel_b = self._robot.data.root_lin_vel_b
        base_angular_vel_b = self._robot.data.root_ang_vel_b
        # Base acceleration
        base_acc_w = self._robot.data.body_acc_w[:, self._base_id, :].view(-1, 6) # [num_envs, [linear_acc, angular_acc]]
        base_acc_b = torch.zeros_like(base_acc_w)
        base_acc_b[:, :3] = quat_apply(base_rot_w_conj, base_acc_w[:, :3])
        base_acc_b[:, 3:] = quat_apply(base_rot_w_conj, base_acc_w[:, 3:]) # Transform base acceleration to body frame

        # Get joint positions and velocities for the joints of interest
        self.joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        self.joint_vel = self._robot.data.joint_vel[:, self._arm_joint_ids]

        # Get current EE poses
        self.update_current_ee_poses() # Retrieves current EE poses in world frame and converts to base frame
                
        # Last actions
        actions = self._previous_actions

        obs = torch.cat(
            (
                target_lin_vel_b, #3
                target_ang_vel_b, #3
                base_linear_vel_b, #3
                base_angular_vel_b, #3
                base_acc_b, #6
                self.joint_pos, #14
                self.joint_vel, #14
                actions, # 20
            ),
            dim=-1, # Total: 54
        )

        # TODO: Consider adding asymmetrics actor-critic

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # -- Task rewards
        sigma_lin_squared = self.cfg.lin_rew_sigma ** 2
        sigma_ang_squared = self.cfg.ang_rew_sigma ** 2
        # Stabilize target
        target_velocity = self._target.data.root_link_vel_w
        target_lin_vel_penalty = torch.exp(-torch.sum(torch.square(target_velocity[:, :3]), dim=-1) / sigma_lin_squared)
        target_ang_vel_penalty = torch.exp(-torch.sum(torch.square(target_velocity[:, 3:]), dim=-1) / sigma_ang_squared)
        # Base velocity penalty to encourage stability
        base_velocity = self._robot.data.root_com_vel_w
        base_linear_vel_penalty = torch.exp(-torch.sum(torch.square(base_velocity[:, :3]), dim=-1) / sigma_lin_squared)
        base_angular_vel_penalty = torch.exp(-torch.sum(torch.square(base_velocity[:, 3:]), dim=-1) / sigma_ang_squared)

        # -- Regularization rewards
        # L1 rewards to encourage fine-grained tracking of target velocities and stability
        # target_lin_vel_fine_grained = torch.linalg.vector_norm(target_velocity[:, :3], ord=1, dim=-1)
        # target_ang_vel_fine_grained= torch.linalg.vector_norm(target_velocity[:, 3:], ord=1, dim=-1)
        # base_lin_vel_fine_grained= torch.linalg.vector_norm(self._robot.data.root_lin_vel_b, ord=1, dim=-1)
        # base_ang_vel_fine_grained= torch.linalg.vector_norm(self._robot.data.root_ang_vel_b, ord=1, dim=-1)
        # Huber loss
        # delta=0.05
        # target_lin_vel_fine_grained = F.huber_loss(target_velocity[:, :3], torch.zeros_like(target_velocity[:, :3]), delta=delta, reduction='none').sum(dim=-1)
        # target_ang_vel_fine_grained= F.huber_loss(target_velocity[:, 3:], torch.zeros_like(target_velocity[:, 3:]), delta=delta, reduction='none').sum(dim=-1)
        # base_lin_vel_fine_grained= F.huber_loss(self._robot.data.root_lin_vel_b, torch.zeros_like(self._robot.data.root_lin_vel_b), delta=delta, reduction='none').sum(dim=-1)
        # base_ang_vel_fine_grained= F.huber_loss(self._robot.data.root_ang_vel_b, torch.zeros_like(self._robot.data.root_ang_vel_b), delta=delta, reduction='none').sum(dim=-1)
        # L0.5 Kernel to favour high gradients near 0.0
        eps = 1e-6
        target_lin_vel_fine_grained = torch.sum(torch.sqrt(torch.abs(target_velocity[:, :3]) + eps), dim=-1)
        target_ang_vel_fine_grained = torch.sum(torch.sqrt(torch.abs(target_velocity[:, 3:]) + eps), dim=-1)
        base_lin_vel_fine_grained = torch.sum(torch.sqrt(torch.abs(self._robot.data.root_lin_vel_b) + eps), dim=-1)
        base_ang_vel_fine_grained = torch.sum(torch.sqrt(torch.abs(self._robot.data.root_ang_vel_b) + eps), dim=-1)

        # Penalize acceleration to avoid oscillations in velocity
        target_acc = self._target.data.body_com_acc_w[:, self._target_base_id, :].view(-1, 6)
        target_lin_acc = target_acc[:, :3]
        target_ang_acc = target_acc[:, 3:]
        base_acc = self._robot.data.body_com_acc_w[:, self._base_id, :].view(-1, 6)
        base_lin_acc = base_acc[:, :3]
        base_ang_acc = base_acc[:, 3:]
        # Compute reward term
        target_lin_acc_penalty = torch.sum(torch.square(target_lin_acc), dim=-1)
        target_ang_acc_penalty = torch.sum(torch.square(target_ang_acc), dim=-1)
        base_lin_acc_penalty = torch.sum(torch.square(base_lin_acc), dim=-1)
        base_ang_acc_penalty = torch.sum(torch.square(base_ang_acc), dim=-1)

        # Penalize arm to be far from default configuration.
        arm_dof_default_pos = self._robot.data.default_joint_pos[:, self._arm_joint_ids]
        arm_deviation_penalty = torch.sum(torch.square(self._robot.data.joint_pos[:, self._arm_joint_ids] - arm_dof_default_pos), dim=-1)

        # Joint velocity penalty to encourage smooth motions
        joint_vel_penalty = torch.sum(torch.square(self.joint_vel), dim=-1)
        # Joint torque penalty to encourage low-effort motions - we can approximate torque with velocity for simplicity since we don't have access to torques in the environment
        joint_torque_penalty = torch.sum(torch.square(self._robot.data.applied_torque[:, self._arm_joint_ids]), dim=-1)
        # Action rate penalty to encourage smoother actions by penalizing large changes in actions between steps
        arm_action_rate_penalty = torch.sum(torch.square(self.arm_actions - self._previous_actions[:, :14]), dim=-1)
        # Action rate penalty for thruster and torque commands
        thruster_action_rate_penalty = torch.sum(torch.square(self.thruster_actions - self._previous_actions[:, 14:17]), dim=-1)
        torque_action_rate_penalty = torch.sum(torch.square(self.torque_actions - self._previous_actions[:, 17:20]), dim=-1)
        # Fuel consumption penalty
        fuel_penalty = torch.norm(self._applied_thruster_forces[:, 0, :].view(-1, 3), p=1, dim=-1) + torch.norm(self._applied_reaction_torques[:, 0, :].view(-1, 3), p=1, dim=-1)
        
        # Collision penalty
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        collision_penalty = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._undesired_collision_body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)

        rewards = {
            "target_linear_vel": target_lin_vel_penalty * self.cfg.target_linear_vel_rew_scale * self.step_dt,
            "target_angular_vel": target_ang_vel_penalty * self.cfg.target_angular_vel_rew_scale * self.step_dt,
            "base_linear_velocity": base_linear_vel_penalty * self.cfg.base_linear_velocity_rew_scale * self.step_dt,
            "base_angular_velocity": base_angular_vel_penalty * self.cfg.base_angular_velocity_rew_scale * self.step_dt,
            "target_linear_vel_fine_grained": target_lin_vel_fine_grained * self.cfg.target_lin_vel_fine_grained_rew_scale * self.step_dt,
            "target_angular_vel_fine_grained": target_ang_vel_fine_grained * self.cfg.target_ang_vel_fine_grained_rew_scale * self.step_dt,
            "base_linear_vel_fine_grained": base_lin_vel_fine_grained * self.cfg.base_lin_vel_fine_grained_rew_scale * self.step_dt,
            "base_angular_vel_fine_grained": base_ang_vel_fine_grained * self.cfg.base_ang_vel_fine_grained_rew_scale * self.step_dt,
            "target_lin_acc": target_lin_acc_penalty * self.cfg.target_lin_acc_rew_scale * self.step_dt,
            "target_ang_acc": target_ang_acc_penalty * self.cfg.target_ang_acc_rew_scale * self.step_dt,
            "base_lin_acc": base_lin_acc_penalty * self.cfg.base_lin_acc_rew_scale * self.step_dt,
            "base_ang_acc": base_ang_acc_penalty * self.cfg.base_ang_acc_rew_scale * self.step_dt,
            "arm_deviation": arm_deviation_penalty * self.cfg.arm_deviation_rew_scale * self.step_dt,
            "joint_velocity": joint_vel_penalty * self.cfg.joint_velocity_rew_scale * self.step_dt,
            "joint_torque": joint_torque_penalty * self.cfg.joint_torque_rew_scale * self.step_dt,
            "arm_action_rate": arm_action_rate_penalty * self.cfg.arm_action_rate_rew_scale * self.step_dt,
            "thruster_action_rate": thruster_action_rate_penalty * self.cfg.thruster_action_rate_rew_scale * self.step_dt,
            "torque_action_rate": torque_action_rate_penalty * self.cfg.torque_action_rate_rew_scale * self.step_dt,
            "fuel_consumption": fuel_penalty * self.cfg.fuel_consumption_rew_scale * self.step_dt,
            "collision": collision_penalty.float() * self.cfg.collision_rew_scale * self.step_dt,
        }
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Log episodic sums for each reward component
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check if maximum episode length has exceeded
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Check for collisions - we consider a collision to be when the net contact force on any of the undesired contact bodies exceeds a threshold of 1.0 N in any direction
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        collision = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._undesired_collision_body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        # Check if robot has lost grip from target
        # An episode is done if a collision occurs, the robot loses grip from the target, or the episode length is exceeded
        return collision, time_out  # terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # TODO: Logging
        # Log rewards
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # Log terminations
        extras = dict()
        extras["Termination/Collision"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Termination/TimeOut"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
        # Log metrics
        # TODO: Add target and base velocity metrics, and maybe arm deviation metric
        extras = dict()
        extras["Metrics/Target_Linear_Vel"] = torch.mean(self._target.data.root_link_vel_w[env_ids, :3].norm(dim=-1)).item()
        extras["Metrics/Target_Angular_Vel"] = torch.mean(self._target.data.root_link_vel_w[env_ids, 3:].norm(dim=-1)).item()
        extras["Metrics/Base_Linear_Vel"] = torch.mean(self._robot.data.root_lin_vel_w[env_ids].norm(dim=-1)).item()
        extras["Metrics/Base_Angular_Vel"] = torch.mean(self._robot.data.root_ang_vel_w[env_ids].norm(dim=-1)).item()
        extras["Curriculum/Curriculum_Factor"] = self.curriculum_factor * 100.0 # Log curriculum factor as percentage for better interpretability
        extras["Curriculum/Max_Lin_Vel"] = self.max_lin_vel
        extras["Curriculum/Max_Ang_Vel"] = self.max_ang_vel
        self.extras["log"].update(extras)

        # -- Reset procedure --
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        # Reset robot state
        # joints
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        # base
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Reset target state
        # pose
        target_init_pose = torch.zeros((len(env_ids), 7), device=self.device)
        target_init_pose[:, :3] = torch.tensor(self.cfg.target_cfg.init_state.pos, device=self.device)
        target_init_pose[:, 3:7] = torch.tensor(self.cfg.target_cfg.init_state.rot, device=self.device)
        # velocity
        target_init_vel = torch.zeros((len(env_ids), 6), device=self.device)

        # reset state in simulation
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._target.write_root_pose_to_sim(target_init_pose, env_ids)
        self._target.write_root_velocity_to_sim(target_init_vel, env_ids)

        # Reset action tensors
        self._target_joint_pos[env_ids] = self._robot.data.default_joint_pos[env_ids][:, self._arm_joint_ids].clone()
        self._applied_thruster_forces[env_ids] = torch.zeros((len(env_ids), 1, 3), device=self.device)
        self._applied_reaction_torques[env_ids] = torch.zeros((len(env_ids), 1, 3), device=self.device)
        self._previous_actions[env_ids] = torch.zeros((len(env_ids), 20), device=self.device)

    def update_target_motion(self, step: int = 0):
        inject_motion_idx = (self.episode_length_buf == step).float()  # Inject a new random target velocity shortly after the beginning of each episode after reset
        env_ids = torch.nonzero(inject_motion_idx, as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return
        else:
            # Curriculum math
            self.curriculum_factor = min(1.0, self.global_step_counter / self.max_curriculum_steps)
            self.max_lin_vel = self.cfg.max_target_lin_vel * self.curriculum_factor
            self.max_ang_vel = self.cfg.max_target_ang_vel * self.curriculum_factor
            r = torch.empty(len(env_ids), device=self.device)   # tensor for sampling
            target_init_vel = torch.zeros((len(env_ids), 6), device=self.device)

            # Compute target velocity
            if self.max_lin_vel == 0.0:
                target_init_vel[:, :3] = self._robot.data.root_lin_vel_w[env_ids]
            else:
                target_init_vel[:, 0] = (2.0 * r.uniform_() - 1.0) * self.max_lin_vel  # sample from [-max, max]
                target_init_vel[:, 1] = (2.0 * r.uniform_() - 1.0) * self.max_lin_vel
                target_init_vel[:, 2] = (2.0 * r.uniform_() - 1.0) * self.max_lin_vel
            if self.max_ang_vel == 0.0:
                target_init_vel[:, 3:] = self._robot.data.root_ang_vel_w[env_ids]
            else:
                target_init_vel[:, 3] = (2.0 * r.uniform_() - 1.0) * self.max_ang_vel
                target_init_vel[:, 4] = (2.0 * r.uniform_() - 1.0) * self.max_ang_vel
                target_init_vel[:, 5] = (2.0 * r.uniform_() - 1.0) * self.max_ang_vel
            
            # Apply velocity to sim
            self._target.write_root_velocity_to_sim(target_init_vel, env_ids)


    '''
    Debugging and visualization functions
    '''
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "thrust_marker") and not hasattr(self, "torque_marker"):
                red_arrow_marker = RED_ARROW_X_MARKER_CFG.copy()
                blue_arrow_marker = BLUE_ARROW_X_MARKER_CFG.copy()
                self.thrust_marker = VisualizationMarkers(red_arrow_marker.replace(prim_path="/Visuals/thrust"))
                self.torque_marker = VisualizationMarkers(blue_arrow_marker.replace(prim_path="/Visuals/torque"))
           
            # set their visibility to true
            self.thrust_marker.set_visibility(True)
            self.torque_marker.set_visibility(True)
        else:
            # set their visibility to false
            if hasattr(self, "thrust_marker"):
                self.thrust_marker.set_visibility(False)
            if hasattr(self, "torque_marker"):
                self.torque_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers for the current EE poses and target poses in world frame
        if self.cfg.debug_vis:
            # Update thrust marker
            thrust_marker_pos = self._robot.data.root_state_w[:, :3]
            thrust_marker_pos[:, 2] += 0.3  # offset the marker slightly above the base for better visibility
            thrust_norm = torch.norm(self._desired_thruster_forces[:, 0, :], dim=-1, keepdim=True)
            thrust_dir = self._desired_thruster_forces[:, 0, :] / (thrust_norm + 1e-6) # normalize to get direction, add small epsilon to avoid division by zero
            torque_norm = torch.norm(self._desired_reaction_torques[:, 0, :], dim=-1, keepdim=True)
            torque_dir = self._desired_reaction_torques[:, 0, :] / (torque_norm + 1e-6) # normalize to get direction, add small epsilon to avoid division by zero
            self.thrust_marker.visualize(
                thrust_marker_pos, 
                quat_from_euler_xyz(thrust_dir[:, 0], thrust_dir[:, 1], thrust_dir[:, 2]),
                scales=torch.tensor([1.0, 0.1, 0.1], device=self.device) * (thrust_norm + 1e-6) # scale the marker length by the magnitude of the thrust, add small epsilon to avoid zero scale
            ) 
            # Update torque marker
            torque_marker_pos = self._robot.data.root_state_w[:, :3]
            torque_marker_pos[:, 2] += 0.3  # offset the marker slightly above the base for better visibility
            self.torque_marker.visualize(
                torque_marker_pos, 
                quat_from_euler_xyz(torque_dir[:, 0], torque_dir[:, 1], torque_dir[:, 2]),
                scales=torch.tensor([1.0, 0.1, 0.1], device=self.device) * (torque_norm + 1e-6) # scale the marker length by the magnitude of the torque, add small epsilon to avoid zero scale
            )

    ''' 
     Custom functions for command generation and updates
    '''
    '''
    def _resample_ee_command(self, env_ids: torch.Tensor | None = None, make_quat_unique: bool = False):
        """Resample a new end-effector target pose command for all environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Tensor for sampling
        r = torch.empty(len(env_ids), device=self.device)    
        # Sample target positions
        self._target_ee_pose_left_b[env_ids, 0] = r.uniform_(self._target_pos_x_range[0], self._target_pos_x_range[1])
        self._target_ee_pose_left_b[env_ids, 1] = r.uniform_(self._target_pos_y_range[0], self._target_pos_y_range[1])
        self._target_ee_pose_left_b[env_ids, 2] = r.uniform_(self._target_pos_z_range[0], self._target_pos_z_range[1])
        self._target_ee_pose_right_b[env_ids, 0] = r.uniform_(self._target_pos_x_range[0], self._target_pos_x_range[1])
        self._target_ee_pose_right_b[env_ids, 1] = r.uniform_(self._target_pos_y_range[0], self._target_pos_y_range[1])
        self._target_ee_pose_right_b[env_ids, 2] = r.uniform_(self._target_pos_z_range[0], self._target_pos_z_range[1])

        # Target poses are relative to the default EE pose to ensure they are reachable and not colliding 
        self._target_ee_pose_left_b[env_ids, :3] += self._default_ee_pos_left_offset
        self._target_ee_pose_right_b[env_ids, :3] += self._default_ee_pos_right_offset

        # Sample target orientation in Euler angles and convert to quaternions
        # Left arm
        euler_angles = torch.zeros_like(self._target_ee_pose_left_b[env_ids, :3])
        euler_angles[:, 0] = r.uniform_(*self.cfg.target_roll_range) + self.cfg.default_ee_rot_left_offset[0]
        euler_angles[:, 1] = r.uniform_(*self.cfg.target_pitch_range) + self.cfg.default_ee_rot_left_offset[1]
        euler_angles[:, 2] = r.uniform_(*self.cfg.target_yaw_range) + self.cfg.default_ee_rot_left_offset[2]
        self._target_ee_pose_left_b[env_ids, 3:] = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        if make_quat_unique:
            self._target_ee_pose_left_b[env_ids, 3:] = quat_unique(self._target_ee_pose_left_b[env_ids, 3:])
        else:
            self._target_ee_pose_left_b[env_ids, 3:] = self._target_ee_pose_left_b[env_ids, 3:]

        # Right arm
        euler_angles = torch.zeros_like(self._target_ee_pose_right_b[env_ids, :3])
        euler_angles[:, 0] = r.uniform_(*self.cfg.target_roll_range) + self.cfg.default_ee_rot_right_offset[0]
        euler_angles[:, 1] = r.uniform_(*self.cfg.target_pitch_range) + self.cfg.default_ee_rot_right_offset[1]
        euler_angles[:, 2] = r.uniform_(*self.cfg.target_yaw_range) + self.cfg.default_ee_rot_right_offset[2]
        self._target_ee_pose_right_b[env_ids, 3:] = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        if make_quat_unique:
            self._target_ee_pose_right_b[env_ids, 3:] = quat_unique(self._target_ee_pose_right_b[env_ids, 3:])
        else:
            self._target_ee_pose_right_b[env_ids, 3:] = self._target_ee_pose_right_b[env_ids, 3:]

        # Base offset pose - follow offset computation as in https://arxiv.org/pdf/2210.10044
        pos_offset = self._robot.data.root_state_w[env_ids, :3].clone()
        # pos_offset[:, 2] = 0.5  # fixed height offset
        rot_offset = self._robot.data.root_pose_w[env_ids, 3:7].clone()
        euler_x, euler_y, euler_z = euler_xyz_from_quat(rot_offset)
        rot_offset = quat_from_euler_xyz(torch.zeros_like(euler_x), torch.zeros_like(euler_y), euler_z) #roll/pitch independent offset

        # Transform target poses to world frame
        self._target_ee_pose_left_w[env_ids, :3], self._target_ee_pose_left_w[env_ids, 3:] = combine_frame_transforms(
            pos_offset,
            rot_offset,
            self._target_ee_pose_left_b[env_ids, :3],
            self._target_ee_pose_left_b[env_ids, 3:],
        )
        self._target_ee_pose_right_w[env_ids, :3], self._target_ee_pose_right_w[env_ids, 3:] = combine_frame_transforms(
            pos_offset,
            rot_offset,
            self._target_ee_pose_right_b[env_ids, :3],
            self._target_ee_pose_right_b[env_ids, 3:],
        )
    
    def _update_command(self):
        # Update ee command in base frame at each step to account for changes in the base pose, for debu
        self._target_ee_pose_left_b[: , :3], self._target_ee_pose_left_b[:, 3:] = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_pose_w[:, 3:7],
            self._target_ee_pose_left_w[:, :3],
            self._target_ee_pose_left_w[:, 3:],
        )
        self._target_ee_pose_right_b[: , :3], self._target_ee_pose_right_b[:, 3:] = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_pose_w[:, 3:7],
            self._target_ee_pose_right_w[:, :3],
            self._target_ee_pose_right_w[:, 3:],
        )
    '''

    def update_current_ee_poses(self):
        # Get current EE poses in world frame
        self.ee_pose_left_w = self._robot.data.body_com_pose_w[:, self._left_ee_id].view(-1, 7)
        self.ee_pose_right_w = self._robot.data.body_com_pose_w[:, self._right_ee_id].view(-1, 7)
        curr_base_pos_w, curr_base_quat_w = self._robot.data.root_pose_w[:, :3], self._robot.data.root_pose_w[:, 3:7]
        # Convert to base frame
        self.ee_pos_left_b, self.ee_quat_left_b = subtract_frame_transforms(
            curr_base_pos_w,
            curr_base_quat_w,
            self.ee_pose_left_w[:, :3],
            self.ee_pose_left_w[:, 3:7],
        )
        self.ee_pose_left_b = torch.cat((self.ee_pos_left_b, self.ee_quat_left_b), dim=-1)
        self.ee_pos_right_b, self.ee_quat_right_b = subtract_frame_transforms(
            curr_base_pos_w,
            curr_base_quat_w,
            self.ee_pose_right_w[:, :3],
            self.ee_pose_right_w[:, 3:7],
        )
        self.ee_pose_right_b = torch.cat((self.ee_pos_right_b, self.ee_quat_right_b), dim=-1)

    '''
    Custom functions for reward computation
    '''
    def _compute_pose_tracking_reward(self, sigma_pos: float = 0.1, sigma_quat: float = 0.5):
        # Compute pose tracking reward based on distance between current EE pose and target EE pose
        pos_error_left = torch.norm(self._robot.data.body_com_pose_w[:, self._left_ee_id, :3].squeeze(1) - self._target_ee_pose_left_w[:, :3], dim=-1)
        pos_error_right = torch.norm(self._robot.data.body_com_pose_w[:, self._right_ee_id, :3].squeeze(1) - self._target_ee_pose_right_w[:, :3], dim=-1)
        quat_error_left = quat_error_magnitude(self._robot.data.body_com_pose_w[:, self._left_ee_id, 3:].squeeze(1), self._target_ee_pose_left_w[:, 3:])
        quat_error_right = quat_error_magnitude(self._robot.data.body_com_pose_w[:, self._right_ee_id, 3:].squeeze(1), self._target_ee_pose_right_w[:, 3:])
        # Compute the position and orientation rewards for each arm
        left_pos_reward = 1.0 - torch.tanh(torch.square(pos_error_left / sigma_pos))
        left_rot_reward = 1.0 - torch.tanh(quat_error_left / sigma_quat)
        right_pos_reward = 1.0 - torch.tanh(torch.square(pos_error_right / sigma_pos))
        right_rot_reward = 1.0 - torch.tanh(quat_error_right / sigma_quat)

        # Compute per-arm reward
        left_arm_reward = left_pos_reward * left_rot_reward
        right_arm_reward = right_pos_reward * right_rot_reward        
        
        return left_arm_reward + right_arm_reward
