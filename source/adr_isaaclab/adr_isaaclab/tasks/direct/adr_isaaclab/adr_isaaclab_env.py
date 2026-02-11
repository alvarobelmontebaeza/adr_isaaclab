# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import compute_pose_error, sample_uniform, subtract_frame_transforms, quat_error_magnitude, quat_from_euler_xyz, euler_xyz_from_quat, combine_frame_transforms, quat_unique

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


from .adr_isaaclab_env_cfg import AdrIsaaclabEnvCfg


class AdrIsaaclabEnv(DirectRLEnv):
    cfg: AdrIsaaclabEnvCfg

    def __init__(self, cfg: AdrIsaaclabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action buffers
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        # Action scales
        self._action_scale = torch.tensor(self.cfg.action_scale, device=self.device)
        # Action: Target joint positions
        self._target_joint_pos = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Buffers for end-effector poses
        self.ee_pose_left_w = torch.zeros((self.num_envs, 7), device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self.ee_pose_right_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.ee_pose_left_b = torch.zeros((self.num_envs, 7), device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self.ee_pose_right_b = torch.zeros((self.num_envs, 7), device=self.device)

        # -- Commands --
        self._default_ee_pos_left_offset = torch.tensor(self.cfg.default_ee_pos_left_offset, device=self.device)
        self._default_ee_pos_right_offset = torch.tensor(self.cfg.default_ee_pos_right_offset, device=self.device)
        # Target poses for each arm in base frame
        self._target_ee_pose_left_b = torch.zeros((self.num_envs, 7), device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self._target_ee_pose_right_b = torch.zeros((self.num_envs, 7), device=self.device)
        # Target poses for each arm in world frame
        self._target_ee_pose_left_w = torch.zeros((self.num_envs, 7), device=self.device)
        self._target_ee_pose_right_w = torch.zeros((self.num_envs, 7), device=self.device)
        
        # Target pose ranges
        self._target_pos_x_range = self.cfg.target_pos_x_range
        self._target_pos_y_range = self.cfg.target_pos_y_range
        self._target_pos_z_range = self.cfg.target_pos_z_range
        self._target_roll_range = self.cfg.target_roll_range
        self._target_pitch_range = self.cfg.target_pitch_range
        self._target_yaw_range = self.cfg.target_yaw_range

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "pose_tracking",
                "base_velocity",
                "joint_velocity",
                "joint_torque",
                "action_rate",
                "collision",
            ]
        }

        # Get indices of the bodies and joints of interest
        self._base_id, _ = self._contact_sensor.find_bodies(".*base")
        self._left_ee_id, _ = self._robot.find_bodies(".*gen3n7_left_end_effector_link")
        self._right_ee_id, _ = self._robot.find_bodies(".*gen3n7_right_end_effector_link")
        self._undesired_collision_body_ids, _ = self._contact_sensor.find_bodies([".*base", ".*link"])

        self._left_arm_joint_ids, _ = self._robot.find_joints(".*left_joint_.*")
        self._right_arm_joint_ids, _ = self._robot.find_joints(".*right_joint_.*")
        self._arm_joint_ids = torch.cat((torch.tensor(self._left_arm_joint_ids), torch.tensor(self._right_arm_joint_ids)), dim=0)

        # Add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        # Add robot and sensors to the scene
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

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
        self._actions = actions.clone()
        # process raw actions into target joint positions
        # apply scaling and add to default joint positions to obtain processed actions as target joint positions
        self._target_joint_pos = self._actions * self._action_scale + self._robot.data.default_joint_pos[:, self._arm_joint_ids]

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._target_joint_pos, joint_ids=self._arm_joint_ids)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # Base twist
        base_linear_vel = self._robot.data.root_lin_vel_b
        base_angular_vel = self._robot.data.root_ang_vel_b
        # Get joint positions and velocities for the joints of interest
        self.joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        self.joint_vel = self._robot.data.joint_vel[:, self._arm_joint_ids]

        # Get current EE poses
        self.update_current_ee_poses() # Retrieves current EE poses in world frame and converts to base frame
        # Last actions
        actions = self._previous_actions

        obs = torch.cat(
            (
                base_linear_vel, #3
                base_angular_vel, #3
                self.joint_pos, #14
                self.joint_vel, #14
                self.ee_pose_left_b, #7
                self.ee_pose_right_b, #7
                self._target_ee_pose_left_b,# 7
                self._target_ee_pose_right_b,# 7
                actions, # 14
            ),
            dim=-1, # Total: 76
        )

        # TODO: Consider adding asymmetrics actor-critic

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Task rewards
        pose_tracking_reward = self._compute_pose_tracking_reward()

        # Regularization rewards
        # Base velocity penalty to avoid body drift due to arms coupled motion
        base_vel_penalty = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=-1) + torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=-1)
        # Joint velocity penalty to encourage smooth motions
        joint_vel_penalty = torch.sum(torch.square(self.joint_vel), dim=-1)
        # Joint torque penalty to encourage low-effort motions - we can approximate torque with velocity for simplicity since we don't have access to torques in the environment
        joint_torque_penalty = torch.sum(torch.square(self._robot.data.applied_torque[:, self._arm_joint_ids]), dim=-1)
        # Action rate penalty to encourage smoother actions by penalizing large changes in actions between steps
        action_rate_penalty = torch.sum(torch.square(self._actions - self._previous_actions), dim=-1)

        # Collision penalty
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        collision_penalty = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._undesired_collision_body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)

        rewards = {
            "pose_tracking": pose_tracking_reward * self.cfg.pose_tracking_rew_scale * self.step_dt,
            "base_velocity": base_vel_penalty * self.cfg.base_velocity_rew_scale * self.step_dt,
            "joint_velocity": joint_vel_penalty * self.cfg.joint_velocity_rew_scale * self.step_dt,
            "joint_torque": joint_torque_penalty * self.cfg.joint_torque_rew_scale * self.step_dt,
            "action_rate": action_rate_penalty * self.cfg.action_rate_rew_scale * self.step_dt,
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
        
        # An episode is done when either a collision occurs or the episode length is exceeded
        return collision, time_out # terminated, truncated

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
        left_pos_error, left_rot_error = compute_pose_error(
            self._robot.data.body_com_pose_w[env_ids, self._left_ee_id, :3].view(-1,3),
            self._robot.data.body_com_pose_w[env_ids, self._left_ee_id, 3:7].view(-1,4),
            self._target_ee_pose_left_w[env_ids, :3],
            self._target_ee_pose_left_w[env_ids, 3:7],
        )
        right_pos_error, right_rot_error = compute_pose_error(
            self._robot.data.body_com_pose_w[env_ids, self._right_ee_id, :3].view(-1,3),
            self._robot.data.body_com_pose_w[env_ids, self._right_ee_id, 3:7].view(-1,4),
            self._target_ee_pose_right_w[env_ids, :3],
            self._target_ee_pose_right_w[env_ids, 3:7],
        )
        extras = dict()
        extras["Metrics/Left_EE_Pos_Error"] = torch.norm(left_pos_error, dim=-1)
        extras["Metrics/Left_EE_Rot_Error"] = torch.norm(left_rot_error, dim=-1)
        extras["Metrics/Right_EE_Pos_Error"] = torch.norm(right_pos_error, dim=-1)
        extras["Metrics/Right_EE_Rot_Error"] = torch.norm(right_rot_error, dim=-1)
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

        # reset state in simulation
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sample new commands
        self._resample_ee_command(env_ids=env_ids, make_quat_unique=True)
        self._update_command() # to update target EE poses in world frame based on the new base pose after reset

    '''
    Debugging and visualization functions
    '''
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "left_ee_marker") and not hasattr(self, "right_ee_marker") and not hasattr(self, "left_target_marker") and not hasattr(self, "right_target_marker"):
                frame_marker_cfg = FRAME_MARKER_CFG.copy()
                frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                self.left_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_ee_current"))
                self.right_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_ee_current"))
                self.left_target_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_ee_goal"))
                self.right_target_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_ee_goal"))
           
            # set their visibility to true
            self.left_ee_marker.set_visibility(True)
            self.right_ee_marker.set_visibility(True)
            self.left_target_marker.set_visibility(True)
            self.right_target_marker.set_visibility(True)
        else:
            if hasattr(self, "left_target_marker"):
                self.left_target_marker.set_visibility(False)
            if hasattr(self, "right_target_marker"):
                self.right_target_marker.set_visibility(False)
            if hasattr(self, "left_ee_marker"):
                self.left_ee_marker.set_visibility(False)
            if hasattr(self, "right_ee_marker"):
                self.right_ee_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers for the current EE poses and target poses in world frame
        self.left_ee_marker.visualize(self.ee_pose_left_w[:, :3], self.ee_pose_left_w[:, 3:7])
        self.right_ee_marker.visualize(self.ee_pose_right_w[:, :3], self.ee_pose_right_w[:, 3:7])
        self.left_target_marker.visualize(self._target_ee_pose_left_w[:, :3], self._target_ee_pose_left_w[:, 3:7])
        self.right_target_marker.visualize(self._target_ee_pose_right_w[:, :3], self._target_ee_pose_right_w[:, 3:7])

    ''' 
     Custom functions for command generation and updates
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
    def _compute_pose_tracking_reward(self, sigma_pos: float = 0.1, sigma_quat: float = 0.1):
        # Compute pose tracking reward based on distance between current EE pose and target EE pose
        pos_error_left = torch.norm(self._robot.data.body_com_pose_w[:, self._left_ee_id, :3].squeeze(1) - self._target_ee_pose_left_w[:, :3], dim=-1)
        pos_error_right = torch.norm(self._robot.data.body_com_pose_w[:, self._right_ee_id, :3].squeeze(1) - self._target_ee_pose_right_w[:, :3], dim=-1)
        quat_error_left = quat_error_magnitude(self._robot.data.body_com_pose_w[:, self._left_ee_id, 3:].squeeze(1), self._target_ee_pose_left_w[:, 3:])
        quat_error_right = quat_error_magnitude(self._robot.data.body_com_pose_w[:, self._right_ee_id, 3:].squeeze(1), self._target_ee_pose_right_w[:, 3:])
        
        # Compute the position and orientation rewards for each arm
        left_pos_reward = 1.0 - torch.tanh(pos_error_left / sigma_pos)
        left_rot_reward = 1.0 - torch.tanh(quat_error_left / sigma_quat)
        right_pos_reward = 1.0 - torch.tanh(pos_error_right / sigma_pos)
        right_rot_reward = 1.0 - torch.tanh(quat_error_right / sigma_quat)

        # Compute per-arm reward
        left_arm_reward = left_pos_reward + left_rot_reward
        right_arm_reward = right_pos_reward + right_rot_reward        
        
        return left_arm_reward + right_arm_reward


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward
