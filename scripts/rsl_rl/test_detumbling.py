# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import adr_isaaclab.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import matplotlib.pyplot as plt

def plot_pose(pos_buff, rot_buff, title, dt=0.02):
    num_steps = pos_buff.shape[0]
    time_axis = [i * dt for i in range(num_steps)]
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.gca().set_prop_cycle('color', ['red', 'green', 'blue'])
    plt.plot(time_axis, pos_buff.cpu().numpy())
    plt.title(f"{title} Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend(["X", "Y", "Z"])
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.gca().set_prop_cycle('color', ['red', 'green', 'blue'])
    plt.plot(time_axis, rot_buff.cpu().numpy())
    plt.title(f"{title} Orientation")
    plt.xlabel("Time (s)")
    plt.ylabel("Orientation (rad)")
    plt.legend(["Roll", "Pitch", "Yaw"])
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_velocity(velocity_buff, title, dt=0.02, ylim=0.04):
    num_steps = velocity_buff.shape[0]
    time_axis = [i * dt for i in range(num_steps)]
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.gca().set_prop_cycle('color', ['red', 'green', 'blue'])
    plt.plot(time_axis, velocity_buff[:, :3].cpu().numpy())
    if ylim is not None:
        plt.ylim(bottom=-ylim, top=ylim)
    plt.title(f"{title} Linear Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Linear Velocity (m/s)")
    plt.legend(["X", "Y", "Z"])
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.gca().set_prop_cycle('color', ['red', 'green', 'blue'])
    plt.plot(time_axis, velocity_buff[:, 3:].cpu().numpy())
    plt.title(f"{title} Angular Velocity")
    if ylim is not None:
        plt.ylim(bottom=-ylim, top=ylim)
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.legend(["Roll", "Pitch", "Yaw"])
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_base_forces(thruster_cmd_buff, torque_cmd_buff, dt=0.02):
    num_steps = thruster_cmd_buff.shape[0]
    time_axis = [i * dt for i in range(num_steps)]
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.gca().set_prop_cycle('color', ['red', 'green', 'blue'])
    plt.plot(time_axis, thruster_cmd_buff.cpu().numpy())
    plt.title("Thruster Commands")
    plt.xlabel("Time (s)")
    plt.ylabel("Thruster Force (N)")
    plt.legend(["X", "Y", "Z"])
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.gca().set_prop_cycle('color', ['red', 'green', 'blue'])
    plt.plot(time_axis, torque_cmd_buff.cpu().numpy())
    plt.title("Reaction Torque Commands")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.legend(["Roll", "Pitch", "Yaw"])
    plt.grid()

    plt.tight_layout()
    plt.show()

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # Remove curriculum for testing
    env.unwrapped.max_curriculum_steps = 1

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # Buffers for logging
    target_pos_buff = torch.zeros((env.num_envs, env.unwrapped.max_episode_length, 3)) #
    target_rot_buff = torch.zeros((env.num_envs, env.unwrapped.max_episode_length, 3)) #
    target_vel_buff = torch.zeros((env.num_envs, env.unwrapped.max_episode_length, 6))  # [num_envs, max_episode_length, [linear_vel, angular_vel]]
    base_vel_buff = torch.zeros((env.num_envs, env.unwrapped.max_episode_length, 6)) # [num_envs, max_episode_length, [linear_vel, angular_vel
    thruster_cmd_buff = torch.zeros((env.num_envs, env.unwrapped.max_episode_length, 3)) # [num_envs, max_episode_length, 3] for logging thruster commands
    torque_cmd_buff = torch.zeros((env.num_envs, env.unwrapped.max_episode_length, 3)) # [num_envs, max_episode_length, 3] for logging torque commands

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            # Data logging for debugging and analysis
            target_pos_buff[:, timestep, :] = env.unwrapped._target.data.root_link_pose_w[:, :3]
            target_euler_x, target_euler_y, target_euler_z = euler_xyz_from_quat(env.unwrapped._target.data.root_link_pose_w[:, 3:])
            target_rot_euler = torch.stack((target_euler_x, target_euler_y, target_euler_z), dim=-1)
            target_rot_buff[:, timestep, :] = target_rot_euler
            target_vel_buff[:, timestep, :3] = env.unwrapped._target.data.root_link_vel_w[:, :3]
            target_vel_buff[:, timestep, 3:] = env.unwrapped._target.data.root_link_vel_w[:, 3:]
            base_vel_buff[:, timestep, :3] = env.unwrapped._robot.data.root_link_vel_w[:, :3]
            base_vel_buff[:, timestep, 3:] = env.unwrapped._robot.data.root_link_vel_w[:, 3:]

            thruster_cmd_buff[:, timestep, :] = env.unwrapped._applied_thruster_forces[:, 0, :] # Assuming last 6 actions are for thrusters and torques
            torque_cmd_buff[:, timestep, :] = env.unwrapped._applied_reaction_torques[:, 0, :]

            timestep += 1
            print(f"[INFO] Timestep: {timestep} / {env.unwrapped.max_episode_length}", end="\r")

        if timestep >= env.unwrapped.max_episode_length - 1:
            print(f"[INFO] Episode finished after {timestep} steps. Resetting environment.")
            break
        
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Plot data for debugging and analysis
    if env.num_envs == 1:
        plot_pose(target_pos_buff[0, :, :], target_rot_buff[0, :, :], title="Target Satellite")
        plot_velocity(target_vel_buff[0, :, :], title="Target Satellite", ylim=None)
        plot_velocity(base_vel_buff[0, :, :], title="Chaser Robot")
        plot_base_forces(thruster_cmd_buff[0, :, :], torque_cmd_buff[0, :, :])
    else:
        mean_target_lin_vel_error = torch.mean(torch.norm(target_vel_buff[:, -5, :3], dim=-1), dim=0)
        std_target_lin_vel_error = torch.std(torch.norm(target_vel_buff[:, -5, :3], dim=-1), dim=0)
        mean_target_ang_vel_error = torch.mean(torch.norm(target_vel_buff[:, -5, 3:], dim=-1), dim=0)
        std_target_ang_vel_error = torch.std(torch.norm(target_vel_buff[:, -5, 3:], dim=-1), dim=0)
        print(f"Mean target linear velocity error: {mean_target_lin_vel_error} +- {std_target_lin_vel_error}")
        print(f"Mean target angular velocity error: {mean_target_ang_vel_error} +- {std_target_ang_vel_error}")

        mean_base_lin_vel_error = torch.mean(torch.norm(base_vel_buff[:, -5, :3], dim=-1), dim=0)
        std_base_lin_vel_error = torch.std(torch.norm(base_vel_buff[:, -5, :3], dim=-1), dim=0)
        mean_base_ang_vel_error = torch.mean(torch.norm(base_vel_buff[:, -5, 3:], dim=-1), dim=0)
        std_base_ang_vel_error = torch.std(torch.norm(base_vel_buff[:, -5, 3:], dim=-1), dim=0)
        print(f"Mean base linear velocity error: {mean_base_lin_vel_error} +- {std_base_lin_vel_error}")
        print(f"Mean base angular velocity error: {mean_base_ang_vel_error} +- {std_base_ang_vel_error}")   
    
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()