# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.utils.math import euler_xyz_from_quat

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

import math
from .quadcopter_isaaclab_env_cfg import QuadcopterIsaaclabEnvCfg  # isort: skip


class QuadcopterIsaaclabEnv(DirectRLEnv):
    cfg: QuadcopterIsaaclabEnvCfg

    def __init__(self, cfg: QuadcopterIsaaclabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self.pos_rel = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_rel_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # self.dt = 0.01   #not defined in the original script, and this is not being used as well, what is being used is self.step_dt

        #not defined goal position

        self.error_prev = torch.zeros((self.num_envs, 3), device=self.device)
        self.integral = torch.zeros((self.num_envs, 3), device=self.device)

        self.action_smoothing_alphas = torch.ones((self.num_envs, 3), device=self.device)
        self.last_smoothed_rates = torch.zeros((self.num_envs, 3), device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "target_rew",
                "smooth_rew",
                "crash_rew",
                "near_rew",
                "stay_rew",
            ]
        }
        # Get specific body indices
        self.num_commands = 3   
        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device)

        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.last_actions[:] = self._actions[:]
        self._actions = actions.clone().clamp(-1.0, 1.0)

        max_ang_vel = 2*math.pi
        desired_rates = self._actions[:, 1:4] * max_ang_vel
        
        # alphas = self.action_smoothing_alphas
        # smoothed_rates = alphas * desired_rates + (1.0 - alphas) * self.last_smoothed_rates
        # self.last_smoothed_rates[:] = smoothed_rates
        
        # --- Original controller logic ---
        #mass = 2.267 # Note: this is for the controller model, not the randomized sim mass
        #g = 9.81
        
        ang_vel = self._robot.data.root_ang_vel_b   #shud this be b?
        thrust = ((self._actions[:,0] + 1)/2)

        # Pass the smoothed rates to the PID controller
        Moment = self.pid(desired_rates, ang_vel, self.error_prev, self.integral)

        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * thrust 
        self._moment[:, 0, :] = self.cfg.moment_scale * Moment

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        robot_quat_w = self._robot.data.body_quat_w[:, self._body_id].view(self.num_envs, 4)
        self.base_euler = torch.stack(euler_xyz_from_quat(robot_quat_w), dim=1)

        self.last_rel_pos = self.pos_rel
        self.pos_rel = self.commands - self._robot.data.root_pos_w

        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.cfg.termination_pitch)
            | (torch.abs(self.base_euler[:, 0]) > self.cfg.termination_roll)
            | (torch.abs(self.pos_rel[:, 0]) > self.cfg.termination_x)
            | (torch.abs(self.pos_rel[:, 1]) > self.cfg.termination_y)
            | (torch.abs(self.pos_rel[:, 2]) > self.cfg.termination_z)
            | (self._robot.data.root_pos_w[:, 2] < self.cfg.termination_ground_close)
        )
        obs = torch.cat(
            [
                torch.clip(self._robot.data.root_lin_vel_w * self.cfg.lin_vel_obs_scale, -1, 1),
                torch.clip(self._robot.data.root_ang_vel_w * self.cfg.ang_vel_obs_scale, -1, 1),
                self._robot.data.root_quat_w,
                torch.clip(self.pos_rel * self.cfg.rel_pos_obs_scale, -1, 1),
                self.last_actions,
            ],
            dim=-1,
        )

        if self.cfg.obs_noise:  # Add observation noise if enabled
            noise_level = self.cfg.noise_scale
            obs = self.add_observation_noise(obs, noise_level)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.pos_rel), dim=1)
        smooth_rew = torch.sum(torch.square(self._actions - self.last_actions), dim=1)
        crash_rew = torch.zeros((self.num_envs,), device=self.device)
        crash_rew[self.crash_condition] = 1
        distance_to_target = torch.norm(self.pos_rel, dim=1)
        near_rew = torch.exp(self.cfg.target_lambda * distance_to_target)
        stay_rew = torch.where(distance_to_target < self.cfg.target_threshold, 1.0, 0.0)

        rewards = {
            "target_rew": target_rew * self.cfg.target_reward_scale * self.step_dt,
            "smooth_rew": smooth_rew * self.cfg.smooth_reward_scale * self.step_dt,
            "crash_rew": crash_rew * self.cfg.crash_reward_scale * self.step_dt,
            "near_rew": near_rew * self.cfg.near_reward_scale * self.step_dt,
            "stay_rew": stay_rew * self.cfg.stay_reward_scale * self.step_dt,
        }

        # if hasattr(self, 'step_count'):
        #     self.step_count += 1
        # else:
        #     self.step_count = 0
            
        # if self.step_count % 100 == 0:  # Log every 100 steps
        #     print(f"ee_reaching1: {rew_reaching_ee.mean().item()}") 
        #     print(f"rew_lifting: {rew_lifting.mean().item()}") 
        #     print(f"rew_goal_tracking_robot1: {rew_goal_tracking.mean().item()}") 
        #     print(f"rew_goal_tracking_fine1: {rew_goal_tracking_fine.mean().item()}") 
        #     print(f"orientation_reward1: {ee_orientation_reward.mean().item()}")
        #     print(f"pen_joint_vel1: {pen_joint_vel1.mean().item()}")
        #     print(f"pen_action1: {pen_action1.mean().item()}")
        #     print(f"total_reward1: {total_reward1.mean().item()}")




        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.crash_condition
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self.commands[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.last_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0

        self.error_prev[env_ids] = 0.0
        self.integral[env_ids] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._resample_commands(env_ids)

        self.pos_rel[env_ids] = self.commands[env_ids] - self._robot.data.root_pos_w[env_ids]
        self.last_rel_pos[env_ids] = self.pos_rel[env_ids]


    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self.commands)

    def pid(self, target, current, error_prev, integral):
    
        kp = torch.tensor([0.00005, 0.00005, 0.00003], device=self.device)
        kd = torch.tensor([0.0000005, 0.0000005, 0.0000003], device=self.device)
        ki = torch.tensor([0.0000005, 0.0000005, 0.0000003], device=self.device)
        
        error = target - current
        derivative = (error - error_prev) / self.step_dt
        integral = integral + error * self.step_dt

        #clamping integral to prevent windup
        integral = torch.clamp(integral, -10.0, 10.0)


        output = kp * error + kd * derivative + ki *integral
        
        self.error_prev[:] = error  # update previous error
        self.integral[:] = integral  # update integral term
        return output
    
    def gs_rand_float(self, lower, upper, shape, device):
        return (upper - lower) * torch.rand(size=shape, device=device) + lower
    
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = self.gs_rand_float(*self.cfg.pos_x_range, (len(envs_idx),), self.device) + self._terrain.env_origins[envs_idx, 0]
        self.commands[envs_idx, 1] = self.gs_rand_float(*self.cfg.pos_y_range, (len(envs_idx),), self.device) + self._terrain.env_origins[envs_idx, 1]
        self.commands[envs_idx, 2] = self.gs_rand_float(*self.cfg.pos_z_range, (len(envs_idx),), self.device) + self._terrain.env_origins[envs_idx, 2]

    def add_observation_noise(self, obs_buf, noise_level=0.01):
        #Adds Gaussian noise to the observation buffer to mimic sensor noise.
        if noise_level > 0:
            noise = torch.randn_like(obs_buf) * noise_level
            return obs_buf + noise
        return obs_buf

