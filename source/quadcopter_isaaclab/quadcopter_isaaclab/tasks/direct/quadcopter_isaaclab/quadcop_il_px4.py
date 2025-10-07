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

import numpy as np


##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

import math
from .quadcopter_isaaclab_env_cfg import QuadcopterIsaaclabEnvCfg  # isort: skip


class QuadcopPX4(DirectRLEnv):
    cfg: QuadcopterIsaaclabEnvCfg

    def __init__(self, cfg: QuadcopterIsaaclabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_space = gym.spaces.Box(low = -1.0, high =1.0, shape = (4,), dtype = np.float32)
        
        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.raw_actions = torch.zeros_like(self.actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                #"smooth_rew",
                #"yaw_rew",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        print(f"weight : {self._robot_weight}")

        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device)

        # --- DOMAIN RANDOMIZATION INITIALIZATION ---
        if self.cfg.dr_enabled:
            # Store base masses for randomization
            self._base_masses = self._robot.root_physx_view.get_masses().clone().to(self.device)
            # Create a buffer to hold the CURRENT randomized masses for all environments
            self._randomized_masses = self._base_masses.clone()
            
            # Buffers for mimicked DR
            # For thrust + 3 moments
            self._actuator_efficiency_scales = torch.ones(self.num_envs, 4, device=self.device)
        
            # Initial randomization for all environments at the start of training
            all_envs_idx = torch.arange(self.num_envs, device=self.device)
            self._apply_randomization(all_envs_idx)

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
        self.last_actions[:] = self.raw_actions[:]
        self.raw_actions = actions.clone().clamp(-1.0, 1.0)

        # Calculate thrust and moment from processed actions
        thrust_val = self._robot_weight * (self.raw_actions[:, 0] + 1.0) / 2.0
        moment_val = self.cfg.moment_scale * self.raw_actions[:, 1:]

        if self.cfg.dr_enabled:
            #(MIMIC KF/KM) Scale final thrust/moment to simulate actuator imperfections
            thrust_val *= self._actuator_efficiency_scales[:, 0]
            moment_val *= self._actuator_efficiency_scales[:, 1:4]

        self._thrust[:, 0, 2] = thrust_val
        self._moment[:, 0, :] = moment_val


    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        robot_quat_w = self._robot.data.body_quat_w[:, self._body_id].view(self.num_envs, 4)
        self.base_euler = torch.stack(euler_xyz_from_quat(robot_quat_w), dim=1)
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )

        if self.cfg.obs_noise:  # Add observation noise if enabled
            noise_level = self.cfg.noise_scale
            obs = self.add_observation_noise(obs, noise_level)
            
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        smooth_rew = torch.sum(torch.square(self.raw_actions - self.last_actions), dim=1)
        yaw = self.base_euler[:, 2]
        #print("yaw in deg" + str(yaw[45]))
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        yaw_rew = torch.exp(self.cfg.yaw_lambda * torch.abs(yaw))
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            #"smooth_rew": smooth_rew * self.cfg.smooth_reward_scale * self.step_dt,
            #"yaw_rew": yaw_rew * self.cfg.yaw_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if self.cfg.dr_enabled:
            self._apply_randomization(env_ids)

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
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
        self.raw_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _apply_randomization(self, env_ids: torch.Tensor):
        """Applies all DR techniques to the specified environments."""
        if len(env_ids) == 0:
            return
        
        # 1. Randomize mass in the simulator
        self._randomize_mass(env_ids)

        # 2. (MIMICKED DR) Generate new parameters for controller mimics
        mimicked_params = self._generate_mimicked_parameters(len(env_ids))

        if 'actuator_scales' in mimicked_params:
            self._actuator_efficiency_scales[env_ids] = mimicked_params['actuator_scales']

    def _randomize_mass(self, env_ids: torch.Tensor):
        """ (REAL DR) Randomizes the mass of the drone's bodies for specified environments."""
        if not hasattr(self.cfg, "mass_scale_range") or len(env_ids) == 0:
            return

        mass_range = self.cfg.mass_scale_range
        num_bodies = self._base_masses.shape[1]
        
        # Generate random scales for each body in each resetting environment
        scales = (mass_range[1] - mass_range[0]) * torch.rand(
            (len(env_ids), num_bodies), device=self.device
        ) + mass_range[0]
        
         # Calculate the new masses for these specific environments based on the non-randomized base mass
        new_masses_subset = self._base_masses[env_ids] * scales
    
        # Update the full mass buffer with the new values for the environments being reset
        self._randomized_masses[env_ids] = new_masses_subset
    
        all_envs_idx = torch.arange(self.num_envs, device=self.device)
        self._robot.root_physx_view.set_masses(self._randomized_masses.cpu(), indices = all_envs_idx.cpu())

    def _generate_mimicked_parameters(self, num_envs_to_reset: int) -> dict:
        """(MIMICKED DR) Generates randomized parameters for mimicking physics changes."""
        mimicked_params = {}

        # --- Mimic kf/km variations by generating actuator efficiency scales ---
        if hasattr(self.cfg, "actuator_scale_range"):
            actuator_range = self.cfg.actuator_scale_range
            # For thrust + 3 moments
            scales = (actuator_range[1] - actuator_range[0]) * torch.rand(
                (num_envs_to_reset, 4), device=self.device
            ) + actuator_range[0]
            mimicked_params['actuator_scales'] = scales

        return mimicked_params

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
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def add_observation_noise(self, obs_buf, noise_level=0.01):
        #Adds Gaussian noise to the observation buffer to mimic sensor noise.
        if noise_level > 0:
            noise = torch.randn_like(obs_buf) * noise_level
            return obs_buf + noise
        return obs_buf
