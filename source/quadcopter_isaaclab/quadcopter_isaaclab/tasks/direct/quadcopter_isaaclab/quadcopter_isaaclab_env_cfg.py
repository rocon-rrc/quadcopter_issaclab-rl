# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg


import math


@configclass
class QuadcopterIsaaclabEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 100.0
    # - spaces definition
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True

    # robot(s)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            #usd_path="/home/amitabh/quadcopter_isaaclab/source/quadcopter_isaaclab/quadcopter_isaaclab/tasks/direct/quadcopter_isaaclab/assets/model/model.usd",
            usd_path="/home/amitabh/quadcopter_isaaclab/source/quadcopter_isaaclab/quadcopter_isaaclab/tasks/direct/quadcopter_isaaclab/assets/F450.SLDASM/F450.SLDASM.usd",
            #usd_path="/home/amitabh/quadcopter_isaaclab/source/quadcopter_isaaclab/quadcopter_isaaclab/tasks/direct/quadcopter_isaaclab/assets/Tarot 650 Assembly_urdf_wts.SLDASM/Tarot 650 Assembly_urdf_wts.SLDASM.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
                max_linear_velocity=5.0,  # Max linear speed in m/s
                max_angular_velocity= math.pi/4, # Max angular speed in rad/s (e.g., 2.5 revolutions/sec * 2*pi)
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                "Motor1": 200.0,
                "Motor2": -200.0,
                "Motor3": 200.0,
                "Motor4": -200.0,
            },
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    sim: SimulationCfg = SimulationCfg(
        dt= 1 / 100.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    thrust_to_weight = 2.25
    moment_scale = 1.0


    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    target_reward_scale = 10.0
    smooth_reward_scale = -5e-3       #CHANGED FROM A PREV VALUE OF -5e-4
    crash_reward_scale = -10.0
    near_reward_scale = 2.0
    stay_reward_scale = 10.0
    yaw_lambda = -10.0
    yaw_reward_scale = 0.01          #CHANGED FROM A PREV VALUE OF 0.01

    target_lambda = -6.0
    target_threshold = 0.05

    termination_pitch = 90
    termination_roll = 90
    termination_x = 11.0
    termination_y = 11.0
    termination_z = 6.0
    termination_ground_close = 0.3

    pos_x_range = [-10.0, 10.0]
    pos_y_range = [-10.0, 10.0]
    pos_z_range = [1.0, 5.0]

    lin_vel_obs_scale = 1/8
    ang_vel_obs_scale = 1/math.pi
    rel_pos_obs_scale = 1/8

    obs_noise = True
    noise_scale = 0.03

    dr_enabled = True
    mass_scale_range = [0.8, 1.2]       # Range for scaling the robot's mass [min, max]
    inertia_scale_range = [0.8, 1.2]
    actuator_scale_range = [0.8, 1.2]      # Range for scaling the actuator efficiency (mimics motor variations) [min, max]
    
    lin_vel_reward_scale = -0.01             #CHANGED FROM A PREV VALUE OF -0.01
    ang_vel_reward_scale = -0.02             #CHANGED FROM A PREV VALUE OF -0.02
    distance_to_goal_reward_scale = 15.0

    # Latency simulation
    action_latency_range = [1, 4]            #Example: [1, 4] means a random delay of 10, 20, or 30ms will be applied.