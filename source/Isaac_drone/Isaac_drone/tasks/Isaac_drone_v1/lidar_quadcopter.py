# python scripts/skrl/train.py --task=Isaac-lidar-drone --algorithm PPO  --enable_cameras  --num_envs=1000 --ml_framework torch
# Only work with skrl 1.4 

# TODO: Curriculum
# Faster domain exploration and exploitation 
#
#

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
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.sensors import (Camera,CameraCfg,RayCaster,RayCasterCfg,TiledCamera,TiledCameraCfg,ContactSensor,ContactSensorCfg, RayCaster, RayCasterCfg, patterns)
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from isaacsim.core.utils.viewports import set_camera_view
#from isaacsim.util.debug_draw import _debug_draw
import einops

'''
class DebugDraw:
    def __init__(self):
        self._draw = _debug_draw.acquire_debug_draw_interface()

    def clear(self):
        self._draw.clear_lines()

    def plot(self, x: torch.Tensor, size=2.0, color=(1., 1., 1., 1.)):
        if not (x.ndim == 2) and (x.shape[1] == 3):
            raise ValueError("x must be a tensor of shape (N, 3).")
        x = x.cpu()
        point_list_0 = x[:-1].tolist()
        point_list_1 = x[1:].tolist()
        sizes = [size] * len(point_list_0)
        colors = [color] * len(point_list_0)
        self._draw.draw_lines(point_list_0, point_list_1, colors, sizes)

    def vector(self, x: torch.Tensor, v: torch.Tensor, size=2.0, color=(0., 1., 1., 1.)):
        x = x.cpu().reshape(-1, 3)
        v = v.cpu().reshape(-1, 3)
        if not (x.shape == v.shape):
            raise ValueError("x and v must have the same shape, got {} and {}.".format(x.shape, v.shape))
        point_list_0 = x.tolist()
        point_list_1 = (x + v).tolist()
        sizes = [size] * len(point_list_0)
        colors = [color] * len(point_list_0)
        self._draw.draw_lines(point_list_0, point_list_1, colors, sizes)
'''

@configclass 
class LidarQuadcopterEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 600.0
    action_scale = 100.0
    debug_vis = True
    action_space = 4
    state_space = 0

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=5000, env_spacing=2.5, replicate_physics=True)

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=5,
            num_cols=5,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(8.0, 8.0),
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=40,
                    obstacle_height_mode="choice",
                    obstacle_width_range=(0.4, 0.8),
                    obstacle_height_range=(3.0, 4.0),
                    platform_width=1.5,
                )
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        debug_vis=False,
    )

    robot = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(1,1,1)))

    lidar_range = 4.0
    lidar_vfov = (
            max(-89., -10),
            min(89., 20)
        )
    ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Robot/body",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=False,
            pattern_cfg=patterns.BpearlPatternCfg(
                vertical_ray_angles=torch.linspace(*lidar_vfov, 4).tolist()
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )


    observation_space = {
    "robot-state": 12,
    "raycaster": [1, 36, 4],
    }
    
    thrust_to_weight = 1.9
    moment_scale = 0.01
    lin_vel_reward_scale =  -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    survive_reward_scale = 0.01


class LidarQuadcopterEnv(DirectRLEnv):

    cfg: LidarQuadcopterEnvCfg
    def __init__(self, cfg: LidarQuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._lidar._initialize_impl()
        self._lidar_resolution = (36, 4)

        self.lidar_range = self.cfg.lidar_range

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                #"survive",
            ]
        }
        #self.debug_draw = DebugDraw()

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.set_debug_vis(self.cfg.debug_vis)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._lidar = self.cfg.ray_caster_cfg.class_type(self.cfg.ray_caster_cfg)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)


    def _get_observations(self) -> dict:

        self.lidar_scan = self.lidar_range - (
            (self._lidar.data.ray_hits_w - self._lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self._lidar_resolution)
        )

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        state = torch.cat([
            self._robot.data.root_lin_vel_b,        
            self._robot.data.root_ang_vel_b,        
            self._robot.data.projected_gravity_b,   
            desired_pos_b,
        ], dim=-1)

        self.debug_draw.clear()
        
        x = self._lidar.data.pos_w[0]
        v = (self._lidar.data.ray_hits_w[0] - x).reshape(*self._lidar_resolution, 3)
        #self.debug_draw.vector(x.expand_as(v[:, 0]), v[:, 0])
        #self.debug_draw.vector(x.expand_as(v[:, -1]), v[:, -1])


        self._lidar.update(self.cfg.sim.dt)
        observation = {
            "policy": {
                "robot-state": state,
                "raycaster": self.lidar_scan,
            }
        }
        return observation

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 7 * (1 - torch.tanh(distance_to_goal / 20)) + 0.3
        #reward_safety = torch.log(self.lidar_range - self.lidar_scan).mean(dim=(1, 2, 3))
        
        
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            #"survive": reward_safety * self.cfg.survive_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Altitude constraints
        low_altitude = self._robot.data.root_pos_w[:, 2] < 0.1
        high_altitude = self._robot.data.root_pos_w[:, 2] > 3.5
        altitude = torch.logical_or(low_altitude, high_altitude)

        # X-Y Plane constraints
        x_constraint = torch.logical_or(self._robot.data.root_pos_w[:, 0] < -22, self._robot.data.root_pos_w[:, 0] > 22)
        y_constraint = torch.logical_or(self._robot.data.root_pos_w[:, 1] < -11, self._robot.data.root_pos_w[:, 1] > 11)
        xy_constraint = torch.logical_or(x_constraint, y_constraint)

        # Domain constraints
        domain_constraint = torch.logical_or(xy_constraint, altitude)

        distance_constraint = (einops.reduce(self.lidar_scan, "n 1 w h -> n", "max") > (self.lidar_range - 0.3))

        # Combine the constraints
        died = torch.logical_or(domain_constraint, distance_constraint)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

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
        if len(env_ids) == self.num_envs: #type: ignore
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, 0] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(19, 21)
        self._desired_pos_w[env_ids, 1] = torch.zeros_like(self._desired_pos_w[env_ids, 1]).uniform_(-1, 1)
        self._desired_pos_w[env_ids, 2] = 3.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, 0] = -20
        default_root_state[:, 1] = torch.zeros_like(default_root_state[:, 1]).uniform_(-10, 10)
        default_root_state[:, 2] = 2
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

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
