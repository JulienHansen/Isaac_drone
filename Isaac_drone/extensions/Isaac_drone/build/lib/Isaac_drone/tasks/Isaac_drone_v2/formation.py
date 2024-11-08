from __future__ import annotations

import torch
import numpy as np
import gymnasium as gym
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import (Camera,CameraCfg,RayCaster,RayCasterCfg,TiledCamera,TiledCameraCfg,patterns,RayCasterCamera,RayCasterCameraCfg)
from omni.isaac.lab_assets import CRAZYFLIE_CFG 
from omni.isaac.lab.markers import CUBOID_MARKER_CFG 
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg, TerrainImporter
from gymnasium.spaces import Box, Dict

@configclass
class QuadcopterScene(InteractiveSceneCfg):
    # Terrain
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
    
    # Define each robot individually
    robot_1 = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1", init_state=ArticulationCfg.InitialStateCfg(pos=(1, 1, 1)))
    robot_2 = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2", init_state=ArticulationCfg.InitialStateCfg(pos=(2, 1, 1)))
    robot_3 = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_3", init_state=ArticulationCfg.InitialStateCfg(pos=(3, 1, 1)))
    robot_4 = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_4", init_state=ArticulationCfg.InitialStateCfg(pos=(4, 1, 1)))
    robot_5 = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_5", init_state=ArticulationCfg.InitialStateCfg(pos=(5, 1, 1)))

    # Dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

@configclass
class formationEnvCfg(DirectMARLEnvCfg):
    # General MARL Environment Configurations
    decimation = 2
    episode_length_s = 15.0
    action_scale = 100.0
    num_states = 0  # Setting state space to None for simplicity
    debug_vis = True
    thrust_to_weight = 1.9
    moment_scale = 0.01
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    
    # Action and Observation Configuration for 5 Drones
    num_channels = 3
    observation_space = 12

    # Define Action and Observation Spaces for Each Drone
    action_spaces = {
        f"drone_{i+1}": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)  # 4-dimensional action space per drone
        for i in range(5)
    }
    observation_spaces = {
        f"drone_{i+1}": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)  # 12-dimensional observation space per drone
        for i in range(5)
    }

    # State Space and Possible Agents
    state_space = -1  # Concatenate all observations for the state space
    possible_agents = [f"drone_{i+1}" for i in range(5)]  # List of agent IDs

    # Define Scene Configuration
    scene: InteractiveSceneCfg = QuadcopterScene(
        num_envs=7000,
        env_spacing=15.0
    )

    # Simulation Configuration
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



class formationEnv(DirectMARLEnv):
    cfg: formationEnvCfg

    def __init__(self, cfg: formationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._robots = [self.scene[f"robot_{i+1}"] for i in range(len(self.cfg.possible_agents))]
        self._terrain = self.scene["terrain"]

        # Initialize actions, thrust, and moments for each drone
        self._actions = torch.zeros(self.num_envs, len(self.cfg.possible_agents), self.cfg.action_spaces["drone_1"].shape[0], device=self.device)
        self._thrust = torch.zeros(self.num_envs, len(self.cfg.possible_agents), 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, len(self.cfg.possible_agents), 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, len(self.cfg.possible_agents), 3, device=self.device)

        # Logging for each drone
        self._episode_sums = {
            key: torch.zeros(self.num_envs, len(self.cfg.possible_agents), dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal"]
        }

        # Robot-specific properties
        self._body_ids = [robot.find_bodies("body")[0] for robot in self._robots]
        self._robot_masses = [robot.root_physx_view.get_masses()[0].sum() for robot in self._robots]
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weights = [(mass * self._gravity_magnitude).item() for mass in self._robot_masses]

        self.set_debug_vis(self.cfg.debug_vis)

    def _pre_physics_step(self, actions: dict):
        # Update each drone's thrust and moment based on actions
        for i, agent_id in enumerate(self.cfg.possible_agents):
            action_tensor = torch.tensor(actions[agent_id], device=self.device)
            self._actions[:, i, :] = action_tensor.clamp(-1.0, 1.0)
            self._thrust[:, i, 2] = self.cfg.thrust_to_weight * self._robot_weights[i] * (self._actions[:, i, 0] + 1.0) / 2.0
            self._moment[:, i, :] = self.cfg.moment_scale * self._actions[:, i, 1:]



    def _apply_action(self):
        # Iterate over each agent and apply force and torque
        for i, robot in enumerate(self._robots):
            try:
                # Debug information for each agent
                print(f"Debug: Applying action for agent {i}")
                print(f"  Thrust shape: {self._thrust[:, i].shape}")
                print(f"  Moment shape: {self._moment[:, i].shape}")


                # Reshape and expand thrust and moment to match expected shape
                thrust = self._thrust[:, i].reshape(1000, 3).to(self.device)
                moment = self._moment[:, i].reshape(1000, 3).to(self.device)

                # Debug to confirm reshaping is as expected
                print(f"  Reshaped thrust shape: {thrust.shape}")
                print(f"  Reshaped moment shape: {moment.shape}")

                # Ensure body_ids are on the correct device
                if len(self._body_ids[i]) == 1:
                    # Replicate the single body ID to match the thrust's first dimension
                    body_ids = torch.tensor(self._body_ids[i], device=self.device).repeat(thrust.shape[0])
                    print(f"  Broadcasted Body IDs shape: {body_ids.shape}")
                elif len(self._body_ids[i]) == thrust.shape[0]:
                    body_ids = torch.tensor(self._body_ids[i], device=self.device)  # Use as-is if they match in size
                else:
                    raise RuntimeError("Body IDs count does not match thrust count")

                # Apply the action with thrust and moment, matching dimensions for body IDs
                robot.set_external_force_and_torque(
                    thrust,
                    moment,
                    body_ids=body_ids
                )
                print(f"  Applied action successfully for agent {i}")

            except RuntimeError as e:
                # Print the error message with debug information for further diagnosis
                print(f"Error applying action for agent {i}: {e}")
                print(f"Debug info - Thrust: {thrust.shape}, Moment: {moment.shape}, Body IDs: {body_ids.shape}")
                raise e






    def _get_observations(self) -> dict:
        # Collect observations for each drone
        observations = {}
        for i, robot in enumerate(self._robots):
            desired_pos_b, _ = subtract_frame_transforms(
                robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], self._desired_pos_w[:, i]
            )
            state = torch.cat([
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                robot.data.projected_gravity_b,
                desired_pos_b,
            ], dim=-1)
            observations[f"drone_{i+1}"] = state
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Calculate rewards for each drone
        rewards = torch.zeros(self.num_envs, len(self._robots), device=self.device)
        for i, robot in enumerate(self._robots):
            lin_vel = torch.sum(torch.square(robot.data.root_lin_vel_b), dim=1)
            ang_vel = torch.sum(torch.square(robot.data.root_ang_vel_b), dim=1)
            distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, i] - robot.data.root_pos_w, dim=1)
            distance_to_goal_mapped = 3 * (1 - torch.tanh(distance_to_goal / 20))
            drone_rewards = {
                "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
                "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
                "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            }
            rewards[:, i] = torch.sum(torch.stack(list(drone_rewards.values())), dim=0)
            # Logging
            for key, value in drone_rewards.items():
                self._episode_sums[key][:, i] += value
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Determine done conditions for each drone
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.stack([torch.logical_or(robot.data.root_pos_w[:, 2] < 0.1, robot.data.root_pos_w[:, 2] > 10.0)
                            for robot in self._robots], dim=1)
        return died.any(dim=1), time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

        for i, robot in enumerate(self._robots):
            final_distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[env_ids, i] - robot.data.root_pos_w[env_ids], dim=1
            ).mean()
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids, i])
                extras["Episode_Reward/" + key + f"_drone_{i+1}"] = episodic_sum_avg / self.max_episode_length_s
                self._episode_sums[key][env_ids, i] = 0.0
            self.extras["log"] = dict()
            self.extras["log"].update(extras)

            # Reset robot state
            robot.reset(env_ids)
        super()._reset_idx(env_ids)


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


