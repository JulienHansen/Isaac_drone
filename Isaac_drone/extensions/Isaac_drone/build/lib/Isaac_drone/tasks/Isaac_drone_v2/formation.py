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
from omni.isaac.lab.sensors import Camera, CameraCfg, RayCaster, RayCasterCfg, TiledCamera, TiledCameraCfg, patterns, RayCasterCamera, RayCasterCameraCfg
from omni.isaac.lab_assets import CRAZYFLIE_CFG 
from omni.isaac.lab.markers import CUBOID_MARKER_CFG 
from gymnasium.spaces import Box, Dict


@configclass
class QuadcopterScene(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )
    
    robot_1 = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
    robot_2 = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
    robot_3 = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_3")
    robot_4 = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_4")
    robot_5 = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_5")

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class formationEnvCfg(DirectMARLEnvCfg):
    decimation = 2
    episode_length_s = 15.0
    action_scale = 100.0
    thrust_to_weight = 1.9
    moment_scale = 0.01
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    num_channels = 3
    observation_space = 12
    action_spaces = {
        f"drone_{i+1}": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        for i in range(5)
    }
    observation_spaces = {
        f"drone_{i+1}": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        for i in range(5)
    }
    state_space = -1
    possible_agents = [f"drone_{i+1}" for i in range(5)]
    scene: InteractiveSceneCfg = QuadcopterScene(num_envs=10, env_spacing=15.0)
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)
    debug_vis = True


class formationEnv(DirectMARLEnv):
    cfg: formationEnvCfg

    def __init__(self, cfg: formationEnvCfg, render_mode: str | None = None, **kwargs):
        print("Initializing formationEnv...")
        super().__init__(cfg, render_mode, **kwargs)
        print(f"Super initialization complete. num_envs: {self.num_envs}")
        
        self._robots = [self.scene[f"robot_{i+1}"] for i in range(len(self.cfg.possible_agents))]
        self._actions = torch.zeros(self.num_envs, len(self.cfg.possible_agents), 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, len(self.cfg.possible_agents), 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, len(self.cfg.possible_agents), 3, device=self.device)
        self.set_debug_vis(self.cfg.debug_vis)
        print(f"Environment initialized with {len(self._robots)} robots.")
        
        # Debugging simulation configuration
        print(f"Simulation configuration: {self.cfg.sim}")
        print(f"Number of environments: {self.num_envs}")
        print(f"Maximum episode length: {self.max_episode_length}")

    def _pre_physics_step(self, actions: dict):
        print(f"Pre-physics step. Actions: {actions}")
        for i, agent_id in enumerate(self.cfg.possible_agents):
            action_tensor = torch.tensor(actions[agent_id], device=self.device)
            self._actions[:, i, :] = action_tensor.clamp(-1.0, 1.0)

    def _apply_action(self):
        print("Applying actions for all robots.")
        for i, robot in enumerate(self._robots):
            print(f"Debug: Applying action for agent {i}")

    def _get_observations(self) -> dict:
        print("Getting observations.")
        observations = {}
        for i, robot in enumerate(self._robots):
            state = torch.zeros(12, device=self.device)  # Dummy state
            print(f"Observation for drone_{i+1}: {state.shape}")
            observations[f"drone_{i+1}"] = state
        return observations

    def _get_rewards(self) -> torch.Tensor:
        print("Calculating rewards.")
        rewards = torch.zeros(self.num_envs, len(self._robots), device=self.device)
        print(f"Rewards: {rewards.shape}")
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        print("Checking if episode is done.")
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros(self.num_envs, len(self._robots), device=self.device)  # No drones die
        print(f"Died: {died.shape}, Time Out: {time_out.shape}")
        return died.any(dim=1), time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        print("Resetting environment indices.")
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES
        for i, robot in enumerate(self._robots):
            robot.reset(env_ids)
        super()._reset_idx(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            print("Debug visualization enabled.")
        else:
            print("Debug visualization disabled.")




