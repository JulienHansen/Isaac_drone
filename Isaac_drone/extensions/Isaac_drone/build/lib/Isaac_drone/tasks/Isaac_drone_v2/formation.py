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


    possible_agents = ["robot_1", "robot_2", "robot_3", "robot_4", "robot_5"]

    # Define action space for each robot (4 actions each in this case)
    action_spaces = {
        "robot_1": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_2": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_3": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_4": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_5": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
    }

    # Define observation space for each robot (for example, 60-dimensional observations)
    observation_spaces = {
        "robot_1": Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
        "robot_2": Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
        "robot_3": Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
        "robot_4": Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
        "robot_5": Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
    }

    # Overall state space (if required)
    state_space = 60 * len(possible_agents)  # For example, concatenating each robot's observation
    
    scene: InteractiveSceneCfg = QuadcopterScene(num_envs=10, env_spacing=15.0)
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
    debug_vis = True


class formationEnv(DirectMARLEnv):
    cfg: formationEnvCfg

    def __init__(self, cfg: formationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._robots = [self.scene[f"robot_{i+1}"] for i in range(len(self.cfg.possible_agents))]
        self._actions = torch.zeros(self.num_envs, len(self.cfg.possible_agents), 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, len(self.cfg.possible_agents), 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, len(self.cfg.possible_agents), 3, device=self.device)
        self.set_debug_vis(self.cfg.debug_vis)

    def _get_observations(self) -> dict:
        print("obs start")
        observations = {}
        for i, robot in enumerate(self._robots):
            state = torch.zeros(60, device=self.device)  # Dummy state
            # Check for NaN values in the observation and replace with zeros
            if torch.any(torch.isnan(state)):
                print(f"Warning: NaN detected in observation for robot_{i+1}. Replacing with zeros.")
                state = torch.zeros_like(state)
            observations[f"robot_{i+1}"] = state
        return observations

    def _get_states(self) -> torch.Tensor:
        print("state start")
        states = torch.zeros(self.num_envs, self.cfg.state_space, device=self.device)
        print(states)
        return states

    def _get_rewards(self) -> dict:
        print("reward start")
        rewards = torch.zeros(self.num_envs, len(self._robots), device=self.device)
        # Check for NaN values in the rewards and replace with zeros
        if torch.any(torch.isnan(rewards)):
            print("Warning: NaN detected in rewards. Replacing with zeros.")
            rewards = torch.zeros_like(rewards)
        rewards_dict = {
            f"robot_{i+1}": rewards[:, i]  # Assigning reward for each robot
            for i in range(len(self._robots))
        }
        return rewards_dict

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        print("done start")
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).bool()
        died = torch.zeros(self.num_envs, len(self._robots), device=self.device, dtype=torch.bool)
        terminated = {f"robot_{i+1}": died[:, i] for i in range(len(self._robots))}
        time_outs = {f"robot_{i+1}": time_out for i in range(len(self._robots))}
        return terminated, time_outs

    def _pre_physics_step(self, actions: dict[str, torch.tensor]) -> None:
        a = 2


    def _apply_action(self) -> None:
        a = 2



    def _reset_idx(self, env_ids: torch.Tensor | None):
        print("CA VAAAAA OU")
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES
        for i, robot in enumerate(self._robots):
            print("INSHALLAH")
            robot.reset(env_ids)
        super()._reset_idx(env_ids)
