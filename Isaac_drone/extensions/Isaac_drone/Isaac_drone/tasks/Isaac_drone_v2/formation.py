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
from collections.abc import Sequence


REGULAR_HEXAGON = [
    [0, 0, 0],
    [1.7321, -1, 0],
    [0, -2, 0],
    [-1.7321, -1, 0],
    [-1.7321, 1.0, 0],
    [0.0, 2.0, 0.0],
    [1.7321, 1.0, 0.0],
]


@configclass
class formationEnvCfg(DirectMARLEnvCfg):
    decimation = 2
    episode_length_s = 150.0
    action_scale = 100.0
    thrust_to_weight = 1.9
    moment_scale = 0.01
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    min_distance_threshold = 5.0


    possible_agents = ["robot_1", "robot_2", "robot_3", "robot_4", "robot_5", "robot_6"]

    # Define action space for each robot (4 actions each in this case)
    action_spaces = {
        "robot_1": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_2": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_3": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_4": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_5": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_6": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
    }
    individual_action_space = 4

    # Define observation space for each robot (for example, 60-dimensional observations)
    observation_spaces = {
        "robot_1": Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
        "robot_2": Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
        "robot_3": Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
        "robot_4": Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
        "robot_5": Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
        "robot_6": Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
    }

    # Overall state space (if required)
    state_space = 60 * len(possible_agents)  # TODO change that 
    
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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=5000, env_spacing=2.5, replicate_physics=True)
    robot_1 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_2 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_2")
    robot_3 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_3")
    robot_4 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_4")
    robot_5 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_5")
    robot_6 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_6")





class formationEnv(DirectMARLEnv):
    cfg: formationEnvCfg

    def __init__(self, cfg: formationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print("Initialization start")

        num_agents = len(self.cfg.possible_agents)
        self._actions = torch.zeros(self.num_envs, num_agents, self.cfg.individual_action_space, device=self.device)
        self._thrust = torch.zeros(self.num_envs, num_agents, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, num_agents, 3, device=self.device)


        # Logging for each robot
        self._episode_sums = {
            key: torch.zeros(self.num_envs, num_agents, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal"]
        }

        self._body_id1 = self._robot1.find_bodies("body")[0]
        self._body_id2 = self._robot2.find_bodies("body")[0]
        self._body_id3 = self._robot3.find_bodies("body")[0]
        self._body_id4 = self._robot4.find_bodies("body")[0]
        self._body_id5 = self._robot5.find_bodies("body")[0]
        self._body_id6 = self._robot6.find_bodies("body")[0]

        print("Body IDs set for all robots")

        self._robot_mass = self._robot1.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        print(f"Gravity and robot weight computed: {self._robot_weight}")


    def _get_observations(self) -> dict[str, torch.Tensor]:
        print("Get observations start")
        print("Actions in get obs", self._actions[0, 0, :])
        observations = {}
        for i, robot in enumerate(self._robots):
            state = torch.zeros(60, device=self.device)  # Dummy state
            if torch.any(torch.isnan(state)):
                print(f"Warning: NaN detected in observation for robot_{i+1}. Replacing with zeros.")
                state = torch.zeros_like(state)
            observations[f"robot_{i+1}"] = state
        print("Get observations done")
        return observations

    def _setup_scene(self):
        self._robot1 = Articulation(self.cfg.robot_1)
        self.scene.articulations["robot1"] = self._robot1
        self._robot2 = Articulation(self.cfg.robot_2)
        self.scene.articulations["robot2"] = self._robot2
        self._robot3 = Articulation(self.cfg.robot_3)
        self.scene.articulations["robot3"] = self._robot3
        self._robot4 = Articulation(self.cfg.robot_4)
        self.scene.articulations["robot4"] = self._robot4
        self._robot5 = Articulation(self.cfg.robot_5)
        self.scene.articulations["robot5"] = self._robot5
        self._robot6 = Articulation(self.cfg.robot_6)
        self.scene.articulations["robot6"] = self._robot6

        self._robots = [self._robot1, self._robot2, self._robot3, self._robot4, self._robot5, self._robot6]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_states(self) -> torch.Tensor:
        states = torch.zeros(self.num_envs, self.cfg.state_space, device=self.device)
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        print("Get rewards start")
        print("Actions in get reward", self._actions[0, 0, :])
        rewards = torch.zeros(self.num_envs, len(self._robots), device=self.device)
        if torch.any(torch.isnan(rewards)):
            print("Warning: NaN detected in rewards. Replacing with zeros.")
            rewards = torch.zeros_like(rewards)
        rewards_dict = {
            f"robot_{i+1}": rewards[:, i]
            for i in range(len(self._robots))
        }
        print("Get rewards done")
        return rewards_dict

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        print("Get dones start")
        print("Actions in get done", self._actions[0, 0, :])
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {}
        time_outs = {}

        # Get positions of all drones (shape: [n_drones, 3])
        drone_positions = torch.stack([robot.data.root_pos_w for robot in self._robots])  # Shape: (n_drones, batch_size, 3)

        # Compute pairwise distances (using broadcasting)
        for i, agent_name in enumerate(self.cfg.possible_agents):
            # Drone i's position
            pos_i = drone_positions[i]  # Shape: (batch_size, 3)

            # Compute distances to all other drones
            distances = torch.norm(drone_positions - pos_i.unsqueeze(0), dim=-1)  # Shape: (n_drones, batch_size)
            
            # Exclude self-distance by masking it (set diagonal to infinity)
            distances[i] = torch.inf
            
            # Check if any distance is below the threshold
            min_distance = distances.min(dim=0).values  # Min distance for each batch
            distance_violation = min_distance > self.cfg.min_distance_threshold  # Threshold check (e.g., 2.0 meters)
            print("====================================")
            print(distance_violation)

            # Altitude constraints
            low_altitude = self._robots[i].data.root_pos_w[:, 2] < 0.1
            high_altitude = self._robots[i].data.root_pos_w[:, 2] > 10
            altitude_violation = torch.logical_or(low_altitude, high_altitude)
            print(altitude_violation)
            # Combine conditions
            done_condition = torch.logical_or(altitude_violation, distance_violation)
            print(done_condition)
            # Save results
            terminated[agent_name] = done_condition
            time_outs[agent_name] = time_out

        print("Get dones done")
        return terminated, time_outs

    
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        print("Pre-physics step")
        for i, action_key in enumerate(actions.keys()):
            action = actions[action_key]
            print(f"Action for {action_key}: {action}")

            temp_act = action.clone().clamp(-1.0, 1.0)
            self._actions[:, i, :] = temp_act
            self._thrust[:, i, 2] = (self.cfg.thrust_to_weight * self._robot_weight * (temp_act[:, 0] + 1.0) / 2.0)
            self._moment[:, i, :] = self.cfg.moment_scale * temp_act[:, 1:]
    
    
    def _apply_action(self) -> None:
        print("Apply action start")
        print("Actions in apply action",self._actions[0, 0, :])
        self._robot1.set_external_force_and_torque(self._thrust[:, self._body_id1, :], self._moment[:, self._body_id1, :], body_ids=self._body_id1)
        self._robot2.set_external_force_and_torque(self._thrust[:, self._body_id2, :], self._moment[:, self._body_id2, :], body_ids=self._body_id2)
        self._robot3.set_external_force_and_torque(self._thrust[:, self._body_id3, :], self._moment[:, self._body_id3, :], body_ids=self._body_id3)
        self._robot4.set_external_force_and_torque(self._thrust[:, self._body_id4, :], self._moment[:, self._body_id4, :], body_ids=self._body_id4)
        self._robot5.set_external_force_and_torque(self._thrust[:, self._body_id5, :], self._moment[:, self._body_id5, :], body_ids=self._body_id5)
        self._robot6.set_external_force_and_torque(self._thrust[:, self._body_id6, :], self._moment[:, self._body_id6, :], body_ids=self._body_id6)
        print("Apply action done")
    
    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        print("Reset start")
        print("Actions in reset idx",self._actions[0, 0, :])
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot1._ALL_INDICES

        super()._reset_idx(env_ids)

        for i, agent_name in enumerate(self.cfg.possible_agents):

            joint_pos = self._robots[i].data.default_joint_pos[env_ids]
            joint_vel = self._robots[i].data.default_joint_vel[env_ids]
            default_root_state = self._robots[i].data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            self._robots[i].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self._robots[i].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self._robots[i].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        print("Reset done")
    

        


def hausdorff_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = torch.cdist(x, y, x=2).min(-1).values.max(-1).values
    return torch.max(torch.sqrt(((x - y) ** 2).sum(-1)), dim=-1).values

def formation_cost(pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    pos = pos - pos.mean(-2, keepdim=True)
    target_pos = target_pos - target_pos.mean(-2, keepdim=True)
    cost = torch.max(hausdorff_distance(pos, target_pos), hausdorff_distance(target_pos, pos))
    return cost.unsqueeze(-1)
    