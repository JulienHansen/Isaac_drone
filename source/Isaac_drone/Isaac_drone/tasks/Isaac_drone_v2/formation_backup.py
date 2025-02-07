#python scripts/skrl/train.py --task=Isaac-drone-formation --algorithm MAPPO  --num_envs=200

from __future__ import annotations

import torch
import numpy as np
import gymnasium as gym
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_assets import CRAZYFLIE_CFG
from gymnasium.spaces import Box
from collections.abc import Sequence

REGULAR_HEXAGON = [
    [0, 0, 0],
    [1.7321, -1, 0],
    [0, -2, 0],
    [-1.7321, -1, 0],
    [-1.7321, 1.0, 0.0],
    [0.0, 2.0, 0.0],
    [1.7321, 1.0, 0.0]
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
    possible_agents = [
        "robot_1",
        "robot_2",
        "robot_3",
        "robot_4",
        "robot_5",
        "robot_6"
    ]
    action_spaces = {
        "robot_1": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_2": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_3": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_4": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_5": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "robot_6": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    }
    individual_action_space = 4
    observation_spaces = {
        "robot_1": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
        "robot_2": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
        "robot_3": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
        "robot_4": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
        "robot_5": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
        "robot_6": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    }
    state_space = 96
    sim = SimulationCfg(
        dt = 1 / 100,
        render_interval = decimation,
        disable_contact_processing = True,
        physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode = "multiply",
            restitution_combine_mode = "multiply",
            static_friction = 1.0,
            dynamic_friction = 1.0,
            restitution = 0.0
        )
    )
    terrain = TerrainImporterCfg(
        prim_path = "/World/ground",
        terrain_type = "plane",
        collision_group = -1,
        physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode = "multiply",
            restitution_combine_mode = "multiply",
            static_friction = 1.0,
            dynamic_friction = 1.0,
            restitution = 0.0
        ),
        debug_vis = False
    )
    scene = InteractiveSceneCfg(
        num_envs = 5000,
        env_spacing = 2.5,
        replicate_physics = True
    )
    robot_1 = CRAZYFLIE_CFG.replace(prim_path = "/World/envs/env_.*/Robot_1")
    robot_2 = CRAZYFLIE_CFG.replace(prim_path = "/World/envs/env_.*/Robot_2")
    robot_3 = CRAZYFLIE_CFG.replace(prim_path = "/World/envs/env_.*/Robot_3")
    robot_4 = CRAZYFLIE_CFG.replace(prim_path = "/World/envs/env_.*/Robot_4")
    robot_5 = CRAZYFLIE_CFG.replace(prim_path = "/World/envs/env_.*/Robot_5")
    robot_6 = CRAZYFLIE_CFG.replace(prim_path = "/World/envs/env_.*/Robot_6")

class formationEnv(DirectMARLEnv):
    cfg: formationEnvCfg

    def __init__(self, cfg: formationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n_agents = len(cfg.possible_agents)
        self._actions = torch.zeros(self.num_envs, n_agents, cfg.individual_action_space, device = self.device)
        self._thrust = torch.zeros(self.num_envs, n_agents, 3, device = self.device)
        self._moment = torch.zeros(self.num_envs, n_agents, 3, device = self.device)
        self._episode_sums = {
            "lin_vel": torch.zeros(self.num_envs, n_agents, device = self.device),
            "ang_vel": torch.zeros(self.num_envs, n_agents, device = self.device),
            "distance_to_goal": torch.zeros(self.num_envs, n_agents, device = self.device)
        }
        self._body_id1 = self._robot1.find_bodies("body")[0]
        self._body_id2 = self._robot2.find_bodies("body")[0]
        self._body_id3 = self._robot3.find_bodies("body")[0]
        self._body_id4 = self._robot4.find_bodies("body")[0]
        self._body_id5 = self._robot5.find_bodies("body")[0]
        self._body_id6 = self._robot6.find_bodies("body")[0]
        self._robot_mass = self._robot1.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device = self.device).norm()
        self._robot_weight = self._robot_mass * self._gravity_magnitude
        self.formation = torch.tensor(REGULAR_HEXAGON[1:], device = self.device, dtype = torch.float)
        self.target_pos = torch.tensor([0.0, 0.0, 2.0], device = self.device, dtype = torch.float)
        self.safe_distance = 0.23
        self.last_cost_h = torch.zeros(self.num_envs, 1, device = self.device)
        self.last_cost_pos = torch.zeros(self.num_envs, 1, device = self.device)

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
        parts = []
        for i, r in enumerate(self._robots):
            part = torch.cat(
                [
                    r.data.root_pos_w,
                    r.data.root_lin_vel_b,
                    r.data.root_ang_vel_b,
                    r.data.projected_gravity_b,
                    self.actions["robot_" + str(i+1)]
                ],
                dim = -1
            )
            parts.append(part)
        return torch.cat(parts, dim = -1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = {}
        for i, r in enumerate(self._robots):
            obs["robot_" + str(i+1)] = torch.cat(
                [
                    r.data.root_pos_w,
                    r.data.root_lin_vel_b,
                    r.data.root_ang_vel_b,
                    r.data.projected_gravity_b
                ],
                dim = -1
            )
        return obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        positions = torch.stack(
            [r.data.root_pos_w for r in self._robots],
            dim = 1
        )
        cost_h = formation_cost(positions, target_pos = self.formation)
        formation_center = positions.mean(dim = 1, keepdim = True)
        distance = torch.norm(formation_center - self.target_pos, dim = -1)
        reward_formation = 1 / (1 + (cost_h * 1.6) ** 2)
        reward_pos = torch.exp(-distance)
        pairwise = torch.cdist(positions, positions, p = 2)
        mask = torch.eye(positions.shape[1], device = self.device).bool().unsqueeze(0)
        pairwise.masked_fill_(mask, float('inf'))
        min_sep = pairwise.min(dim = 2).values.min(dim = 1, keepdim = True).values
        reward_separation = (min_sep / self.safe_distance) ** 2
        reward = reward_separation * (
            reward_formation + reward_formation * reward_pos + 0.4 * reward_pos
        )
        self.last_cost_h[:] = cost_h
        self.last_cost_pos[:] = distance ** 2
        rewards = reward.expand(-1, len(self._robots))
        rewards = rewards.clone()
        rewards[torch.isnan(rewards)] = 0
        result = {}
        for i in range(len(self._robots)):
            result["robot_" + str(i+1)] = rewards[:, i]
        return result

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated = {}
        time_outs = {}
        drone_pos = torch.stack(
            [r.data.root_pos_w for r in self._robots]
        )
        for i, name in enumerate(self.cfg.possible_agents):
            pos_i = drone_pos[i]
            dists = torch.norm(drone_pos - pos_i.unsqueeze(0), dim = -1)
            dists[i] = float('inf')
            min_dist = dists.min(dim = 0).values
            violation = min_dist > self.cfg.min_distance_threshold
            alt = self._robots[i].data.root_pos_w[:, 2]
            terminated[name] = (alt < 0.1) | (alt > 10) | violation
            time_outs[name] = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_outs

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for i, key in enumerate(actions):
            act = actions[key].clone().clamp(-1, 1)
            self._actions[:, i, :] = act
            self._thrust[:, i, 2] = self.cfg.thrust_to_weight * self._robot_weight * (act[:, 0] + 1) / 2
            self._moment[:, i, :] = self.cfg.moment_scale * act[:, 1:]

    def _apply_action(self) -> None:
        body_ids = [
            self._body_id1,
            self._body_id2,
            self._body_id3,
            self._body_id4,
            self._body_id5,
            self._body_id6
        ]
        for r, bid in zip(self._robots, body_ids):
            r.set_external_force_and_torque(
                self._thrust[:, bid, :],
                self._moment[:, bid, :],
                body_ids = bid
            )

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot1._ALL_INDICES

        # Spread out the resets to avoid spikes in training when many environments reset at a similar time
        self.episode_length_buf[env_ids] = torch.randint(
            low=0,
            high=int(self.max_episode_length),
            size=(len(env_ids),), #type: ignore
            device=self.device
        )

        for r in self._robots:
            r.reset(env_ids)

        super()._reset_idx(env_ids)
        self._actions[env_ids] = 0

        for r in self._robots:
            jp = r.data.default_joint_pos[env_ids]
            jv = r.data.default_joint_vel[env_ids]
            drs = r.data.default_root_state[env_ids].clone()
            drs[:, :3] += self._terrain.env_origins[env_ids].to(self.device)
            r.write_root_pose_to_sim(drs[:, :7], env_ids)
            r.write_root_velocity_to_sim(drs[:, 7:], env_ids)
            r.write_joint_state_to_sim(jp, jv, None, env_ids)


def hausdorff_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if y.ndim == 2:
        y = y.unsqueeze(0)
    d = torch.cdist(x, y, p = 2)
    d_x = d.min(dim = -1).values.max(dim = -1).values
    d_y = d.min(dim = -2).values.max(dim = -1).values
    return torch.max(d_x, d_y)

def formation_cost(pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    pos = pos - pos.mean(dim = -2, keepdim = True)
    target_pos = target_pos - target_pos.mean(dim = -2, keepdim = True)
    if target_pos.ndim == 2:
        target_pos = target_pos.unsqueeze(0)
    cost = torch.max(
        hausdorff_distance(pos, target_pos),
        hausdorff_distance(target_pos, pos)
    )
    return cost.unsqueeze(-1)
