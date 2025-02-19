#python scripts/skrl/train.py --task=Isaac-drone-formation --algorithm MAPPO  --num_envs=200

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from gymnasium.spaces import Box
from collections.abc import Sequence
##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


@configclass
class DefenseEnvCfg(DirectMARLEnvCfg):
    decimation = 2
    episode_length_s = 150.0
    action_scale = 100.0
    thrust_to_weight = 1.9
    moment_scale = 0.01
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01

    possible_agents = [
        "drone_attack",
        "drone_defense",
    ]
    action_spaces = {
        "drone_attack": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "drone_defense": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
    }
    individual_action_space = 4
    observation_spaces = {
        "drone_attack": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32), # Need to be modified 
        "drone_defense": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32), # Need to be modified
    }
    state_space = 32
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
        num_envs = 1000,
        env_spacing = 2.5,
        replicate_physics = True
    )
    drone_attack = CRAZYFLIE_CFG.replace(prim_path = "/World/envs/env_.*/drone_attack")
    drone_defense = CRAZYFLIE_CFG.replace(prim_path = "/World/envs/env_.*/drone_defense")

class DefenseEnv(DirectMARLEnv):
    cfg: DefenseEnvCfg

    def __init__(self, cfg: DefenseEnvCfg, render_mode: str | None = None, **kwargs):
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
        self._body_id1 = self._drone_attack.find_bodies("body")[0]
        self._body_id2 = self._drone_defense.find_bodies("body")[0]

        self._robot_mass = self._drone_defense.root_physx_view.get_masses()[0].sum() # Same mass for both 
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device = self.device).norm()
        self._robot_weight = self._robot_mass * self._gravity_magnitude

    def _setup_scene(self):
        self._drone_attack = Articulation(self.cfg.drone_attack)
        self.scene.articulations["drone_attack"] = self._drone_attack
        self._drone_defense = Articulation(self.cfg.drone_defense)
        self.scene.articulations["drone_defense"] = self._drone_defense

        self._robots = [self._drone_attack, self._drone_defense]

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
        # Build state for drone_attack:
        state_drone_attack = torch.cat(
            [
                self._drone_attack.data.root_pos_w,      
                self._drone_attack.data.root_lin_vel_b,    
                self._drone_attack.data.root_ang_vel_b,    
                self._drone_attack.data.projected_gravity_b, 
                self._actions[:, 0, :]                     # TODO: check if that is needed
            ],
            dim=-1
        )
        # Build state for drone_defense:
        state_drone_defense = torch.cat(
            [
                self._drone_defense.data.root_pos_w,
                self._drone_defense.data.root_lin_vel_b,
                self._drone_defense.data.root_ang_vel_b,
                self._drone_defense.data.projected_gravity_b,
                self._actions[:, 1, :]                     # TODO: check if that is needed
            ],
            dim=-1
        )
        return torch.cat([state_drone_attack, state_drone_defense], dim=-1)


    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = {
            "drone_attack": torch.cat(
                [
                    self._drone_attack.data.root_pos_w,            # TODO: modify that 
                    self._drone_attack.data.root_lin_vel_b,
                    self._drone_attack.data.root_ang_vel_b,
                    self._drone_attack.data.projected_gravity_b
                ],
                dim=-1
            ),
            "drone_defense": torch.cat(
                [
                    torch.zero_like(self._drone_attack.data.root_pos_w),            # The defender know the position of the attacker
                    torch.zero_like(self._drone_defense.data.root_lin_vel_b),
                    torch.zero_like(self._drone_defense.data.root_ang_vel_b),
                    torch.zero_like(self._drone_defense.data.projected_gravity_b)
                ],
                dim=-1
            )
        }
        print(obs)
        return obs


    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards = {}
        for agent in self.cfg.possible_agents:
            if agent == "drone_attack":
                lin_vel = torch.sum(torch.square(self._drone_attack.data.root_lin_vel_b), dim=1)
                ang_vel = torch.sum(torch.square(self._drone_attack.data.root_ang_vel_b), dim=1)
                distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._drone_attack.data.root_pos_w, dim=1)
                distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
                reward = (
                    lin_vel * self.cfg.lin_vel_reward_scale +
                    ang_vel * self.cfg.ang_vel_reward_scale +
                    distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale
                ) * self.step_dt
            elif agent == "drone_defense":
                reward = torch.zeros((self.num_envs), self.device)
            else:
                raise ValueError(f"Unknown agent key: {agent}")
            
            rewards[agent] = reward
            # Logging
            #self._episode_sums[f"{agent}_reward"] += reward
        print(rewards)
        return rewards


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
            alt = self._robots[i].data.root_pos_w[:, 2]
            terminated[name] = (alt < 0.2) | (alt > 10) 
            time_outs[name] = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_outs

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # Expected keys: "drone_attack" and "drone_defense"
        for agent in self.cfg.possible_agents:
            act = actions[agent].clone().clamp(-1, 1)
            i = self.cfg.possible_agents.index(agent)
            self._actions[:, i, :] = act
            self._thrust[:, i, 2] = self.cfg.thrust_to_weight * self._robot_weight * ((act[:, 0] + 1) / 2)
            self._moment[:, i, :] = self.cfg.moment_scale * act[:, 1:]


    def _apply_action(self) -> None:
        for agent in self.cfg.possible_agents:
            i = self.cfg.possible_agents.index(agent)
            if agent == "drone_attack":
                robot_instance = self._drone_attack
                body_id = self._body_id1  
            elif agent == "drone_defense":
                robot_instance = self._drone_defense
                body_id = self._body_id2
            else:
                raise ValueError(f"Unknown agent key: {agent}")
            robot_instance.set_external_force_and_torque(
                self._thrust[:, i, :],
                self._moment[:, i, :],
                body_ids=body_id
            )


    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        # If no env_ids are provided, use the indices from one agent (they should be the same)
        if env_ids is None:
            env_ids = self._drone_attack._ALL_INDICES

        # Reset each drone
        for robot in self._robots:
            robot.reset(env_ids)

        # Call the superclass reset to perform additional resets if needed
        super()._reset_idx(env_ids)
        
        # Clear stored actions for these environments
        self._actions[env_ids] = 0.0

        # Reset each agent's state with different spawn positions
        for agent, robot in zip(self.cfg.possible_agents, self._robots):
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids].clone()

            # Set x-coordinate based on the agent type
            if agent == "drone_attack":
                default_root_state[:, 0] = -2  # Attacker spawns at x = -20
            elif agent == "drone_defense":
                default_root_state[:, 0] = 2   # Defender spawns at x = 20

            # Set y and z coordinates (y is randomized, z is fixed)
            default_root_state[:, 1] = 0.2
            default_root_state[:, 2] = 1

            # Write the new states to simulation
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)



    '''
    def _get_observations(self) -> dict[str, torch.Tensor]:
        desired_pos_b, _ = subtract_frame_transforms(
            self._drone_attack.data.root_state_w[:, :3], self._drone_attack.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        obs = {
            "drone_attack": torch.cat(
                [
                    desired_pos_b,          
                    self._drone_attack.data.root_lin_vel_b,
                    self._drone_attack.data.root_ang_vel_b,
                    self._drone_attack.data.projected_gravity_b
                ],
                dim=-1
            ),
            
            "drone_defense": torch.cat(
                [
                    self._drone_attack.data.root_pos_w,
                    self._drone_defense.data.root_lin_vel_b,
                    self._drone_defense.data.root_ang_vel_b,
                    self._drone_defense.data.projected_gravity_b
                ],
                dim=-1
            )
        }
        return obs
        '''
    '''
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # Still need to add huge penalty for going out of bound 
        rewards = {}

        attacker_pos = self._drone_attack.data.root_pos_w
        defender_pos = self._drone_defense.data.root_pos_w

        distance_between_drones = torch.linalg.norm(attacker_pos - defender_pos, dim=1)
        defenser_win = torch.where(distance_between_drones < .5, torch.tensor(10.0, device=distance_between_drones.device), torch.tensor(0.0, device=distance_between_drones.device))

        lin_vel_attack = torch.sum(torch.square(self._drone_attack.data.root_lin_vel_b), dim=1)
        ang_vel_attack = torch.sum(torch.square(self._drone_attack.data.root_ang_vel_b), dim=1)
        lin_vel_defense = torch.sum(torch.square(self._drone_defense.data.root_lin_vel_b), dim=1)
        ang_vel_defense = torch.sum(torch.square(self._drone_defense.data.root_ang_vel_b), dim=1)

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - attacker_pos, dim=1)
        attacker_win = torch.where(distance_to_goal < .5, torch.tensor(10.0, device=distance_to_goal.device), torch.tensor(0.0, device=distance_to_goal.device))

        reward_attack = (
            lin_vel_attack * self.cfg.lin_vel_reward_scale +
            ang_vel_attack * self.cfg.ang_vel_reward_scale +
            attacker_win - defenser_win
        ) * self.step_dt

        reward_defense = (
            lin_vel_defense * self.cfg.lin_vel_reward_scale +
            ang_vel_defense * self.cfg.ang_vel_reward_scale +
            defenser_win - attacker_win
        ) * self.step_dt

        rewards["drone_attack"] = reward_attack
        rewards["drone_defense"] = reward_defense  

        return rewards
    '''