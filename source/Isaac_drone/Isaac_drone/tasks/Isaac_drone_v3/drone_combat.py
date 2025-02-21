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
from isaaclab.markers import VisualizationMarkers
from gymnasium.spaces import Box
from collections.abc import Sequence
##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

@configclass
class CombatEnvCfg(DirectMARLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0
    thrust_to_weight = 1.9
    moment_scale = 0.01
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    distance_from_defender_reward_scale = 5.0
    distance_to_attacker_reward_scale = 5.0
    debug_vis = True

    possible_agents = [
        "drone_attack_1",
        "drone_attack_2",
        "drone_attack_3",
        "drone_defense_1",
        "drone_defense_2",
        "drone_defense_3",
    ]
    action_spaces = {
        "drone_attack_1": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "drone_attack_2": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "drone_attack_3": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "drone_defense_1": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "drone_defense_2": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "drone_defense_3": Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
    }

    individual_action_space = 4

    observation_spaces = {
        "drone_attack_1": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32), 
        "drone_attack_2": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32), 
        "drone_attack_3": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32), 
        "drone_defense_1": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32), 
        "drone_defense_2": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32), 
        "drone_defense_3": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32), 
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
        env_spacing = 5,
        replicate_physics = True
    )

    drone_attack_1 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/drone_attack_1")
    drone_attack_2 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/drone_attack_2")
    drone_attack_3 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/drone_attack_3")

    drone_defense_1 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/drone_defense_1")
    drone_defense_2 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/drone_defense_2")
    drone_defense_3 = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/drone_defense_3")


class CombatEnv(DirectMARLEnv):
    cfg: CombatEnvCfg

    def __init__(self, cfg: CombatEnvCfg, render_mode: str | None = None, **kwargs):
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
        self._body_id1 = self._drone_attack_1.find_bodies("body")[0]
        self._body_id2 = self._drone_attack_2.find_bodies("body")[0]
        self._body_id3 = self._drone_attack_3.find_bodies("body")[0]

        self._body_id4 = self._drone_defense_1.find_bodies("body")[0]
        self._body_id5 = self._drone_defense_2.find_bodies("body")[0]
        self._body_id6 = self._drone_defense_3.find_bodies("body")[0]

        self._robot_mass = self._drone_defense_1.root_physx_view.get_masses()[0].sum() # Same mass for all
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device = self.device).norm()
        self._robot_weight = self._robot_mass * self._gravity_magnitude

        #Goal position 
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):

        self._drone_attack_1 = Articulation(self.cfg.drone_attack_1)
        self.scene.articulations["drone_attack_1"] = self._drone_attack_1
        self._drone_defense_1 = Articulation(self.cfg.drone_defense_1)
        self.scene.articulations["drone_defense_1"] = self._drone_defense_1

        self._drone_attack_2 = Articulation(self.cfg.drone_attack_2)
        self.scene.articulations["drone_attack_2"] = self._drone_attack_2
        self._drone_defense_2 = Articulation(self.cfg.drone_defense_2)
        self.scene.articulations["drone_defense_2"] = self._drone_defense_2

        self._drone_attack_3 = Articulation(self.cfg.drone_attack_3)
        self.scene.articulations["drone_attack_3"] = self._drone_attack_3
        self._drone_defense_3 = Articulation(self.cfg.drone_defense_3)
        self.scene.articulations["drone_defense_3"] = self._drone_defense_3

        self._robots = [self._drone_attack_1, self._drone_defense_1, self._drone_attack_2, self._drone_defense_2, self._drone_attack_3, self._drone_defense_3]

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
        state_drone_attack_1 = torch.cat(
            [
                self._drone_attack_1.data.root_pos_w,      
                self._drone_attack_1.data.root_lin_vel_b,    
                self._drone_attack_1.data.root_ang_vel_b,    
                self._drone_attack_1.data.projected_gravity_b, 
                self._actions[:, 0, :]                     # TODO: check if that is needed
            ],
            dim=-1
        )
        state_drone_attack_2 = torch.cat(
            [
                self._drone_attack_2.data.root_pos_w,      
                self._drone_attack_2.data.root_lin_vel_b,    
                self._drone_attack_2.data.root_ang_vel_b,    
                self._drone_attack_2.data.projected_gravity_b, 
                self._actions[:, 1, :]                     # TODO: check if that is needed
            ],
            dim=-1
        )
        state_drone_attack_3 = torch.cat(
            [
                self._drone_attack_3.data.root_pos_w,      
                self._drone_attack_3.data.root_lin_vel_b,    
                self._drone_attack_3.data.root_ang_vel_b,    
                self._drone_attack_3.data.projected_gravity_b, 
                self._actions[:, 2, :]                     # TODO: check if that is needed
            ],
            dim=-1
        )
        # Build state for drone_defense:
        state_drone_defense_1 = torch.cat(
            [
                self._drone_defense_1.data.root_pos_w,
                self._drone_defense_1.data.root_lin_vel_b,
                self._drone_defense_1.data.root_ang_vel_b,
                self._drone_defense_1.data.projected_gravity_b,
                self._actions[:, 3, :]                     # TODO: check if that is needed
            ],
            dim=-1
        )
        state_drone_defense_2 = torch.cat(
            [
                self._drone_defense_2.data.root_pos_w,
                self._drone_defense_2.data.root_lin_vel_b,
                self._drone_defense_2.data.root_ang_vel_b,
                self._drone_defense_2.data.projected_gravity_b,
                self._actions[:, 4, :]                     # TODO: check if that is needed
            ],
            dim=-1
        )
        state_drone_defense_3 = torch.cat(
            [
                self._drone_defense_3.data.root_pos_w,
                self._drone_defense_3.data.root_lin_vel_b,
                self._drone_defense_3.data.root_ang_vel_b,
                self._drone_defense_3.data.projected_gravity_b,
                self._actions[:, 5, :]                     # TODO: check if that is needed
            ],
            dim=-1
        )
        return torch.cat([state_drone_attack_1, state_drone_attack_2, state_drone_attack_3, state_drone_defense_1, state_drone_defense_2, state_drone_defense_3], dim=-1)


    def _get_observations(self) -> dict[str, torch.Tensor]:
        desired_pos_b, _ = subtract_frame_transforms(
            self._drone_attack_1.data.root_state_w[:, :3], self._drone_attack_1.data.root_state_w[:, 3:7], self._desired_pos_w # TODO: Need to be changed 
        )
        obs = {
            "drone_attack_1": torch.cat(
                [
                    desired_pos_b,          
                    self._drone_attack_1.data.root_lin_vel_b,
                    self._drone_attack_1.data.root_ang_vel_b,
                    self._drone_attack_1.data.projected_gravity_b
                ],
                dim=-1
            ),

            "drone_attack_2": torch.cat(
                [
                    desired_pos_b,          
                    self._drone_attack_2.data.root_lin_vel_b,
                    self._drone_attack_2.data.root_ang_vel_b,
                    self._drone_attack_2.data.projected_gravity_b
                ],
                dim=-1
            ),

            "drone_attack_3": torch.cat(
                [
                    desired_pos_b,          
                    self._drone_attack_3.data.root_lin_vel_b,
                    self._drone_attack_3.data.root_ang_vel_b,
                    self._drone_attack_3.data.projected_gravity_b
                ],
                dim=-1
            ),
            
            "drone_defense_1": torch.cat(
                [
                    desired_pos_b,  
                    self._drone_defense_1.data.root_lin_vel_b,
                    self._drone_defense_1.data.root_ang_vel_b,
                    self._drone_defense_1.data.projected_gravity_b
                ],
                dim=-1
            ),

            "drone_defense_2": torch.cat(
                [
                    desired_pos_b,  
                    self._drone_defense_2.data.root_lin_vel_b,
                    self._drone_defense_2.data.root_ang_vel_b,
                    self._drone_defense_2.data.projected_gravity_b
                ],
                dim=-1
            ),

            "drone_defense_3": torch.cat(
                [
                    desired_pos_b,  
                    self._drone_defense_3.data.root_lin_vel_b,
                    self._drone_defense_3.data.root_ang_vel_b,
                    self._drone_defense_3.data.projected_gravity_b
                ],
                dim=-1
            ),
        }
        return obs


    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # Still need to add huge penalty for going out of bound 
        rewards = {}

        attacker_pos = self._drone_attack_1.data.root_pos_w
        defender_pos = self._drone_defense_1.data.root_pos_w

        distance_between_drones = torch.linalg.norm(attacker_pos - defender_pos, dim=1)
        defenser_win = torch.where(distance_between_drones < .5, torch.tensor(10.0, device=distance_between_drones.device), torch.tensor(0.0, device=distance_between_drones.device))

        lin_vel_attack = torch.sum(torch.square(self._drone_attack_1.data.root_lin_vel_b), dim=1)
        ang_vel_attack = torch.sum(torch.square(self._drone_attack_1.data.root_ang_vel_b), dim=1)
        lin_vel_defense = torch.sum(torch.square(self._drone_defense_1.data.root_lin_vel_b), dim=1)
        ang_vel_defense = torch.sum(torch.square(self._drone_defense_1.data.root_ang_vel_b), dim=1)

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

        rewards["drone_attack_1"] = reward_attack
        rewards["drone_defense_1"] = reward_defense

        # TODO: modify this part ! 
        rewards["drone_attack_2"] = reward_attack
        rewards["drone_defense_2"] = reward_defense

        rewards["drone_attack_3"] = reward_attack
        rewards["drone_defense_3"] = reward_defense 

        return rewards


    
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated = {}
        time_outs = {}  
        for i, name in enumerate(self.cfg.possible_agents):
            alt = self._robots[i].data.root_pos_w[:, 2]
            terminated[name] = (alt < 0.2)

            # Terminate if maximum episode length is reached
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
        body_ids = [
            self._body_id1,
            self._body_id2,
            self._body_id3,
            self._body_id4,
            self._body_id5,
            self._body_id6,
        ]
        for r, bid in zip(self._robots, body_ids):
            r.set_external_force_and_torque(
                self._thrust[:, bid, :],
                self._moment[:, bid, :],
                body_ids = bid
            )


    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._drone_attack_1._ALL_INDICES

        # Reset each drone individually.
        self._drone_attack_1.reset(env_ids)
        self._drone_attack_2.reset(env_ids)
        self._drone_attack_3.reset(env_ids)
        self._drone_defense_1.reset(env_ids)
        self._drone_defense_2.reset(env_ids)
        self._drone_defense_3.reset(env_ids)

        # Call parent reset and clear actions.
        super()._reset_idx(env_ids)
        self._actions[env_ids] = 0.0

        # Set a common desired goal position (if needed).
        self._desired_pos_w[env_ids, 0] = 7.0
        self._desired_pos_w[env_ids, 1] = 0.0
        self._desired_pos_w[env_ids, 2] = 3.0

        # Map agent names to their corresponding drone objects.
        drone_mapping = {
            "drone_attack_1": self._drone_attack_1,
            "drone_attack_2": self._drone_attack_2,
            "drone_attack_3": self._drone_attack_3,
            "drone_defense_1": self._drone_defense_1,
            "drone_defense_2": self._drone_defense_2,
            "drone_defense_3": self._drone_defense_3,
        }

        # Define fixed spawn positions (x, y, z) for each drone.
        spawn_positions = {
            "drone_attack_1": (-5.0, -3.0, 3.0),
            "drone_attack_2": (-6.0, -3.0, 3.0),
            "drone_attack_3": (-7.0, -3.0, 3.0),
            "drone_defense_1": (5.0, 3.0, 3.0),
            "drone_defense_2": (6.0, 3.0, 3.0),
            "drone_defense_3": (7.0, 3.0, 3.0),
        }

        # For each drone, update its root state with the fixed spawn position.
        for agent, drone in drone_mapping.items():
            default_root_state = drone.data.default_root_state[env_ids].clone()
            pos = spawn_positions[agent]
            default_root_state[:, 0] = pos[0]  # X coordinate.
            default_root_state[:, 1] = pos[1]  # Y coordinate.
            default_root_state[:, 2] = pos[2]  # Z coordinate.

            drone.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            drone.write_joint_state_to_sim(
                drone.data.default_joint_pos[env_ids],
                drone.data.default_joint_vel[env_ids],
                None,
                env_ids
            )



    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            # Create separate markers with unique prim paths for each drone.
            for agent in self.cfg.possible_agents:
                prim_path = f"/Visuals/Markers/cube_{agent}"  # Unique prim path
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
                # Set color based on agent type.
                if "attack" in agent:
                    marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
                else:
                    marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
                marker_cfg.prim_path = prim_path  # Assign the unique prim path.
                setattr(self, f"{agent}_cube_marker", VisualizationMarkers(marker_cfg))
                getattr(self, f"{agent}_cube_marker").set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            for agent in self.cfg.possible_agents:
                marker_attr = f"{agent}_cube_marker"
                if hasattr(self, marker_attr):
                    getattr(self, marker_attr).set_visibility(False)



    def _debug_vis_callback(self, event):
        # update the goal position marker
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

        # For each agent, update its cube marker
        drone_positions = {
            "drone_attack_1": self._drone_attack_1.data.root_pos_w.clone(),
            "drone_attack_2": self._drone_attack_2.data.root_pos_w.clone(),
            "drone_attack_3": self._drone_attack_3.data.root_pos_w.clone(),
            "drone_defense_1": self._drone_defense_1.data.root_pos_w.clone(),
            "drone_defense_2": self._drone_defense_2.data.root_pos_w.clone(),
            "drone_defense_3": self._drone_defense_3.data.root_pos_w.clone(),
        }
        offset = 0.1  # upward offset
        for agent, pos in drone_positions.items():
            pos[:, 2] += offset
            marker = getattr(self, f"{agent}_cube_marker")
            marker.visualize(pos)
