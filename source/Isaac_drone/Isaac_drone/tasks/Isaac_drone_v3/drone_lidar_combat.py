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
class CombatLidarEnvCfg(DirectMARLEnvCfg):
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
        "drone_defense": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32), # Need to be modified
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
    drone_attack = CRAZYFLIE_CFG.replace(prim_path = "/World/envs/env_.*/drone_attack")
    drone_defense = CRAZYFLIE_CFG.replace(prim_path = "/World/envs/env_.*/drone_defense")

class CombatLidarEnv(DirectMARLEnv):
    cfg: CombatLidarEnvCfg

    def __init__(self, cfg: CombatLidarEnvCfg, render_mode: str | None = None, **kwargs):
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

        #Goal position 
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

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
                torch.zeros_like(self._drone_defense.data.root_pos_w),
                torch.zeros_like(self._drone_defense.data.root_lin_vel_b),
                torch.zeros_like(self._drone_defense.data.root_ang_vel_b),
                torch.zeros_like(self._drone_defense.data.projected_gravity_b),
                torch.zeros_like(self._actions[:, 1, :])                     # TODO: check if that is needed
            ],
            dim=-1
        )
        # Build state for drone_defense:
        state_drone_defense = torch.cat(
            [
                torch.zeros_like(self._drone_defense.data.root_pos_w),
                torch.zeros_like(self._drone_defense.data.root_lin_vel_b),
                torch.zeros_like(self._drone_defense.data.root_ang_vel_b),
                torch.zeros_like(self._drone_defense.data.projected_gravity_b),
                torch.zeros_like(self._actions[:, 1, :])                     # TODO: check if that is needed
            ],
            dim=-1
        )
        return torch.cat([state_drone_attack, state_drone_defense], dim=-1)


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
                    torch.zeros_like(self._drone_defense.data.root_pos_w),
                ],
                dim=-1
            )
        }
        return obs


    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards = {}

        attacker_pos = self._drone_attack.data.root_pos_w
        defender_pos = self._drone_defense.data.root_pos_w
        distance_between_drones = torch.linalg.norm(attacker_pos - defender_pos, dim=1)

        # Compute rewards for drone_attack
        lin_vel = torch.sum(torch.square(self._drone_attack.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._drone_attack.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - attacker_pos, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        reward_attack = (
            lin_vel * self.cfg.lin_vel_reward_scale +
            ang_vel * self.cfg.ang_vel_reward_scale +
            distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale
        ) * self.step_dt

        # Assign rewards
        rewards["drone_attack"] = reward_attack
        rewards["drone_defense"] = torch.zeros_like(reward_attack)  # Placeholder for drone_defense

        return rewards


    
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated = {}
        time_outs = {}
        # Define a distance threshold either from your config or default to 1.0
        threshold = getattr(self.cfg, 'distance_threshold', 1.0)
        
        # Stack positions of all drones
        drone_pos = torch.stack([r.data.root_pos_w for r in self._robots])
        
        for i, name in enumerate(self.cfg.possible_agents):
            pos_i = drone_pos[i]
            # Compute distances from drone i to all drones
            dists = torch.norm(drone_pos - pos_i.unsqueeze(0), dim=-1)
            # Ignore self-distance by setting it to infinity
            dists[i] = float('inf')
            min_dist = dists.min(dim=0).values
            alt = self._robots[i].data.root_pos_w[:, 2]
            
            terminated[name] = (alt < 0.2) | (min_dist < threshold)
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
        ]
        for r, bid in zip(self._robots, body_ids):
            r.set_external_force_and_torque(
                self._thrust[:, bid, :],
                self._moment[:, bid, :],
                body_ids = bid
            )


    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._drone_attack._ALL_INDICES

        for robot in self._robots:
            robot.reset(env_ids)

        super()._reset_idx(env_ids)
        self._actions[env_ids] = 0.0

        # Sample new commands
        self._desired_pos_w[env_ids, 0] = 20.0
        self._desired_pos_w[env_ids, 1] = 0
        self._desired_pos_w[env_ids, 2] = 3.0

        for agent, robot in zip(self.cfg.possible_agents, self._robots):
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids].clone()


            if agent == "drone_attack":
                default_root_state[:, 0] = 0  
            elif agent == "drone_defense":
                default_root_state[:, 0] = 19  

            # Set y and z coordinates (y is randomized, z is fixed)
            default_root_state[:, 1] = 0.0
            default_root_state[:, 2] = 3.0
            
            '''
            default_root_state[:, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-5.0, 5.0)
            default_root_state[:, :2] += self._terrain.env_origins[env_ids, :2]
            default_root_state[:, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
            '''

            # Write the new states to simulation
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                 # Green goal marker 
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "drone_defense_cube_marker"):
                # Create a marker for the blue cube on top of the defense drone
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)  
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
                marker_cfg.prim_path = "/Visuals/Drone/cube"
                self.drone_defense_cube_marker = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "drone_attack_cube_marker"):
                # Create a marker for the red cube on top of the attack drone
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)  
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
                marker_cfg.prim_path = "/Visuals/Drone/cube"
                self.drone_attack_cube_marker = VisualizationMarkers(marker_cfg)

            self.goal_pos_visualizer.set_visibility(True)
            self.drone_defense_cube_marker.set_visibility(True)
            self.drone_attack_cube_marker.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "drone_defense_cube_marker"):
                self.drone_defense_cube_marker.set_visibility(False)
            if hasattr(self, "drone_attack_cube_marker"):
                self.drone_attack_cube_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the goal position marker
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

        # update the drones blue and red cube marker:
        drone_positions_defense = self._drone_defense.data.root_pos_w.clone()
        drone_positions_attack = self._drone_attack.data.root_pos_w.clone()

        # Add an upward offset so the cube sits on top of the drone
        offset = 0.1 
        drone_positions_defense[:, 2] += offset
        drone_positions_attack[:, 2] += offset

        self.drone_defense_cube_marker.visualize(drone_positions_defense)
        self.drone_attack_cube_marker.visualize(drone_positions_attack)