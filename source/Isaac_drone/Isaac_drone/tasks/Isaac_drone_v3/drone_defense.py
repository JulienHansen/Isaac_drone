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
class DefenseEnvCfg(DirectMARLEnvCfg):
    decimation = 2
    episode_length_s = 50.0
    action_scale = 100.0
    thrust_to_weight = 1.9
    moment_scale = 0.01
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 10
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
        "drone_attack": Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32), # Need to be modified 
        "drone_defense": Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32), # Need to be modified
    }

    state_space = 38

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

class DefenseEnv(DirectMARLEnv):
    cfg: DefenseEnvCfg

    def __init__(self, cfg: DefenseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n_agents = len(cfg.possible_agents)
        self._actions = torch.zeros(self.num_envs, n_agents, cfg.individual_action_space, device = self.device)
        self._thrust = torch.zeros(self.num_envs, n_agents, 3, device = self.device)
        self._moment = torch.zeros(self.num_envs, n_agents, 3, device = self.device)

        self._episode_sums = {
            "drone_attack_distance": torch.zeros(self.num_envs, device=self.device),
            "drone_attack_lin_vel": torch.zeros(self.num_envs, device=self.device),
            "drone_attack_ang_vel": torch.zeros(self.num_envs, device=self.device),
            "drone_attack_win": torch.zeros(self.num_envs, device=self.device),
            "drone_attack_penalty": torch.zeros(self.num_envs, device=self.device),
            "drone_defense_distance": torch.zeros(self.num_envs, device=self.device),
            "drone_defense_lin_vel": torch.zeros(self.num_envs, device=self.device),
            "drone_defense_ang_vel": torch.zeros(self.num_envs, device=self.device),
            "drone_defense_win": torch.zeros(self.num_envs, device=self.device),
            "drone_defense_penalty": torch.zeros(self.num_envs, device=self.device),
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
                self._desired_pos_w,
                self._drone_attack.data.root_pos_w,
                self._drone_attack.data.root_lin_vel_b,
                self._drone_attack.data.root_ang_vel_b,
                self._drone_attack.data.projected_gravity_b,
                self._actions[:, 0, :]                    # TODO: check if that is needed
            ],
            dim=-1
        )
        # Build state for drone_defense:
        state_drone_defense = torch.cat(
            [
                self._desired_pos_w,
                self._drone_defense.data.root_pos_w,
                self._drone_defense.data.root_lin_vel_b,
                self._drone_defense.data.root_ang_vel_b,
                self._drone_defense.data.projected_gravity_b,
                self._actions[:, 1, :]                    # TODO: check if that is needed
            ],
            dim=-1
        )
        return torch.cat([state_drone_attack, state_drone_defense], dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        desired_pos_b, _ = subtract_frame_transforms(
            self._drone_attack.data.root_state_w[:, :3], self._drone_attack.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        relative_position, _ = subtract_frame_transforms(
            self._drone_defense.data.root_state_w[:, :3],  
            self._drone_defense.data.root_state_w[:, 3:7], 
            self._drone_attack.data.root_state_w[:, :3],   
            self._drone_attack.data.root_state_w[:, 3:7] 
        )

        obs = {
            "drone_attack": torch.cat(
                [
                    desired_pos_b,
                    self._drone_attack.data.root_pos_w,           
                    self._drone_attack.data.root_lin_vel_b,
                    self._drone_attack.data.root_ang_vel_b,
                    self._drone_attack.data.projected_gravity_b
                ],
                dim=-1
            ),
            
            "drone_defense": torch.cat(
                [
                    relative_position,
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
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # --------------------- Drone Attack Components --------------------- #
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._drone_attack.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 5*(1 - torch.tanh(distance_to_goal / 5))
        attack_distance_term = distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt

        lin_vel_attack = torch.sum(torch.square(self._drone_attack.data.root_lin_vel_b), dim=1)
        attack_lin_vel_term = lin_vel_attack * self.cfg.lin_vel_reward_scale * self.step_dt

        ang_vel_attack = torch.sum(torch.square(self._drone_attack.data.root_ang_vel_b), dim=1)
        attack_ang_vel_term = ang_vel_attack * self.cfg.ang_vel_reward_scale * self.step_dt

        # Win bonus
        attacker_win = torch.where(
            distance_to_goal < 0.5,
            torch.tensor(10.0, device=distance_to_goal.device),
            torch.tensor(0.0, device=distance_to_goal.device)
        )
        attack_win_term = attacker_win * self.step_dt

        # --------------------- Drone Defense Components --------------------- #
        attacker_pos = self._drone_attack.data.root_pos_w
        defender_pos = self._drone_defense.data.root_pos_w
        distance_between_drones = torch.linalg.norm(attacker_pos - defender_pos, dim=1)
        distance_to_attacker_mapped = 5*(1 - torch.tanh(distance_between_drones / 5))
        defense_distance_term = distance_to_attacker_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt

        lin_vel_defense = torch.sum(torch.square(self._drone_defense.data.root_lin_vel_b), dim=1)
        defense_lin_vel_term = lin_vel_defense * self.cfg.lin_vel_reward_scale * self.step_dt

        ang_vel_defense = torch.sum(torch.square(self._drone_defense.data.root_ang_vel_b), dim=1)
        defense_ang_vel_term = ang_vel_defense * self.cfg.ang_vel_reward_scale * self.step_dt

        # Win bonus
        defenser_win = torch.where(
            distance_between_drones < 0.5,
            torch.tensor(10.0, device=distance_between_drones.device),
            torch.tensor(0.0, device=distance_between_drones.device)
        )
        defense_win_term = defenser_win * self.step_dt

        # --------------------- Penalty Terms --------------------- #
        attack_penalty = -defenser_win * self.step_dt
        defense_penalty = -attacker_win * self.step_dt

        # --------------------- Total Rewards --------------------- #
        total_attack = (attack_distance_term +
                        attack_lin_vel_term +
                        attack_ang_vel_term +
                        attack_win_term +
                        attack_penalty)

        total_defense = (defense_distance_term +
                        defense_lin_vel_term +
                        defense_ang_vel_term +
                        defense_win_term +
                        defense_penalty)

        self._episode_sums["drone_attack_distance"] += attack_distance_term
        self._episode_sums["drone_attack_lin_vel"] += attack_lin_vel_term
        self._episode_sums["drone_attack_ang_vel"] += attack_ang_vel_term
        self._episode_sums["drone_attack_win"] += attack_win_term
        self._episode_sums["drone_attack_penalty"] += attack_penalty

        self._episode_sums["drone_defense_distance"] += defense_distance_term
        self._episode_sums["drone_defense_lin_vel"] += defense_lin_vel_term
        self._episode_sums["drone_defense_ang_vel"] += defense_ang_vel_term
        self._episode_sums["drone_defense_win"] += defense_win_term
        self._episode_sums["drone_defense_penalty"] += defense_penalty

        rewards = {
            "drone_attack": total_attack,
            "drone_defense": total_defense,
        }
        return rewards
    '''
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # --------------------- Drone Attack Components --------------------- #
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._drone_attack.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 5)
        attack_distance_term = distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt

        lin_vel_attack = torch.sum(torch.square(self._drone_attack.data.root_lin_vel_b), dim=1)
        attack_lin_vel_term = lin_vel_attack * self.cfg.lin_vel_reward_scale * self.step_dt

        ang_vel_attack = torch.sum(torch.square(self._drone_attack.data.root_ang_vel_b), dim=1)
        attack_ang_vel_term = ang_vel_attack * self.cfg.ang_vel_reward_scale * self.step_dt

        # Base win bonus (10.0) for attacker if near the goal
        attacker_win_flag = distance_to_goal < 0.5
        attack_win_term = torch.where(
            attacker_win_flag,
            torch.tensor(10.0, device=distance_to_goal.device) * self.step_dt,
            torch.tensor(0.0, device=distance_to_goal.device)
        )

        # --------------------- Drone Defense Components --------------------- #
        attacker_pos = self._drone_attack.data.root_pos_w
        defender_pos = self._drone_defense.data.root_pos_w
        distance_between_drones = torch.linalg.norm(attacker_pos - defender_pos, dim=1)
        distance_to_attacker_mapped = 1 - torch.tanh(distance_between_drones / 5)
        defense_distance_term = distance_to_attacker_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt

        lin_vel_defense = torch.sum(torch.square(self._drone_defense.data.root_lin_vel_b), dim=1)
        defense_lin_vel_term = lin_vel_defense * self.cfg.lin_vel_reward_scale * self.step_dt

        ang_vel_defense = torch.sum(torch.square(self._drone_defense.data.root_ang_vel_b), dim=1)
        defense_ang_vel_term = ang_vel_defense * self.cfg.ang_vel_reward_scale * self.step_dt

        # Base win bonus for defender if drones are very close
        defender_win_flag = distance_between_drones < 0.5
        defense_win_term = torch.where(
            defender_win_flag,
            torch.tensor(10.0, device=distance_between_drones.device) * self.step_dt,
            torch.tensor(0.0, device=distance_between_drones.device)
        )

        # --------------------- Penalty Terms --------------------- #
        # Penalties: if one wins the other gets a penalty equal in magnitude to the win bonus.
        attack_penalty = -defender_win_flag.to(torch.float32) * self.step_dt * 10.0
        defense_penalty = -attacker_win_flag.to(torch.float32) * self.step_dt * 10.0

        # --------------------- Total Rewards --------------------- #
        total_attack = (attack_distance_term +
                        attack_lin_vel_term +
                        attack_ang_vel_term +
                        attack_win_term +
                        attack_penalty)

        total_defense = (defense_distance_term +
                        defense_lin_vel_term +
                        defense_ang_vel_term +
                        defense_win_term +
                        defense_penalty)

        # --------------------- Override with Huge Rewards/Penalties on Win --------------------- #
        # When a win condition is met, give a huge reward to the winner and huge penalty to the other.
        huge_reward = torch.tensor(100.0, device=distance_to_goal.device) * self.step_dt
        huge_penalty = -huge_reward

        # If the attacker wins, override rewards:
        total_attack = torch.where(attacker_win_flag, huge_reward, total_attack)
        total_defense = torch.where(attacker_win_flag, huge_penalty, total_defense)
        # If the defender wins, override rewards:
        total_attack = torch.where(defender_win_flag, huge_penalty, total_attack)
        total_defense = torch.where(defender_win_flag, huge_reward, total_defense)

        # --------------------- Logging --------------------- #
        self._episode_sums["drone_attack_distance"] += attack_distance_term
        self._episode_sums["drone_attack_lin_vel"] += attack_lin_vel_term
        self._episode_sums["drone_attack_ang_vel"] += attack_ang_vel_term
        self._episode_sums["drone_attack_win"] += attack_win_term
        self._episode_sums["drone_attack_penalty"] += attack_penalty

        self._episode_sums["drone_defense_distance"] += defense_distance_term
        self._episode_sums["drone_defense_lin_vel"] += defense_lin_vel_term
        self._episode_sums["drone_defense_ang_vel"] += defense_ang_vel_term
        self._episode_sums["drone_defense_win"] += defense_win_term
        self._episode_sums["drone_defense_penalty"] += defense_penalty

        rewards = {
            "drone_attack": total_attack,
            "drone_defense": total_defense,
        }
        return rewards



    '''
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated = {}
        time_outs = {}
        for i, name in enumerate(self.cfg.possible_agents):
            alt = self._robots[i].data.root_pos_w[:, 2]
            terminated[name] = (alt < 0.2)

            # Terminate if maximum episode length is reached
            time_outs[name] = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_outs
    '''

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated = {}
        time_outs = {}

        # Compute win conditions (same as in rewards)
        attacker_win_flag = torch.linalg.norm(self._desired_pos_w - self._drone_attack.data.root_pos_w, dim=1) < 0.5
        defender_win_flag = torch.linalg.norm(self._drone_attack.data.root_pos_w - self._drone_defense.data.root_pos_w, dim=1) < 0.5
        win_flag = attacker_win_flag | defender_win_flag

        for i, name in enumerate(self.cfg.possible_agents):
            alt = self._robots[i].data.root_pos_w[:, 2]
            # Terminate if altitude is too low OR if any win condition is met
            terminated[name] = (alt < 0.2) | win_flag
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
        for i, (r, body_id) in enumerate(zip(self._robots, [self._body_id1, self._body_id2])):
            r.set_external_force_and_torque(
                self._thrust[:, i, :].unsqueeze(1), 
                self._moment[:, i, :].unsqueeze(1),    
                body_ids=body_id
            )



    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        # If env_ids is None or covers all environments, reset all.
        if env_ids is None or (isinstance(env_ids, torch.Tensor) and len(env_ids) == self.num_envs):
            env_ids = self._drone_attack._ALL_INDICES

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._drone_attack.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode_Termination/died"] = (
            torch.count_nonzero(self.reset_terminated[env_ids]).item()
            if hasattr(self, "reset_terminated") else 0
        )
        extras["Episode_Termination/time_out"] = (
            torch.count_nonzero(self.reset_time_outs[env_ids]).item()
            if hasattr(self, "reset_time_outs") else 0
        )
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        for robot in self._robots:
            robot.reset(env_ids)

        super()._reset_idx(env_ids)
        if isinstance(env_ids, torch.Tensor) and len(env_ids) == self.num_envs: #type: ignore
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0

        self._desired_pos_w[env_ids, 0] = 5.0
        self._desired_pos_w[env_ids, 1] = 5.0
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        for agent, robot in zip(self.cfg.possible_agents, self._robots):
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids].clone()

            if agent == "drone_attack":
                default_root_state[:, 0] = 0.0
                default_root_state[:, 1] = 0.0
                default_root_state[:, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
            elif agent == "drone_defense":
                default_root_state[:, 0] = 0.0
                default_root_state[:, 1] = 5.0
                default_root_state[:, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

            # Write the new states to simulation.
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