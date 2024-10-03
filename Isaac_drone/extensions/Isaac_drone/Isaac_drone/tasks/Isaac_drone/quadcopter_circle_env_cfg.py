from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 2
    num_actions = 4
    num_observations = 12
    num_states = 0
    debug_vis = True

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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    effort_reward_scale = 0.2
    action_smoothness_reward_scale = 0.1


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Desired positions for the track
        self._desired_pos_w = torch.zeros(self.num_envs, self.cfg.num_observations, 3, device=self.device)
        self.traj_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # For trajectory calculations
        self.trajectory = self._generate_trajectory(self.num_envs, steps=self.max_episode_length)
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _generate_trajectory(self, num_envs, steps):
        """Generate a simple trajectory, like a circular path."""
        t = torch.linspace(0, 2 * torch.pi, steps, device=self.device)
        trajectory = torch.stack([torch.sin(t), torch.cos(t), torch.ones_like(t)], dim=-1)
        return trajectory.unsqueeze(0).expand(num_envs, -1, -1)  # (num_envs, steps, 3)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        current_step = self.traj_step % self.trajectory.shape[1]
        target_pos = self.trajectory[torch.arange(self.num_envs), current_step]
        
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], target_pos
        )

        obs = torch.cat([self._robot.data.root_lin_vel_b,
                         self._robot.data.root_ang_vel_b,
                         desired_pos_b], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        current_step = self.traj_step % self.trajectory.shape[1]
        target_pos = self.trajectory[torch.arange(self.num_envs), current_step]
        distance_to_goal = torch.linalg.norm(target_pos - self._robot.data.root_pos_w, dim=1)

        rewards = {
            "distance_to_goal": -distance_to_goal * self.cfg.distance_to_goal_reward_scale,
            "effort": -self._actions.abs().sum(dim=1) * self.cfg.effort_reward_scale,
            "action_smoothness": -torch.sum(torch.abs(self._actions[:, :-1] - self._actions[:, 1:]), dim=1) * self.cfg.action_smoothness_reward_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        self.traj_step += 1
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self.traj_step[env_ids] = 0  # Reset trajectory step
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_joint_state_to_sim(self._robot.data.default_joint_pos[env_ids], self._robot.data.default_joint_vel[env_ids], None, env_ids)
