"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .camera_quadcopter import CameraQuadcopterEnv, CameraQuadcopterEnvCfg
from .lidar_quadcopter import LidarQuadcopterEnv, LidarQuadcopterEnvCfg
##
# Register Gym environments.
##


gym.register(
    id="Isaac-camera-drone",
    entry_point="Isaac_drone.tasks.Isaac_drone_v1.camera_quadcopter:CameraQuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CameraQuadcopterEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cnn_cfg.yaml",
    },
)

gym.register(
    id="Isaac-lidar-drone",
    entry_point="Isaac_drone.tasks.Isaac_drone_v1.lidar_quadcopter:LidarQuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LidarQuadcopterEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_lidar_cnn_cfg.yaml",
    },
)
