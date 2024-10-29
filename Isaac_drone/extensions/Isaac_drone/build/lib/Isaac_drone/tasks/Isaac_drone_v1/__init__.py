"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .basic_terrain import QuadcopterEnv, QuadcopterEnvCfg
from .complex_terrain import CustomQuadcopterEnv, CustomQuadcopterEnvCfg
from .lidar_quadcopter import LidarQuadcopterEnv, LidarQuadcopterEnvCfg
##
# Register Gym environments.
##

gym.register(
    id="Isaac-basic-terrain",
    entry_point="Isaac_drone.tasks.Isaac_drone_v1.basic_terrain:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-complex-terrain",
    entry_point="Isaac_drone.tasks.Isaac_drone_v1.complex_terrain:CustomQuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CustomQuadcopterEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
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
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)