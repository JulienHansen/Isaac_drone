"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .drone_defense import DefenseEnv, DefenseEnvCfg
from .drone_combat import CombatEnv, CombatEnvCfg
from .drone_lidar_combat import CombatLidarEnv, CombatLidarEnvCfg

##
# Register Gym environments.
##
gym.register(
    id="Isaac-drone-defense-v0",
    entry_point="Isaac_drone.tasks.Isaac_drone_v3.drone_defense:DefenseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DefenseEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-drone-defense-v3",
    entry_point="Isaac_drone.tasks.Isaac_drone_v3.drone_combat:CombatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CombatEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-drone-defense-final-lidar",
    entry_point="Isaac_drone.tasks.Isaac_drone_v3.drone_lidar_combat:CombatLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CombatLidarEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cnn_cfg.yaml",
    },
)
