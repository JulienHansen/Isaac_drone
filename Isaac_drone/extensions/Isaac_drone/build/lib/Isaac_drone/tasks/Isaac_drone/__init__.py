"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .quadcopter_circle_env_cfg import QuadcopterCircleEnv, QuadcopterCircleEnvCfg
from .quadcopter_env_cfg import QuadcopterEnv, QuadcopterEnvCfg
##
# Register Gym environments.
##

gym.register(
    id="Isaac-drone-v0",
    entry_point="Isaac_drone.tasks.Isaac_drone.quadcopter_env_cfg:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-drone-circle-v0",
    entry_point="Isaac_drone.tasks.Isaac_drone.quadcopter_circle_env_cfg:QuadcopterCircleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterCircleEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)