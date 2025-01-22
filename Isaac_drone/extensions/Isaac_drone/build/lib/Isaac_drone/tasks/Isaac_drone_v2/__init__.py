"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .formation import formationEnv, formationEnvCfg 

##
# Register Gym environments.
##
gym.register(
    id="Isaac-drone-formation",
    entry_point="Isaac_drone.tasks.Isaac_drone_v2.formation:formationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": formationEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
