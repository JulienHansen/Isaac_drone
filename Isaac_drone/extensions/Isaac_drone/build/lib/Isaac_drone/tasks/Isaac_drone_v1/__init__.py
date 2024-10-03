"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .po_quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-po-drone-v1",
    entry_point="Isaac_drone.tasks.Isaac_drone_v1.po_quadcopter_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)