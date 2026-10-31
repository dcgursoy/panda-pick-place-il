"""MuJoCo scene (Menagerie Panda + table, cube, bin, cameras), the gymnasium environment,
differential IK, camera clocks and the task definition."""

from gymnasium.envs.registration import register

from panda_il.config import EnvConfig
from panda_il.env.panda_env import (
    ACTION_DIM,
    ENV_VERSION,
    PROPRIO_DIM,
    STATE_DIM,
    PandaPickPlaceEnv,
    make_env,
)
from panda_il.env.task import InitialCondition

register(id="PandaPickPlace-v0", entry_point="panda_il.env.panda_env:PandaPickPlaceEnv")

__all__ = [
    "ACTION_DIM",
    "ENV_VERSION",
    "PROPRIO_DIM",
    "STATE_DIM",
    "EnvConfig",
    "InitialCondition",
    "PandaPickPlaceEnv",
    "make_env",
]
