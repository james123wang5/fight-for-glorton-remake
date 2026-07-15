"""Optional reinforcement-learning tools for the Glorton remake.

The environment is imported lazily so ``python -m training.play_level21`` can
initialize the ordinary visible SDL display before loading headless tools.
"""

from typing import Any

__all__ = ["PeachLeagueEnv", "PeachVsLevel20Env", "TacticalPeachEnv"]


def __getattr__(name: str) -> Any:
    if name == "PeachVsLevel20Env":
        from .peach_env import PeachVsLevel20Env

        return PeachVsLevel20Env
    if name == "PeachLeagueEnv":
        from .league_env import PeachLeagueEnv

        return PeachLeagueEnv
    if name == "TacticalPeachEnv":
        from .tactical_env import TacticalPeachEnv

        return TacticalPeachEnv
    raise AttributeError(name)
