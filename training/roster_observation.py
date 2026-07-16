from __future__ import annotations

from typing import Any

import numpy as np

from .roster_contract import encode_roster_context
from .v5_env import V5_OBSERVATION_SIZE, encode_v5_observation
from .v5_options import PurposefulOptionController


ROSTER_OBSERVATION_VERSION = "glorton-roster-purpose-v6"
ROSTER_CONTEXT_SIZE = 186
ROSTER_OBSERVATION_SIZE = V5_OBSERVATION_SIZE + ROSTER_CONTEXT_SIZE


def encode_roster_observation(
    runtime: Any,
    fighter: Any,
    opponent: Any,
    controller: PurposefulOptionController,
    *,
    episode_ticks: int,
    max_ticks: int,
    spawns_swapped: bool,
    curriculum: str = "duel",
    wall_stall_steps: int = 0,
) -> np.ndarray:
    """Keep the frozen 294-value v5 prefix and append roster/map context."""

    legacy = encode_v5_observation(
        runtime,
        fighter,
        opponent,
        controller,
        episode_ticks=episode_ticks,
        max_ticks=max_ticks,
        spawns_swapped=spawns_swapped,
        curriculum=curriculum,
        wall_stall_steps=wall_stall_steps,
    )
    context = encode_roster_context(runtime, fighter, opponent)
    observation = np.concatenate((legacy, context)).astype(np.float32, copy=False)
    if observation.shape != (ROSTER_OBSERVATION_SIZE,):
        raise RuntimeError(f"roster observation changed: {observation.shape}")
    return observation
