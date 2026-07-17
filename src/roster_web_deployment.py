from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from training.roster_observation import ROSTER_OBSERVATION_SIZE, encode_roster_observation
from training.roster_options import roster_purpose_action_mask

from .v5_web_deployment import WebV5AIController, WebV5Policy


class RosterWebAIController(WebV5AIController):
    """NumPy-only version of the approved 480-value roster controller."""

    def __init__(
        self,
        runtime: Any,
        player: Any,
        stage: Any,
        model_path: str | Path,
        *,
        level: int,
        policy: WebV5Policy | None = None,
        reaction_delay_decisions: int = 1,
    ) -> None:
        super().__init__(
            runtime,
            player,
            stage,
            model_path,
            level=level,
            policy=policy,
            reaction_delay_decisions=reaction_delay_decisions,
        )
        if self.policy.observation_size != ROSTER_OBSERVATION_SIZE:
            raise RuntimeError(
                f"{player.fighter_name} {level}级轻量策略需要 {ROSTER_OBSERVATION_SIZE} 维输入，"
                f"实际 {self.policy.observation_size}"
            )

    def _observation(self) -> np.ndarray:
        if self.victim is None:
            return np.zeros(ROSTER_OBSERVATION_SIZE, dtype=np.float32)
        return encode_roster_observation(
            self.runtime,
            self.player,
            self.victim,
            self.option,
            episode_ticks=min(self.episode_ticks, 7200),
            max_ticks=7200,
            spawns_swapped=self.spawns_swapped,
            curriculum="duel",
            wall_stall_steps=self.option.no_progress_steps,
        )

    def _action_mask(self) -> np.ndarray:
        return roster_purpose_action_mask(
            self.runtime,
            self.player,
            self.victim,
            self.option,
            curriculum="duel",
        )
