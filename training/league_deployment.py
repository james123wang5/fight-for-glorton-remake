from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .deployment import _load_ppo_model
from .human_input import HumanInputAdapter
from .league_env import LEAGUE_OBSERVATION_SIZE, encode_league_observation


class LeagueTrainedAIController:
    """Deploy a v2 league policy with the same delayed, humanized input path."""

    force_victim = False
    uses_simulation_controls = True

    def __init__(
        self,
        runtime: Any,
        player: Any,
        stage: Any,
        model_path: str | Path,
        *,
        level: int,
        model: Any | None = None,
        reaction_delay_decisions: int = 2,
    ) -> None:
        self.runtime = runtime
        self.player = player
        self.stage = stage
        self.level = int(level)
        self.model_path = Path(model_path).expanduser().resolve()
        self.model = model if model is not None else _load_ppo_model(self.model_path)
        expected_shape = tuple(
            getattr(getattr(self.model, "observation_space", None), "shape", ()) or ()
        )
        if expected_shape != (LEAGUE_OBSERVATION_SIZE,):
            raise RuntimeError(
                f"{self.level}级AI模型观察维度不兼容: "
                f"需要 ({LEAGUE_OBSERVATION_SIZE},), 实际 {expected_shape}"
            )
        self.reaction_delay_decisions = max(0, int(reaction_delay_decisions))
        self.input_adapter = HumanInputAdapter()
        self.victim: Any | None = None
        self.control_pair: tuple[dict[str, bool], dict[str, bool]] = ({}, {})
        self.action_phase = 0
        self.episode_ticks = 0
        self.spawns_swapped = False
        self.observations: deque[np.ndarray] = deque()
        self.active = False
        self.was_dead = False

    def _pick_opponent(self, fighters: Sequence[Any]) -> Any | None:
        candidates = [fighter for fighter in fighters if fighter is not self.player and not fighter.dead]
        if not candidates:
            candidates = [fighter for fighter in fighters if fighter is not self.player]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda fighter: math.hypot(
                fighter.pos.x - self.player.pos.x,
                fighter.pos.y - self.player.pos.y,
            ),
        )

    def _observation(self) -> np.ndarray:
        if self.victim is None:
            return np.zeros(LEAGUE_OBSERVATION_SIZE, dtype=np.float32)
        return encode_league_observation(
            self.runtime,
            self.player,
            self.victim,
            self.input_adapter,
            episode_ticks=min(self.episode_ticks, 7200),
            max_ticks=7200,
            spawns_swapped=self.spawns_swapped,
        )

    def _begin_match(self, fighters: Sequence[Any]) -> None:
        self.victim = self._pick_opponent(fighters)
        spawn_p1 = self.stage.spawn_point("SpawnP1")
        spawn_p2 = self.stage.spawn_point("SpawnP2")
        self.spawns_swapped = self.player.pos.distance_to(spawn_p2) < self.player.pos.distance_to(
            spawn_p1
        )
        self.input_adapter.reset()
        self.control_pair = ({}, {})
        self.action_phase = 0
        self.episode_ticks = 0
        current = self._observation()
        self.observations = deque(
            [current.copy() for _ in range(self.reaction_delay_decisions + 1)],
            maxlen=self.reaction_delay_decisions + 1,
        )
        self.active = True
        self.was_dead = False

    def _decide(self, fighters: Sequence[Any]) -> None:
        self.victim = self._pick_opponent(fighters)
        self.observations.append(self._observation())
        action, _state = self.model.predict(self.observations[0], deterministic=True)
        candidate = np.asarray(action, dtype=np.int64).reshape(-1)
        if candidate.shape != (3,):
            raise RuntimeError(f"{self.level}级AI模型返回了无效动作: {candidate!r}")
        candidate = np.asarray(
            [
                int(np.clip(candidate[0], 0, 2)),
                int(np.clip(candidate[1], 0, 2)),
                int(np.clip(candidate[2], 0, 3)),
            ],
            dtype=np.int64,
        )
        self.control_pair = self.input_adapter.begin_decision(candidate)

    def controls_for_tick(self, fighters: list[Any]) -> dict[str, bool]:
        if self.runtime.match_state != "playing":
            self.active = False
            return {}
        if not self.active:
            self._begin_match(fighters)
        if self.player.dead:
            self.was_dead = True
            return {}
        if self.was_dead:
            self._begin_match(fighters)
        if self.action_phase == 0:
            self._decide(fighters)
        controls = self.control_pair[self.action_phase]
        self.action_phase = (self.action_phase + 1) % 2
        self.episode_ticks += 1
        return controls

    def fixed_tick(self, fighters: list[Any]) -> None:
        return None
