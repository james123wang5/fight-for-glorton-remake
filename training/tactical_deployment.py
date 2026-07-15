from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .tactical_env import (
    TACTICAL_OBSERVATION_SIZE,
    encode_tactical_observation,
    tactical_action_mask,
)
from .tactical_input import TacticalInputAdapter


_MODEL_CACHE: dict[Path, Any] = {}


def _load_tactical_model(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    cached = _MODEL_CACHE.get(resolved)
    if cached is not None:
        return cached
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise RuntimeError(
            "v3战术AI需要训练环境。请用 .venv-train/bin/python -m training.play_tactical 启动。"
        ) from exc
    if not resolved.is_file():
        raise RuntimeError(f"找不到v3战术AI模型: {resolved}")
    model = MaskablePPO.load(str(resolved), device="cpu")
    _MODEL_CACHE[resolved] = model
    return model


class TacticalTrainedAIController:
    """Run a v3 masked tactical policy through real four-tick controls."""

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
        reaction_delay_decisions: int = 1,
    ) -> None:
        self.runtime = runtime
        self.player = player
        self.stage = stage
        self.level = int(level)
        self.model_path = Path(model_path).expanduser().resolve()
        self.model = model if model is not None else _load_tactical_model(self.model_path)
        expected_shape = tuple(
            getattr(getattr(self.model, "observation_space", None), "shape", ()) or ()
        )
        if expected_shape != (TACTICAL_OBSERVATION_SIZE,):
            raise RuntimeError(
                f"{self.level}级v3 AI模型观察维度不兼容: "
                f"需要 ({TACTICAL_OBSERVATION_SIZE},), 实际 {expected_shape}"
            )
        self.reaction_delay_decisions = max(0, int(reaction_delay_decisions))
        self.input_adapter = TacticalInputAdapter(frame_skip=4)
        self.victim: Any | None = None
        self.control_sequence: tuple[dict[str, bool], ...] = ({}, {}, {}, {})
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
            return np.zeros(TACTICAL_OBSERVATION_SIZE, dtype=np.float32)
        return encode_tactical_observation(
            self.runtime,
            self.player,
            self.victim,
            self.input_adapter,
            episode_ticks=min(self.episode_ticks, 7200),
            max_ticks=7200,
            spawns_swapped=self.spawns_swapped,
            curriculum="duel",
        )

    def _begin_match(self, fighters: Sequence[Any]) -> None:
        self.victim = self._pick_opponent(fighters)
        spawn_p1 = self.stage.spawn_point("SpawnP1")
        spawn_p2 = self.stage.spawn_point("SpawnP2")
        self.spawns_swapped = self.player.pos.distance_to(spawn_p2) < self.player.pos.distance_to(
            spawn_p1
        )
        self.input_adapter.reset()
        self.control_sequence = ({}, {}, {}, {})
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
        if self.victim is None:
            self.control_sequence = ({}, {}, {}, {})
            return
        mask = tactical_action_mask(
            self.runtime,
            self.player,
            self.victim,
            self.input_adapter,
            curriculum="duel",
        )
        action, _state = self.model.predict(
            self.observations[0],
            action_masks=mask,
            deterministic=True,
        )
        candidate = np.asarray(action, dtype=np.int64).reshape(-1)
        if candidate.shape != (2,):
            raise RuntimeError(f"{self.level}级v3 AI模型返回了无效动作: {candidate!r}")
        self.control_sequence = self.input_adapter.begin_decision(
            candidate,
            fighter=self.player,
            opponent=self.victim,
            action_mask=mask,
        )

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
        controls = self.control_sequence[self.action_phase]
        self.action_phase = (self.action_phase + 1) % 4
        self.episode_ticks += 1
        return controls

    def fixed_tick(self, fighters: list[Any]) -> None:
        return None
