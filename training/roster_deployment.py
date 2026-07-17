from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .roster_observation import (
    ROSTER_OBSERVATION_SIZE,
    encode_roster_observation,
)
from .roster_options import roster_purpose_action_mask
from .v5_env import V5_FRAME_SKIP
from .v5_options import PURPOSE_COUNT, Purpose, PurposefulOptionController
from .v5_runtime_helpers import is_offstage


_MODEL_CACHE: dict[Path, Any] = {}


def _load_roster_model(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    cached = _MODEL_CACHE.get(resolved)
    if cached is not None:
        return cached
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise RuntimeError(
            "全角色AI需要训练环境，请用 .venv-train/bin/python -m "
            "training.play_roster_battle 启动。"
        ) from exc
    if not resolved.is_file():
        raise RuntimeError(f"找不到全角色AI模型: {resolved}")
    model = MaskablePPO.load(str(resolved), device="cpu")
    _MODEL_CACHE[resolved] = model
    return model


class RosterTrainedAIController:
    """Use one role's 480-value v6 policy in the ordinary rendered battle."""

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
        self.model = model if model is not None else _load_roster_model(self.model_path)
        expected_shape = tuple(
            getattr(getattr(self.model, "observation_space", None), "shape", ()) or ()
        )
        if expected_shape != (ROSTER_OBSERVATION_SIZE,):
            raise RuntimeError(
                f"{self.player.fighter_name} {self.level}级模型观察维度不兼容: "
                f"需要 ({ROSTER_OBSERVATION_SIZE},)，实际 {expected_shape}"
            )
        action_count = int(getattr(getattr(self.model, "action_space", None), "n", -1))
        if action_count not in {-1, PURPOSE_COUNT}:
            raise RuntimeError(
                f"{self.player.fighter_name} {self.level}级模型动作空间不兼容: "
                f"需要 Discrete({PURPOSE_COUNT})"
            )
        self.reaction_delay_decisions = max(0, int(reaction_delay_decisions))
        self.option = PurposefulOptionController(runtime)
        self.victim: Any | None = None
        self.control_sequence: tuple[dict[str, bool], ...] = ({}, {}, {}, {})
        self.action_phase = 0
        self.episode_ticks = 0
        self.spawns_swapped = False
        self.observations: deque[np.ndarray] = deque()
        self.active = False
        self.was_dead = False
        self.mutual_idle_decisions = 0
        self.stalemate_breaks = 0
        self._last_activity_sample: tuple[float, ...] | None = None

    def _pick_opponent(self, fighters: Sequence[Any]) -> Any | None:
        candidates = [
            fighter
            for fighter in fighters
            if fighter is not self.player and not fighter.dead
        ]
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

    def _begin_match(self, fighters: Sequence[Any]) -> None:
        self.victim = self._pick_opponent(fighters)
        spawn_p1 = self.stage.spawn_point("SpawnP1")
        spawn_p2 = self.stage.spawn_point("SpawnP2")
        self.spawns_swapped = self.player.pos.distance_to(
            spawn_p2
        ) < self.player.pos.distance_to(spawn_p1)
        self.option.reset()
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
        self.mutual_idle_decisions = 0
        self._last_activity_sample = self._activity_sample()

    def _activity_sample(self) -> tuple[float, ...] | None:
        if self.victim is None:
            return None
        return (
            float(self.player.pos.x),
            float(self.player.pos.y),
            float(self.player.damage_amnt),
            float(self.victim.pos.x),
            float(self.victim.pos.y),
            float(self.victim.damage_amnt),
        )

    def _mutual_stalemate(self) -> bool:
        """Detect a real playing-state duel stall, not a normal short pause."""

        sample = self._activity_sample()
        previous = self._last_activity_sample
        self._last_activity_sample = sample
        if sample is None or previous is None or self.victim is None:
            self.mutual_idle_decisions = 0
            return False
        valid = bool(
            self.runtime.match_state == "playing"
            and not self.player.dead
            and not self.victim.dead
            and self.player.state not in {"spawn", "thrown", "ko", "dead"}
            and self.victim.state not in {"spawn", "thrown", "ko", "dead"}
            and not self.player.current_attack
            and not self.victim.current_attack
            and not is_offstage(self.runtime, self.player)
            and not is_offstage(self.runtime, self.victim)
        )
        moved = math.hypot(sample[0] - previous[0], sample[1] - previous[1])
        moved += math.hypot(sample[3] - previous[3], sample[4] - previous[4])
        damage_changed = bool(sample[2] != previous[2] or sample[5] != previous[5])
        if valid and moved < 1.0 and not damage_changed:
            self.mutual_idle_decisions += 1
        else:
            self.mutual_idle_decisions = 0
        # Decisions run at 10 Hz. P1 breaks symmetry after 1.0 s; later slots
        # wait a little longer so both fighters do not mirror the same jump.
        try:
            slot = self.runtime.fighters.index(self.player)
        except ValueError:
            slot = 0
        return self.mutual_idle_decisions >= 10 + min(3, slot) * 6

    def _decide(self, fighters: Sequence[Any]) -> None:
        self.victim = self._pick_opponent(fighters)
        if self.victim is None:
            self.control_sequence = ({}, {}, {}, {})
            return
        self.observations.append(self._observation())
        mask = roster_purpose_action_mask(
            self.runtime,
            self.player,
            self.victim,
            self.option,
            curriculum="duel",
        )
        if self._mutual_stalemate():
            # Attack immediately when already in a genuine hit window.
            # Otherwise execute one asymmetric approach+jump to cross the
            # building geometry, even if a stale learned plan requested the
            # same failed navigation intention again.
            if mask[Purpose.MELEE]:
                forced = Purpose.MELEE
                self.control_sequence = self.option.begin_decision(
                    int(forced),
                    fighter=self.player,
                    opponent=self.victim,
                    action_mask=mask,
                )
            elif mask[Purpose.AIMED_SHOT]:
                forced = Purpose.AIMED_SHOT
                self.control_sequence = self.option.begin_decision(
                    int(forced),
                    fighter=self.player,
                    opponent=self.victim,
                    action_mask=mask,
                )
            else:
                self.control_sequence = self.option.begin_stalemate_break(
                    self.player,
                    self.victim,
                )
            self.mutual_idle_decisions = 0
            self.stalemate_breaks += 1
            return
        action, _state = self.model.predict(
            self.observations[0],
            action_masks=mask,
            deterministic=True,
        )
        candidate = np.asarray(action, dtype=np.int64).reshape(-1)
        if candidate.shape != (1,):
            raise RuntimeError(
                f"{self.player.fighter_name} {self.level}级模型返回无效意图: "
                f"{candidate!r}"
            )
        self.control_sequence = self.option.begin_decision(
            int(candidate[0]),
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
            if self.victim is not None and self.episode_ticks > 0:
                self.option.observe_result(self.player, self.victim)
            self._decide(fighters)
        controls = self.control_sequence[self.action_phase]
        self.action_phase = (self.action_phase + 1) % V5_FRAME_SKIP
        self.episode_ticks += 1
        return controls

    def fixed_tick(self, fighters: list[Any]) -> None:
        return None
