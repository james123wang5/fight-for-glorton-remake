from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from training.v5_options import PURPOSE_COUNT, PurposefulOptionController, purpose_action_mask
from training.v5_runtime_observation import (
    V5_FRAME_SKIP,
    V5_OBSERVATION_SIZE,
    encode_v5_runtime_observation,
)


class WebV5Policy:
    """Small NumPy-only inference copy of the MaskablePPO actor network."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as weights:
            self.w1 = weights["w1"].astype(np.float32, copy=False)
            self.b1 = weights["b1"].astype(np.float32, copy=False)
            self.w2 = weights["w2"].astype(np.float32, copy=False)
            self.b2 = weights["b2"].astype(np.float32, copy=False)
            self.wa = weights["wa"].astype(np.float32, copy=False)
            self.ba = weights["ba"].astype(np.float32, copy=False)
        expected = ((256, 294), (256,), (256, 256), (256,), (14, 256), (14,))
        actual = tuple(value.shape for value in (self.w1, self.b1, self.w2, self.b2, self.wa, self.ba))
        if actual != expected:
            raise RuntimeError(f"网页v5策略权重不兼容: {actual!r}")

    def predict(self, observation: np.ndarray, action_mask: Sequence[bool]) -> int:
        vector = np.asarray(observation, dtype=np.float32).reshape(V5_OBSERVATION_SIZE)
        hidden = np.tanh(self.w1 @ vector + self.b1)
        hidden = np.tanh(self.w2 @ hidden + self.b2)
        logits = self.wa @ hidden + self.ba
        mask = np.asarray(action_mask, dtype=bool).reshape(PURPOSE_COUNT)
        if not bool(mask.any()):
            raise RuntimeError("网页v5策略没有合法动作")
        return int(np.argmax(np.where(mask, logits, np.float32(-1.0e8))))


_POLICY_CACHE: dict[Path, WebV5Policy] = {}


def _load_policy(path: str | Path) -> WebV5Policy:
    resolved = Path(path).expanduser().resolve()
    policy = _POLICY_CACHE.get(resolved)
    if policy is None:
        if not resolved.is_file():
            raise RuntimeError(f"找不到网页v5策略: {resolved}")
        policy = WebV5Policy(resolved)
        _POLICY_CACHE[resolved] = policy
    return policy


class WebV5AIController:
    """Browser-safe v5 controller; no Torch, Gymnasium or SB3 dependency."""

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
        policy: WebV5Policy | None = None,
        reaction_delay_decisions: int = 1,
    ) -> None:
        self.runtime = runtime
        self.player = player
        self.stage = stage
        self.level = int(level)
        self.model_path = Path(model_path).expanduser().resolve()
        self.policy = policy or _load_policy(self.model_path)
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
            return np.zeros(V5_OBSERVATION_SIZE, dtype=np.float32)
        return encode_v5_runtime_observation(
            self.runtime,
            self.player,
            self.victim,
            self.option,
            episode_ticks=min(self.episode_ticks, 7200),
            max_ticks=7200,
            spawns_swapped=self.spawns_swapped,
            wall_stall_steps=self.option.no_progress_steps,
        )

    def _begin_match(self, fighters: Sequence[Any]) -> None:
        self.victim = self._pick_opponent(fighters)
        spawn_p1 = self.stage.spawn_point("SpawnP1")
        spawn_p2 = self.stage.spawn_point("SpawnP2")
        self.spawns_swapped = self.player.pos.distance_to(spawn_p2) < self.player.pos.distance_to(spawn_p1)
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

    def _decide(self, fighters: Sequence[Any]) -> None:
        self.victim = self._pick_opponent(fighters)
        if self.victim is None:
            self.control_sequence = ({}, {}, {}, {})
            return
        self.observations.append(self._observation())
        mask = purpose_action_mask(
            self.runtime,
            self.player,
            self.victim,
            self.option,
            curriculum="duel",
        )
        action = self.policy.predict(self.observations[0], mask)
        self.control_sequence = self.option.begin_decision(
            action,
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
