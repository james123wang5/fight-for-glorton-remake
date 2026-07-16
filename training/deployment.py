from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .peach_env import encode_runtime_observation


_MODEL_CACHE: dict[Path, Any] = {}


def _load_ppo_model(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    cached = _MODEL_CACHE.get(resolved)
    if cached is not None:
        return cached
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "21级AI需要训练环境。请用 .venv-train/bin/python -m training.play_level21 启动。"
        ) from exc
    if not resolved.is_file():
        raise RuntimeError(f"找不到21级AI模型: {resolved}")
    model = PPO.load(str(resolved), device="cpu")
    _MODEL_CACHE[resolved] = model
    return model


class TrainedAIController:
    """Run the stage-one PPO policy inside an ordinary rendered battle.

    The original levels 1--20 continue to use ``AIController``.  This adapter
    exists only when the special launcher exposes level 21 and points it at a
    model.  It mirrors the training action cadence: one decision per two 25 ms
    combat ticks, with press edges on the first tick.
    """

    level = 21
    force_victim = False
    uses_simulation_controls = True

    def __init__(
        self,
        runtime: Any,
        player: Any,
        stage: Any,
        model_path: str | Path,
        *,
        model: Any | None = None,
    ) -> None:
        self.runtime = runtime
        self.player = player
        self.stage = stage
        self.model_path = Path(model_path).expanduser().resolve()
        self.model = model if model is not None else _load_ppo_model(self.model_path)
        expected_shape = getattr(getattr(self.model, "observation_space", None), "shape", (142,))
        if tuple(expected_shape or ()) != (142,):
            raise RuntimeError(
                f"21级AI模型观察维度不兼容: 需要 (142,), 实际 {expected_shape}"
            )
        self.victim: Any | None = None
        self.current_action = np.zeros(3, dtype=np.int64)
        self.previous_combat = 0
        self.action_phase = 0
        self.episode_ticks = 0
        self.spawns_swapped = False
        self.active = False

    def _begin_match(self, fighters: Sequence[Any]) -> None:
        self.victim = self._pick_opponent(fighters)
        spawn_p1 = self.stage.spawn_point("SpawnP1")
        spawn_p2 = self.stage.spawn_point("SpawnP2")
        self.spawns_swapped = self.player.pos.distance_to(spawn_p2) < self.player.pos.distance_to(spawn_p1)
        self.current_action[:] = 0
        self.previous_combat = 0
        self.action_phase = 0
        self.episode_ticks = 0
        self.active = True

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

    def _decide(self, fighters: Sequence[Any]) -> None:
        self.victim = self._pick_opponent(fighters)
        if self.victim is None:
            self.current_action[:] = 0
            return
        observation = encode_runtime_observation(
            self.runtime,
            self.player,
            self.victim,
            episode_ticks=min(self.episode_ticks, 7200),
            max_ticks=7200,
            spawns_swapped=self.spawns_swapped,
        )
        action, _state = self.model.predict(observation, deterministic=True)
        candidate = np.asarray(action, dtype=np.int64).reshape(-1)
        if candidate.shape != (3,):
            raise RuntimeError(f"21级AI模型返回了无效动作: {candidate!r}")
        self.current_action = np.asarray(
            [
                int(np.clip(candidate[0], 0, 2)),
                int(np.clip(candidate[1], 0, 2)),
                int(np.clip(candidate[2], 0, 3)),
            ],
            dtype=np.int64,
        )

    def _controls_for_action(self, *, press_tick: bool) -> dict[str, bool]:
        horizontal, vertical, combat = (int(value) for value in self.current_action)
        controls = {
            "left": horizontal == 1,
            "right": horizontal == 2,
            "down": vertical == 2,
        }
        if not press_tick:
            return controls
        controls["up_trace"] = vertical == 1 and combat in {1, 2}
        controls["jump_pressed"] = vertical == 1 and combat == 0
        controls["punch_pressed"] = combat == 1
        controls["special_pressed"] = combat == 2
        controls["shield_pressed"] = combat == 3 and self.previous_combat != 3
        controls["shield_released"] = combat != 3 and self.previous_combat == 3
        self.previous_combat = combat
        return controls

    def controls_for_tick(self, fighters: list[Any]) -> dict[str, bool]:
        if self.runtime.match_state != "playing":
            self.active = False
            return {}
        if not self.active:
            self._begin_match(fighters)
        if self.player.dead:
            return {}
        press_tick = self.action_phase == 0
        if press_tick:
            self._decide(fighters)
        controls = self._controls_for_action(press_tick=press_tick)
        self.action_phase = (self.action_phase + 1) % 2
        self.episode_ticks += 1
        return controls

    def fixed_tick(self, fighters: list[Any]) -> None:
        # BattleSimulation consumes controls_for_tick() before fighter updates.
        # Keep this method for the legacy controller interface and dense-array
        # iteration, but never apply the policy a second time afterward.
        return None
