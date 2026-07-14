from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from .runtime import RuntimeApp


INPUT_FIELDS = (
    "left",
    "right",
    "up_trace",
    "down",
    "jump_pressed",
    "punch_pressed",
    "special_pressed",
    "shield_pressed",
    "shield_released",
)

FIGHTER_SCALAR_FIELDS = (
    "team_index",
    "draw_depth",
    "color_frame",
    "xinc",
    "yinc",
    "state",
    "jumpstate",
    "on_ground",
    "has_control",
    "ctrl_loss",
    "paralized",
    "electrocuted_ms",
    "damage_amnt",
    "combo",
    "time_ko",
    "lives",
    "deaths",
    "sds",
    "kos",
    "death_order",
    "dead",
    "last_death_type",
    "spawn_invincible_ms",
    "invincible",
    "blinky_cos",
    "blinking",
    "facing",
    "attack_facing",
    "current_label",
    "animation_frame",
    "animation_time_ms",
    "current_attack",
    "attack_frame",
    "attack_done",
    "attack_pending_finish",
    "bullet_shot",
    "garbage_variant",
    "move_queue",
    "spec_up_ok",
    "shielded",
    "shield_size",
    "current_item",
    "last_collision",
    "dead_reason",
    "out_of_camera",
    "spawn_reveal_ms",
    "spawn_reveal_frame",
    "spawn_age_ms",
    "spawn_effect_kind",
    "spawn_visual_alpha",
    "spawn_white_offset",
    "spawn_fighter_visible",
    "osd_damage_age_ms",
    "osd_score_event",
    "osd_score_age_ms",
    "intro_visible",
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if hasattr(value, "x") and hasattr(value, "y"):
        return [float(value.x), float(value.y)]
    return repr(value)


def _vector(value: Any) -> list[float]:
    return [round(float(value.x), 6), round(float(value.y), 6)]


def _rect(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [round(float(item), 6) for item in value]
    return [
        round(float(value.x), 6),
        round(float(value.y), 6),
        round(float(value.w), 6),
        round(float(value.h), 6),
    ]


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _entity_index(items: Sequence[Any], target: Any) -> int | None:
    if target is None:
        return None
    try:
        return items.index(target)
    except ValueError:
        return None


class BattleSimulation:
    """Deterministic, fixed-step battle facade with no display loop.

    RuntimeApp remains the renderer and backwards-compatible host. This class
    owns tick scheduling, the isolated pseudo-random stream, snapshots and
    input recordings, so a server or test can advance combat without opening a
    game window or reading the keyboard.
    """

    SCHEMA = "glorton-battle-snapshot-v1"
    RECORDING_SCHEMA = "glorton-input-recording-v1"

    def __init__(self, runtime: RuntimeApp, seed: int = 0, tick_ms: int = 25) -> None:
        self.runtime = runtime
        self.seed = int(seed)
        self.tick_ms = int(tick_ms)
        self.tick_index = 0
        self._rng = random.Random(self.seed)
        self._recording_inputs: list[list[dict[str, bool]]] | None = None
        self._recording_metadata: dict[str, Any] = {}
        self._recording_initial_digest = ""
        self._recording_initial_snapshot: dict[str, Any] | None = None
        self._previous_controls: list[dict[str, bool]] = []

    @classmethod
    def headless(
        cls,
        *,
        seed: int = 0,
        match_config: Mapping[str, Any] | None = None,
    ) -> BattleSimulation:
        """Create a hidden one-pixel SDL host and return its simulation."""

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        from .runtime import RuntimeApp, Stage

        if not pygame.get_init():
            pygame.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
        runtime = RuntimeApp(random_seed=seed)
        runtime.audio = None
        if match_config is not None:
            runtime.match_config = copy.deepcopy(dict(match_config))
            stage_name = str(
                runtime.match_config.get(
                    "selected_stage",
                    runtime.match_config.get("stage", runtime.stage.name),
                )
            )
            runtime.stage = Stage(runtime.manifest, stage_name)
            runtime.simulation.reset(seed)
        return runtime.simulation

    @property
    def fighters(self) -> list[Any]:
        return self.runtime.fighters

    @property
    def stage(self) -> Any:
        return self.runtime.stage

    @contextmanager
    def _activate_rng(self) -> Iterable[None]:
        """Temporarily route legacy module-level random calls to this match."""

        previous = random.getstate()
        random.setstate(self._rng.getstate())
        try:
            yield
        finally:
            self._rng.setstate(random.getstate())
            random.setstate(previous)

    @staticmethod
    def normalize_inputs(
        inputs: Sequence[Mapping[str, Any]] | None,
        player_count: int,
    ) -> list[dict[str, bool]]:
        source = list(inputs or [])
        return [
            {
                field: bool(source[index].get(field, False)) if index < len(source) else False
                for field in INPUT_FIELDS
            }
            for index in range(player_count)
        ]

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
        self._rng.seed(self.seed)
        self.tick_index = 0
        with self._activate_rng():
            self.runtime._reset_match()
        self._previous_controls = self.normalize_inputs(None, len(self.runtime.fighters))
        self._recording_inputs = None
        self._recording_metadata = {}
        self._recording_initial_digest = ""
        self._recording_initial_snapshot = None
        return self.snapshot()

    def step(
        self,
        inputs: Sequence[Mapping[str, Any]] | None = None,
        *,
        advance_clock: bool = True,
    ) -> dict[str, Any]:
        """Advance exactly one 25 ms combat tick and return a state snapshot."""

        runtime = self.runtime
        controls = self.normalize_inputs(inputs, len(runtime.fighters))
        with self._activate_rng():
            self._apply_control_edges(controls)
            if advance_clock:
                runtime._advance_battle_time(self.tick_ms)
                runtime.accumulator = max(0, runtime.accumulator - self.tick_ms)
            runtime.stage.set_time(runtime.stage_time_ms)
            if runtime.match_state in {"loading", "countdown"}:
                self._fixed_tick_countdown(controls)
            elif runtime.match_state == "playing":
                self._fixed_tick_match(controls)
        if self._recording_inputs is not None:
            self._recording_inputs.append(copy.deepcopy(controls))
        self._previous_controls = copy.deepcopy(controls)
        self.tick_index += 1
        return self.snapshot()

    def _apply_control_edges(self, controls: list[dict[str, bool]]) -> None:
        runtime = self.runtime
        previous = self._previous_controls
        for index, fighter in enumerate(runtime.fighters):
            if index in runtime.ai_controllers:
                continue
            current = controls[index]
            old = previous[index] if index < len(previous) else {}
            direction = "stop"
            if current.get("left") and not current.get("right"):
                direction = "left"
            elif current.get("right") and not current.get("left"):
                direction = "right"
            old_direction = "stop"
            if old.get("left") and not old.get("right"):
                old_direction = "left"
            elif old.get("right") and not old.get("left"):
                old_direction = "right"
            if direction != old_direction:
                fighter.move(direction)

    def _fixed_tick_countdown(self, controls_by_player: list[dict[str, bool]]) -> None:
        runtime = self.runtime
        # ItemGen is attached when the fight frame loads, before GameOn. Its
        # two-second timer therefore advances throughout loading/countdown in
        # the AVM1 source instead of starting only after GO! disappears.
        runtime._fixed_tick_items()
        source_computers = list(runtime.ai_controllers.values())
        for index, fighter in enumerate(runtime.fighters):
            controls = (
                controls_by_player[index]
                if index < len(controls_by_player) and index not in runtime.ai_controllers
                else {}
            )
            fighter.advance_pre_game_tick(controls)
            fighter.advance_intro_tick()
            if index < len(source_computers):
                source_computers[index].fixed_tick(runtime.fighters)
        if runtime.match_state == "countdown":
            runtime._step_camera()

    def _fixed_tick_match(self, player_controls: list[dict[str, bool]]) -> None:
        runtime = self.runtime
        runtime._fixed_tick_items()
        # Preserve the dense Computers[] indexing quirk from the AVM1 source.
        source_computers = list(runtime.ai_controllers.values())
        for index, fighter in enumerate(runtime.fighters):
            controls = (
                player_controls[index]
                if index < len(player_controls) and index not in runtime.ai_controllers
                else {}
            )
            fighter.fixed_tick(
                runtime.stage,
                controls,
                runtime.bullets,
                runtime.rockets,
                runtime.special_projectiles,
            )
            if fighter.pending_stage_boom:
                runtime._start_explosion(fighter.pos, None, 4)
                fighter.pending_stage_boom = False
            runtime._collect_fighter_sounds(fighter)
            runtime._collect_puffs(fighter)
            runtime._collect_death_event(fighter)
            if fighter.resolve_attack_this_tick:
                runtime._resolve_melee_hits(fighter)
            fighter.finish_post_collision_tick()
            if index < len(source_computers):
                source_computers[index].fixed_tick(runtime.fighters)
        for bullet in runtime.bullets:
            bullet.fixed_tick(runtime.stage)
        for rocket in runtime.rockets:
            rocket.fixed_tick(runtime.stage)
        for projectile in runtime.special_projectiles:
            projectile.fixed_tick(runtime.stage)
        runtime._resolve_bullet_hits()
        runtime._resolve_rocket_hits()
        runtime._resolve_special_projectile_hits()
        runtime.bullets = [bullet for bullet in runtime.bullets if bullet.alive]
        runtime.rockets = [rocket for rocket in runtime.rockets if rocket.alive]
        runtime.special_projectiles = [
            projectile for projectile in runtime.special_projectiles if projectile.alive
        ]
        runtime._resolve_item_collisions()
        runtime._tick_explosions()
        for fighter in runtime.fighters:
            runtime._collect_fighter_sound_stops(fighter)
        runtime._tick_death_effects()
        runtime._tick_spawn_effects()
        runtime._update_match_state()
        runtime._step_camera()

    def _fighter_snapshot(self, fighter: Any) -> dict[str, Any]:
        fighters = self.runtime.fighters
        items = self.runtime.items
        state = {
            "name": fighter.name,
            "fighter": fighter.fighter_name,
            "character": fighter.character_name,
            "pos": _vector(fighter.pos),
            "prev_pos": _vector(fighter.prev_pos),
            "last_sender": _entity_index(fighters, fighter.last_sender),
            "throw_victim": _entity_index(fighters, fighter.throw_victim),
            "ground_platform": getattr(fighter.ground_platform, "name", None),
            "go_through_platform": getattr(fighter.go_through_platform, "name", None),
            "current_item_index": _entity_index(items, fighter.current_item_obj),
            "hit_targets": sorted(int(value) for value in fighter.hit_targets),
        }
        state.update(
            {
                field: _json_safe(getattr(fighter, field))
                for field in FIGHTER_SCALAR_FIELDS
                if hasattr(fighter, field)
            }
        )
        return state

    def _projectile_snapshot(self, projectile: Any) -> dict[str, Any]:
        state = {
            "type": type(projectile).__name__,
            "pos": _vector(projectile.pos),
            "prev_pos": _vector(getattr(projectile, "prev_pos", projectile.pos)),
            "sender": _entity_index(self.runtime.fighters, getattr(projectile, "sender", None)),
        }
        for field in (
            "kind",
            "xinc",
            "yinc",
            "rotation",
            "source_scale",
            "display_scale",
            "facing",
            "mirrored",
            "variant",
            "age",
            "life",
            "alive",
        ):
            if hasattr(projectile, field):
                state[field] = _json_safe(getattr(projectile, field))
        return state

    def _item_snapshot(self, item: Any) -> dict[str, Any]:
        return {
            "kind": item.kind,
            "pos": _vector(item.pos),
            "prev_pos": _vector(getattr(item, "prev_pos", item.pos)),
            "sender": _entity_index(self.runtime.fighters, item.sender),
            "source_scale": item.source_scale,
            "life_ms": item.life_ms,
            "age_ms": item.age_ms,
            "state": item.state,
            "alive": item.alive,
            "xinc": item.xinc,
            "yinc": item.yinc,
            "rotation": item.rotation,
            "active_ms": item.active_ms,
            "active_platform": getattr(item.active_platform, "name", None),
            "active_offset": _vector(item.active_offset) if item.active_offset is not None else None,
            "influenced": sorted(item.influenced or set()),
        }

    def snapshot(self) -> dict[str, Any]:
        runtime = self.runtime
        rng_state = _json_safe(self._rng.getstate())
        snapshot = {
            "schema": self.SCHEMA,
            "seed": self.seed,
            "tick_ms": self.tick_ms,
            "tick": self.tick_index,
            "rng_state": rng_state,
            "match_config": _json_safe(runtime.match_config),
            "app_state": runtime.app_state,
            "match_state": runtime.match_state,
            "stage": runtime.stage.name,
            "stage_time_ms": runtime.stage_time_ms,
            "match_time_remaining_ms": runtime.match_time_remaining_ms,
            "game_time_seconds": runtime.game_time_seconds,
            "ready_set": runtime.ready_set,
            "ready_text": runtime.ready_text,
            "match_winner": _entity_index(runtime.fighters, runtime.match_winner),
            "camera_view": _rect(runtime.camera_view),
            "camera_target_view": _rect(runtime.camera_target_view),
            "item_gen_timer_ms": runtime.item_gen_timer_ms,
            "accumulator": runtime.accumulator,
            "previous_controls": copy.deepcopy(self._previous_controls),
            "fighters": [self._fighter_snapshot(fighter) for fighter in runtime.fighters],
            "bullets": [self._projectile_snapshot(item) for item in runtime.bullets],
            "rockets": [self._projectile_snapshot(item) for item in runtime.rockets],
            "special_projectiles": [
                self._projectile_snapshot(item) for item in runtime.special_projectiles
            ],
            "items": [self._item_snapshot(item) for item in runtime.items],
            "explosions": [
                {
                    "pos": _vector(item.pos),
                    "size": item.size,
                    "sender": _entity_index(runtime.fighters, item.sender),
                    "age_ms": item.age_ms,
                    "matter_offsets": [_vector(offset) for offset in item.matter_offsets],
                    "influenced": sorted(item.influenced or set()),
                }
                for item in runtime.explosions
            ],
        }
        return snapshot

    @staticmethod
    def digest(snapshot: Mapping[str, Any]) -> str:
        payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def state_digest(self) -> str:
        return self.digest(self.snapshot())

    def start_recording(self, metadata: Mapping[str, Any] | None = None) -> None:
        self._recording_inputs = []
        self._recording_metadata = copy.deepcopy(dict(metadata or {}))
        self._recording_initial_snapshot = self.snapshot()
        self._recording_initial_digest = self.digest(self._recording_initial_snapshot)

    def stop_recording(self) -> dict[str, Any]:
        if self._recording_inputs is None:
            raise RuntimeError("No battle recording is active")
        recording = {
            "schema": self.RECORDING_SCHEMA,
            "seed": self.seed,
            "tick_ms": self.tick_ms,
            "stage": self.runtime.stage.name,
            "match_config": _json_safe(self.runtime.match_config),
            "metadata": _json_safe(self._recording_metadata),
            "initial_digest": self._recording_initial_digest,
            "initial_snapshot": self._recording_initial_snapshot,
            "inputs": self._recording_inputs,
            "final_digest": self.state_digest(),
        }
        self._recording_inputs = None
        self._recording_initial_snapshot = None
        return recording

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """Restore a JSON snapshot suitable for replay or state correction.

        Dynamic entities are deliberately required to be empty at the restore
        boundary for now. A recording may create them after its first input;
        this keeps the initial-state contract deterministic without silently
        constructing incomplete projectile or item objects.
        """

        if snapshot.get("schema") != self.SCHEMA:
            raise ValueError("Unsupported Glorton snapshot schema")
        dynamic_fields = ("bullets", "rockets", "special_projectiles", "items", "explosions")
        nonempty = [field for field in dynamic_fields if snapshot.get(field)]
        if nonempty:
            raise ValueError(
                "Snapshot restore currently requires an entity-free boundary; "
                f"found {', '.join(nonempty)}"
            )

        from .runtime import Stage

        runtime = self.runtime
        self.seed = int(snapshot.get("seed", self.seed))
        self.tick_ms = int(snapshot.get("tick_ms", self.tick_ms))
        match_config = snapshot.get("match_config")
        if isinstance(match_config, Mapping):
            runtime.match_config = copy.deepcopy(dict(match_config))
        stage_name = str(snapshot.get("stage", runtime.stage.name))
        if runtime.stage.name != stage_name:
            runtime.stage = Stage(runtime.manifest, stage_name)
            runtime._reset_match()
        fighters = list(snapshot.get("fighters", []))
        if len(fighters) != len(runtime.fighters):
            raise ValueError(
                f"Snapshot has {len(fighters)} fighters, runtime has {len(runtime.fighters)}"
            )
        runtime.bullets = []
        runtime.rockets = []
        runtime.special_projectiles = []
        runtime.items = []
        runtime.explosions = []
        platforms = {getattr(item, "name", None): item for item in runtime.stage.platforms}

        for fighter, state in zip(runtime.fighters, fighters):
            fighter.pos.update(state["pos"])
            fighter.prev_pos.update(state["prev_pos"])
            for field in FIGHTER_SCALAR_FIELDS:
                if field in state and hasattr(fighter, field):
                    setattr(fighter, field, copy.deepcopy(state[field]))
            fighter.ground_platform = platforms.get(state.get("ground_platform"))
            fighter.go_through_platform = platforms.get(state.get("go_through_platform"))
            fighter.current_item_obj = None
            fighter.hit_targets = set(int(value) for value in state.get("hit_targets", []))

        for fighter, state in zip(runtime.fighters, fighters):
            last_sender = state.get("last_sender")
            throw_victim = state.get("throw_victim")
            fighter.last_sender = (
                runtime.fighters[int(last_sender)] if last_sender is not None else None
            )
            fighter.throw_victim = (
                runtime.fighters[int(throw_victim)] if throw_victim is not None else None
            )

        for field in (
            "app_state",
            "match_state",
            "stage_time_ms",
            "match_time_remaining_ms",
            "game_time_seconds",
            "ready_set",
            "ready_text",
            "item_gen_timer_ms",
            "accumulator",
        ):
            if field in snapshot:
                setattr(runtime, field, copy.deepcopy(snapshot[field]))
        winner = snapshot.get("match_winner")
        runtime.match_winner = runtime.fighters[int(winner)] if winner is not None else None

        def restore_rect(value: Any) -> list[float] | None:
            return [float(item) for item in value] if value is not None else None

        runtime.camera_view = restore_rect(snapshot.get("camera_view"))
        runtime.camera_target_view = restore_rect(snapshot.get("camera_target_view"))
        runtime.stage.set_time(runtime.stage_time_ms)
        self.tick_index = int(snapshot.get("tick", 0))
        self._rng.setstate(_tuple_tree(snapshot["rng_state"]))
        self._previous_controls = self.normalize_inputs(
            snapshot.get("previous_controls"),
            len(runtime.fighters),
        )
        return self.snapshot()

    def replay(self, recording: Mapping[str, Any], *, strict: bool = True) -> dict[str, Any]:
        if recording.get("schema") != self.RECORDING_SCHEMA:
            raise ValueError("Unsupported Glorton recording schema")
        match_config = recording.get("match_config")
        if isinstance(match_config, Mapping):
            self.runtime.match_config = copy.deepcopy(dict(match_config))
            from .runtime import Stage

            stage_name = str(recording.get("stage", self.runtime.stage.name))
            self.runtime.stage = Stage(self.runtime.manifest, stage_name)
        self.reset(int(recording.get("seed", 0)))
        initial_snapshot = recording.get("initial_snapshot")
        if isinstance(initial_snapshot, Mapping):
            self.restore_snapshot(initial_snapshot)
        initial = self.state_digest()
        expected_initial = str(recording.get("initial_digest", ""))
        if strict and expected_initial and initial != expected_initial:
            raise ValueError(
                "Recording initial state does not match this match configuration "
                f"({initial[:12]} != {expected_initial[:12]})"
            )
        for controls in recording.get("inputs", []):
            self.step(controls)
        final = self.state_digest()
        expected_final = str(recording.get("final_digest", ""))
        if strict and expected_final and final != expected_final:
            raise AssertionError(
                f"Replay diverged ({final[:12]} != {expected_final[:12]})"
            )
        return self.snapshot()

    @staticmethod
    def save_recording(recording: Mapping[str, Any], path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(recording, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def load_recording(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
