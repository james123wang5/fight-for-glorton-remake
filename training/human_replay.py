from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.simulation import BattleSimulation

from .v5_env import V5_FRAME_SKIP, V5_OBSERVATION_SIZE, encode_v5_observation
from .v5_options import (
    PURPOSE_COUNT,
    Purpose,
    PurposefulOptionController,
    purpose_action_mask,
)
from .v5_runtime_helpers import is_offstage, tactical_context


@dataclass(frozen=True)
class HumanReplayDataset:
    observations: np.ndarray
    actions: np.ndarray
    masks: np.ndarray
    action_counts: Mapping[str, int]
    accepted_files: tuple[str, ...]
    skipped_files: Mapping[str, str]

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


def _pressed(window: Sequence[Sequence[Mapping[str, Any]]], slot: int, key: str) -> bool:
    return any(
        slot < len(frame) and bool(frame[slot].get(key, False))
        for frame in window
    )


def _held_direction(
    window: Sequence[Sequence[Mapping[str, Any]]], slot: int
) -> int:
    left = sum(
        int(slot < len(frame) and bool(frame[slot].get("left", False)))
        for frame in window
    )
    right = sum(
        int(slot < len(frame) and bool(frame[slot].get("right", False)))
        for frame in window
    )
    if left == right:
        return 0
    return -1 if left > right else 1


def infer_human_purpose(
    runtime: Any,
    fighter: Any,
    opponent: Any,
    controller: PurposefulOptionController,
    window: Sequence[Sequence[Mapping[str, Any]]],
    *,
    slot: int,
) -> Purpose | None:
    """Translate a 100 ms human input window into a conservative v5 intent.

    Only explicit, contextually legal-looking choices become demonstrations.
    Quiet frames and obvious accidental crouches are left to PPO instead of
    teaching the candidate to idle.
    """

    jump = _pressed(window, slot, "jump_pressed")
    punch = _pressed(window, slot, "punch_pressed")
    special = _pressed(window, slot, "special_pressed")
    up = _pressed(window, slot, "up_trace")
    down = _pressed(window, slot, "down")
    shield = _pressed(window, slot, "shield_pressed")
    direction = _held_direction(window, slot)
    active = bool(jump or punch or special or down or shield or direction)
    if not active or fighter.dead or fighter.state in {"spawn", "dead"}:
        return None

    context = tactical_context(runtime, fighter, opponent)
    route = controller.navigator.route(fighter, opponent)
    if fighter.state == "thrown" or fighter.ctrl_loss > 0:
        return Purpose.HITSTUN_ESCAPE
    if is_offstage(runtime, fighter):
        return Purpose.RECOVER
    if shield:
        return Purpose.SHIELD
    if special and up:
        return Purpose.ROCKET
    if punch and up:
        return Purpose.ANTI_AIR
    if punch:
        if (
            fighter.on_ground
            and float(context["distance"]) <= 32.0
            and bool(context["target_behind"])
        ):
            return Purpose.BACK_THROW
        if (
            not fighter.on_ground
            or not opponent.on_ground
            or opponent.state == "thrown"
        ):
            return Purpose.AIR_CHASE
        return Purpose.MELEE
    if special:
        return Purpose.AIMED_SHOT
    if down and not fighter.on_ground:
        return Purpose.LAND
    if jump:
        if route.requires_jump or route.blocked:
            return Purpose.NAVIGATE
        if not opponent.on_ground or opponent.state == "thrown":
            return Purpose.AIR_CHASE
        return Purpose.CHASE
    if direction:
        toward = 1 if opponent.pos.x >= fighter.pos.x else -1
        if direction == toward:
            return Purpose.NAVIGATE if route.requires_jump or route.blocked else Purpose.CHASE
        if float(context["threat_score"]) >= 0.20:
            return Purpose.EVADE
    return None


def _legal_purpose(
    requested: Purpose | None,
    mask: np.ndarray,
) -> Purpose | None:
    if requested is None:
        return None
    if bool(mask[int(requested)]):
        return requested
    fallbacks: dict[Purpose, tuple[Purpose, ...]] = {
        Purpose.ROCKET: (Purpose.AIMED_SHOT, Purpose.CHASE),
        Purpose.AIMED_SHOT: (Purpose.CHASE,),
        Purpose.ANTI_AIR: (Purpose.MELEE, Purpose.AIR_CHASE, Purpose.CHASE),
        Purpose.BACK_THROW: (Purpose.MELEE, Purpose.CHASE),
        Purpose.MELEE: (Purpose.AIR_CHASE, Purpose.CHASE),
        Purpose.AIR_CHASE: (Purpose.NAVIGATE, Purpose.CHASE),
        Purpose.EVADE: (Purpose.LAND, Purpose.CHASE),
        Purpose.HITSTUN_ESCAPE: (Purpose.RECOVER, Purpose.CONTINUE),
        Purpose.RECOVER: (Purpose.HITSTUN_ESCAPE, Purpose.CONTINUE),
        Purpose.LAND: (Purpose.RECOVER, Purpose.CHASE),
        Purpose.NAVIGATE: (Purpose.CHASE,),
    }
    for candidate in fallbacks.get(requested, ()):
        if bool(mask[int(candidate)]):
            return candidate
    return None


def examples_from_recording(
    recording: Mapping[str, Any],
    *,
    strict: bool = True,
) -> tuple[list[np.ndarray], list[int], list[np.ndarray]]:
    if recording.get("schema") != BattleSimulation.RECORDING_SCHEMA:
        raise ValueError("需要 glorton-input-recording-v2 真人录像")
    if not bool(recording.get("authoritative_inputs", False)):
        raise ValueError("录像未包含权威双方输入")
    metadata = recording.get("metadata", {})
    human_slots = [int(value) for value in metadata.get("human_slots", [])]
    if not human_slots:
        raise ValueError("录像没有 human_slots")
    fighters = list(metadata.get("fighter_names", []))
    if fighters and any(fighters[slot] != "PeachPlayer" for slot in human_slots):
        raise ValueError("当前只学习 PeachPlayer 真人录像")
    if str(recording.get("stage", "")) != "Mogadishu":
        raise ValueError("当前只学习 Mogadishu 真人录像")

    simulation = BattleSimulation.headless(
        seed=int(recording.get("seed", 0)),
        match_config=recording.get("match_config", {}),
    )
    initial = recording.get("initial_snapshot")
    if not isinstance(initial, Mapping):
        raise ValueError("录像缺少 initial_snapshot")
    simulation.restore_snapshot(initial)
    if strict and simulation.state_digest() != str(recording.get("initial_digest", "")):
        raise ValueError("录像初始状态摘要不一致")
    runtime = simulation.runtime
    controllers = {
        slot: PurposefulOptionController(runtime)
        for slot in human_slots
        if 0 <= slot < len(runtime.fighters)
    }
    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    raw_inputs = list(recording.get("inputs", []))

    for start in range(0, len(raw_inputs), V5_FRAME_SKIP):
        window = raw_inputs[start : start + V5_FRAME_SKIP]
        decisions: list[tuple[int, PurposefulOptionController, Any, Any]] = []
        if runtime.match_state == "playing":
            for slot, controller in controllers.items():
                fighter = runtime.fighters[slot]
                opponents = [
                    item
                    for index, item in enumerate(runtime.fighters)
                    if index != slot and not item.dead
                ]
                if not opponents:
                    continue
                opponent = min(opponents, key=lambda item: fighter.pos.distance_to(item.pos))
                mask = purpose_action_mask(
                    runtime,
                    fighter,
                    opponent,
                    controller,
                    curriculum="duel",
                )
                requested = infer_human_purpose(
                    runtime,
                    fighter,
                    opponent,
                    controller,
                    window,
                    slot=slot,
                )
                purpose = _legal_purpose(requested, mask)
                if purpose is not None:
                    observations.append(
                        encode_v5_observation(
                            runtime,
                            fighter,
                            opponent,
                            controller,
                            episode_ticks=start,
                            max_ticks=max(1, len(raw_inputs)),
                            spawns_swapped=False,
                            curriculum="duel",
                            wall_stall_steps=controller.no_progress_steps,
                        )
                    )
                    actions.append(int(purpose))
                    masks.append(mask.copy())
                    controller.begin_decision(
                        int(purpose),
                        fighter=fighter,
                        opponent=opponent,
                        action_mask=mask,
                    )
                elif controller.has_active_plan:
                    controller.begin_decision(
                        int(Purpose.CONTINUE),
                        fighter=fighter,
                        opponent=opponent,
                        action_mask=mask,
                    )
                decisions.append((slot, controller, fighter, opponent))

        for controls in window:
            simulation._advance_once(
                controls,
                advance_clock=True,
                authoritative_inputs=True,
            )
        for _slot, controller, fighter, opponent in decisions:
            controller.observe_result(fighter, opponent)

    if strict and simulation.state_digest() != str(recording.get("final_digest", "")):
        raise ValueError("录像逐帧回放摘要不一致")
    return observations, actions, masks


def build_human_dataset(
    paths: Iterable[str | Path],
    *,
    wins_only: bool = False,
) -> HumanReplayDataset:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    accepted: list[str] = []
    skipped: dict[str, str] = {}
    for source in paths:
        path = Path(source)
        try:
            recording = BattleSimulation.load_recording(path)
            metadata = recording.get("metadata", {})
            if wins_only:
                human_slots = [int(value) for value in metadata.get("human_slots", [])]
                names = list(metadata.get("fighter_names", []))
                human_names = {
                    f"P{slot + 1}" for slot in human_slots if slot < len(names)
                }
                if str(metadata.get("winner", "")) not in human_names:
                    raise ValueError("真人没有获胜")
            file_observations, file_actions, file_masks = examples_from_recording(recording)
            if not file_actions:
                raise ValueError("没有可学习的合法真人决策")
        except (OSError, ValueError, AssertionError) as exc:
            skipped[str(path)] = str(exc)
            continue
        observations.extend(file_observations)
        actions.extend(file_actions)
        masks.extend(file_masks)
        accepted.append(str(path))

    action_counts = Counter(Purpose(value).name.lower() for value in actions)
    return HumanReplayDataset(
        observations=(
            np.asarray(observations, dtype=np.float32)
            if observations
            else np.empty((0, V5_OBSERVATION_SIZE), dtype=np.float32)
        ),
        actions=np.asarray(actions, dtype=np.int64),
        masks=(
            np.asarray(masks, dtype=bool)
            if masks
            else np.empty((0, PURPOSE_COUNT), dtype=bool)
        ),
        action_counts=dict(action_counts),
        accepted_files=tuple(accepted),
        skipped_files=skipped,
    )
