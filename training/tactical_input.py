from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


MOVEMENT_LABELS = ("stop", "approach", "retreat", "continue")
COMBAT_LABELS = (
    "none",
    "jump",
    "drop",
    "melee",
    "uppercut",
    "back_throw",
    "shoot",
    "rocket",
    "shield",
)
TACTICAL_ACTION_NVECS = np.asarray([len(MOVEMENT_LABELS), len(COMBAT_LABELS)], dtype=np.int64)


@dataclass
class TacticalInputAdapter:
    """Convert persistent tactical intentions into legal source-style key edges."""

    frame_skip: int = 4
    movement_commitment_decisions: int = 2
    combat_cooldown_decisions: int = 2
    shield_min_hold_decisions: int = 2
    shield_rearm_decisions: int = 3

    def __post_init__(self) -> None:
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        self.reset()

    def reset(self) -> None:
        self.movement_intent = 0
        self.previous_combat = 0
        self.movement_lock = 0
        self.combat_cooldown = 0
        self.shield_hold = 0
        self.shield_rearm = 0
        self.last_horizontal = 0
        self.last_action = np.zeros(2, dtype=np.int64)
        self.invalid_requests = 0
        self.combat_edges = 0
        self.shield_starts = 0

    def action_mask_prefix(self) -> list[bool]:
        if self.movement_lock <= 0:
            return [True] * len(MOVEMENT_LABELS)
        allowed = [False] * len(MOVEMENT_LABELS)
        allowed[self.movement_intent] = True
        allowed[3] = True
        return allowed

    def _accept_movement(self, requested: int) -> int:
        self.movement_lock = max(0, self.movement_lock - 1)
        if requested == 3:
            return self.movement_intent
        if requested != self.movement_intent and self.movement_lock <= 0:
            self.movement_intent = requested
            self.movement_lock = self.movement_commitment_decisions
        return self.movement_intent

    @staticmethod
    def _horizontal_for_intent(intent: int, fighter: Any, opponent: Any) -> int:
        if intent == 0:
            return 0
        target_right = float(opponent.pos.x) >= float(fighter.pos.x)
        if intent == 1:
            return 2 if target_right else 1
        if intent == 2:
            return 1 if target_right else 2
        return 0

    @staticmethod
    def _held_controls(horizontal: int, *, down: bool = False) -> dict[str, bool]:
        return {
            "left": horizontal == 1,
            "right": horizontal == 2,
            "down": bool(down),
        }

    def begin_decision(
        self,
        action: np.ndarray | Sequence[int],
        *,
        fighter: Any,
        opponent: Any,
        action_mask: np.ndarray | Sequence[bool],
    ) -> tuple[dict[str, bool], ...]:
        candidate = np.asarray(action, dtype=np.int64).reshape(-1)
        if candidate.shape != (2,):
            raise ValueError(f"expected two tactical action components, got {candidate!r}")
        movement = int(np.clip(candidate[0], 0, len(MOVEMENT_LABELS) - 1))
        combat = int(np.clip(candidate[1], 0, len(COMBAT_LABELS) - 1))
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.shape != (len(MOVEMENT_LABELS) + len(COMBAT_LABELS),):
            raise ValueError(f"invalid tactical action mask shape: {mask.shape}")
        if not mask[movement] or not mask[len(MOVEMENT_LABELS) + combat]:
            self.invalid_requests += 1
            movement = 3 if mask[3] else self.movement_intent
            combat = 0

        movement = self._accept_movement(movement)
        horizontal = self._horizontal_for_intent(movement, fighter, opponent)
        self.combat_cooldown = max(0, self.combat_cooldown - 1)
        self.shield_rearm = max(0, self.shield_rearm - 1)

        if combat == 0:
            self.previous_combat = 0
        if combat != 8 and bool(fighter.shielded):
            if self.shield_hold > 0:
                self.shield_hold -= 1
                combat = 8
            else:
                first = self._held_controls(horizontal)
                first["shield_released"] = True
                # The release decision itself starts the first 100 ms interval;
                # storing N-1 here makes N=3 reopen exactly 300 ms later.
                self.shield_rearm = max(0, self.shield_rearm_decisions - 1)
                self.previous_combat = 0
                controls = [first]
                controls.extend(self._held_controls(horizontal) for _ in range(self.frame_skip - 1))
                self.last_horizontal = horizontal
                self.last_action = np.asarray([movement, 0], dtype=np.int64)
                return tuple(controls)

        first = self._held_controls(horizontal, down=combat == 2)
        if combat == 8:
            if not bool(fighter.shielded) and self.shield_rearm <= 0:
                first["shield_pressed"] = True
                self.shield_hold = self.shield_min_hold_decisions
                self.shield_starts += 1
            elif self.shield_hold > 0:
                self.shield_hold -= 1
            self.previous_combat = 8
        elif combat and self.combat_cooldown <= 0 and combat != self.previous_combat:
            if combat == 1:
                first["jump_pressed"] = True
            elif combat == 3:
                first["punch_pressed"] = True
            elif combat == 4:
                horizontal = 0
                first = self._held_controls(horizontal)
                first["up_trace"] = True
                first["punch_pressed"] = True
            elif combat == 5:
                # Back throw requires the victim to remain behind attack_facing.
                # Move away from that victim on the punch edge, matching the
                # source direction+punch combination without turning into it.
                horizontal = 1 if opponent.pos.x >= fighter.pos.x else 2
                first = self._held_controls(horizontal)
                first["punch_pressed"] = True
            elif combat == 6:
                first["special_pressed"] = True
            elif combat == 7:
                first["up_trace"] = True
                first["special_pressed"] = True
            if combat not in {1, 2, 8}:
                self.combat_cooldown = self.combat_cooldown_decisions
            self.previous_combat = combat
            self.combat_edges += 1

        controls = [first]
        controls.extend(
            self._held_controls(horizontal, down=combat == 2)
            for _ in range(self.frame_skip - 1)
        )
        self.last_horizontal = horizontal
        self.last_action = np.asarray([movement, combat], dtype=np.int64)
        return tuple(controls)

    def features(self) -> list[float]:
        """Legacy-compatible 13 features for the frozen v2 opponent."""

        horizontal = [float(self.last_horizontal == index) for index in range(3)]
        vertical_class = 1 if self.previous_combat in {1, 4, 7} else (2 if self.previous_combat == 2 else 0)
        vertical = [float(vertical_class == index) for index in range(3)]
        if self.previous_combat in {3, 4, 5}:
            combat_class = 1
        elif self.previous_combat in {6, 7}:
            combat_class = 2
        elif self.previous_combat == 8:
            combat_class = 3
        else:
            combat_class = 0
        combat = [float(combat_class == index) for index in range(4)]
        return [
            *horizontal,
            *vertical,
            *combat,
            self.movement_lock / max(1, self.movement_commitment_decisions),
            self.shield_hold / max(1, self.shield_min_hold_decisions),
            self.shield_rearm / max(1, self.shield_rearm_decisions),
        ]

    def tactical_features(self) -> list[float]:
        values = [float(self.movement_intent == index) for index in range(len(MOVEMENT_LABELS))]
        values.extend(float(self.previous_combat == index) for index in range(len(COMBAT_LABELS)))
        values.extend(
            [
                self.movement_lock / max(1, self.movement_commitment_decisions),
                self.combat_cooldown / max(1, self.combat_cooldown_decisions),
                self.shield_hold / max(1, self.shield_min_hold_decisions),
                self.shield_rearm / max(1, self.shield_rearm_decisions),
            ]
        )
        return values
