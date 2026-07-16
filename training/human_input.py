from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class HumanInputAdapter:
    """Turn policy key states into human-like press edges and short commitments."""

    direction_commitment_decisions: int = 2
    shield_min_hold_decisions: int = 1
    shield_rearm_decisions: int = 4

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.horizontal = 0
        self.previous_vertical = 0
        self.previous_combat = 0
        self.direction_lock = 0
        self.shield_hold = 0
        self.shield_cooldown = 0
        self.last_action = np.zeros(3, dtype=np.int64)

    def _accept_horizontal(self, requested: int) -> int:
        self.direction_lock = max(0, self.direction_lock - 1)
        if requested != self.horizontal:
            if self.direction_lock == 0:
                self.horizontal = requested
                self.direction_lock = self.direction_commitment_decisions
        return self.horizontal

    def _accept_combat(self, requested: int) -> int:
        if self.previous_combat != 3:
            self.shield_cooldown = max(0, self.shield_cooldown - 1)
        if self.previous_combat == 3:
            if self.shield_hold > 0:
                self.shield_hold -= 1
                return 3
            if requested != 3:
                self.shield_cooldown = self.shield_rearm_decisions
                return requested
            return 3
        if requested == 3:
            if self.shield_cooldown > 0:
                return 0
            self.shield_hold = self.shield_min_hold_decisions
            return 3
        return requested

    def begin_decision(
        self,
        action: np.ndarray | Sequence[int],
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        candidate = np.asarray(action, dtype=np.int64).reshape(-1)
        if candidate.shape != (3,):
            raise ValueError(f"expected three action components, got {candidate!r}")
        requested_horizontal = int(np.clip(candidate[0], 0, 2))
        vertical = int(np.clip(candidate[1], 0, 2))
        requested_combat = int(np.clip(candidate[2], 0, 3))
        horizontal = self._accept_horizontal(requested_horizontal)
        combat = self._accept_combat(requested_combat)

        punch_edge = combat == 1 and self.previous_combat != 1
        special_edge = combat == 2 and self.previous_combat != 2
        shield_edge = combat == 3 and self.previous_combat != 3
        shield_release = combat != 3 and self.previous_combat == 3
        jump_edge = vertical == 1 and self.previous_vertical != 1 and combat == 0
        first = {
            "left": horizontal == 1,
            "right": horizontal == 2,
            "down": vertical == 2,
            "up_trace": vertical == 1 and (punch_edge or special_edge),
            "jump_pressed": jump_edge,
            "punch_pressed": punch_edge,
            "special_pressed": special_edge,
            "shield_pressed": shield_edge,
            "shield_released": shield_release,
        }
        second = {
            "left": horizontal == 1,
            "right": horizontal == 2,
            "down": vertical == 2,
        }
        self.previous_vertical = vertical
        self.previous_combat = combat
        self.last_action = np.asarray([horizontal, vertical, combat], dtype=np.int64)
        return first, second

    def features(self) -> list[float]:
        values = [float(self.horizontal == index) for index in range(3)]
        values.extend(float(self.previous_vertical == index) for index in range(3))
        values.extend(float(self.previous_combat == index) for index in range(4))
        values.extend(
            [
                self.direction_lock / max(1, self.direction_commitment_decisions),
                self.shield_hold / max(1, self.shield_min_hold_decisions),
                self.shield_cooldown / max(1, self.shield_rearm_decisions),
            ]
        )
        return values
