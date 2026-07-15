from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from training.tactical_input import TacticalInputAdapter


def _fighter(x: float, *, facing: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        pos=SimpleNamespace(x=x),
        facing=facing,
        shielded=False,
        current_attack="",
    )


class TacticalInputAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = TacticalInputAdapter(frame_skip=4)
        self.fighter = _fighter(100)
        self.opponent = _fighter(200, facing=-1)
        self.all_legal = np.ones(13, dtype=bool)

    def decide(self, movement: int, combat: int, mask: np.ndarray | None = None):
        return self.adapter.begin_decision(
            np.asarray([movement, combat]),
            fighter=self.fighter,
            opponent=self.opponent,
            action_mask=self.all_legal if mask is None else mask,
        )

    def test_one_decision_persists_for_four_source_ticks(self) -> None:
        controls = self.decide(1, 6)
        self.assertEqual(len(controls), 4)
        self.assertTrue(controls[0]["right"])
        self.assertTrue(controls[0]["special_pressed"])
        self.assertTrue(all(item["right"] for item in controls))
        self.assertFalse(any(item.get("special_pressed", False) for item in controls[1:]))

    def test_movement_commitment_rejects_instant_direction_flip(self) -> None:
        self.decide(1, 0)
        mask = np.asarray([*self.adapter.action_mask_prefix(), *([True] * 9)])
        controls = self.decide(2, 0, mask)
        self.assertTrue(all(item["right"] for item in controls))
        self.assertEqual(self.adapter.invalid_requests, 1)

    def test_back_throw_combo_moves_away_and_keeps_victim_behind(self) -> None:
        self.fighter.pos.x = 100
        self.fighter.facing = 1
        self.opponent.pos.x = 90
        controls = self.decide(0, 5)
        self.assertTrue(controls[0]["right"])
        self.assertTrue(controls[0]["punch_pressed"])
        self.assertFalse(controls[0]["left"])

    def test_shield_has_hold_and_rearm_windows(self) -> None:
        opened = self.decide(0, 8)
        self.assertTrue(opened[0]["shield_pressed"])
        self.fighter.shielded = True
        held = self.decide(0, 0)
        self.assertFalse(held[0].get("shield_released", False))
        self.decide(0, 0)
        released = self.decide(0, 0)
        self.assertTrue(released[0]["shield_released"])
        self.fighter.shielded = False
        mask = self.all_legal.copy()
        mask[4 + 8] = False
        reopened = self.decide(0, 8, mask)
        self.assertFalse(reopened[0].get("shield_pressed", False))


if __name__ == "__main__":
    unittest.main()
