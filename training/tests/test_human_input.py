from __future__ import annotations

import unittest

import numpy as np

from training.human_input import HumanInputAdapter


class HumanInputAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = HumanInputAdapter()

    def test_held_attack_must_be_released_before_it_can_trigger_again(self) -> None:
        first, _ = self.adapter.begin_decision(np.asarray([0, 0, 1]))
        held, _ = self.adapter.begin_decision(np.asarray([0, 0, 1]))
        released, _ = self.adapter.begin_decision(np.asarray([0, 0, 0]))
        pressed_again, _ = self.adapter.begin_decision(np.asarray([0, 0, 1]))
        self.assertTrue(first["punch_pressed"])
        self.assertFalse(held["punch_pressed"])
        self.assertFalse(released["punch_pressed"])
        self.assertTrue(pressed_again["punch_pressed"])

    def test_direction_attack_and_up_special_are_same_tick_combinations(self) -> None:
        throw_attempt, _ = self.adapter.begin_decision(np.asarray([2, 0, 1]))
        self.adapter.begin_decision(np.asarray([2, 0, 0]))
        rocket, _ = self.adapter.begin_decision(np.asarray([2, 1, 2]))
        self.assertTrue(throw_attempt["right"])
        self.assertTrue(throw_attempt["punch_pressed"])
        self.assertTrue(rocket["up_trace"])
        self.assertTrue(rocket["special_pressed"])
        self.assertFalse(rocket["jump_pressed"])

    def test_jump_and_shield_use_real_press_edges(self) -> None:
        jump, _ = self.adapter.begin_decision(np.asarray([0, 1, 0]))
        held_jump, _ = self.adapter.begin_decision(np.asarray([0, 1, 0]))
        self.adapter.begin_decision(np.asarray([0, 0, 0]))
        shield, _ = self.adapter.begin_decision(np.asarray([0, 0, 3]))
        held_shield, _ = self.adapter.begin_decision(np.asarray([0, 0, 0]))
        release, _ = self.adapter.begin_decision(np.asarray([0, 0, 0]))
        instant_reopen, _ = self.adapter.begin_decision(np.asarray([0, 0, 3]))
        self.assertTrue(jump["jump_pressed"])
        self.assertFalse(held_jump["jump_pressed"])
        self.assertTrue(shield["shield_pressed"])
        self.assertFalse(held_shield.get("shield_released", False))
        self.assertTrue(release["shield_released"])
        self.assertFalse(instant_reopen["shield_pressed"])


if __name__ == "__main__":
    unittest.main()
