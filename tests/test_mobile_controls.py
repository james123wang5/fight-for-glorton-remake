from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.display import recommended_window_size
from src.mobile_controls import MobileControlLayout, MobileControls


class MobileControlsTests(unittest.TestCase):
    SIZE = (1600, 800)

    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    @staticmethod
    def finger_event(kind: int, finger_id: int, pos: pygame.Vector2) -> pygame.event.Event:
        return pygame.event.Event(
            kind,
            finger_id=finger_id,
            x=pos.x / MobileControlsTests.SIZE[0],
            y=pos.y / MobileControlsTests.SIZE[1],
        )

    def test_desktop_path_is_disabled_without_mobile_flag(self) -> None:
        with patch.dict(os.environ, {"GLORTON_MOBILE": "0"}):
            controls = MobileControls()
        self.assertFalse(controls.enabled)
        self.assertFalse(any(controls.controls().values()))

    def test_stick_is_continuous_but_jump_and_up_trace_are_one_edge(self) -> None:
        controls = MobileControls(enabled=True)
        layout = MobileControlLayout.for_size(self.SIZE)
        up_right = layout.stick_center + pygame.Vector2(
            layout.stick_radius * 0.7,
            -layout.stick_radius * 0.7,
        )
        controls.handle_battle_event(self.finger_event(pygame.FINGERDOWN, 10, up_right), self.SIZE)

        first = controls.controls()
        second = controls.controls()
        self.assertTrue(first["right"])
        self.assertTrue(first["jump_pressed"])
        self.assertTrue(first["up_trace"])
        self.assertTrue(second["right"])
        self.assertFalse(second["jump_pressed"])
        self.assertFalse(second["up_trace"])

    def test_up_and_attack_same_tick_preserve_upper_attack_combo(self) -> None:
        controls = MobileControls(enabled=True)
        layout = MobileControlLayout.for_size(self.SIZE)
        up = layout.stick_center + pygame.Vector2(0, -layout.stick_radius)
        controls.handle_battle_event(self.finger_event(pygame.FINGERDOWN, 1, up), self.SIZE)
        controls.handle_battle_event(
            self.finger_event(pygame.FINGERDOWN, 2, layout.special_center),
            self.SIZE,
        )

        sample = controls.controls()
        self.assertTrue(sample["jump_pressed"])
        self.assertTrue(sample["up_trace"])
        self.assertTrue(sample["special_pressed"])

    def test_three_action_buttons_are_independent_multitouch_edges(self) -> None:
        controls = MobileControls(enabled=True)
        layout = MobileControlLayout.for_size(self.SIZE)
        controls.handle_battle_event(
            self.finger_event(pygame.FINGERDOWN, 1, layout.punch_center), self.SIZE
        )
        controls.handle_battle_event(
            self.finger_event(pygame.FINGERDOWN, 2, layout.special_center), self.SIZE
        )
        controls.handle_battle_event(
            self.finger_event(pygame.FINGERDOWN, 3, layout.shield_center), self.SIZE
        )
        sample = controls.controls()
        self.assertTrue(sample["punch_pressed"])
        self.assertTrue(sample["special_pressed"])
        self.assertTrue(sample["shield_pressed"])

        controls.handle_battle_event(
            self.finger_event(pygame.FINGERUP, 3, layout.shield_center), self.SIZE
        )
        self.assertTrue(controls.controls()["shield_released"])

    def test_pause_is_a_separate_edge(self) -> None:
        controls = MobileControls(enabled=True)
        layout = MobileControlLayout.for_size(self.SIZE)
        point = pygame.Vector2(layout.pause_rect.center)
        controls.handle_battle_event(self.finger_event(pygame.FINGERDOWN, 7, point), self.SIZE)
        self.assertTrue(controls.take_pause_toggle())
        self.assertFalse(controls.take_pause_toggle())

    def test_mobile_canvas_size_is_bounded_and_desktop_default_is_unchanged(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(recommended_window_size((1280, 760)), (1280, 760))
        with patch.dict(
            os.environ,
            {"GLORTON_WEB_WIDTH": "2532", "GLORTON_WEB_HEIGHT": "1170"},
            clear=True,
        ):
            width, height = recommended_window_size((1280, 760))
        self.assertLessEqual(width, 1920)
        self.assertLessEqual(height, 1080)
        self.assertGreater(width, height)


if __name__ == "__main__":
    unittest.main()
