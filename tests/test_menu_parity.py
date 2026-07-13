from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.menu import MainMenu
from src.runtime import ROOT, load_manifest


class MenuParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))
        cls.manifest = load_manifest()
        cls.menu = MainMenu(ROOT)

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_dynamic_text_uses_the_exported_futura_md_bt_font(self) -> None:
        expected = pygame.font.Font(str(ROOT / "assets/fonts/2_Futura Md BT.ttf"), 80)
        expected.set_bold(True)
        actual = self.menu._font(20, True)
        self.assertEqual(actual.size("The Fight for Glorton"), expected.size("The Fight for Glorton"))

    def test_preloader_waits_for_exact_play_button_before_sponsor_intro(self) -> None:
        self.menu.reset_to_intro()
        outside = pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(100, 100))
        self.menu.handle_event(outside, (600, 400))
        self.assertEqual(self.menu.scene, "preloader")

        rect = self.manifest["menu"]["preloader"]["play_rect"]
        inside = pygame.event.Event(
            pygame.MOUSEBUTTONUP,
            button=1,
            pos=(round(rect["x"] + rect["w"] / 2), round(rect["y"] + rect["h"] / 2)),
        )
        self.menu.handle_event(inside, (600, 400))
        self.assertEqual(self.menu.scene, "sponsor_intro")
        self.assertEqual(self.menu.sponsor_frame, 1)

    def test_sponsor_intro_plays_all_eighty_one_source_frames_before_opening(self) -> None:
        self.menu.scene = "sponsor_intro"
        self.menu.sponsor_frame = 1
        self.menu.sponsor_elapsed_ms = 0
        sponsor = self.manifest["menu"]["sponsor_intro"]
        self.assertEqual(sponsor["root_symbol_id"], 928)
        self.assertEqual(sponsor["frame_count"], 81)
        self.assertEqual(len(self.menu.sponsor_intro_frames), 81)
        self.assertTrue(all(path.is_file() for path in self.menu.sponsor_intro_frames.paths))
        self.menu.update(80 * 1000 // 30)
        self.assertEqual(self.menu.scene, "sponsor_intro")
        self.assertEqual(self.menu.sponsor_frame, 80)
        self.menu.update(34)
        self.assertEqual(self.menu.scene, "opening")
        self.assertEqual(self.menu.opening_frame, 3)

    def test_sponsor_armor_button_uses_source_frames_and_hit_bounds(self) -> None:
        sponsor = self.manifest["menu"]["sponsor_intro"]
        self.assertEqual(
            (sponsor["armor_button_active_start"], sponsor["armor_button_active_stop"]),
            (23, 81),
        )
        self.assertEqual(
            sponsor["armor_button_rect"],
            {"x": 194.05, "y": 110.5, "w": 212.0, "h": 212.0},
        )
        self.menu.scene = "sponsor_intro"
        self.menu.sponsor_frame = 23
        action = self.menu.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(300, 216)),
            (600, 400),
        )
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, "open_url")
        self.assertEqual(action.payload, {"url": "http://www.armorgames.com"})

    def test_opening_plays_source_frames_3_through_39_at_30_fps(self) -> None:
        self.menu.scene = "opening"
        self.menu.opening_frame = 3
        self.menu.opening_elapsed_ms = 0
        self.menu.update(1200)
        self.assertEqual(self.menu.scene, "opening")
        self.assertEqual(self.menu.opening_frame, 39)
        self.menu.update(34)
        self.assertEqual(self.menu.scene, "intro")

    def test_root_button_hit_rects_are_ffdec_bounds(self) -> None:
        self.menu.scene = "main"
        multi = next(button for button in self.menu.BUTTONS["main"] if button.symbol_id == 965)
        rect = self.menu._button_rect(multi)
        self.assertAlmostEqual(rect.x, 217.1)
        self.assertAlmostEqual(rect.y, 230.25)
        self.assertAlmostEqual(rect.w, 165.0)
        self.assertAlmostEqual(rect.h, 26.0)

    def test_main_button_down_state_uses_source_one_pixel_shift(self) -> None:
        self.menu.scene = "main"
        multi = next(button for button in self.menu.BUTTONS["main"] if button.symbol_id == 965)
        rect = self.menu._button_rect(multi)
        self.menu.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=(round(rect.x + rect.w / 2), round(rect.y + rect.h / 2)),
            ),
            (600, 400),
        )
        self.assertEqual(self.menu.pressed_button, 965)
        self.menu.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONUP,
                button=1,
                pos=(round(rect.x + rect.w / 2), round(rect.y + rect.h / 2)),
            ),
            (600, 400),
        )
        self.assertIsNone(self.menu.pressed_button)

    def test_main_button_release_requires_matching_source_press(self) -> None:
        self.menu.scene = "main"
        multi = next(button for button in self.menu.BUTTONS["main"] if button.symbol_id == 965)
        rect = self.menu._button_rect(multi)
        center = (round(rect.x + rect.w / 2), round(rect.y + rect.h / 2))

        self.menu.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=center),
            (600, 400),
        )
        self.assertEqual(self.menu.scene, "main")

        self.menu.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=center),
            (600, 400),
        )
        self.menu.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(10, 10)),
            (600, 400),
        )
        self.assertEqual(self.menu.scene, "main")

    def test_hover_replaces_white_up_glyph_over_exact_source_texture(self) -> None:
        self.menu.scene = "main"
        multi = next(button for button in self.menu.BUTTONS["main"] if button.symbol_id == 965)
        image = self.menu.buttons[multi.symbol_id].copy()
        image.fill((255, 204, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
        offset_x, offset_y = self.menu.button_hover_offsets[multi.symbol_id]
        position = (
            round(multi.center[0] * 4 - image.get_width() / 2 + offset_x),
            round(multi.center[1] * 4 - image.get_height() / 2 + offset_y),
        )
        area = pygame.Rect(position, image.get_size())
        expected = self.menu.menu_background.subsurface(area).copy()
        expected.blit(image, (0, 0))

        canvas = self.menu._base_canvas()
        self.menu._draw_button_hover(canvas, multi.center)
        actual = canvas.subsurface(area)

        self.assertEqual(pygame.image.tobytes(actual, "RGBA"), pygame.image.tobytes(expected, "RGBA"))

    def test_default_limit_picker_keeps_exact_root_frame_pixels(self) -> None:
        self.menu._start_player_select("vsmode", 4, 4)
        self.menu.limit_mode = "stock"
        self.menu.limit_value = 5
        canvas = self.menu._base_canvas()
        self.menu._draw_player_select(canvas, None)
        picker = pygame.Rect(109 * 4, 0, 376 * 4, 36 * 4)
        expected = self.menu._frame(46).subsurface(picker)
        actual = canvas.subsurface(picker)
        self.assertEqual(pygame.image.tobytes(actual, "RGBA"), pygame.image.tobytes(expected, "RGBA"))

    def test_all_six_fighters_have_four_complete_run_previews(self) -> None:
        expected = {name for name, _ in self.menu.FIGHTERS}
        self.assertEqual(set(self.menu.selection_previews), expected)
        for fighter in expected:
            colors = self.menu.selection_previews[fighter]
            self.assertEqual(len(colors), 4)
            for run in colors:
                self.assertEqual(len(run["frames"]), 18)
                self.assertEqual(run["playback"], {"loop_from": 4, "loop_at": 18})

    def test_time_buttons_follow_source_piecewise_steps(self) -> None:
        self.menu.limit_mode = "time"
        cases = (
            (30, -1, 0),
            (120, -1, 90),
            (300, -1, 240),
            (600, -1, 300),
            (1200, -1, 600),
            (0, 1, 30),
            (120, 1, 150),
            (300, 1, 360),
            (600, 1, 900),
            (1200, 1, 1800),
        )
        for value, direction, expected in cases:
            with self.subTest(value=value, direction=direction):
                self.menu.limit_value = value
                self.menu._adjust_limit(direction)
                self.assertEqual(self.menu.limit_value, expected)

    def test_player_toggle_cycle_and_ai_level_bounds_match_source(self) -> None:
        self.menu._start_player_select("vsmode", 1, 1)
        self.assertFalse(self.menu.computer_players[0])
        self.assertTrue(self.menu.player_enabled[0])
        self.assertEqual(self.menu.player_levels[0], 7)

        self.menu._toggle_player(0)
        self.assertTrue(self.menu.computer_players[0])
        self.assertTrue(self.menu.player_enabled[0])
        self.menu._toggle_player(0)
        self.assertFalse(self.menu.player_enabled[0])
        self.menu._toggle_player(0)
        self.assertFalse(self.menu.computer_players[0])
        self.assertTrue(self.menu.player_enabled[0])

        self.menu.player_levels[0] = 1
        self.menu.player_levels[0] = max(1, self.menu.player_levels[0] - 1)
        self.assertEqual(self.menu.player_levels[0], 1)
        self.menu.player_levels[0] = 20
        self.menu.player_levels[0] = min(20, self.menu.player_levels[0] + 1)
        self.assertEqual(self.menu.player_levels[0], 20)

    def test_duplicate_fighters_receive_next_unused_source_color(self) -> None:
        self.menu._start_player_select("vsmode", 4, 4)
        for index in range(4):
            self.menu.selected_fighters[index] = "PeachPlayer"
            self.menu._resolve_duplicate_color(index)
        self.assertEqual(self.menu.selected_colors, [0, 1, 2, 3])

    def test_generated_player_coin_assets_keep_original_twenty_six_unit_bounds(self) -> None:
        assets = self.manifest["menu"]["player_select"]["coin_assets"]
        self.assertEqual(set(assets), {"1", "2", "3", "4"})
        for item in assets.values():
            self.assertEqual(item["offset"], {"x": -13.0, "y": -13.0})
            self.assertEqual(item["logical_size"], {"w": 26.0, "h": 26.0})

    def test_options_are_persisted_when_source_main_button_is_released(self) -> None:
        self.menu.scene = "options"
        with patch.object(self.menu, "_save_settings") as save:
            self.menu._handle_options_click((109.0, 180.0))
            self.assertEqual(self.menu.quality, "LOW")
            save.assert_not_called()
            rect = self.menu._main_menu_rect()
            self.menu._handle_options_click((rect.x + rect.w / 2, rect.y + rect.h / 2))
            save.assert_called_once_with()
            self.assertEqual(self.menu.scene, "main")

    def test_options_render_only_the_source_selected_checkbox_frames(self) -> None:
        self.menu.scene = "options"
        self.menu.quality = "MEDIUM"
        self.menu.sound_on = True
        canvas = self.menu._base_canvas()
        self.menu._draw_options(canvas)

        def center_pixel(y: float) -> pygame.Color:
            return canvas.get_at((round(99.85 * 4 + 31), round((y - 1.0) * 4 + 31)))

        self.assertLess(max(center_pixel(124.25)[:3]), 200)
        self.assertGreater(min(center_pixel(152.25)[:3]), 240)
        self.assertLess(max(center_pixel(179.25)[:3]), 200)
        self.assertGreater(min(center_pixel(231.15)[:3]), 240)
        self.assertLess(max(center_pixel(259.15)[:3]), 200)

    def test_control_key_confirmation_plays_source_frames_before_hiding(self) -> None:
        self.menu.scene = "controls"
        self.menu.listening_control = (0, 4)
        with patch.object(self.menu, "_save_settings"):
            self.menu.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN),
                (600, 400),
            )
        self.assertIsNone(self.menu.listening_control)
        self.assertEqual(self.menu.control_confirmation, (0, 4, "ENTER"))
        self.assertEqual(self.menu.control_keys[0][4], pygame.K_RETURN)
        self.menu.update(633)
        self.assertIsNotNone(self.menu.control_confirmation)
        self.menu.update(1)
        self.assertIsNone(self.menu.control_confirmation)

    def test_control_key_names_use_the_source_labels(self) -> None:
        expected = {
            pygame.K_RETURN: "ENTER",
            pygame.K_ESCAPE: "ESC",
            pygame.K_PAGEUP: "PGUP",
            pygame.K_PAGEDOWN: "PGDN",
            pygame.K_INSERT: "INS",
            pygame.K_DELETE: "DEL",
        }
        for key, name in expected.items():
            with self.subTest(key=key):
                self.assertEqual(self.menu._key_name(key), name)


if __name__ == "__main__":
    unittest.main()
