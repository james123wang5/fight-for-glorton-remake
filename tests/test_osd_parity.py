from __future__ import annotations

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.runtime import ROOT, RuntimeApp, load_manifest


class OSDParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))
        cls.manifest = load_manifest()

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_damage_layout_matches_symbols_780_779_and_778(self) -> None:
        layout = self.manifest["ui"]["layout"]
        self.assertEqual(layout["damage_origin"], {"x": 60, "y": 340})
        self.assertEqual(layout["damage_spacing"], 150)
        self.assertEqual(layout["damage_font"], {"name": "Arial", "size": 23, "bold": True})
        self.assertEqual(layout["damage_field"], {"right": 35.4, "top": -14.85, "align": "right"})
        self.assertEqual(
            layout["damage_glow"],
            {"blur_x": 2.0, "blur_y": 2.0, "strength": 3.328125},
        )
        self.assertEqual(len(layout["damage_pulse"]), 8)
        self.assertEqual(layout["damage_pulse"][1]["scale"], 1.4526215)
        self.assertEqual(layout["damage_pulse"][7]["x"], 5.25)

    def test_fifty_percent_bigicon_alpha_preserves_transparent_pixels(self) -> None:
        source = pygame.image.load(str(ROOT / "assets/ui/osd_bigicon/003.png")).convert_alpha()
        app = RuntimeApp.__new__(RuntimeApp)
        tinted = app._team_tinted_osd_icon(source, 0)

        self.assertEqual(source.get_at((0, 0)).a, 0)
        self.assertEqual(tinted.get_at((0, 0)).a, 0)
        opaque = next(
            (x, y)
            for y in range(source.get_height())
            for x in range(source.get_width())
            if source.get_at((x, y)).a == 255
        )
        self.assertEqual(tinted.get_at(opaque).a, 128)
        self.assertEqual(tinted.get_at(opaque)[:3], (255, 0, 0))

    def test_far_indicator_uses_the_full_dynamic_text_canvas_registration(self) -> None:
        frames = self.manifest["effects"]["FarIndicator"]["frames"]
        self.assertTrue(all(frame["offset"] == {"x": -11.25, "y": -83.0} for frame in frames))
        config = self.manifest["ui"]["layout"]["far_indicator"]
        self.assertEqual(config["source_canvas"], {"x": -16, "y": -90, "w": 34, "h": 49})
        self.assertEqual(
            config["team_colors"],
            [[255, 0, 0], [51, 102, 255], [102, 204, 0], [255, 204, 0]],
        )

    def test_embedded_futura_is_not_synthetically_emboldened_twice(self) -> None:
        app = RuntimeApp.__new__(RuntimeApp)
        app._ui_font_cache = {}
        font = app._ui_font("Futura Md BT", 20, True)
        self.assertFalse(font.get_bold())

    def test_off_camera_ring_is_the_original_conditional_pos_indicator(self) -> None:
        effect = self.manifest["effects"]["PosIndicator"]
        self.assertEqual(effect["symbol_id"], 787)
        self.assertEqual(len(effect["frames"]), 4)
        self.assertTrue(
            all(frame["offset"] == {"x": -44.0, "y": -31.05} for frame in effect["frames"])
        )


if __name__ == "__main__":
    unittest.main()
