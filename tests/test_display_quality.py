from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.display import aligned_aspect_rect, detect_display_metrics, snap_to_device_pixel
from src.runtime import RuntimeApp


class DisplayQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((600, 400))

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_retina_override_and_device_pixel_alignment(self) -> None:
        surface = pygame.display.get_surface()
        with patch.dict(os.environ, {"GLORTON_PIXEL_RATIO": "2"}):
            metrics = detect_display_metrics(surface)
        self.assertEqual(metrics.pixel_ratio, 2.0)
        self.assertTrue(metrics.native_pixels)
        self.assertEqual(snap_to_device_pixel(1.26, 2), 1.5)
        self.assertEqual(aligned_aspect_rect((1281, 760), pixel_ratio=2), pygame.Rect(70, 0, 1140, 760))

    def test_high_is_a_distinct_two_pass_filter(self) -> None:
        runtime = RuntimeApp(random_seed=3)

        class MenuQuality:
            quality = "MEDIUM"

        runtime.menu = MenuQuality()
        source = pygame.Surface((3, 3), pygame.SRCALPHA)
        source.fill((0, 0, 0, 255))
        source.set_at((1, 1), (255, 255, 255, 255))
        medium = runtime._quality_scale(source, (8, 8))
        runtime.menu.quality = "HIGH"
        high = runtime._quality_scale(source, (8, 8))
        self.assertNotEqual(
            pygame.image.tobytes(medium, "RGBA"),
            pygame.image.tobytes(high, "RGBA"),
        )


if __name__ == "__main__":
    unittest.main()
