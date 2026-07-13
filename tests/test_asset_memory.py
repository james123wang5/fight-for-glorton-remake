from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.assets import LazySurfaceMap, LazySurfaceSequence, SURFACE_CACHE, SurfaceCache
from src.runtime import PeachFighter, Stage, load_manifest


class AssetMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))

    @classmethod
    def tearDownClass(cls) -> None:
        SURFACE_CACHE.clear()
        pygame.quit()

    @staticmethod
    def make_png(root: Path, name: str, color: tuple[int, int, int]) -> Path:
        path = root / name
        surface = pygame.Surface((10, 10), pygame.SRCALPHA)
        surface.fill((*color, 255))
        pygame.image.save(surface, path)
        return path

    def test_surface_cache_is_byte_bounded_and_lru(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                self.make_png(root, f"{index}.png", (index * 30, 0, 0))
                for index in range(1, 4)
            ]
            cache = SurfaceCache(max_bytes=800)
            cache.get(paths[0])
            cache.get(paths[1])
            cache.get(paths[0])
            cache.get(paths[2])

            self.assertLessEqual(cache.current_bytes, cache.max_bytes)
            self.assertIn(paths[0], cache._surfaces)
            self.assertNotIn(paths[1], cache._surfaces)
            self.assertIn(paths[2], cache._surfaces)

    def test_lazy_sequence_decodes_only_on_first_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_png(Path(directory), "frame.png", (20, 40, 60))
            SURFACE_CACHE.clear()
            original_load = pygame.image.load
            with patch("pygame.image.load", wraps=original_load) as load:
                frames = LazySurfaceSequence([path])
                self.assertEqual(load.call_count, 0)
                self.assertEqual(frames[0].get_size(), (10, 10))
                self.assertEqual(frames[0].get_size(), (10, 10))
                self.assertEqual(load.call_count, 1)

    def test_fighter_animation_maps_keep_paths_until_a_frame_is_drawn(self) -> None:
        manifest = load_manifest()
        stage = Stage(manifest)
        SURFACE_CACHE.clear()
        fighter = PeachFighter(manifest, stage.spawn_point("SpawnP1"))

        self.assertIsInstance(fighter.animations["run"]["frames"], LazySurfaceMap)
        self.assertEqual(SURFACE_CACHE.current_bytes, 0)
        self.assertGreater(fighter.current_image().get_width(), 0)
        self.assertGreater(SURFACE_CACHE.current_bytes, 0)
        self.assertLessEqual(SURFACE_CACHE.current_bytes, SURFACE_CACHE.max_bytes)


if __name__ == "__main__":
    unittest.main()
