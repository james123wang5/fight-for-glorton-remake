from __future__ import annotations

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.runtime import Stage, load_manifest


class AllStagesParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))
        cls.manifest = load_manifest()

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_frame_51_stage_symbols_and_bounds(self) -> None:
        expected = {
            "Rooftop": (721, (100, -100, 950, 450), (-50, -200, 1200, 700)),
            "Mogadishu": (827, (-900, -100, 2600, 500), (-1000, -200, 2700, 700)),
            "B52": (868, (50, -200, 950, 550), (-50, -300, 1200, 700)),
            "Space": (881, (-800, -400, 1500, 1000), (-900, -500, 1700, 1200)),
        }
        self.assertEqual(set(self.manifest["stages"]), set(expected))
        for name, (symbol_id, bounds_cam, bounds) in expected.items():
            data = self.manifest["stages"][name]
            self.assertEqual(data["symbol_id"], symbol_id)
            self.assertEqual(tuple(data["bounds_cam"].values()), bounds_cam)
            self.assertEqual(tuple(data["bounds"].values()), bounds)

    def test_all_stages_have_four_fixed_source_spawn_points(self) -> None:
        for name in self.manifest["stages"]:
            stage = Stage(self.manifest, name)
            object_names = {item["name"] for item in stage.data["objects"]}
            self.assertTrue({"SpawnP1", "SpawnP2", "SpawnP3", "SpawnP4"}.issubset(object_names))
            for index in range(1, 5):
                self.assertNotEqual(stage.spawn_point(f"SpawnP{index}"), pygame.Vector2(stage.bounds_cam.center))

    def test_environment_timelines_and_hazards_are_not_flattened(self) -> None:
        stages = self.manifest["stages"]
        self.assertEqual(len(stages["Rooftop"]["helicopter"]["frames"]), 408)
        self.assertEqual(len(stages["B52"]["dynamic_layer"]["frames"]), 170)
        self.assertTrue(stages["B52"]["dynamic_above_foreground"])
        space_layers = stages["Space"]["background_animation"]["layers"]
        self.assertEqual([len(layer["frames"]) for layer in space_layers], [300])
        self.assertEqual([layer["loop_from"] for layer in space_layers], [151])
        object_layers = stages["Space"]["background_animation"]["object_layers"]
        self.assertEqual(len(object_layers), 1)
        self.assertEqual(len(object_layers[0]["frames"]), 200)
        self.assertEqual(len(object_layers[0]["frames"][0]["matrices"]), 252)
        self.assertEqual(len(Stage(self.manifest, "Mogadishu").boom_rects), 2)
        self.assertEqual(len(Stage(self.manifest, "B52").killer_rects), 1)

    def test_background_registration_uses_the_source_symbol_bounds(self) -> None:
        expected = {
            "Rooftop": (-247.94991248800002, -106.03146821499998),
            "Mogadishu": (-1388.1, -146.98053549600002),
            "B52": (-189.0, -302.95),
            # Space uses the transformed radial base shape rather than the
            # complete animated symbol bounds.
            "Space": (-844.25, -612.75),
        }
        for name, offset in expected.items():
            data = self.manifest["stages"][name]["background_offset"]
            self.assertAlmostEqual(data["x"], offset[0])
            self.assertAlmostEqual(data["y"], offset[1])

    def test_moving_platform_timelines_follow_each_stage_outer_timeline(self) -> None:
        expected_lengths = {"Rooftop": 408, "Mogadishu": 1, "B52": 170, "Space": 1}
        for name, length in expected_lengths.items():
            platforms = self.manifest["stages"][name]["moving_platforms"]["platforms"]
            self.assertTrue(platforms)
            self.assertTrue(all(len(frames) == length for frames in platforms.values()))


if __name__ == "__main__":
    unittest.main()
