from __future__ import annotations

import os
import random
import unittest
from unittest.mock import call, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.menu import MainMenu
from src.runtime import PeachFighter, RuntimeApp, Stage, TICK_MS, load_manifest


class RooftopStageParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))
        cls.manifest = load_manifest()

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def stage(self) -> Stage:
        return Stage(self.manifest)

    def fighter(self, stage: Stage, name: str = "P1", color: int = 0) -> PeachFighter:
        fighter = PeachFighter(self.manifest, stage.spawn_point("SpawnP1"), name, color)
        fighter.intro_visible = True
        return fighter

    def camera_runtime(self, stage: Stage, fighters: list[PeachFighter]) -> RuntimeApp:
        runtime = RuntimeApp.__new__(RuntimeApp)
        runtime.stage = stage
        runtime.fighters = fighters
        runtime.player = fighters[0]
        runtime.match_state = "playing"
        runtime.countdown_focus_indices = []
        runtime.camera_view = None
        runtime.camera_tricks = []
        runtime.camera_shake_start_ms = 0
        runtime.camera_shake_until_ms = 0
        runtime.stage_time_ms = 0
        return runtime

    def test_rooftop_source_bounds_and_spawn_points(self) -> None:
        stage = self.stage()
        self.assertEqual(stage.bounds, pygame.Rect(-50, -200, 1200, 700))
        self.assertEqual(stage.bounds_cam, pygame.Rect(100, -100, 950, 450))
        self.assertEqual(stage.spawn_point("SpawnP1"), pygame.Vector2(528.65, 152.5))
        self.assertEqual(stage.spawn_point("SpawnP2"), pygame.Vector2(585.3, 152.5))

    def test_widescreen_uses_the_original_three_to_two_stage_view(self) -> None:
        self.assertEqual(MainMenu._screen_rect((1280, 760)), pygame.Rect(70, 0, 1140, 760))
        self.assertEqual(MainMenu._screen_rect((2048, 1200)), pygame.Rect(124, 0, 1800, 1200))

    def test_camera_focus_keeps_real_out_of_bounds_hit_testers(self) -> None:
        stage = self.stage()
        left = self.fighter(stage)
        right = self.fighter(stage, "P2", 1)
        left.pos.update(-49, 0)
        right.pos.update(1149, 0)
        for fighter in (left, right):
            fighter.prev_pos.update(fighter.pos)
            fighter._check_bounds(stage)
            self.assertTrue(fighter.out_of_camera)
        runtime = self.camera_runtime(stage, [left, right])

        target = runtime._camera_target(pygame.Rect(0, 0, 600, 400))

        self.assertLessEqual(target[0], left.body_rect().left - 50)
        self.assertGreaterEqual(target[0] + target[2], right.body_rect().right + 50)
        self.assertGreater(target[2], stage.bounds_cam.w)

    def test_mogadishu_extreme_separation_uses_unclamped_source_zoom(self) -> None:
        stage = Stage(self.manifest, "Mogadishu")
        left = self.fighter(stage)
        right = self.fighter(stage, "P2", 1)
        left.pos.update(stage.bounds_cam.left + 1, 0)
        right.pos.update(stage.bounds_cam.right - 1, 0)
        for fighter in (left, right):
            fighter.prev_pos.update(fighter.pos)
            fighter._check_bounds(stage)
        runtime = self.camera_runtime(stage, [left, right])
        viewport = pygame.Rect(0, 0, 600, 400)

        cam, zoom = runtime._camera(viewport)

        self.assertLess(zoom, 0.35)
        self.assertAlmostEqual(
            zoom,
            min(viewport.w / runtime.camera_view[2], viewport.h / runtime.camera_view[3]),
        )
        for fighter in (left, right):
            screen_pos = runtime._world_to_screen(fighter.pos, cam, zoom, viewport)
            self.assertTrue(viewport.inflate(2, 2).collidepoint(screen_pos))

    def test_camera_first_step_matches_zeroed_vcamera_init(self) -> None:
        stage = self.stage()
        fighter = self.fighter(stage)
        runtime = self.camera_runtime(stage, [fighter])
        runtime.match_state = "countdown"
        runtime.countdown_focus_indices = [0]
        runtime.camera_view = [0.0, 0.0, 0.0, 0.0]
        target = runtime._camera_target(pygame.Rect(0, 0, 600, 400))

        runtime._step_camera()

        expected = [value / 5 for value in target]
        expected = runtime._clamp_camera_view(expected)
        for actual, wanted in zip(runtime.camera_view, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_helicopter_platform_carries_a_grounded_fighter(self) -> None:
        stage = self.stage()
        stage.set_time(0)
        platform = next(item for item in stage.platforms if item.name == "Moving4")
        fighter = self.fighter(stage)
        fighter.pos.update(platform.rect.centerx, platform.rect.top)
        fighter.prev_pos.update(fighter.pos)
        fighter.on_ground = True
        fighter.ground_platform = platform
        old_pos = pygame.Vector2(fighter.pos)

        stage.set_time(134)
        dx = platform.rect.x - platform.prev_rect.x
        self.assertNotEqual(dx, 0)
        fighter._carry_with_moving_platform()

        self.assertEqual(fighter.pos.x, old_pos.x + dx)
        self.assertEqual(fighter.pos.y, platform.rect.top)

    def test_item_generator_uses_source_spawn_index_and_truncated_width(self) -> None:
        stage = self.stage()
        zone = next(item for item in stage.data["objects"] if item["name"] == "SpawnH3")
        zone["estimated_rect"]["w"] = 10.9

        with patch("src.runtime.random.randrange", side_effect=[2, 9]) as source_random:
            point = stage.item_spawn_point()

        self.assertEqual(
            point,
            pygame.Vector2(float(zone["matrix"]["x"]) + 9, float(zone["matrix"]["y"])),
        )
        self.assertEqual(source_random.call_args_list, [call(4), call(10)])

    def test_respawn_waits_on_cloud_then_can_leave_while_invincible(self) -> None:
        random.seed(17)
        stage = self.stage()
        fighter = self.fighter(stage)
        fighter.lives = 2
        fighter.pos.update(300, stage.bounds.bottom + 1)
        fighter.die("bot", stage)
        fighter.spawn_effect_kind = 1
        fighter.spawn_reveal_frame = 10
        spawn_pos = pygame.Vector2(fighter.pos)

        while not fighter.has_control:
            fighter.fixed_tick(stage, {})

        self.assertEqual(fighter.state, "spawn")
        self.assertEqual(fighter.pos, spawn_pos)
        self.assertTrue(fighter.invincible)
        fighter.move("right")
        fighter.fixed_tick(stage, {})
        self.assertEqual(fighter.state, "goright")
        self.assertGreater(fighter.pos.x, spawn_pos.x)
        self.assertTrue(fighter.invincible)

    def test_idle_respawn_starts_falling_only_after_source_timeout(self) -> None:
        random.seed(23)
        stage = self.stage()
        fighter = self.fighter(stage)
        fighter.lives = 2
        fighter.pos.update(300, stage.bounds.bottom + 1)
        fighter.die("bot", stage)
        spawn_pos = pygame.Vector2(fighter.pos)

        for _ in range(120):
            fighter.fixed_tick(stage, {})
        self.assertEqual(fighter.state, "spawn")
        self.assertEqual(fighter.pos, spawn_pos)
        fighter.fixed_tick(stage, {})
        self.assertEqual(fighter.state, "stop")
        self.assertEqual(fighter.pos, spawn_pos)
        self.assertGreater(fighter.yinc, 0)
        self.assertFalse(fighter.invincible)
        fighter.fixed_tick(stage, {})
        self.assertGreater(fighter.pos.y, spawn_pos.y)


if __name__ == "__main__":
    unittest.main()
