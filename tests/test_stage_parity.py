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

    def test_mogadishu_source_top_line_keeps_a_normal_double_jump_alive(self) -> None:
        stage = Stage(self.manifest, "Mogadishu")
        self.assertEqual(stage.bounds.top, -200)
        self.assertEqual(stage.bounds_cam.top, -100)
        platform = next(item for item in stage.platforms if item.name == "Moving2")
        fighter = self.fighter(stage)
        fighter.pos.update(platform.rect.centerx, platform.rect.top)
        fighter.prev_pos.update(fighter.pos)
        fighter.on_ground = True
        fighter.ground_platform = platform
        fighter.state = "stop"
        fighter.has_control = True
        fighter.fixed_tick(stage, {"jump_pressed": True})
        minimum_y = float(fighter.pos.y)
        second_jump = False
        for _tick in range(80):
            controls = {}
            if not second_jump and fighter.jumpstate == 1 and fighter.yinc >= -3.0:
                controls = {"jump_pressed": True}
                second_jump = True
            fighter.fixed_tick(stage, controls)
            minimum_y = min(minimum_y, float(fighter.pos.y))
            if fighter.dead:
                break
        self.assertTrue(second_jump)
        self.assertEqual(minimum_y, -192.5)
        self.assertFalse(fighter.dead)

    def test_mogadishu_narrow_walls_use_source_midpoint_and_corner_probes(self) -> None:
        stage = Stage(self.manifest, "Mogadishu")
        wall = next(item for item in stage.platforms if item.name == "Fixed12")
        side_body = pygame.Rect(wall.rect.left - 23, wall.rect.top + 12, 28, 29)
        side = stage.collision_probe_flags(side_body, wall)
        self.assertTrue(side.right)
        self.assertFalse(side.left)

        corner_body = pygame.Rect(wall.rect.left - 23, wall.rect.top - 28, 28, 29)
        corner = stage.collision_probe_flags(corner_body, wall)
        self.assertFalse(corner.side)
        self.assertTrue(corner.right_bottom)

    def test_mogadishu_fixed12_and_fixed13_stop_high_speed_side_tunneling(self) -> None:
        stage = Stage(self.manifest, "Mogadishu")
        base = next(item for item in stage.platforms if item.name == "Fixed1")
        for wall_name in ("Fixed12", "Fixed13"):
            wall = next(item for item in stage.platforms if item.name == wall_name)
            for direction in (-1, 1):
                fighter = self.fighter(stage)
                start_x = (
                    wall.rect.left - fighter.body_half_width - 50
                    if direction > 0
                    else wall.rect.right + fighter.body_half_width + 50
                )
                fighter.pos.update(start_x, base.rect.top)
                fighter.prev_pos.update(fighter.pos)
                fighter.xinc = 120.0 * direction
                fighter.yinc = 0.0
                fighter.state = "thrown"
                old_x, old_y = fighter.pos

                fighter._move_with_stage(stage, old_x, old_y)

                expected = (
                    wall.rect.left - fighter.body_half_width
                    if direction > 0
                    else wall.rect.right + fighter.body_half_width
                )
                self.assertEqual(fighter.pos.x, expected)
                self.assertFalse(fighter.body_rect().colliderect(wall.rect))

    def test_mogadishu_foot_only_corner_graze_does_not_become_a_side_wall(self) -> None:
        stage = Stage(self.manifest, "Mogadishu")
        wall = next(item for item in stage.platforms if item.name == "Fixed12")
        fighter = self.fighter(stage)
        old_x = wall.rect.left - fighter.body_half_width - 5
        new_x = old_x + 12
        old_body = fighter.body_rect_at(old_x, wall.rect.top + 1)
        new_body = fighter.body_rect_at(new_x, wall.rect.top + 1)

        collision = stage.find_side_crossing(
            old_x,
            new_x,
            old_body.top,
            old_body.bottom,
            new_body.top,
            new_body.bottom,
            fighter.body_half_width,
        )

        self.assertIsNone(collision)

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
