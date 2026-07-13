from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.menu import MainMenu
from src.runtime import AIController, RuntimeApp


ROOT = Path(__file__).resolve().parents[1]


class MultiplayerAIParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))
        cls.runtime = RuntimeApp()

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def configure_four(self) -> RuntimeApp:
        runtime = self.runtime
        runtime.match_config = {
            "type": "vsmode",
            "players": [
                {"fighter": "PeachPlayer", "color": 0, "computer": False, "enabled": True, "level": 7},
                {"fighter": "TrashPlayer", "color": 1, "computer": True, "enabled": True, "level": 9},
                {"fighter": "CoffeePlayer", "color": 2, "computer": False, "enabled": True, "level": 7},
                {"fighter": "DefaultPlayer", "color": 3, "computer": True, "enabled": True, "level": 12},
            ],
            "limit_mode": "stock",
            "limit_value": 5,
        }
        runtime._reset_match()
        return runtime

    def test_disabled_player_is_not_passed_from_selection_to_init_area(self) -> None:
        menu = MainMenu(ROOT)
        menu._start_player_select("vsmode", 4, 4)
        menu.selected_fighters = ["PeachPlayer", "TrashPlayer", "CoffeePlayer", "DefaultPlayer"]
        menu.selected_colors = [0, 1, 2, 3]
        menu.player_enabled[2] = False
        action = menu._start_game_action()
        self.assertEqual([item["fighter"] for item in action.payload["players"]], ["PeachPlayer", "TrashPlayer", "DefaultPlayer"])
        self.assertEqual([item["team_index"] for item in action.payload["players"]], [0, 1, 3])

    def test_runtime_builds_four_source_spawn_slots_and_cpu_controllers(self) -> None:
        runtime = self.configure_four()
        self.assertEqual([fighter.name for fighter in runtime.fighters], ["P1", "P2", "P3", "P4"])
        self.assertEqual([fighter.fighter_name for fighter in runtime.fighters], ["PeachPlayer", "TrashPlayer", "CoffeePlayer", "DefaultPlayer"])
        self.assertEqual(set(runtime.ai_controllers), {1, 3})
        self.assertEqual(runtime.ai_controllers[1].level, 9)
        self.assertEqual(runtime.ai_controllers[3].level, 12)

    def test_match_reset_discards_surface_id_bound_caches(self) -> None:
        runtime = self.runtime
        runtime._surface_bounds_cache[123] = pygame.Rect(1, 2, 3, 4)
        runtime._stage_surface_cache[(123, 4, 5, "auto")] = pygame.Surface((4, 5))
        runtime._reset_match()
        self.assertEqual(runtime._surface_bounds_cache, {})
        self.assertEqual(runtime._stage_surface_cache, {})

    def test_four_player_countdown_reveals_p1_p2_p3_p4_in_source_order(self) -> None:
        runtime = self.configure_four()
        for fighter in runtime.fighters:
            fighter.intro_visible = False
        expected = {5: [0], 4: [1], 3: [2], 2: [3], 1: [0, 1, 2, 3]}
        for ready_set, focus in expected.items():
            runtime.ready_set = ready_set
            runtime._apply_ready_step()
            self.assertEqual(runtime.countdown_focus_indices, focus)

    def test_countdown_advances_nested_spawn_timeline_past_transparent_lead_frames(self) -> None:
        runtime = self.runtime
        runtime.match_config = {
            "type": "vsmode",
            "players": [
                {"fighter": "SBLPlayer", "color": 0, "computer": False, "enabled": True},
                {"fighter": "PeachPlayer", "color": 1, "computer": False, "enabled": True},
            ],
            "limit_mode": "stock",
            "limit_value": 5,
        }
        runtime._reset_match()
        runtime._start_match_countdown()
        fighter = runtime.fighters[0]

        self.assertEqual(fighter.current_label, "spawn")
        self.assertEqual(fighter._timeline_frame("spawn"), 1)
        self.assertLessEqual(fighter.current_image().get_bounding_rect().w, 1)

        for _ in range(6):
            runtime._fixed_tick_countdown([{}, {}])

        self.assertEqual(fighter._timeline_frame("spawn"), 5)
        self.assertGreater(fighter.current_image().get_bounding_rect().w, 1)

    def test_pregame_keycombi_updates_commands_but_gameon_freezes_physics(self) -> None:
        runtime = self.runtime
        runtime.match_config = {
            "type": "vsmode",
            "players": [
                {"fighter": "PeachPlayer", "color": 0, "computer": False, "enabled": True},
                {"fighter": "PeachPlayer", "color": 1, "computer": False, "enabled": True},
            ],
            "limit_mode": "stock",
            "limit_value": 5,
        }
        runtime._reset_match()
        fighter = runtime.fighters[0]
        start = pygame.Vector2(fighter.pos)

        runtime._handle_keydown(next(iter(runtime.inputs[0].right_keys)))
        runtime._handle_keydown(next(iter(runtime.inputs[0].punch_keys)))
        self.assertTrue(fighter.has_control)
        self.assertEqual((fighter.state, fighter.xinc), ("goright", fighter.move_xinc))
        self.assertEqual(fighter.current_attack, "punchAir")

        animation_time = fighter.animation_time_ms
        runtime._fixed_tick_countdown([{}, {}])

        self.assertEqual(fighter.pos, start)
        self.assertGreater(fighter.animation_time_ms, animation_time)
        self.assertEqual((runtime.bullets, runtime.rockets, runtime.special_projectiles), ([], [], []))

    def test_ai_action_queue_uses_exact_inverse_level_delay(self) -> None:
        runtime = self.configure_four()
        controller = AIController(runtime.fighters[1], runtime.stage, level=10)
        with patch("src.runtime.random.randrange", return_value=125):
            controller.act("punch", "none")
        self.assertEqual(controller.action_delay_ms, 175)
        self.assertEqual(controller.queued_action, ("punch", "none"))

    def test_source_p1_cpu_first_candidate_index_quirk_is_preserved(self) -> None:
        runtime = self.configure_four()
        controller = AIController(runtime.fighters[0], runtime.stage, level=7)
        runtime.fighters[0].pos.update(0, 0)
        runtime.fighters[1].pos.update(10, 0)
        runtime.fighters[2].pos.update(1000, 0)
        runtime.fighters[3].pos.update(1200, 0)
        with patch("src.runtime.random.randrange", return_value=50):
            controller.pick_fighter(runtime.fighters)
        self.assertIs(controller.victim, runtime.fighters[0])

    def test_ai_queue_keeps_ticking_while_fighter_temporarily_lacks_control(self) -> None:
        runtime = self.configure_four()
        fighter = runtime.fighters[1]
        controller = runtime.ai_controllers[1]
        controller.victim = runtime.fighters[0]
        controller.action_delay_ms = 50
        controller.queued_action = ("punch", "none")
        fighter.has_control = False
        with patch("src.runtime.random.randrange", return_value=199):
            controller.fixed_tick(runtime.fighters)
        self.assertEqual(controller.action_delay_ms, 25)
        self.assertEqual(controller.queued_action, ("punch", "none"))

    def test_dense_source_computer_array_runs_cpu_before_its_fighter_tick(self) -> None:
        runtime = self.configure_four()
        order: list[str] = []
        for index, fighter in enumerate(runtime.fighters):
            original = fighter.fixed_tick
            fighter.fixed_tick = Mock(
                side_effect=lambda *args, slot=index, method=original, **kwargs: (
                    order.append(f"fighter{slot + 1}"),
                    method(*args, **kwargs),
                )[-1]
            )
        for player_index, controller in runtime.ai_controllers.items():
            controller.fixed_tick = Mock(
                side_effect=lambda _fighters, slot=player_index: order.append(f"cpu{slot + 1}")
            )
        runtime._fixed_tick_items = Mock()

        runtime._fixed_tick_match([{}, {}, {}, {}])

        self.assertLess(order.index("cpu2"), order.index("fighter2"))
        self.assertLess(order.index("cpu4"), order.index("fighter3"))

    def test_endurance_replaces_defeated_cpu_and_increments_level(self) -> None:
        runtime = self.runtime
        runtime.match_config = {
            "type": "endurance",
            "players": [
                {"fighter": "PeachPlayer", "color": 0, "computer": False, "enabled": True, "level": 7}
            ],
            "limit_mode": "stock",
            "limit_value": 5,
        }
        runtime._reset_match()
        first = runtime.fighters[1]
        first.dead = True
        first.lives = 0
        runtime._update_match_state()
        self.assertIsNot(runtime.fighters[1], first)
        self.assertEqual(runtime.endurance_level, 2)
        self.assertEqual(runtime.fighters[1].lives, 1)
        self.assertTrue(runtime.ai_controllers[1].force_victim)

    def test_single_remaining_player_only_ends_after_that_player_dies(self) -> None:
        runtime = self.runtime
        runtime.match_config = {
            "type": "vsmode",
            "players": [
                {"fighter": "PeachPlayer", "color": 0, "computer": False, "enabled": True, "level": 7}
            ],
            "limit_mode": "stock",
            "limit_value": 5,
        }
        runtime._reset_match()
        runtime.match_state = "playing"
        runtime._update_match_state()
        self.assertEqual(runtime.match_state, "playing")
        runtime.fighters[0].dead = True
        runtime._update_match_state()
        self.assertEqual(runtime.match_state, "game_set")


if __name__ == "__main__":
    unittest.main()
