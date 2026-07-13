from __future__ import annotations

import math
import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.assets import SURFACE_CACHE
from src.menu import MainMenu, MatchResults
from src.runtime import ROOT, TICK_MS, RuntimeApp


class FullFlowParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))
        cls.runtime = RuntimeApp()
        cls.runtime.audio = None
        cls.runtime.menu = MainMenu(ROOT, cls.runtime.manifest)
        cls.runtime.results = MatchResults(ROOT, cls.runtime.manifest)
        cls.screen = pygame.Surface((600, 400))
        cls.font = pygame.font.SysFont("menlo", 14)

    @classmethod
    def tearDownClass(cls) -> None:
        SURFACE_CACHE.clear()
        pygame.quit()

    def advance_battle(self, elapsed_ms: int) -> None:
        app = self.runtime
        app._advance_battle_time(elapsed_ms)
        while not app.paused and app.accumulator >= TICK_MS:
            app.stage.set_time(app.stage_time_ms)
            if app.match_state == "countdown":
                app._fixed_tick_countdown()
            elif app.match_state == "playing":
                app._fixed_tick_match([{} for _ in app.inputs])
            app.accumulator -= TICK_MS

    def test_preloader_to_two_player_match_results_and_main(self) -> None:
        app = self.runtime
        menu = app.menu
        assert menu is not None
        results = app.results
        assert results is not None

        menu.reset_to_intro()
        play_rect = app.manifest["menu"]["preloader"]["play_rect"]
        play_pos = (
            round(play_rect["x"] + play_rect["w"] / 2),
            round(play_rect["y"] + play_rect["h"] / 2),
        )
        menu.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=play_pos),
            self.screen.get_size(),
        )
        self.assertEqual(menu.scene, "sponsor_intro")
        menu.update(math.ceil(81 * 1000 / 30))
        self.assertEqual(menu.scene, "opening")
        menu.update(math.ceil(37 * 1000 / 30))
        self.assertEqual(menu.scene, "intro")
        menu.handle_event(
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN),
            self.screen.get_size(),
        )
        self.assertEqual(menu.scene, "main")

        menu._activate_button("multi")
        menu._activate_button("vsmode")
        self.assertEqual(menu.scene, "player_select")
        menu.selected_fighters = ["PeachPlayer", "SBLPlayer", None, None]
        menu.selected_colors = [0, 1, None, None]
        menu.player_enabled = [True, True, False, False]
        menu.computer_players = [False, False, False, False]
        menu.player_levels = [7, 7, 7, 7]
        menu.limit_mode = "stock"
        menu.limit_value = 1
        menu.scene = "stage_select"
        action = menu._handle_stage_select_event(
            pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(100, 130)),
            (100, 130),
        )
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, "start_game")

        self.assertTrue(app._handle_menu_action(action))
        self.assertEqual((app.app_state, app.match_state), ("battle", "loading"))
        self.assertEqual([fighter.fighter_name for fighter in app.fighters], ["PeachPlayer", "SBLPlayer"])
        for _ in range(7):
            self.advance_battle(1000)
        self.assertEqual(app.match_state, "playing")
        self.assertTrue(all(fighter.has_control for fighter in app.fighters))
        app._draw(self.screen, self.font)
        self.assertGreater(self.screen.get_bounding_rect().w, 0)

        loser = app.fighters[1]
        loser.lives = 1
        loser.die("bot", app.stage)
        app._update_match_state()
        self.assertEqual(app.match_state, "game_set")
        self.assertIs(app.match_winner, app.fighters[0])

        app.match_end_elapsed_ms = math.ceil(app._pre_end_duration_ms())
        results.start(
            app.fighters,
            app.match_winner,
            str(app.match_config["limit_mode"]),
            str(app.match_config["type"]),
            app.killed_players,
            app.game_time_seconds,
        )
        app.app_state = "results"
        results.update((170 - 101) * 1000 / 30)
        results.draw(self.screen)
        self.assertEqual(results.frame_number, 170)

        return_action = results.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(536, 380)),
            self.screen.get_size(),
        )
        self.assertIsNotNone(return_action)
        self.assertTrue(app._handle_menu_action(return_action))
        self.assertEqual((app.app_state, menu.scene), ("menu", "main"))


if __name__ == "__main__":
    unittest.main()
