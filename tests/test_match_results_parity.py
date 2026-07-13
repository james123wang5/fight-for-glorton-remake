from __future__ import annotations

import os
import copy
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.menu import MatchResults
from src.runtime import ROOT, PeachFighter, RuntimeApp, Stage, load_manifest


class MatchResultsParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))
        cls.manifest = load_manifest()
        cls.stage = Stage(cls.manifest)

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def fighter(self, name: str, color: int) -> PeachFighter:
        return PeachFighter(
            self.manifest,
            self.stage.spawn_point("SpawnP1"),
            name,
            color,
        )

    def game_set_runtime(self) -> RuntimeApp:
        runtime = RuntimeApp.__new__(RuntimeApp)
        runtime.manifest = self.manifest
        runtime.match_state = "playing"
        runtime.match_winner = None
        runtime.match_end_elapsed_ms = 99
        runtime.game_set_audio_played = True
        runtime.accumulator = 25
        runtime.stage_time_ms = 1000
        runtime.paused = False
        runtime.fight_timer_accumulator_ms = 0
        runtime.game_time_seconds = 0
        runtime.match_loading_elapsed_ms = 0
        return runtime

    def test_pre_end_is_source_frames_52_through_100(self) -> None:
        runtime = self.game_set_runtime()
        self.assertAlmostEqual(runtime._pre_end_duration_ms(), 49 * 1000 / 30)

    def test_game_set_freezes_stage_timeline(self) -> None:
        runtime = self.game_set_runtime()
        winner = self.fighter("P1", 0)

        runtime._begin_game_set(winner)
        runtime._advance_battle_time(250)

        self.assertEqual(runtime.match_state, "game_set")
        self.assertIs(runtime.match_winner, winner)
        self.assertEqual(runtime.stage_time_ms, 1000)
        self.assertEqual(runtime.accumulator, 0)
        self.assertEqual(runtime.match_end_elapsed_ms, 0)
        self.assertFalse(runtime.game_set_audio_played)

    def test_source_fight_timer_sequence_and_immediate_first_time_tick(self) -> None:
        runtime = self.game_set_runtime()
        runtime.manifest = copy.deepcopy(self.manifest)
        runtime.manifest["match"]["limit_mode"] = "time"
        runtime.match_state = "loading"
        runtime.match_time_remaining_ms = 300000
        runtime.ready_set = 5
        runtime.ready_timer_ms = 0
        runtime.ready_text = ""
        runtime.camera_view = None
        runtime.camera_target_view = None
        runtime.countdown_focus_indices = []
        runtime.fighters = [self.fighter("P1", 0), self.fighter("P2", 1)]
        for fighter in runtime.fighters:
            fighter.intro_visible = False
            fighter.has_control = False

        shown = []
        for _ in range(7):
            runtime._advance_fight_timer(1000)
            shown.append(runtime.ready_text)

        self.assertEqual(shown, ["5", "4", "3", "2", "1", "GO!", ""])
        self.assertEqual(runtime.match_state, "playing")
        self.assertEqual(runtime.match_time_remaining_ms, 299000)

    def test_paused_countdown_keeps_timer_phase_without_deducting_time(self) -> None:
        runtime = self.game_set_runtime()
        runtime.manifest = copy.deepcopy(self.manifest)
        runtime.manifest["match"]["limit_mode"] = "time"
        runtime.match_state = "loading"
        runtime.match_time_remaining_ms = 300000
        runtime.ready_set = 5
        runtime.ready_timer_ms = 0
        runtime.ready_text = ""
        runtime.camera_view = None
        runtime.camera_target_view = None
        runtime.countdown_focus_indices = []
        runtime.fighters = [self.fighter("P1", 0), self.fighter("P2", 1)]
        for fighter in runtime.fighters:
            fighter.intro_visible = False
            fighter.has_control = False
        runtime.paused = True

        runtime._advance_fight_timer(7000)

        self.assertEqual(runtime.match_state, "playing")
        self.assertEqual(runtime.match_time_remaining_ms, 300000)
        runtime.paused = False
        runtime._advance_fight_timer(1000)
        self.assertEqual(runtime.match_time_remaining_ms, 299000)

    def test_pause_freezes_world_timeline_and_physics_accumulator(self) -> None:
        runtime = self.game_set_runtime()
        runtime.stage = Stage(self.manifest)
        runtime.paused = True
        start_time = runtime.stage_time_ms
        start_accumulator = runtime.accumulator

        runtime._advance_battle_time(750)

        self.assertEqual(runtime.stage_time_ms, start_time)
        self.assertEqual(runtime.accumulator, start_accumulator)

    def test_stock_ranking_uses_descending_source_death_order(self) -> None:
        p1 = self.fighter("P1", 0)
        p2 = self.fighter("P2", 1)
        p1.death_order = 0
        p1.dead = True
        results = MatchResults(ROOT, self.manifest)

        results.start([p1, p2], p1, "stock")

        self.assertEqual([entry["player"] for entry in results.ranking], [2, 1])
        self.assertEqual(results.winner_player, 2)

    def test_time_winner_is_highest_ko_minus_death_score(self) -> None:
        p1 = self.fighter("P1", 0)
        p2 = self.fighter("P2", 1)
        p1.kos, p1.deaths = 5, 4
        p2.kos, p2.deaths = 3, 0
        results = MatchResults(ROOT, self.manifest)

        results.start([p1, p2], p1, "time")

        self.assertEqual([entry["player"] for entry in results.ranking], [2, 1])
        self.assertEqual(results.winner_player, 2)

    def test_score_upper_manifest_uses_original_labels_and_frames(self) -> None:
        data = self.manifest["ui"]["OSDScoreUpper"]
        self.assertEqual(data["symbol_id"], 776)
        self.assertEqual(data["timeline"]["frame_count"], 70)
        self.assertEqual(
            data["timeline"]["labels"],
            [{"frame": 2, "name": "Plus1"}, {"frame": 35, "name": "Minus1"}],
        )
        self.assertEqual(len(data["frames"]), 70)

    def test_podium_fighters_begin_on_source_frame_143(self) -> None:
        self.assertEqual(self.manifest["results"]["podium_start_frame"], 143)

    def test_source_pause_and_result_button_hit_rects(self) -> None:
        self.assertEqual(
            self.manifest["ui"]["layout"]["pause_end_button"],
            {"x": 251.8, "y": 202.7, "w": 99.0, "h": 23.25},
        )
        self.assertEqual(
            self.manifest["results"]["more_games_button"],
            {"x": 8.35, "y": 367.2, "w": 160.5, "h": 23.5},
        )

    def test_result_buttons_only_activate_on_stopped_frame_170(self) -> None:
        results = MatchResults(ROOT, self.manifest)
        results.elapsed_ms = 0
        early = pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(536, 380))
        self.assertIsNone(results.handle_event(early, (600, 400)))

        results.elapsed_ms = (170 - 101) * 1000 / 30
        main = results.handle_event(early, (600, 400))
        more = results.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(88, 377)),
            (600, 400),
        )
        self.assertEqual(main.kind, "return_main")
        self.assertEqual(more.kind, "open_url")
        self.assertEqual(more.payload, {"url": "http://www.armorgames.com/"})


if __name__ == "__main__":
    unittest.main()
