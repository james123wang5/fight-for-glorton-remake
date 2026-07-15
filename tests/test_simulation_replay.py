from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.runtime import RuntimeApp, Stage
from src.simulation import BattleSimulation, INPUT_FIELDS


class BattleSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    @staticmethod
    def scripted_inputs(tick: int) -> list[dict[str, bool]]:
        p1: dict[str, bool] = {}
        if 2 <= tick < 8:
            p1["right"] = True
        if tick == 4:
            p1["jump_pressed"] = True
        if tick == 10:
            p1["punch_pressed"] = True
        if tick == 18:
            p1["special_pressed"] = True
        return [p1, {}]

    def test_step_has_a_complete_stable_input_contract(self) -> None:
        runtime = RuntimeApp(random_seed=7)
        snapshot = runtime.simulation.step([{"left": True}])
        self.assertEqual(snapshot["schema"], BattleSimulation.SCHEMA)
        self.assertEqual(snapshot["tick"], 1)
        runtime.simulation.start_recording()
        runtime.simulation.step([{"punch_pressed": True}])
        recording = runtime.simulation.stop_recording()
        self.assertEqual(set(recording["inputs"][0][0]), set(INPUT_FIELDS))
        json.dumps(snapshot)
        json.dumps(recording)

    def test_seeded_simulations_do_not_depend_on_process_random_state(self) -> None:
        first = RuntimeApp(random_seed=1337)
        first.match_state = "playing"
        for fighter in first.fighters:
            fighter.intro_visible = True
        first_digests = []
        for tick in range(30):
            random.seed(9000 + tick)
            first.simulation.step(self.scripted_inputs(tick))
            first_digests.append(first.simulation.state_digest())

        second = RuntimeApp(random_seed=1337)
        second.match_state = "playing"
        for fighter in second.fighters:
            fighter.intro_visible = True
        second_digests = []
        for tick in range(30):
            random.seed(100 + tick)
            second.simulation.step(self.scripted_inputs(tick))
            second_digests.append(second.simulation.state_digest())

        self.assertEqual(first_digests, second_digests)

    def test_recording_replays_to_the_exact_final_snapshot(self) -> None:
        runtime = RuntimeApp(random_seed=2026)
        simulation = runtime.simulation
        simulation.start_recording({"case": "countdown-smoke"})
        for tick in range(80):
            simulation.step(self.scripted_inputs(tick))
        recording = simulation.stop_recording()
        expected = recording["final_digest"]

        simulation.replay(recording)

        self.assertEqual(simulation.state_digest(), expected)

    def test_recording_restores_a_custom_entity_free_starting_snapshot(self) -> None:
        runtime = RuntimeApp(random_seed=55)
        simulation = runtime.simulation
        runtime.match_state = "playing"
        fighter = runtime.fighters[0]
        fighter.intro_visible = True
        fighter.pos.update(321.5, 177.25)
        fighter.prev_pos.update(fighter.pos)
        fighter.facing = -1
        simulation.start_recording({"case": "custom-start"})
        for tick in range(12):
            simulation.step(self.scripted_inputs(tick))
        recording = simulation.stop_recording()
        self.assertEqual(recording["initial_snapshot"]["fighters"][0]["pos"], [321.5, 177.25])

        fighter.pos.update(-999, -999)
        simulation.replay(recording)

        self.assertEqual(simulation.state_digest(), recording["final_digest"])

    def test_headless_factory_never_creates_a_visible_game_window(self) -> None:
        simulation = BattleSimulation.headless(seed=41)
        self.assertEqual(pygame.display.get_surface().get_size(), (1, 1))
        before = simulation.snapshot()["stage_time_ms"]
        simulation.step([])
        self.assertEqual(simulation.snapshot()["stage_time_ms"], before + 25)

    def test_fast_step_matches_snapshot_step_without_returning_a_snapshot(self) -> None:
        regular = BattleSimulation.headless(seed=73)
        fast = BattleSimulation.headless(seed=73)
        regular.runtime.match_state = "playing"
        fast.runtime.match_state = "playing"
        for simulation in (regular, fast):
            for fighter in simulation.fighters:
                fighter.intro_visible = True

        for tick in range(40):
            controls = self.scripted_inputs(tick)
            regular.step(controls)
            self.assertIsNone(fast.step_fast(controls))

        self.assertEqual(fast.state_digest(), regular.state_digest())

    def test_human_vs_ai_capture_saves_complete_training_material_without_live_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"GLORTON_HUMAN_REPLAY_DIR": directory},
        ):
            runtime = RuntimeApp(random_seed=84)
            runtime.match_config = {
                "type": "vsmode",
                "selected_stage": "Mogadishu",
                "players": [
                    {
                        "fighter": "PeachPlayer",
                        "color": 0,
                        "computer": False,
                        "enabled": True,
                        "level": 7,
                    },
                    {
                        "fighter": "PeachPlayer",
                        "color": 1,
                        "computer": True,
                        "enabled": True,
                        "level": 20,
                    },
                ],
                "limit_mode": "stock",
                "limit_value": 3,
            }
            runtime.stage = Stage(runtime.manifest, "Mogadishu")
            runtime.simulation.reset()
            runtime.match_state = "playing"
            for fighter in runtime.fighters:
                fighter.intro_visible = True

            self.assertTrue(runtime._maybe_start_human_recording())
            runtime.simulation.step_fast([{"right": True}, {}])
            runtime.simulation.step_fast([{"punch_pressed": True}, {}])
            path = runtime._finish_human_recording("test_complete")

            self.assertIsNotNone(path)
            self.assertTrue(Path(path).is_file())
            recording = BattleSimulation.load_recording(path)
            self.assertEqual(len(recording["inputs"]), 2)
            self.assertEqual(recording["metadata"]["human_slots"], [0])
            self.assertEqual(recording["metadata"]["ai_levels"], {"1": 20})
            self.assertEqual(recording["metadata"]["end_reason"], "test_complete")
            self.assertFalse(runtime._human_recording_active)


if __name__ == "__main__":
    unittest.main()
