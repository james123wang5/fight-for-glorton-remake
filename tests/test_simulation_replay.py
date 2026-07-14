from __future__ import annotations

import json
import os
import random
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.runtime import RuntimeApp
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


if __name__ == "__main__":
    unittest.main()
