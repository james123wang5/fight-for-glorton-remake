from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.runtime import RuntimeApp, Stage
from src.simulation import BattleSimulation
from training.human_replay import build_human_dataset
from training.v5_env import V5_OBSERVATION_SIZE
from training.v5_options import PURPOSE_COUNT


class HumanReplayDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    @staticmethod
    def _record_human_inputs() -> dict[str, object]:
        runtime = RuntimeApp(random_seed=902)
        runtime.match_config = {
            "type": "vsmode",
            "selected_stage": "Mogadishu",
            "players": [
                {"fighter": "PeachPlayer", "color": 0, "computer": False, "enabled": True},
                {"fighter": "PeachPlayer", "color": 1, "computer": False, "enabled": True},
            ],
            "limit_mode": "stock",
            "limit_value": 3,
        }
        runtime.stage = Stage(runtime.manifest, "Mogadishu")
        runtime.simulation.reset()
        runtime.match_state = "playing"
        for fighter in runtime.fighters:
            fighter.intro_visible = True
        runtime.simulation.start_recording(
            {
                "kind": "human_vs_ai",
                "human_slots": [0],
                "ai_slots": [1],
                "ai_levels": {"1": 22},
                "fighter_names": ["PeachPlayer", "PeachPlayer"],
            }
        )
        for tick in range(24):
            direction = (
                "right"
                if runtime.fighters[1].pos.x >= runtime.fighters[0].pos.x
                else "left"
            )
            controls: dict[str, bool] = {direction: True}
            if tick in {2, 14}:
                controls["punch_pressed"] = True
            if tick == 9:
                controls["jump_pressed"] = True
            runtime.simulation.step_fast([controls, {}])
        return runtime.simulation.stop_recording()

    def test_verified_v2_recording_becomes_masked_purpose_examples(self) -> None:
        recording = self._record_human_inputs()
        self.assertTrue(recording["authoritative_inputs"])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            valid = directory / "human_vs_ai_valid.json"
            legacy = directory / "human_vs_ai_legacy.json"
            BattleSimulation.save_recording(recording, valid)
            old = copy.deepcopy(recording)
            old["schema"] = BattleSimulation.LEGACY_RECORDING_SCHEMA
            BattleSimulation.save_recording(old, legacy)

            dataset = build_human_dataset([valid, legacy])

        self.assertGreater(dataset.size, 0)
        self.assertEqual(dataset.observations.shape[1], V5_OBSERVATION_SIZE)
        self.assertEqual(dataset.masks.shape[1], PURPOSE_COUNT)
        self.assertEqual(dataset.accepted_files, (str(valid),))
        self.assertIn(str(legacy), dataset.skipped_files)
        self.assertTrue(dataset.masks[range(dataset.size), dataset.actions].all())


if __name__ == "__main__":
    unittest.main()
