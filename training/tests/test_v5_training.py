from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.play_v5 import resolve_model
from training.train_v5 import behavior_gate


def passing_report() -> dict[str, object]:
    return {
        "episodes": 40,
        "decisive_finish_rate": 0.70,
        "win_rate": 0.45,
        "resolved_projectiles": 12,
        "quality": {
            "projectile_accuracy": 0.25,
            "projectiles_per_minute": 8.0,
            "false_shield_rate": 0.10,
            "shield_hold_fraction": 0.04,
            "far_idle_fraction": 0.08,
            "wall_stall_fraction": 0.02,
            "plan_completion_rate": 0.50,
            "purposeful_jump_rate": 0.95,
            "jump_down_reversal_rate": 0.005,
            "air_chase_opportunity_use_rate": 0.45,
            "air_chase_hit_rate": 0.30,
        },
    }


class V5TrainingGateTests(unittest.TestCase):
    def test_purposeful_active_report_passes(self) -> None:
        passed, failures = behavior_gate(passing_report())
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_wall_stall_and_jump_down_spam_fail(self) -> None:
        report = passing_report()
        quality = report["quality"]
        assert isinstance(quality, dict)
        quality["wall_stall_fraction"] = 0.30
        quality["jump_down_reversal_rate"] = 0.20
        passed, failures = behavior_gate(report)
        self.assertFalse(passed)
        self.assertTrue(any("墙体" in failure for failure in failures))
        self.assertTrue(any("快落" in failure for failure in failures))

    def test_candidate_requires_explicit_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidate = directory / "candidate_level21_model.zip"
            candidate.touch()
            with self.assertRaises(FileNotFoundError):
                resolve_model(directory, 21, allow_candidate=False)
            self.assertEqual(resolve_model(directory, 21, allow_candidate=True), candidate.resolve())

    def test_qualified_champion_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            champion = directory / "champion_level22_model.zip"
            champion.touch()
            (directory / "champions.json").write_text(
                json.dumps(
                    {"levels": {"22": {"qualified": True, "path": champion.name}}}
                ),
                encoding="utf-8",
            )
            self.assertEqual(resolve_model(directory, 22, allow_candidate=False), champion.resolve())


if __name__ == "__main__":
    unittest.main()
