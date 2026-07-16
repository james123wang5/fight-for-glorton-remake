from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.play_v4 import resolve_model
from training.train_v4 import behavior_gate


def passing_report() -> dict[str, object]:
    return {
        "episodes": 30,
        "decisive_finish_rate": 0.70,
        "win_rate": 0.45,
        "resolved_projectiles": 12,
        "melee_opportunities": 12,
        "quality": {
            "far_idle_fraction": 0.08,
            "wall_stall_fraction": 0.05,
            "ground_crouches_per_minute": 0.0,
            "shield_hold_fraction": 0.03,
            "false_shield_rate": 0.10,
            "shield_activations_per_minute": 4.0,
            "projectiles_per_minute": 8.0,
            "projectile_accuracy": 0.30,
            "melee_opportunity_use_rate": 0.40,
        },
    }


class V4TrainingGateTests(unittest.TestCase):
    def test_complete_active_report_passes(self) -> None:
        passed, failures = behavior_gate(passing_report())
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_timeout_and_camping_cannot_pass(self) -> None:
        report = passing_report()
        report["decisive_finish_rate"] = 0.20
        report["quality"]["far_idle_fraction"] = 0.50  # type: ignore[index]
        passed, failures = behavior_gate(report)
        self.assertFalse(passed)
        self.assertTrue(any("胜负" in failure for failure in failures))
        self.assertTrue(any("发呆" in failure for failure in failures))

    def test_unqualified_candidate_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidate = directory / "candidate_level21_model.zip"
            candidate.touch()
            with self.assertRaises(FileNotFoundError):
                resolve_model(directory, 21, allow_candidate=False)
            self.assertEqual(
                resolve_model(directory, 21, allow_candidate=True),
                candidate.resolve(),
            )

    def test_qualified_manifest_selects_champion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            champion = directory / "champion_level22_model.zip"
            champion.touch()
            (directory / "champions.json").write_text(
                json.dumps(
                    {
                        "levels": {
                            "22": {
                                "qualified": True,
                                "path": champion.name,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_model(directory, 22, allow_candidate=False),
                champion.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
