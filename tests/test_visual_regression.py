from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.visual_regression import (
    DEFAULT_MANIFEST,
    DEFAULT_SCENARIOS,
    _scripted_controls,
    compare_pair,
    coverage_cases,
    load_json,
)


class VisualRegressionTests(unittest.TestCase):
    def test_coverage_matrix_contains_all_ninety_nine_source_cases(self) -> None:
        cases = coverage_cases(load_json(DEFAULT_MANIFEST))
        self.assertEqual(len(cases), 99)
        ids = {case["id"] for case in cases}
        self.assertIn("attacks/PeachPlayer/punchGround", ids)
        self.assertIn("roster/AuberginePlayer/color_4", ids)
        self.assertIn("damage/SBLPlayer", ids)
        self.assertIn("items/Grenade", ids)
        self.assertIn("stages/Space", ids)

    def test_every_attack_uses_the_committed_fixed_input_script(self) -> None:
        scenarios = load_json(DEFAULT_SCENARIOS)
        manifest = load_json(DEFAULT_MANIFEST)
        fighter = next(iter(manifest["fighters"].values()))
        self.assertEqual(set(scenarios["attack_inputs"]), set(fighter["attacks"]))
        controls = _scripted_controls(scenarios["attack_inputs"]["specialUp"], 0)
        self.assertTrue(controls[0]["up_trace"])
        self.assertTrue(controls[0]["special_pressed"])

    def test_frame_diff_reports_identical_and_changed_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            black = root / "black.png"
            changed = root / "changed.png"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(black)
            image = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
            image.putpixel((2, 1), (255, 0, 0, 255))
            image.save(changed)
            identical = compare_pair(black, black, root / "same-diff.png")
            different = compare_pair(black, changed, root / "changed-diff.png")
            self.assertEqual(identical["status"], "identical")
            self.assertEqual(identical["changed_ratio"], 0)
            self.assertEqual(different["changed_pixels"], 1)
            self.assertEqual(different["bbox"], [2, 1, 3, 2])


if __name__ == "__main__":
    unittest.main()
