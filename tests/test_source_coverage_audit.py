from __future__ import annotations

import unittest

from tools.audit_source_coverage import COVERAGE, UNREACHABLE, audit


class SourceCoverageAuditTests(unittest.TestCase):
    def test_every_exported_actionscript_class_is_classified(self) -> None:
        self.assertEqual(audit(), [])
        self.assertEqual(len(COVERAGE), 27)
        self.assertEqual(UNREACHABLE, {
            "RedGuy": "Registered as symbol 700, but never placed by the root timeline or any selectable-fighter list."
        })


if __name__ == "__main__":
    unittest.main()
