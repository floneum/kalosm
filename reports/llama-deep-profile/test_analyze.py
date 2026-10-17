"""Dispatch attribution must reject mismatches, not silently produce percentages."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class AttributionTests(unittest.TestCase):
    def trace(self):
        operation = {"kind": "group", "members": [
            {"kind": "contract", "family": "Coop", "m": 2, "n": 4, "k": 8,
             "batch": 1, "macs": 64, "schedule": "test", "a": [{"dtype": "F32", "logical_bytes_native_quant": 64}],
             "b": [{"dtype": "F32", "logical_bytes_native_quant": 128}]},
            {"kind": "fold", "shape": [2, 4], "axis": 1}]}
        meta = {"launches": [
            {"index": 0, "body": "unused", "grid": [0, 1, 1], "operation": {}},
            {"index": 1, "body": "abcd", "grid": [2, 1, 1], "operation": operation}]}
        row = {"key": "example", "phase": "prefill", "warmup": False, "step": 0, "dispatches": 1}
        return "\n".join(["PHASE_BEGIN example", "RESOLVE_META " + json.dumps(meta),
                          "BODY abcd grid=[2, 1, 1]", "TSPAN 0 kgroup 10.0 L1 grid=[2, 1, 1]",
                          "PHASE_END " + json.dumps(row)])

    def analyze(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.log"
            path.write_text(text)
            run = subprocess.run([sys.executable, str(Path(__file__).with_name("analyze.py")), str(path)],
                                 text=True, capture_output=True)
            summary = path.with_suffix(".summary.json")
            return run, json.loads(summary.read_text()) if summary.exists() else None

    def test_composite_is_one_timed_launch_and_zero_grid_is_not_executed(self):
        run, report = self.analyze(self.trace())
        self.assertEqual(run.returncode, 0, run.stderr)
        groups = report["prefill"]["groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["operation"]["kind"], "group")
        self.assertEqual(groups[0]["us"], 10)
        self.assertEqual(groups[0]["macs"], 64)
        self.assertEqual(groups[0]["operand_bytes"], 192)
        self.assertEqual(groups[0]["percent"], 100)

    def test_wrong_body_grid_or_launch_rejected(self):
        for old, new in [("BODY abcd", "BODY dcba"),
                         ("L1 grid=[2, 1, 1]", "L1 grid=[3, 1, 1]"),
                         ("L1 grid=", "L0 grid=")]:
            with self.subTest(new=new):
                run, report = self.analyze(self.trace().replace(old, new))
                self.assertNotEqual(run.returncode, 0)
                self.assertIsNone(report)

    def test_missing_timestamp_rejected(self):
        run, report = self.analyze("\n".join(line for line in self.trace().splitlines() if not line.startswith("TSPAN ")))
        self.assertNotEqual(run.returncode, 0)
        self.assertIsNone(report)


if __name__ == "__main__":
    unittest.main()
