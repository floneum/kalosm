"""Resource regression tests: tiny subprocesses, no model or GPU allocation."""
import dataclasses
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import benchmark_guard as guard

GIB = 1024**3
HEALTHY = guard.Snapshot(available_bytes=24*GIB, compressed_bytes=0,
                         swapout_bytes=0, disk_free_bytes=32*GIB)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = self.root / "lock"
        self.marker = self.root / "started"
        self.command = [sys.executable, "-c",
                        "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(30)",
                        str(self.marker)]

    def run_child(self, **kwargs):
        return guard.run_guarded(self.command, cwd=self.root, timeout=2,
                                 lock_path=self.lock, report_path=self.root / "resources.jsonl",
                                 poll_seconds=.01, **kwargs)

    def test_refuses_before_start_when_headroom_is_low(self):
        with patch.object(guard, "read_snapshot", return_value=dataclasses.replace(
                HEALTHY, available_bytes=8*GIB)):
            with self.assertRaisesRegex(guard.ResourceLimitError, "memory"):
                self.run_child()
        self.assertFalse(self.marker.exists())

    def test_refuses_large_model_before_start(self):
        with patch.object(guard, "read_snapshot", return_value=HEALTHY):
            with self.assertRaisesRegex(guard.ResourceLimitError, "memory"):
                self.run_child(model_bytes=12*GIB)
        self.assertFalse(self.marker.exists())

    def test_refuses_low_disk_and_compression(self):
        for changes, reason in [({"disk_free_bytes": 7*GIB}, "disk"),
                                ({"compressed_bytes": 3*GIB}, "compress")]:
            with self.subTest(reason=reason), patch.object(guard, "read_snapshot",
                    return_value=dataclasses.replace(HEALTHY, **changes)):
                with self.assertRaisesRegex(guard.ResourceLimitError, reason):
                    self.run_child()
                self.assertFalse(self.marker.exists())

    def test_stops_own_child_during_loading_on_pressure_or_monitor_failure(self):
        # Pressure starts as soon as the child begins loading, before any timing
        # or READY event. A monitor failure must also stop the child.
        for failure in [dataclasses.replace(HEALTHY, available_bytes=5*GIB),
                        dataclasses.replace(HEALTHY, swapout_bytes=128*1024**2),
                        dataclasses.replace(HEALTHY, compressed_bytes=GIB),
                        dataclasses.replace(HEALTHY, disk_free_bytes=3*GIB),
                        OSError("monitor unavailable")]:
            self.marker.unlink(missing_ok=True)
            children = []
            real_popen = subprocess.Popen

            def spawn(*args, **kwargs):
                child = real_popen(*args, **kwargs)
                children.append(child)
                return child

            def sample(_):
                if not self.marker.exists():
                    return HEALTHY
                if isinstance(failure, Exception):
                    raise failure
                return failure

            with self.subTest(failure=failure), patch.object(guard, "read_snapshot", side_effect=sample), \
                    patch.object(guard.subprocess, "Popen", side_effect=spawn):
                with self.assertRaises((guard.ResourceLimitError, OSError)):
                    self.run_child()
            self.assertTrue(self.marker.exists())
            self.assertEqual(len(children), 1, "never retry a resource failure")
            self.assertIsNotNone(children[0].poll(), "child must be reaped before unlocking")
            with guard.single_benchmark(self.lock):
                pass

    def test_global_lock_rejects_a_second_process(self):
        code = "import sys; from pathlib import Path; from benchmark_guard import single_benchmark; " \
               "ctx = single_benchmark(Path(sys.argv[1])); ctx.__enter__()"
        with guard.single_benchmark(self.lock):
            child = subprocess.run([sys.executable, "-c", code, str(self.lock)],
                                   cwd=Path(__file__).parent, capture_output=True, text=True, timeout=2)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("already running", child.stderr)

    def test_timeout_reaps_child(self):
        with patch.object(guard, "read_snapshot", return_value=HEALTHY):
            with self.assertRaises(subprocess.TimeoutExpired):
                guard.run_guarded(self.command, cwd=self.root, timeout=.15,
                                  lock_path=self.lock, poll_seconds=.01)
        with guard.single_benchmark(self.lock):
            pass

    def test_success_and_child_failure_are_preserved(self):
        with patch.object(guard, "read_snapshot", return_value=HEALTHY):
            result = guard.run_guarded([sys.executable, "-c", "pass"], cwd=self.root,
                                       timeout=2, lock_path=self.lock, poll_seconds=.01)
            self.assertEqual(result.returncode, 0)
            with self.assertRaises(subprocess.CalledProcessError):
                guard.run_guarded([sys.executable, "-c", "raise SystemExit(7)"], cwd=self.root,
                                  timeout=2, lock_path=self.lock, poll_seconds=.01)

    def test_monitor_sigterm_reaps_its_child(self):
        child_code = "import os,pathlib,sys,time; " \
                     "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
        controller_code = """
import pathlib, sys
import benchmark_guard as guard
guard.read_snapshot = lambda _: guard.Snapshot(24*guard.GIB, 0, 0, 32*guard.GIB)
guard.run_guarded([sys.executable, '-c', sys.argv[3], sys.argv[1]],
                  cwd=pathlib.Path(sys.argv[1]).parent, timeout=5,
                  lock_path=pathlib.Path(sys.argv[2]), poll_seconds=.01)
"""
        controller = subprocess.Popen([sys.executable, "-c", controller_code,
                                        str(self.marker), str(self.lock), child_code],
                                       cwd=Path(__file__).parent,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child_pid = None
        try:
            deadline = time.monotonic() + 2
            while not self.marker.exists() or not self.marker.read_text():
                self.assertLess(time.monotonic(), deadline, "test child failed to start")
                time.sleep(.01)
            child_pid = int(self.marker.read_text())
            controller.send_signal(signal.SIGTERM)
            self.assertNotEqual(controller.wait(timeout=3), 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
            with guard.single_benchmark(self.lock):
                pass
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.wait(timeout=2)
            if child_pid:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_unknown_monitor_data_fails_closed(self):
        with patch.object(guard.subprocess, "check_output", return_value="unexpected output"), \
                self.assertRaises(guard.ResourceLimitError):
            guard.read_snapshot(self.root)

    def test_disabled_controllers_never_spawn(self):
        root = Path(__file__).resolve().parents[3]
        for script in [root / "fusor/fusor-visualize/tools/pair_qmatmul.py",
                       root / "reports/llama-decode-2x/interleaved_wall.py"]:
            result = subprocess.run([sys.executable, str(script)], capture_output=True,
                                    text=True, timeout=2)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Disabled after memory exhaustion", result.stderr)

    def test_wall_harness_rejects_before_model_launch(self):
        root = Path(__file__).resolve().parents[3]
        script = root / "reports/llama-decode-2x/wall_ab.py"
        model = self.root / "model.gguf"
        binary = self.root / "unused-binary"
        config = self.root / "config.json"
        model.write_bytes(b"fixture")
        binary.write_bytes(b"must never execute")
        config.write_text("{}")
        argv = [str(script), str(model), "--output", str(self.root / "report"),
                "--binary", str(binary), "--baseline-config", str(config), "--config", str(config)]
        real_lock = guard.single_benchmark
        with patch.object(sys, "argv", argv), patch.object(guard, "read_snapshot",
                return_value=dataclasses.replace(HEALTHY, available_bytes=GIB)), \
                patch.object(guard, "single_benchmark", side_effect=lambda _: real_lock(self.lock)), \
                patch.object(guard.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(guard.ResourceLimitError, "memory"):
                runpy.run_path(str(script), run_name="__main__")
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
