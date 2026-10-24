import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from compare_decode import compare, compare_bracketed


class DecodeComparison(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / 'base.log'
        self.candidate = Path(self.temp.name) / 'candidate.log'

    def write(self, path, gaps, output='same text', rounds=2, warmup_bytes=2):
        lines = ['llama_forward_config warmup=1 measured=3 seed=0 repeats=2 chat=false prompt="p"']
        for i in range(rounds):
            values = [900, 900, 900] if i == 0 else gaps
            lines += [f'llama_forward_profile chunks=3 requested=3 warmup_bytes={warmup_bytes}',
                      'llama_forward_gaps_ms ' + json.dumps(values),
                      'llama_forward_output ' + json.dumps(output)]
        path.write_text('\n'.join(lines))

    def test_reports_slow_tail_without_discarding_it(self):
        self.write(self.base, [20, 20, 100])
        self.write(self.candidate, [10, 10, 80])
        result = compare(self.base, self.candidate)
        self.assertEqual(result['elapsed_speedup'], 1.4)
        self.assertEqual(result['p95_speedup'], 1.25)
        self.assertEqual(result['candidate']['max_ms'], 80)
        self.assertEqual(result['candidate']['chunks'], 3)
        self.assertEqual(result['candidate']['slowest_chunks'][0],
                         dict(round=1, chunk=2, ms=80))
        self.assertEqual(len(result['candidate']['slowest_chunks']), 3)

    def test_refuses_different_outputs_and_boundaries(self):
        self.write(self.base, [20, 20, 100])
        for output, boundary in [('different text', 2), ('same text', 3)]:
            self.write(self.candidate, [10, 10, 50], output=output, warmup_bytes=boundary)
            with self.assertRaisesRegex(ValueError, 'differs'):
                compare(self.base, self.candidate)

    def test_refuses_partial_or_invalid_samples(self):
        self.write(self.base, [20, 20, 100])
        self.write(self.candidate, [10, 10, 50], rounds=1)
        with self.assertRaisesRegex(ValueError, 'rounds'):
            compare(self.base, self.candidate)
        for gaps in [[10, 10], [10, 0, 10], [10, float('nan'), 10], [10, True, 10], 10]:
            self.write(self.candidate, gaps)
            with self.assertRaisesRegex(ValueError, 'latency'):
                compare(self.base, self.candidate)

    def test_bracketed_run_uses_the_faster_baseline(self):
        after = Path(self.temp.name) / 'after.log'
        self.write(self.base, [40, 40, 40])
        self.write(self.candidate, [10, 10, 10])
        self.write(after, [20, 20, 20])
        result = compare_bracketed(self.base, self.candidate, after)
        self.assertEqual(result['elapsed_speedup_min'], 2)
        self.assertEqual(result['p95_speedup_min'], 2)
        self.assertEqual(result['baseline_elapsed_ratio'], 2)

    def test_bracketed_run_checks_the_second_baseline_output(self):
        after = Path(self.temp.name) / 'after.log'
        self.write(self.base, [40, 40, 40])
        self.write(self.candidate, [10, 10, 10])
        self.write(after, [20, 20, 20], output='different text')
        with self.assertRaisesRegex(ValueError, 'differs'):
            compare_bracketed(self.base, self.candidate, after)

    def test_cli_target_checks_every_baseline(self):
        after = Path(self.temp.name) / 'after.log'
        self.write(self.base, [40, 40, 40])
        self.write(self.candidate, [10, 10, 10])
        self.write(after, [15, 15, 15])
        command = [sys.executable, str(Path(__file__).with_name('compare_decode.py')),
                   str(self.base), str(self.candidate), '--minimum-speedup', '2']
        good = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        bad = subprocess.run(command + ['--baseline-after', str(after)],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)
        self.assertIn('1.500x speedup < 2.000x target', bad.stderr)
        self.assertEqual(json.loads(bad.stdout)['elapsed_speedup_min'], 1.5)

    def write_tokens(self, path, gaps, token_ids=None):
        self.write(path, gaps)
        rows = path.read_text().replace('chunks=3', 'tokens=3').replace(
            'warmup_bytes=2', 'warmup=1')
        ids = token_ids if token_ids is not None else [10, 20, 30, 40]
        rows = rows.replace('llama_forward_output ',
                            'llama_forward_token_ids ' + json.dumps(ids) + '\nllama_forward_output ')
        path.write_text(rows)

    def test_actual_tokens_have_throughput_and_keep_slow_samples(self):
        self.write_tokens(self.base, [20, 20, 20])
        self.write_tokens(self.candidate, [10, 10, 30])
        result = compare(self.base, self.candidate)
        self.assertEqual(result['candidate']['tokens_per_second'], 60)
        self.assertEqual(result['candidate']['tokens'], 3)
        self.assertEqual(result['candidate']['slowest_tokens'][0],
                         dict(round=1, token=2, ms=30))

    def test_refuses_different_token_ids_despite_identical_text(self):
        self.write_tokens(self.base, [20, 20, 20])
        self.write_tokens(self.candidate, [10, 10, 10], [10, 20, 31, 40])
        with self.assertRaisesRegex(ValueError, 'differs'):
            compare(self.base, self.candidate)

    def test_refuses_missing_partial_and_invalid_token_ids(self):
        self.write_tokens(self.base, [20, 20, 20])
        for ids in [[10, 20, 30], [10, 20, 30, -1], [10, 20, 30, 2**32],
                    [10, 20, 30, True]]:
            self.write_tokens(self.candidate, [10, 10, 10], ids)
            with self.assertRaisesRegex(ValueError, 'token'):
                compare(self.base, self.candidate)
        self.write_tokens(self.candidate, [10, 10, 10])
        self.candidate.write_text('\n'.join(line for line in self.candidate.read_text().splitlines()
                                             if not line.startswith('llama_forward_token_ids ')))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            compare(self.base, self.candidate)

    def test_tps_target_cannot_use_text_chunks(self):
        command = [sys.executable, str(Path(__file__).with_name('compare_decode.py')),
                   str(self.base), str(self.candidate), '--minimum-tps', '60']
        self.write(self.base, [20, 20, 20])
        self.write(self.candidate, [1, 1, 1])
        chunks = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(chunks.returncode, 1)
        self.assertIn('requires actual token', chunks.stderr)
        self.write_tokens(self.base, [20, 20, 20])
        self.write_tokens(self.candidate, [16, 16, 16])
        passed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        self.write_tokens(self.candidate, [16, 16, 19])
        failed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(failed.returncode, 1)
        self.assertIn('58.824 tokens/sec < 60.000', failed.stderr)
