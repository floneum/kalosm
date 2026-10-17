"""Fixture checks for attribution coverage, warmups and missing samples."""
import json
from pathlib import Path
import tempfile
import unittest
from analyze_samples import analyze


def phase(key, warmup=False, dispatches=2, sample_index=0):
    meta = {'launches': [
        {'index': 0, 'grid': [2, 1, 1], 'operation': {'kind': 'first'}},
        {'index': 1, 'grid': [0, 1, 1], 'operation': {'kind': 'empty'}},
        {'index': 2, 'grid': [1, 1, 1], 'operation': {'kind': 'unsampled'}},
    ]}
    row = dict(key=key, warmup=warmup, phase='decode', step=2, dispatches=dispatches)
    return '\n'.join([
        f'PHASE_BEGIN {key}', 'RESOLVE_META ' + json.dumps(meta),
        f'TSAMPLE 0 kernel 12.0 L{sample_index} grid=[2, 1, 1]',
        'PHASE_END ' + json.dumps(row),
    ]) + '\n'


class Samples(unittest.TestCase):
    def run_log(self, content):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'samples.log'
            path.write_text(content)
            return analyze(path)

    def test_warmup_zero_grid_and_unsampled_are_distinct(self):
        result = self.run_log(phase('warmup', warmup=True) + phase('measured'))
        self.assertEqual(result['steady_tokens'], 1)
        self.assertEqual(result['sampled_dispatches'], 1)
        self.assertEqual(result['unsampled_groups'], 1)
        self.assertEqual(result['estimated_sampled_groups_ms'], .012)
        groups = {g['operation']['kind']: g for g in result['groups']}
        self.assertEqual(set(groups), {'first', 'unsampled'})
        self.assertIsNone(groups['unsampled']['median_us'])
        self.assertIsNone(groups['unsampled']['estimated_ms_per_token'])

    def test_rejects_missing_or_misaligned_evidence(self):
        for content in [phase('bad', dispatches=3), phase('bad', sample_index=2),
                        phase('truncated').split('PHASE_END')[0]]:
            with self.subTest(content=content), self.assertRaises(AssertionError):
                self.run_log(content)

    def test_cached_metadata_has_explicit_identity_and_full_coverage(self):
        first = phase('first').replace('RESOLVE_META {', 'RESOLVE_META {"metadata_id": 3, ')
        second = phase('second')
        second = '\n'.join('RESOLVE_META_REF {"metadata_id": 3}' if line.startswith('RESOLVE_META ')
                           else line for line in second.splitlines()) + '\n'
        result = self.run_log(first + second)
        self.assertEqual(result['steady_tokens'], 2)
        self.assertEqual(result['sampled_dispatches'], 2)
        with self.assertRaisesRegex(AssertionError, 'missing metadata reference'):
            self.run_log(second)
        with self.assertRaisesRegex(AssertionError, 'duplicate metadata identity'):
            self.run_log(first + first.replace('first', 'duplicate'))


if __name__ == '__main__':
    unittest.main()
