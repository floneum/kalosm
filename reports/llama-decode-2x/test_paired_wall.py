import copy
import unittest
from paired_wall import summarize


def pair(a, b, prompt=208):
    return dict(baseline=dict(samples=15, median_ms=a),
                candidate=dict(samples=15, median_ms=b),
                prompt_tokens=prompt, text_matches=True)


class PairedWallTests(unittest.TestCase):
    def test_reverse_pair_is_reoriented(self):
        result = summarize(pair(32, 15), pair(16, 34), 2)
        self.assertTrue(result['target_pass'])
        self.assertEqual(result['baseline_run_medians_ms'], [32, 34])
        self.assertEqual(result['candidate_run_medians_ms'], [15, 16])

    def test_one_fast_order_cannot_pass(self):
        result = summarize(pair(40, 15), pair(20, 30), 2)
        self.assertGreater(result['median_run_speedup'], 1.99)
        self.assertFalse(result['target_pass'])

    def test_text_prompt_or_samples_cannot_change(self):
        for key, value in [('text_matches', False), ('prompt_tokens', 4008)]:
            changed = pair(15, 32)
            changed[key] = value
            with self.assertRaises(AssertionError):
                summarize(pair(32, 15), changed, 2)
        changed = copy.deepcopy(pair(15, 32))
        changed['candidate']['samples'] = 14
        with self.assertRaises(AssertionError):
            summarize(pair(32, 15), changed, 2)


if __name__ == '__main__':
    unittest.main()
