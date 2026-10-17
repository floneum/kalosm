import unittest
from paired_model_geometry import validate_pairs


def phases():
    return [dict(round=r,step=s,warmup=r==0 or s<=1,phase='prefill' if s==0 else 'decode',
                 order=[0,1] if s%2==0 else [1,0],prompt_tokens=208,dispatches=[403,403],
                 matched=[32,32] if s else [0,0],baseline_ms=19.,candidate_ms=18.,speedup=19/18)
            for r in range(2) for s in range(3)]


class PairedModelProtocol(unittest.TestCase):
    def test_first_prefill_can_initialize_shared_weights(self):
        rows=phases()
        rows[0]['dispatches']=[25799,25000]
        validate_pairs(rows,1,2)

    def test_steady_dispatch_difference_is_rejected(self):
        rows=phases()
        rows[-1]['dispatches']=[403,402]
        with self.assertRaisesRegex(ValueError,'dispatch counts'):
            validate_pairs(rows,1,2)

    def test_candidate_must_match_projections(self):
        rows=phases()
        rows[-1]['matched']=[32,0]
        with self.assertRaisesRegex(ValueError,'same decode projections'):
            validate_pairs(rows,1,2)

    def test_missing_phase_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing or reordered'):
            validate_pairs(phases()[:-1],1,2)

    def test_order_must_alternate(self):
        rows=phases()
        rows[-1]['order']=[1,0]
        with self.assertRaisesRegex(ValueError,'identity'):
            validate_pairs(rows,1,2)


if __name__=='__main__':
    unittest.main()
