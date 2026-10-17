import unittest
from benchmark_protocol import validate_decode_resource_scope

class PrefillScopeTests(unittest.TestCase):
    def test_rejects_observed_original_decode_throttling(self):
        rows=[dict(phase='prefill',dispatches=25799),dict(phase='decode',dispatches=1270)]
        limits=dict(FUSOR_PROFILE_LARGE_PLAN_INFLIGHT_CHUNKS='1',FUSOR_PROFILE_LARGE_PLAN_SUBMIT_SIZE='32')
        with self.assertRaisesRegex(ValueError, '1270.*1024'):
            validate_decode_resource_scope(rows,limits)
        validate_decode_resource_scope(rows,dict(limits,FUSOR_PROFILE_LARGE_PLAN_MIN_DISPATCHES='4096'))
        validate_decode_resource_scope(rows,{})
        validate_decode_resource_scope([dict(phase='decode',dispatches=403)],limits)

if __name__ == '__main__': unittest.main()
