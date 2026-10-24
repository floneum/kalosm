import array
from pathlib import Path
import tempfile
import unittest

from check_model_logits import compare_vector


class LogitChecks(unittest.TestCase):
    def test_corruption_outside_top_logits_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root)/"a", Path(root)/"b"
            values = array.array("f", [10.] + [0.]*256)
            a.write_bytes(values.tobytes())
            b.write_bytes(values.tobytes())
            self.assertTrue(compare_vector(a, b)["passed"])
            values[129] = 1.
            b.write_bytes(values.tobytes())
            result = compare_vector(a, b)
            self.assertTrue(result["argmax_equal"])
            self.assertFalse(result["passed"])
            values[129] = float("nan")
            b.write_bytes(values.tobytes())
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                compare_vector(a, b)


if __name__ == "__main__":
    unittest.main()
