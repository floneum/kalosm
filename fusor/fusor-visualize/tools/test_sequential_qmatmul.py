"""The comparison must inspect every saved value, including column 129."""
import array
from pathlib import Path
import tempfile
import unittest

from sequential_qmatmul import compare_outputs


class OutputChecks(unittest.TestCase):
    def test_full_vectors_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory)/"left", Path(directory)/"right"
            left.mkdir()
            right.mkdir()
            data = array.array("f", [i/256 for i in range(257)])
            for seed in range(3):
                (left/f"{seed}.f32").write_bytes(data.tobytes())
                (right/f"{seed}.f32").write_bytes(data.tobytes())
            self.assertEqual(compare_outputs(left, right, 257)["values"], 771)
            data[129] += 1
            (right/"2.f32").write_bytes(data.tobytes())
            with self.assertRaisesRegex(ValueError, "column 129"):
                compare_outputs(left, right, 257)
            data[129] = float("nan")
            (right/"2.f32").write_bytes(data.tobytes())
            with self.assertRaisesRegex(ValueError, "column 129"):
                compare_outputs(left, right, 257)


if __name__ == "__main__":
    unittest.main()
