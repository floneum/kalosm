import unittest
from shader_liveness import analyze


class ShaderLivenessTests(unittest.TestCase):
    def test_long_vector_lifetime_and_last_use(self):
        result = analyze('''fn main() {
    let a: vec4<f32> = vec4<f32>(1.);
    let b: f32 = a.x;
    let c: f32 = b + 2.;
    output[0] = c;
    output[1] = a.y;
}''')
        self.assertEqual(result['peak_named_words'], 6)
        self.assertEqual(result['peak_float_words'], 6)
        self.assertEqual(result['steps'][-1]['live'], ['a'])

    def test_comments_and_unused_values_do_not_extend_lifetimes(self):
        result = analyze('''fn main() {
    let a: u32 = 1u;
    let unused: vec4<f32> = vec4<f32>(1.);
    output[0] = a;
    // unused a
}''')
        self.assertEqual(result['unused_temporaries'], 1)
        self.assertEqual(result['peak_named_words'], 1)
        self.assertEqual(result['peak_float_words'], 0)

    def test_rejects_ambiguous_shadowing_and_multiple_functions(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            analyze('fn main() {\nlet a: f32 = 1.;\nlet a: f32 = 2.;\n}')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            analyze('fn helper() {}\nfn main() {}')


if __name__ == '__main__':
    unittest.main()
