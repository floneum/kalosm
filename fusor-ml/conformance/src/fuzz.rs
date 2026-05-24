use std::{
    ops::{Range, RangeInclusive},
    sync::Arc,
};

use fusor::{DataType, Device, SimdElement, Tensor};
use rand::{
    RngCore, SeedableRng,
    distr::{Distribution, StandardUniform, Uniform},
    rngs::StdRng,
};

use crate::random_tensor;

#[derive(Clone, Debug)]
pub enum FuzzSizeSpec {
    Fixed(usize),
    Choices(Arc<[usize]>),
    Range { start: usize, end_exclusive: usize },
}

impl FuzzSizeSpec {
    fn sample(&self, rng: &mut StdRng) -> usize {
        match self {
            FuzzSizeSpec::Fixed(size) => *size,
            FuzzSizeSpec::Choices(choices) => {
                assert!(
                    !choices.is_empty(),
                    "fuzz size choice list must contain at least one size"
                );
                let index = (rng.next_u64() as usize) % choices.len();
                choices[index]
            }
            FuzzSizeSpec::Range {
                start,
                end_exclusive,
            } => {
                assert!(
                    start < end_exclusive,
                    "fuzz size range must not be empty: {start}..{end_exclusive}"
                );
                Uniform::new(*start, *end_exclusive)
                    .expect("validated non-empty size range")
                    .sample(rng)
            }
        }
    }
}

impl From<usize> for FuzzSizeSpec {
    fn from(value: usize) -> Self {
        Self::Fixed(value)
    }
}

impl<const N: usize> From<[usize; N]> for FuzzSizeSpec {
    fn from(value: [usize; N]) -> Self {
        Self::Choices(Arc::from(value))
    }
}

impl From<Vec<usize>> for FuzzSizeSpec {
    fn from(value: Vec<usize>) -> Self {
        Self::Choices(Arc::from(value.into_boxed_slice()))
    }
}

impl From<Box<[usize]>> for FuzzSizeSpec {
    fn from(value: Box<[usize]>) -> Self {
        Self::Choices(Arc::from(value))
    }
}

impl From<Range<usize>> for FuzzSizeSpec {
    fn from(value: Range<usize>) -> Self {
        Self::Range {
            start: value.start,
            end_exclusive: value.end,
        }
    }
}

impl From<RangeInclusive<usize>> for FuzzSizeSpec {
    fn from(value: RangeInclusive<usize>) -> Self {
        let (start, end) = value.into_inner();
        Self::Range {
            start,
            end_exclusive: end
                .checked_add(1)
                .expect("inclusive fuzz size range upper bound overflowed"),
        }
    }
}

pub trait IntoFuzzShape<const R: usize> {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R];
}

impl<const R: usize> IntoFuzzShape<R> for [usize; R] {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R] {
        self.map(FuzzSizeSpec::from)
    }
}

impl<const R: usize> IntoFuzzShape<R> for [FuzzSizeSpec; R] {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R] {
        self
    }
}

impl<const R: usize, const N: usize> IntoFuzzShape<R> for [[usize; N]; R] {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R] {
        self.map(FuzzSizeSpec::from)
    }
}

impl<const R: usize> IntoFuzzShape<R> for [Range<usize>; R] {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R] {
        self.map(FuzzSizeSpec::from)
    }
}

impl<const R: usize> IntoFuzzShape<R> for [RangeInclusive<usize>; R] {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R] {
        self.map(FuzzSizeSpec::from)
    }
}

#[derive(Clone)]
pub struct FuzzGenerator<const R: usize, T: SimdElement> {
    value_seed: u64,
    shape_seed: u64,
    distribution: Arc<dyn Fn(&mut rand::rngs::StdRng) -> T + Send + Sync>,
    shape_specs: [FuzzSizeSpec; R],
    phantom: std::marker::PhantomData<T>,
}

impl<const R: usize, T: SimdElement + DataType> FuzzGenerator<R, T> {
    pub fn new(shape: impl IntoFuzzShape<R>) -> Self
    where
        StandardUniform: rand::distr::Distribution<T>,
    {
        Self::with_sampler(shape, |rng| StandardUniform.sample(rng))
    }

    /// Construct a fuzz generator from an explicit sampler closure.
    ///
    /// Use this for dtypes (e.g. `f16`) where `StandardUniform` is not implemented.
    pub fn with_sampler(
        shape: impl IntoFuzzShape<R>,
        sampler: impl Fn(&mut StdRng) -> T + Send + Sync + 'static,
    ) -> Self {
        Self {
            value_seed: 0,
            shape_seed: 0,
            distribution: Arc::new(sampler),
            shape_specs: shape.into_shape_specs(),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.value_seed = seed;
        self
    }

    pub fn with_rng(mut self, mut rng: impl RngCore) -> Self {
        self.value_seed = rng.next_u64();
        self
    }

    pub fn with_shape_seed(mut self, seed: u64) -> Self {
        self.shape_seed = seed;
        self
    }

    pub fn with_distribution(
        mut self,
        distribution: impl Distribution<T> + Send + Sync + 'static,
    ) -> Self {
        self.distribution = Arc::new(move |rng| distribution.sample(rng));
        self
    }

    fn value_seed_for_run(&self, run: usize) -> u64 {
        self.value_seed
            ^ (run as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (R as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
    }

    fn shape_seed_for_run(&self, run: usize) -> u64 {
        self.shape_seed
            ^ (run as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ (R as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
    }

    fn sample_shape(&self, rng: &mut StdRng) -> [usize; R] {
        self.shape_specs
            .clone()
            .map(|shape_spec| shape_spec.sample(rng))
    }

    #[cfg(test)]
    fn shape_for_run(&self, run: usize) -> [usize; R] {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.shape_seed_for_run(run));
        self.sample_shape(&mut rng)
    }

    fn generate_for_run(&self, device: &Device, run: usize) -> Tensor<R, T> {
        let mut shape_rng = rand::rngs::StdRng::seed_from_u64(self.shape_seed_for_run(run));
        let shape = self.sample_shape(&mut shape_rng);
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.value_seed_for_run(run));
        let base = random_tensor(device, shape, &mut rng, &*self.distribution);
        // Vary layout based on run index: even runs stay contiguous, odd runs
        // get a non-contiguous layout so operations are tested with both.
        let strategy = run % 3;
        match strategy {
            0 => base,
            1 => make_transposed(base, &mut rng, &*self.distribution),
            _ => make_sliced(base, &mut rng, &*self.distribution),
        }
    }
}

/// Generate a contiguous tensor with the last two dimensions swapped, then
/// transpose it so the result has the correct shape but non-contiguous strides.
///
/// On GPU the lazy transpose view is preserved, so the op under test sees a
/// non-contiguous stride layout. On CPU `to_concrete()` materializes the view
/// into a contiguous backing buffer, so CPU only exercises the contiguous path.
/// The CPU `to_concrete()` is not an oversight: the CPU backend is contiguous-only
/// by design (see the CPU fusion backing in `cpu/src/lib.rs`), so non-contig stride
/// coverage on CPU is not reachable from conformance.
fn make_transposed<const R: usize, T: SimdElement + DataType + Default>(
    tensor: Tensor<R, T>,
    rng: &mut StdRng,
    sample: &dyn Fn(&mut StdRng) -> T,
) -> Tensor<R, T> {
    if R < 2 {
        return tensor;
    }
    let shape = tensor.shape();
    // Build a shape with the last two dims swapped.
    let transposed_shape: [usize; R] = std::array::from_fn(|i| {
        if i == R - 2 {
            shape[R - 1]
        } else if i == R - 1 {
            shape[R - 2]
        } else {
            shape[i]
        }
    });
    let device = tensor.device();
    // Generate fresh contiguous data in the transposed shape, then
    // transpose so the logical shape matches `self.shape` but strides
    // are non-contiguous (the last two dims' strides are swapped).
    let contiguous = random_tensor(&device, transposed_shape, rng, sample);
    contiguous.transpose(R - 2, R - 1).to_concrete()
}

/// Generate a larger tensor and narrow it back to the original shape,
/// producing a tensor with a non-zero offset in the underlying buffer.
///
/// Same materialization caveat as [`make_transposed`]: on GPU the narrowed
/// view reaches the op under test, but on CPU `to_concrete()` materializes it
/// into a fresh contiguous buffer (CPU backend is contiguous-only by design).
fn make_sliced<const R: usize, T: SimdElement + DataType + Default>(
    tensor: Tensor<R, T>,
    rng: &mut StdRng,
    sample: &dyn Fn(&mut StdRng) -> T,
) -> Tensor<R, T> {
    if R == 0 {
        return tensor;
    }
    let shape = tensor.shape();
    // Pick the last dimension to pad. We prepend `pad` extra elements
    // along that dimension so the resulting narrow has a non-zero offset.
    let pad_dim = R - 1;
    let pad = 1;
    let padded_size = shape[pad_dim] + pad;
    let padded_shape: [usize; R] =
        std::array::from_fn(|i| if i == pad_dim { padded_size } else { shape[i] });
    let device = tensor.device();
    let padded = random_tensor(&device, padded_shape, rng, sample);
    // Narrow away the extra padding, creating an offset view.
    padded.narrow(pad_dim, pad, shape[pad_dim]).to_concrete()
}

impl<const R: usize> FuzzGenerator<R, f32> {
    pub fn with_positive(mut self) -> Self {
        self.distribution =
            Arc::new(move |rng| Uniform::new(0.0, 1.0).expect("0.0 < 1.0").sample(rng));
        self
    }
}

#[doc(hidden)]
pub trait GenerateFromDevice {
    type Output;
    fn generate(&mut self, device: &Device, run: usize) -> Self::Output;
}

impl<F, O> GenerateFromDevice for F
where
    F: FnMut(&Device) -> O,
{
    type Output = O;
    fn generate(&mut self, device: &Device, _run: usize) -> Self::Output {
        (self)(device)
    }
}

impl<const R: usize, T: SimdElement + DataType> GenerateFromDevice for FuzzGenerator<R, T> {
    type Output = Tensor<R, T>;
    fn generate(&mut self, device: &Device, run: usize) -> Self::Output {
        self.generate_for_run(device, run)
    }
}

#[cfg(test)]
mod api_tests {
    use fusor::{Device, Tensor};

    use crate::{FuzzGenerator, FuzzSizeSpec};

    #[tokio::test]
    async fn test_api() {
        crate::assert(async |x: fusor::Tensor<1, f32>| x.sin().to_concrete())
            .arg(FuzzGenerator::<1, f32>::new([63..=65]))
            .equal_to_resolved_with_device(async |vec: Vec<f32>, device: Device| {
                let expected = vec.iter().map(|&v| v.sin()).collect::<Vec<_>>();
                Tensor::new(&device, &expected)
            })
            .runs(10)
            .await
            .unwrap();
    }

    #[test]
    fn fuzz_generator_accepts_size_choices_and_ranges() {
        let choice_generator =
            FuzzGenerator::<2, f32>::new([[255, 256, 257], [31, 32, 33]]).with_seed(1234);
        for run in 0..24 {
            let shape = choice_generator.shape_for_run(run);
            assert!([255, 256, 257].contains(&shape[0]));
            assert!([31, 32, 33].contains(&shape[1]));
        }

        let range_generator = FuzzGenerator::<2, f32>::new([255..=257, 31..=33]).with_seed(5678);
        for run in 0..24 {
            let shape = range_generator.shape_for_run(run);
            assert!((255..=257).contains(&shape[0]));
            assert!((31..=33).contains(&shape[1]));
        }

        let mixed_generator = FuzzGenerator::<2, f32>::new([
            FuzzSizeSpec::from([255, 256, 257]),
            FuzzSizeSpec::from(63..=65),
        ])
        .with_seed(9012);
        for run in 0..24 {
            let shape = mixed_generator.shape_for_run(run);
            assert!([255, 256, 257].contains(&shape[0]));
            assert!((63..=65).contains(&shape[1]));
        }
    }

    #[test]
    fn fuzz_generator_shapes_do_not_depend_on_value_seed() {
        let first = FuzzGenerator::<2, f32>::new([255..=257, 63..=65]).with_seed(1);
        let second = FuzzGenerator::<2, f32>::new([255..=257, 63..=65]).with_seed(2);
        for run in 0..24 {
            assert_eq!(first.shape_for_run(run), second.shape_for_run(run));
        }
    }
}
