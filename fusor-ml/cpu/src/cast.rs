//! Type casting operations for tensors

use crate::{ConcreteTensor, LazyBacking, SimdElement};

/// Trait for numeric types that can be cast to another type
pub trait CastTo<T>: SimdElement {
    fn cast(self) -> T;
}

// Implement CastTo for all numeric type pairs using a macro
macro_rules! impl_cast {
    ($from:ty => $($to:ty),*) => {
        $(
            impl CastTo<$to> for $from {
                #[inline(always)]
                fn cast(self) -> $to {
                    self as $to
                }
            }
        )*
    };
}

// f32 casts
impl_cast!(f32 => f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// f64 casts
impl_cast!(f64 => f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// i8 casts
impl_cast!(i8 => f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// i16 casts
impl_cast!(i16 => f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// i32 casts
impl_cast!(i32 => f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// i64 casts
impl_cast!(i64 => f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// u8 casts
impl_cast!(u8 => f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// u16 casts
impl_cast!(u16 => f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// u32 casts
impl_cast!(u32 => f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// u64 casts
impl_cast!(u64 => f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// half::f16 casts - these require special handling since f16 isn't a primitive
impl CastTo<f32> for half::f16 {
    #[inline(always)]
    fn cast(self) -> f32 {
        self.to_f32()
    }
}

impl CastTo<f64> for half::f16 {
    #[inline(always)]
    fn cast(self) -> f64 {
        self.to_f64()
    }
}

impl CastTo<half::f16> for half::f16 {
    #[inline(always)]
    fn cast(self) -> half::f16 {
        self
    }
}

impl CastTo<half::f16> for f32 {
    #[inline(always)]
    fn cast(self) -> half::f16 {
        half::f16::from_f32(self)
    }
}

impl CastTo<half::f16> for f64 {
    #[inline(always)]
    fn cast(self) -> half::f16 {
        half::f16::from_f64(self)
    }
}

/// Cast a tensor from one element type to another
pub(crate) fn cast_tensor<T, T2, const R: usize>(
    input: &ConcreteTensor<T, R>,
) -> ConcreteTensor<T2, R>
where
    T: SimdElement + CastTo<T2>,
    T2: SimdElement,
{
    let shape: [usize; R] = input
        .layout()
        .shape()
        .try_into()
        .expect("Shape length mismatch");
    ConcreteTensor::from_fn(shape, |i| input.eval_scalar(i).cast())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cast_f32_to_i32() {
        let a = ConcreteTensor::<f32, 1>::from_slice([4], &[1.5, 2.7, -3.2, 4.9]);
        let b: ConcreteTensor<i32, 1> = cast_tensor(&a);

        assert_eq!(b.get([0]), 1); // 1.5 -> 1
        assert_eq!(b.get([1]), 2); // 2.7 -> 2
        assert_eq!(b.get([2]), -3); // -3.2 -> -3
        assert_eq!(b.get([3]), 4); // 4.9 -> 4
    }

    #[test]
    fn test_cast_i32_to_f64() {
        let a = ConcreteTensor::<i32, 1>::from_slice([3], &[1, -2, 3]);
        let b: ConcreteTensor<f64, 1> = cast_tensor(&a);

        assert_eq!(b.get([0]), 1.0);
        assert_eq!(b.get([1]), -2.0);
        assert_eq!(b.get([2]), 3.0);
    }
}
