//! Conditional tensor operations: where_cond
//! Selects elements based on condition tensor != 0

use crate::expr::linear_to_indices;
use crate::{ConcreteTensor, ResolvedTensor, SimdElement};

/// Helper trait for types that can be compared to zero
pub trait IsNonZero: SimdElement {
    fn is_nonzero(&self) -> bool;
}

macro_rules! impl_is_nonzero {
    ($($ty:ty => $zero:expr),*) => {
        $(
            impl IsNonZero for $ty {
                fn is_nonzero(&self) -> bool {
                    *self != $zero
                }
            }
        )*
    };
}

impl_is_nonzero!(
    f32 => 0.0, f64 => 0.0,
    i8 => 0, i16 => 0, i32 => 0, i64 => 0,
    u8 => 0, u16 => 0, u32 => 0, u64 => 0
);

/// Conditional selection: where condition != 0, select on_true, else on_false
#[inline(always)]
pub(crate) fn where_cond_ref<E, const R: usize>(
    cond: &ConcreteTensor<E, R>,
    on_true: &ConcreteTensor<E, R>,
    on_false: &ConcreteTensor<E, R>,
) -> ConcreteTensor<E, R>
where
    E: SimdElement + IsNonZero,
{
    let shape: [usize; R] = cond
        .layout()
        .shape()
        .try_into()
        .expect("Shape length mismatch");

    debug_assert_eq!(
        cond.layout().shape(),
        on_true.layout().shape(),
        "where_cond: cond and on_true shape mismatch"
    );
    debug_assert_eq!(
        cond.layout().shape(),
        on_false.layout().shape(),
        "where_cond: cond and on_false shape mismatch"
    );

    let all_contiguous = cond.layout().is_contiguous()
        && on_true.layout().is_contiguous()
        && on_false.layout().is_contiguous();

    if all_contiguous {
        let cond_data = cond.data();
        let true_data = on_true.data();
        let false_data = on_false.data();
        ConcreteTensor::from_fn(shape, |i| {
            if cond_data[i].is_nonzero() {
                true_data[i]
            } else {
                false_data[i]
            }
        })
    } else {
        ConcreteTensor::from_fn(shape, |out_idx| {
            let indices = linear_to_indices::<R>(out_idx, &shape);

            let cond_idx = cond.layout().linear_index(&indices);
            let true_idx = on_true.layout().linear_index(&indices);
            let false_idx = on_false.layout().linear_index(&indices);

            let cond_val = cond.data()[cond_idx];
            if cond_val.is_nonzero() {
                on_true.data()[true_idx]
            } else {
                on_false.data()[false_idx]
            }
        })
    }
}
