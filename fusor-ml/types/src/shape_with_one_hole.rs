//! Reshape helpers: `ShapeWithOneHole` trait for inferring a single missing dimension
//! from the total element count.
//!
//! Used by `Tensor::reshape` to support shapes like `(2usize, (), 3usize)` where the
//! hole `()` is filled in from the original shape's product.

/// A shape with at most one unknown ("hole") dimension that can be inferred
/// from the original tensor's total element count.
///
/// Implemented for `[usize; R]` (no hole — fully specified) and for tuples of
/// `usize` values containing exactly one `()` placeholder, e.g. `(2usize, (), 3usize)`.
pub trait ShapeWithOneHole<const R: usize> {
    fn resolve_shape(&self, original_shape: &[usize]) -> [usize; R];
}

impl<const R: usize> ShapeWithOneHole<R> for [usize; R] {
    fn resolve_shape(&self, _original_shape: &[usize]) -> [usize; R] {
        *self
    }
}

impl ShapeWithOneHole<1> for ((),) {
    fn resolve_shape(&self, original_shape: &[usize]) -> [usize; 1] {
        [original_shape.iter().product()]
    }
}

pub(crate) trait IndexTuple<const INDEX: usize> {
    type Output;
    fn const_index(&self) -> &Self::Output;
}

macro_rules! impl_index_tuple {
    // Internal: generate a single impl
    (@impl [$($T:ident),+] $idx:tt $Ti:ident) => {
        impl<$($T),+> IndexTuple<$idx> for ($($T,)+) {
            type Output = $Ti;
            fn const_index(&self) -> &Self::Output {
                &self.$idx
            }
        }
    };

    // Internal: recursively process parallel lists of indices and types
    (@step [$($T:ident),+] [$idx:tt $(, $rest_idx:tt)*] [$curr:ident $(, $rest:ident)*]) => {
        impl_index_tuple!(@impl [$($T),+] $idx $curr);
        impl_index_tuple!(@step [$($T),+] [$($rest_idx),*] [$($rest),*]);
    };

    // Base case: both lists exhausted
    (@step [$($T:ident),+] [] []) => {};

    // Entry point: [indices] followed by types
    ([$($idx:tt),+] $($T:ident),+ $(,)?) => {
        impl_index_tuple!(@step [$($T),+] [$($idx),+] [$($T),+]);
    };
}

impl_index_tuple!([0] T);
impl_index_tuple!([0, 1] T1, T2);
impl_index_tuple!([0, 1, 2] T1, T2, T3);
impl_index_tuple!([0, 1, 2, 3] T1, T2, T3, T4);
impl_index_tuple!([0, 1, 2, 3, 4] T1, T2, T3, T4, T5);
impl_index_tuple!([0, 1, 2, 3, 4, 5] T1, T2, T3, T4, T5, T6);
impl_index_tuple!([0, 1, 2, 3, 4, 5, 6] T1, T2, T3, T4, T5, T6, T7);
impl_index_tuple!([0, 1, 2, 3, 4, 5, 6, 7] T1, T2, T3, T4, T5, T6, T7, T8);
impl_index_tuple!([0, 1, 2, 3, 4, 5, 6, 7, 8] T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_index_tuple!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
impl_index_tuple!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);

macro_rules! impl_shape_with_one_hole {
    ($($name:ident),+) => {
        impl_shape_with_one_hole!(@push_forward (), $($name,)+);
    };
    (@push_forward $($before:ident,)* (), $next:ident, $($after:ident,)*) => {
        impl_shape_with_one_hole!(@impl_tuple $($before,)* (), $next, $($after,)*);
        impl_shape_with_one_hole!(@push_forward $($before,)* $next, (), $($after,)*);
    };
    (@push_forward $($before:ident,)* (),) => {
        impl_shape_with_one_hole!(@impl_tuple $($before,)* (),);
    };
    (@usize $($t:tt)*) => {
        usize
    };
    (@one $($t:ident)*) => {
        1
    };
    (@tuple_size $($before:ident,)* (), $($after:ident,)*) => {
        $(impl_shape_with_one_hole!(@one $before) + )* $(impl_shape_with_one_hole!(@one $after) + )* 1
    };
    (@known_size $first:ident, $($before:ident,)* (), $($after:ident,)* = $sum:expr) => {
        const $first: usize = $sum;
        impl_shape_with_one_hole!(@known_size $($before,)* (), $($after,)* = $sum + 1);
    };
    (@known_size (), $first:ident, $($after:ident,)* = $sum:expr) => {
        const $first: usize = $sum + 1;
        impl_shape_with_one_hole!(@known_size (), $($after,)* = $sum + 1);
    };
    (@known_size (), = $sum:expr) => {};
    (@impl_tuple $($before:ident,)* (), $($after:ident,)*) => {
        #[allow(non_snake_case)]
        impl ShapeWithOneHole<{impl_shape_with_one_hole!(@tuple_size $($before,)* (), $($after,)*)}> for ($(impl_shape_with_one_hole!(@usize $before),)* (), $(impl_shape_with_one_hole!(@usize $after),)*) {
            fn resolve_shape(&self, original_shape: &[usize]) -> [usize; impl_shape_with_one_hole!(@tuple_size $($before,)* (), $($after,)*)] {
                let total_size = original_shape.iter().product::<usize>();
                impl_shape_with_one_hole!(@known_size $($before,)* (), $($after,)* = 0);
                let known_size = {
                    let mut size = 1;
                    $(
                        size *= *IndexTuple::<{$before}>::const_index(self);
                    )*
                    $(
                        size *= *IndexTuple::<{$after}>::const_index(self);
                    )*
                    size
                };
                let hole_size = total_size / known_size;
                [
                    $(
                        *IndexTuple::<{$before}>::const_index(self),
                    )*
                    hole_size,
                    $(
                        *IndexTuple::<{$after}>::const_index(self),
                    )*
                ]
            }
        }
    };
}

impl_shape_with_one_hole!(A);
impl_shape_with_one_hole!(A, B);
impl_shape_with_one_hole!(A, B, C);
impl_shape_with_one_hole!(A, B, C, D);
impl_shape_with_one_hole!(A, B, C, D, E);
impl_shape_with_one_hole!(A, B, C, D, E, F);
impl_shape_with_one_hole!(A, B, C, D, E, F, G);
impl_shape_with_one_hole!(A, B, C, D, E, F, G, H);
impl_shape_with_one_hole!(A, B, C, D, E, F, G, H, I);
impl_shape_with_one_hole!(A, B, C, D, E, F, G, H, I, J);
