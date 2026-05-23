//! Rank and dimension helpers for tensor operations

/// A trait for resolving dimension indices at compile time or runtime.
/// Allows using either concrete `usize` values or symbolic dimension types like `D::Minus1`.
pub trait Dim<const R: usize>: Copy {
    fn resolve(self) -> usize;
}

impl<const R: usize> Dim<R> for usize {
    fn resolve(self) -> usize {
        self
    }
}

/// Dimension helpers for symbolic dimension access
#[allow(non_snake_case)]
pub mod D {
    use super::*;

    /// The last dimension (index R-1)
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct Minus1;

    impl<const R: usize> Dim<R> for Minus1 {
        fn resolve(self) -> usize {
            const {
                assert!(R > 0);
            }
            R - 1
        }
    }

    /// The second to last dimension (index R-2)
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct Minus2;

    impl<const R: usize> Dim<R> for Minus2 {
        fn resolve(self) -> usize {
            const {
                assert!(R > 1);
            }
            R - 2
        }
    }
}

/// Internal helper macro: generate the per-rank `impl` block for one rank row.
/// Used by `impl_rank_traits!`; not part of the public API.
///
/// The traits `NextRankInner`, `LastRankInner`, `SmallerRankInner`,
/// `LargerRankInner`, and `MaxRankInner` must be in scope at the call site
/// (they are defined locally in each consumer crate to satisfy the orphan
/// rule when impl'ing for tuples).
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_next_last_row {
    ($tensor:ident; $($smaller:literal, )* [0] $(, $larger:literal)*) => {
        $(
            impl<D> SmallerRankInner<{0 - $smaller}> for $tensor<0, D> {
                type SmallerRank = $tensor<$smaller, D>;
                type SmallerByArray = [usize; {0 - $smaller}];
            }
        )*

        impl<D> NextRankInner for $tensor<0, D> {
            type NextRank = $tensor<1, D>;
        }

        $(
            impl<D> LargerRankInner<{$larger - 0}> for $tensor<0, D> {
                type LargerRank = $tensor<$larger, D>;
                type LargerByArray = [usize; {$larger - 0}];
            }

            impl<D> MaxRankInner for ($tensor<0, D>, $tensor<$larger, D>) {
                type MaxRank = $tensor<$larger, D>;
            }

            impl<D> MaxRankInner for ($tensor<$larger, D>, $tensor<0, D>) {
                type MaxRank = $tensor<$larger, D>;
            }
        )*
    };

    ($tensor:ident; $($smaller:literal, )* [$R:literal] $(, $larger:literal)*) => {
        $(
            impl<D> SmallerRankInner<{$R - $smaller}> for $tensor<$R, D> {
                type SmallerRank = $tensor<$smaller, D>;
                type SmallerByArray = [usize; {$R - $smaller}];
            }
        )*

        impl<D> NextRankInner for $tensor<$R, D> {
            type NextRank = $tensor<{ $R + 1 }, D>;
        }

        impl<D> LastRankInner for $tensor<$R, D> {
            type LastRank = $tensor<{ $R - 1 }, D>;
        }

        $(
            impl<D> LargerRankInner<{$larger - $R}> for $tensor<$R, D> {
                type LargerRank = $tensor<$larger, D>;
                type LargerByArray = [usize; {$larger - $R}];
            }

            impl<D> MaxRankInner for ($tensor<$R, D>, $tensor<$larger, D>) {
                type MaxRank = $tensor<$larger, D>;
            }

            impl<D> MaxRankInner for ($tensor<$larger, D>, $tensor<$R, D>) {
                type MaxRank = $tensor<$larger, D>;
            }
        )*
    };
}

/// Generate the rank-relation `impl`s (`NextRankInner`, `LastRankInner`,
/// `SmallerRankInner`, `LargerRankInner`, `MaxRankInner`) for a tensor type
/// across ranks 0-21.
///
/// The five `*Inner` traits must be defined locally in the consumer crate and
/// in scope at the macro call site (the orphan rule requires the impls live
/// in the same crate as the trait, since the macro generates impls for
/// tuples of `$tensor<N, D>`).
///
/// ```ignore
/// use crate::{LargerRankInner, LastRankInner, MaxRankInner, NextRankInner, SmallerRankInner, Tensor};
/// fusor_types::impl_rank_traits!(Tensor);
/// ```
#[macro_export]
macro_rules! impl_rank_traits {
    ($tensor:ident) => {
        impl<const N: usize, D> MaxRankInner for ($tensor<N, D>, $tensor<N, D>) {
            type MaxRank = $tensor<N, D>;
        }

        impl<D> LastRankInner for $tensor<21, D> {
            type LastRank = $tensor<20, D>;
        }

        impl<D> NextRankInner for $tensor<21, D> {
            type NextRank = $tensor<21, D>;
        }

        #[rustfmt::skip]
        const _: () = {
            $crate::__impl_next_last_row!($tensor; [0], 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, [1], 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, [2], 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, [3], 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, [4], 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, [5], 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, [6], 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, [7], 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, [8], 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, [9], 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, [10], 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, [11], 12, 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, [12], 13, 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, [13], 14, 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, [14], 15, 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, [15], 16, 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, [16], 17, 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, [17], 18, 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, [18], 19, 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, [19], 20);
            $crate::__impl_next_last_row!($tensor; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, [20]);
        };
    };
}
