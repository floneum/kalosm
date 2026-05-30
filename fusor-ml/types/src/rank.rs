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

#[doc(hidden)]
#[macro_export]
macro_rules! __impl_type_rank_last {
    ($tensor:ident, $elem_trait:path; $($R:literal),* $(,)?) => {
        $(
            impl<T: $elem_trait> LastRankInner for $tensor<T, $R> {
                type LastRank = $tensor<T, { $R - 1 }>;
            }
        )*
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __impl_type_rank_next {
    ($tensor:ident, $elem_trait:path; $($R:literal),* $(,)?) => {
        $(
            impl<T: $elem_trait> NextRankInner for $tensor<T, $R> {
                type NextRank = $tensor<T, { $R + 1 }>;
            }
        )*
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __impl_type_rank_smaller_row {
    ($tensor:ident, $elem_trait:path; $R:literal; $($DIFF:literal => $OUT:literal),* $(,)?) => {
        $(
            impl<T: $elem_trait> SmallerRankInner<$DIFF> for $tensor<T, $R> {
                type SmallerRank = $tensor<T, $OUT>;
            }
        )*
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __impl_type_rank_larger_row {
    ($tensor:ident, $elem_trait:path; $R:literal; $($DIFF:literal => $OUT:literal),* $(,)?) => {
        $(
            impl<T: $elem_trait> LargerRankInner<$DIFF> for $tensor<T, $R> {
                type LargerRank = $tensor<T, $OUT>;
            }
        )*
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __impl_type_rank_max_pair {
    ($tensor:ident, $elem_trait:path; $R1:literal, $R2:literal) => {
        impl<T: $elem_trait> MaxRankInner for ($tensor<T, $R1>, $tensor<T, $R2>) {
            type MaxRank = $tensor<T, $R2>;
        }

        impl<T: $elem_trait> MaxRankInner for ($tensor<T, $R2>, $tensor<T, $R1>) {
            type MaxRank = $tensor<T, $R2>;
        }
    };
}

/// Generate rank-relation impls for tensor types whose generic order is
/// element-then-rank, such as `ConcreteTensor<T, R>`.
///
/// The traits `NextRankInner`, `LastRankInner`, `SmallerRankInner`,
/// `LargerRankInner`, and `MaxRankInner` must be defined locally and in scope.
#[macro_export]
macro_rules! impl_type_rank_traits {
    ($tensor:ident, $elem_trait:path) => {
        $crate::__impl_type_rank_last!($tensor, $elem_trait; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        $crate::__impl_type_rank_next!($tensor, $elem_trait; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

        $crate::__impl_type_rank_smaller_row!($tensor, $elem_trait; 1; 1 => 0);
        $crate::__impl_type_rank_smaller_row!($tensor, $elem_trait; 2; 1 => 1, 2 => 0);
        $crate::__impl_type_rank_smaller_row!($tensor, $elem_trait; 3; 1 => 2, 2 => 1, 3 => 0);
        $crate::__impl_type_rank_smaller_row!($tensor, $elem_trait; 4; 1 => 3, 2 => 2, 3 => 1, 4 => 0);
        $crate::__impl_type_rank_smaller_row!($tensor, $elem_trait; 5; 1 => 4, 2 => 3, 3 => 2, 4 => 1, 5 => 0);
        $crate::__impl_type_rank_smaller_row!($tensor, $elem_trait; 6; 1 => 5, 2 => 4, 3 => 3, 4 => 2, 5 => 1, 6 => 0);
        $crate::__impl_type_rank_smaller_row!($tensor, $elem_trait; 7; 1 => 6, 2 => 5, 3 => 4, 4 => 3, 5 => 2, 6 => 1, 7 => 0);
        $crate::__impl_type_rank_smaller_row!($tensor, $elem_trait; 8; 1 => 7, 2 => 6, 3 => 5, 4 => 4, 5 => 3, 6 => 2, 7 => 1, 8 => 0);
        $crate::__impl_type_rank_smaller_row!($tensor, $elem_trait; 9; 1 => 8, 2 => 7, 3 => 6, 4 => 5, 5 => 4, 6 => 3, 7 => 2, 8 => 1, 9 => 0);
        $crate::__impl_type_rank_smaller_row!($tensor, $elem_trait; 10; 1 => 9, 2 => 8, 3 => 7, 4 => 6, 5 => 5, 6 => 4, 7 => 3, 8 => 2, 9 => 1, 10 => 0);

        $crate::__impl_type_rank_larger_row!($tensor, $elem_trait; 0; 1 => 1, 2 => 2, 3 => 3, 4 => 4, 5 => 5, 6 => 6, 7 => 7, 8 => 8, 9 => 9, 10 => 10);
        $crate::__impl_type_rank_larger_row!($tensor, $elem_trait; 1; 1 => 2, 2 => 3, 3 => 4, 4 => 5, 5 => 6, 6 => 7, 7 => 8, 8 => 9, 9 => 10);
        $crate::__impl_type_rank_larger_row!($tensor, $elem_trait; 2; 1 => 3, 2 => 4, 3 => 5, 4 => 6, 5 => 7, 6 => 8, 7 => 9, 8 => 10);
        $crate::__impl_type_rank_larger_row!($tensor, $elem_trait; 3; 1 => 4, 2 => 5, 3 => 6, 4 => 7, 5 => 8, 6 => 9, 7 => 10);
        $crate::__impl_type_rank_larger_row!($tensor, $elem_trait; 4; 1 => 5, 2 => 6, 3 => 7, 4 => 8, 5 => 9, 6 => 10);
        $crate::__impl_type_rank_larger_row!($tensor, $elem_trait; 5; 1 => 6, 2 => 7, 3 => 8, 4 => 9, 5 => 10);
        $crate::__impl_type_rank_larger_row!($tensor, $elem_trait; 6; 1 => 7, 2 => 8, 3 => 9, 4 => 10);
        $crate::__impl_type_rank_larger_row!($tensor, $elem_trait; 7; 1 => 8, 2 => 9, 3 => 10);
        $crate::__impl_type_rank_larger_row!($tensor, $elem_trait; 8; 1 => 9, 2 => 10);
        $crate::__impl_type_rank_larger_row!($tensor, $elem_trait; 9; 1 => 10);

        impl<const N: usize, T: $elem_trait> MaxRankInner for ($tensor<T, N>, $tensor<T, N>) {
            type MaxRank = $tensor<T, N>;
        }

        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 0, 1);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 0, 2);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 0, 3);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 0, 4);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 0, 5);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 0, 6);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 0, 7);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 0, 8);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 0, 9);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 0, 10);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 1, 2);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 1, 3);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 1, 4);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 1, 5);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 1, 6);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 1, 7);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 1, 8);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 1, 9);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 1, 10);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 2, 3);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 2, 4);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 2, 5);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 2, 6);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 2, 7);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 2, 8);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 2, 9);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 2, 10);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 3, 4);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 3, 5);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 3, 6);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 3, 7);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 3, 8);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 3, 9);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 3, 10);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 4, 5);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 4, 6);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 4, 7);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 4, 8);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 4, 9);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 4, 10);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 5, 6);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 5, 7);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 5, 8);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 5, 9);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 5, 10);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 6, 7);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 6, 8);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 6, 9);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 6, 10);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 7, 8);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 7, 9);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 7, 10);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 8, 9);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 8, 10);
        $crate::__impl_type_rank_max_pair!($tensor, $elem_trait; 9, 10);
    };
}
