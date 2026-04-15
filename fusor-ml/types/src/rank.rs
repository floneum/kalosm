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
