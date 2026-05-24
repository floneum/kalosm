use crate::{ConcreteTensor, SimdElement};

// Trait for mapping tensor to its one-rank-smaller type (for axis reductions)
pub trait LastRankInner {
    type LastRank;
}

pub trait LastRank<const R: usize, T: SimdElement>:
    LastRankInner<LastRank = ConcreteTensor<T, R>>
{
}

impl<const R: usize, T: SimdElement, X> LastRank<R, T> for X where
    X: LastRankInner<LastRank = ConcreteTensor<T, R>>
{
}

// Trait for mapping tensor to its next-higher rank type (for unsqueeze)
pub trait NextRankInner {
    type NextRank;
}

pub trait NextRank<const R: usize, T: SimdElement>:
    NextRankInner<NextRank = ConcreteTensor<T, R>>
{
}

impl<const R: usize, T: SimdElement, X> NextRank<R, T> for X where
    X: NextRankInner<NextRank = ConcreteTensor<T, R>>
{
}

// Trait for mapping tensor to a smaller rank (for squeeze, reduce)
pub trait SmallerRankInner<const DIFF: usize> {
    type SmallerRank;
}

pub trait SmallerRank<const R: usize, const DIFF: usize, T: SimdElement>:
    SmallerRankInner<DIFF, SmallerRank = ConcreteTensor<T, R>>
{
}

impl<const R: usize, const DIFF: usize, T: SimdElement, X> SmallerRank<R, DIFF, T> for X where
    X: SmallerRankInner<DIFF, SmallerRank = ConcreteTensor<T, R>>
{
}

// Trait for mapping tensor to a larger rank (for unsqueeze, expand)
pub trait LargerRankInner<const DIFF: usize> {
    type LargerRank;
}

pub trait LargerRank<const R: usize, const DIFF: usize, T: SimdElement>:
    LargerRankInner<DIFF, LargerRank = ConcreteTensor<T, R>>
{
}

impl<const R: usize, const DIFF: usize, T: SimdElement, X> LargerRank<R, DIFF, T> for X where
    X: LargerRankInner<DIFF, LargerRank = ConcreteTensor<T, R>>
{
}

// Trait for mapping two tensors to their max rank (for broadcasting operations)
pub trait MaxRankInner {
    type MaxRank;
}

pub trait MaxRank<const R: usize, T: SimdElement>:
    MaxRankInner<MaxRank = ConcreteTensor<T, R>>
{
}

impl<const R: usize, T: SimdElement, X> MaxRank<R, T> for X where
    X: MaxRankInner<MaxRank = ConcreteTensor<T, R>>
{
}

fusor_types::impl_type_rank_traits!(ConcreteTensor, SimdElement);
