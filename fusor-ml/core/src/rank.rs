use crate::Tensor;

// Re-export dimension helpers from fusor-types.
pub use fusor_types::{D, Dim};

pub trait NextRankInner {
    type NextRank: LastRankInner + NextRankInner;
}

pub trait NextRank<const R: usize, D>: NextRankInner<NextRank = Tensor<R, D>> {}

impl<const R: usize, D, T> NextRank<R, D> for T where T: NextRankInner<NextRank = Tensor<R, D>> {}

pub trait SmallerRankInner<const R: usize> {
    type SmallerRank;
    type SmallerByArray;
}

pub trait SmallerRank<const R: usize, const S: usize, D>:
    SmallerRankInner<R, SmallerRank = Tensor<S, D>, SmallerByArray = [usize; R]>
{
}

impl<const R: usize, const S: usize, D, T> SmallerRank<R, S, D> for T where
    T: SmallerRankInner<R, SmallerRank = Tensor<S, D>, SmallerByArray = [usize; R]>
{
}

pub trait LastRankInner {
    type LastRank: NextRankInner;
}

pub trait LastRank<const R: usize, D>: LastRankInner<LastRank = Tensor<R, D>> {}

impl<const R: usize, D, T> LastRank<R, D> for T where T: LastRankInner<LastRank = Tensor<R, D>> {}

pub trait LargerRankInner<const R: usize> {
    type LargerRank;
    type LargerByArray;
}

pub trait LargerRank<const R: usize, const L: usize, D>:
    LargerRankInner<R, LargerRank = Tensor<L, D>, LargerByArray = [usize; R]>
{
}

impl<const R: usize, const L: usize, D, T> LargerRank<R, L, D> for T where
    T: LargerRankInner<R, LargerRank = Tensor<L, D>, LargerByArray = [usize; R]>
{
}

pub trait MaxRankInner {
    type MaxRank;
}

pub trait MaxRank<const R: usize, D>: MaxRankInner<MaxRank = Tensor<R, D>> {}

impl<const R: usize, D, T> MaxRank<R, D> for T where T: MaxRankInner<MaxRank = Tensor<R, D>> {}

fusor_types::impl_rank_traits!(Tensor);
