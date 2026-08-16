//! `pool_max` and `pool_min` as macro ops over `Window` + `Fold`.
//!
//! Because `Window` carries `(window, step)` as integers, a non-overlapping
//! pool's adjoint is provably an elementwise mask-and-broadcast — so the
//! trainer's reshape-as-maxpool workaround deletes, and it deletes on the
//! symbolic-shape path too, where injectivity of a relative stride
//! composition is undecidable and a `Restride`-based pool would have to
//! degrade to a scatter.

use fusor2_autograd::tape::{GraphTape, TapeExt, accum_dtype};
use fusor2_ir::autograd::{Tape, Val};
use fusor2_ir::ir::logical::Logical;
use fusor2_ir::scalar::BinOp;
use fusor2_ir::shape::SlidingWindow;
use fusor2_ir::{Error, Result};
use smallvec::SmallVec;

use crate::composite::{MacroAttr, MacroOp, PoolReduce, macro_op};
use crate::tensor::Tensor;

/// One pooled axis. `From<usize>` makes the stride equal the window, which is
/// the non-overlapping case whose adjoint is a mask.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PoolSize {
    size: u32,
    stride: u32,
}

impl PoolSize {
    /// `usize` arguments, as the reference spelled them; the `u32` fields are
    /// an internal detail.
    pub const fn new(size: usize, stride: usize) -> Self {
        assert!(size <= u32::MAX as usize && stride <= u32::MAX as usize);
        Self {
            size: size as u32,
            stride: stride as u32,
        }
    }

    /// Whether adjacent windows do not overlap.
    pub const fn is_non_overlapping(self) -> bool {
        self.stride >= self.size
    }
}

impl From<usize> for PoolSize {
    fn from(size: usize) -> Self {
        Self::new(size, size)
    }
}

impl From<u32> for PoolSize {
    fn from(size: u32) -> Self {
        Self::new(size as usize, size as usize)
    }
}

impl From<(usize, usize)> for PoolSize {
    fn from((size, stride): (usize, usize)) -> Self {
        Self::new(size, stride)
    }
}

impl From<[usize; 2]> for PoolSize {
    fn from([size, stride]: [usize; 2]) -> Self {
        Self::new(size, stride)
    }
}

/// Window the trailing `pools.len()` axes and reduce every window axis.
///
/// Each window axis is folded separately rather than flattened into one: max
/// and min are associative, so successive folds are the same value, and a
/// flatten of two window axes would need a contiguity proof the windowed view
/// cannot supply.
fn pool_defn(
    t: &mut GraphTape<'_>,
    x: Val,
    specs: &[SlidingWindow],
    reduce: PoolReduce,
) -> Result<Val> {
    let dtype = t.dtype_of(x);
    let windowed = t.add(Logical::Window {
        specs: specs.iter().copied().collect(),
        x,
    })?;
    let rank = t.rank_of(windowed);
    let count = specs.len();

    let combine = match reduce {
        PoolReduce::Max => BinOp::Max,
        PoolReduce::Min => BinOp::Min,
        PoolReduce::Mean => BinOp::Add,
    };
    let acc = match reduce {
        PoolReduce::Mean => accum_dtype(dtype),
        _ => dtype,
    };

    // The window axes are the last `count`; folding from the back keeps every
    // remaining axis index stable.
    let mut v = windowed;
    for i in 0..count {
        let axis = (rank - 1 - i) as u32;
        v = t.fold_binop(combine, axis, acc, v)?;
    }
    if matches!(reduce, PoolReduce::Mean) {
        let n: u64 = specs.iter().map(|w| w.window as u64).product();
        v = t.cast(dtype, v)?;
        v = t.mul_scalar(v, 1.0 / n.max(1) as f32)?;
    }
    Ok(v)
}

fn pool_with(x: &Tensor, pools: &[PoolSize], reduce: PoolReduce) -> Result<Tensor> {
    let rank = x.graph.facts(x.id).rank();
    if pools.is_empty() || pools.len() > rank {
        return Err(Error::Shape(format!(
            "pooling {} axes of a rank-{rank} value",
            pools.len()
        )));
    }
    let first = (rank - pools.len()) as u32;
    let specs: SmallVec<[SlidingWindow; 3]> = pools
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if p.size == 0 || p.stride == 0 {
                return Err(Error::Shape("a pool window and stride must be nonzero".into()));
            }
            Ok(SlidingWindow::new(first + i as u32, p.size, p.stride))
        })
        .collect::<Result<_>>()?;

    let xid = x.id;
    let attrs = MacroAttr::Pool {
        windows: specs.clone(),
        reduce,
    };
    macro_op(&x.graph, MacroOp::Pool, attrs, &[xid], move |t| {
        pool_defn(t, xid, &specs, reduce)
    })
}

/// The generic form: window the trailing axes and reduce with `with`.
pub fn pool(x: &Tensor, pools: &[PoolSize], with: PoolReduce) -> Result<Tensor> {
    pool_with(x, pools, with)
}

/// Maximum pooling over the trailing axes.
pub fn pool_max(x: &Tensor, pools: &[PoolSize]) -> Result<Tensor> {
    pool_with(x, pools, PoolReduce::Max)
}

/// Minimum pooling over the trailing axes.
pub fn pool_min(x: &Tensor, pools: &[PoolSize]) -> Result<Tensor> {
    pool_with(x, pools, PoolReduce::Min)
}

/// Average pooling. The reference has none; it is the same node with
/// an `Add` carrier and a scale, which is the point of the reduction being an
/// attribute.
pub fn pool_avg(x: &Tensor, pools: &[PoolSize]) -> Result<Tensor> {
    pool_with(x, pools, PoolReduce::Mean)
}
