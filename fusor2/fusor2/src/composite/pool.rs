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
use fusor2_ir::ir::level0::L0;
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
    pub size: u32,
    pub stride: u32,
}

impl PoolSize {
    pub const fn new(size: u32, stride: u32) -> Self {
        Self { size, stride }
    }

    pub const fn is_non_overlapping(self) -> bool {
        self.stride >= self.size
    }
}

impl From<usize> for PoolSize {
    fn from(size: usize) -> Self {
        Self::new(size as u32, size as u32)
    }
}

impl From<u32> for PoolSize {
    fn from(size: u32) -> Self {
        Self::new(size, size)
    }
}

impl From<(usize, usize)> for PoolSize {
    fn from((size, stride): (usize, usize)) -> Self {
        Self::new(size as u32, stride as u32)
    }
}

impl From<[usize; 2]> for PoolSize {
    fn from([size, stride]: [usize; 2]) -> Self {
        Self::new(size as u32, stride as u32)
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
    let windowed = t.add(L0::Window {
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

pub fn pool_max(x: &Tensor, pools: &[PoolSize]) -> Result<Tensor> {
    pool_with(x, pools, PoolReduce::Max)
}

pub fn pool_min(x: &Tensor, pools: &[PoolSize]) -> Result<Tensor> {
    pool_with(x, pools, PoolReduce::Min)
}

/// Average pooling. The reference has none; it is the same node with
/// an `Add` carrier and a scale, which is the point of the reduction being an
/// attribute.
pub fn pool_avg(x: &Tensor, pools: &[PoolSize]) -> Result<Tensor> {
    pool_with(x, pools, PoolReduce::Mean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::session::{Backend, Session};
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::ir::Op;
    use fusor2_ir::ir::level1::L1;
    use fusor2_ir::shape::Dim;

    fn graph() -> Graph {
        Graph::new(&Session::new(Backend::cpu().unwrap()).unwrap())
    }

    fn leaf(g: &Graph, shape: &[u64]) -> Tensor {
        let dims: Vec<Dim> = shape.iter().map(|d| Dim::Const(*d)).collect();
        g.leaf("x", &dims, Dtype::F32).unwrap()
    }

    #[test]
    fn pool_size_conversions_default_the_stride_to_the_window() {
        assert_eq!(PoolSize::from(4usize), PoolSize::new(4, 4));
        assert_eq!(PoolSize::from((4usize, 2usize)), PoolSize::new(4, 2));
        assert_eq!(PoolSize::from([4usize, 2usize]), PoolSize::new(4, 2));
        assert!(PoolSize::from(4usize).is_non_overlapping());
        assert!(!PoolSize::new(4, 2).is_non_overlapping());
    }

    #[test]
    fn max_pooling_a_last_axis_divides_it_by_the_window() {
        let g = graph();
        let x = leaf(&g, &[8, 64, 768]);
        let y = pool_max(&x, &[PoolSize::from(4usize)]).unwrap();
        assert_eq!(
            &g.handle().facts(y.id()).shape[..],
            &[Dim::Const(8), Dim::Const(64), Dim::Const(192)]
        );
    }

    #[test]
    fn two_dimensional_pooling_reduces_both_window_axes() {
        let g = graph();
        let x = leaf(&g, &[1, 3, 8, 8]);
        let y = pool_min(&x, &[PoolSize::from(2usize), PoolSize::from(2usize)]).unwrap();
        assert_eq!(
            &g.handle().facts(y.id()).shape[..],
            &[Dim::Const(1), Dim::Const(3), Dim::Const(4), Dim::Const(4)]
        );
    }

    #[test]
    fn a_pool_class_holds_both_the_sugar_and_a_marked_defn() {
        let g = graph();
        let x = leaf(&g, &[2, 4, 16]);
        let y = pool_max(&x, &[PoolSize::from(4usize)]).unwrap();
        let (n, sugars, defns) = g
            .handle()
            .with_egraph(|eg| {
                let ms = eg.members(eg.class_of(y.id()));
                let s = ms
                    .iter()
                    .filter(|m| matches!(eg.node(**m).op, Op::L1(L1::Ext { .. })))
                    .count();
                let d = ms.iter().filter(|m| eg.is_defn(**m)).count();
                Ok((ms.len(), s, d))
            })
            .unwrap();
        assert!(n >= 2);
        assert_eq!(sugars, 1);
        assert_eq!(defns, 1);
    }

    #[test]
    fn the_sugar_node_carries_the_window_geometry_the_adjoint_reads() {
        let g = graph();
        let x = leaf(&g, &[2, 4, 16]);
        let y = pool_max(&x, &[PoolSize::new(4, 4)]).unwrap();
        let attrs = g
            .handle()
            .with_egraph(|eg| {
                let ms = eg.members(eg.class_of(y.id()));
                Ok(ms.iter().find_map(|m| match &eg.node(*m).op {
                    Op::L1(L1::Ext { attrs, .. }) => Some(*attrs),
                    _ => None,
                }))
            })
            .unwrap()
            .expect("a sugar node");
        match g.handle().attrs_of(attrs).unwrap() {
            MacroAttr::Pool { windows, reduce } => {
                assert_eq!(reduce, PoolReduce::Max);
                assert!(windows[0].is_non_overlapping());
            }
            other => panic!("expected pool attributes, got {other:?}"),
        }
    }

    #[test]
    fn pooling_more_axes_than_the_value_has_is_refused() {
        let g = graph();
        let x = leaf(&g, &[4]);
        assert!(pool_max(&x, &[PoolSize::from(2usize), PoolSize::from(2usize)]).is_err());
        assert!(pool_max(&x, &[]).is_err());
    }
}
