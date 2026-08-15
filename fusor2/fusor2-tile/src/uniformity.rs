//! Uniformity analysis. A bottom-up classification of every [`TileExpr`],
//! then a statement walk carrying a predicate-uniformity stack. A `Barrier`
//! (or `StorageBarrier`) under a non-uniform predicate is
//! [`LowerError::NonUniformBarrier`][fusor2_ir::ir::level2::LowerError::NonUniformBarrier].
//!
//! The classification is conservative in the direction that fails lowering
//! rather than racing: anything read from memory, any lane-indexed builtin,
//! any subgroup collective and any cooperative fragment is `NonUniform`.

use fusor2_ir::Result;
use fusor2_ir::error::Error;
use fusor2_ir::ir::level2::{
    Accumulator, Builtin, KernelIr, Local, LowerError, ReduceKind, Stmt, TileExpr, TileExprKind,
};
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::liveness::for_each_child;

/// Whether a value is provably identical across every invocation of the
/// group. Unknown is treated as `NonUniform`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Uniformity {
    Uniform,
    NonUniform,
}

impl Uniformity {
    /// `Uniform` only when both are.
    pub const fn meet(self, other: Self) -> Self {
        match (self, other) {
            (Self::Uniform, Self::Uniform) => Self::Uniform,
            _ => Self::NonUniform,
        }
    }
}

type LocalKey = usize;

fn local_key(local: &Local) -> LocalKey {
    Arc::as_ptr(local) as *const () as usize
}

/// Per-local classification plus a per-node memo keyed on `structural_hash`.
#[derive(Default)]
struct Ctx {
    locals: FxHashMap<LocalKey, Uniformity>,
    memo: FxHashMap<u64, Uniformity>,
}

impl Ctx {
    fn local(&self, local: &Local) -> Uniformity {
        // A local nothing ever assigns is a malformed kernel; treat it as
        // non-uniform so it can never license a barrier.
        self.locals
            .get(&local_key(local))
            .copied()
            .unwrap_or(Uniformity::NonUniform)
    }

    fn classify(&mut self, expr: &TileExpr) -> Uniformity {
        if let Some(cached) = self.memo.get(&expr.structural_hash()) {
            return *cached;
        }
        let result = self.classify_uncached(expr);
        self.memo.insert(expr.structural_hash(), result);
        result
    }

    fn classify_uncached(&mut self, expr: &TileExpr) -> Uniformity {
        use TileExprKind as K;
        match expr.kind() {
            K::Literal(_) => Uniformity::Uniform,
            K::Builtin(builtin) => match builtin {
                // Uniform over the workgroup.
                Builtin::ProgramId(_)
                | Builtin::NumWorkgroups(_)
                | Builtin::SubgroupSize
                | Builtin::NumSubgroups => {
                    Uniformity::Uniform
                }
                // `SubgroupId` is uniform only *within* a subgroup, so at
                // workgroup scope it is not.
                Builtin::Lane | Builtin::SubgroupLane | Builtin::SubgroupId => {
                    Uniformity::NonUniform
                }
            },
            K::LoadLocal(local) => self.local(local),
            K::Load { .. }
            | K::LoadTile { .. }
            | K::CoopLoad { .. }
            | K::CoopMma { .. } => Uniformity::NonUniform,
            K::Reduce { kind, value, .. } => match kind.as_ref() {
                ReduceKind::Subgroup => Uniformity::NonUniform,
                ReduceKind::Workgroup { .. } | ReduceKind::Loop { .. } => self.classify(value),
            },
            _ => {
                let mut children: Vec<TileExpr> = Vec::new();
                for_each_child(expr.kind(), &mut |child| children.push(child.clone()));
                let mut result = Uniformity::Uniform;
                for child in &children {
                    result = result.meet(self.classify(child));
                }
                result
            }
        }
    }
}

/// A `Barrier` may not appear under an `If` whose predicate is non-uniform
/// over the group.
pub fn verify_uniformity(ir: &KernelIr) -> Result<()> {
    let mut ctx = Ctx::default();
    classify_locals(&ir.body, &mut ctx);
    let mut path: Vec<u32> = Vec::new();
    walk(&ir.body, Uniformity::Uniform, &mut ctx, &mut path)
}

/// Every value assigned to a local, plus the locals that are uniform by
/// construction (loop counters).
fn collect_assignments(
    body: &[Stmt],
    assignments: &mut Vec<(LocalKey, TileExpr)>,
    counters: &mut Vec<LocalKey>,
) {
    for stmt in body {
        match stmt {
            Stmt::StoreLocal { dst, value } => assignments.push((local_key(dst), value.clone())),
            Stmt::If { accept, reject, .. } => {
                collect_assignments(accept, assignments, counters);
                collect_assignments(reject, assignments, counters);
            }
            Stmt::Loop {
                index,
                accumulators,
                body,
                ..
            } => {
                if let Some(index) = index {
                    counters.push(local_key(index));
                }
                for Accumulator {
                    local,
                    init,
                    update,
                } in accumulators
                {
                    assignments.push((local_key(local), init.clone()));
                    assignments.push((local_key(local), update.clone()));
                }
                collect_assignments(body, assignments, counters);
            }
            _ => {}
        }
    }
}

/// `ReduceKind::Loop` carries its own counter local, which is uniform for the
/// same reason a counted loop's index is.
fn collect_reduce_counters(body: &[Stmt], counters: &mut Vec<LocalKey>) {
    let mut seen = rustc_hash::FxHashSet::default();
    crate::verify_l2::for_each_root_expr(body, &mut |expr| {
        crate::verify_l2::visit_unique(expr, &mut seen, &mut |node| {
            if let TileExprKind::Reduce { kind, .. } = node.kind()
                && let ReduceKind::Loop { index, .. } = kind.as_ref()
            {
                counters.push(local_key(index));
            }
        });
    });
    // The N-ary form carries its kind on the statement, not inside an
    // expression, so the expression walk above cannot see its counter.
    crate::verify_l2::for_each_stmt(body, &mut |stmt| {
        if let Stmt::Reduce { kind, .. } = stmt
            && let ReduceKind::Loop { index, .. } = kind.as_ref()
        {
            counters.push(local_key(index));
        }
    });
}

/// Fixpoint: start every assigned local `Uniform` and downgrade it the moment
/// any assignment is non-uniform. Monotone, so it terminates; a loop-carried
/// local settles after at most one extra pass per dependency edge.
fn classify_locals(body: &[Stmt], ctx: &mut Ctx) {
    let mut assignments = Vec::new();
    let mut counters = Vec::new();
    collect_assignments(body, &mut assignments, &mut counters);
    collect_reduce_counters(body, &mut counters);
    for (key, _) in &assignments {
        ctx.locals.entry(*key).or_insert(Uniformity::Uniform);
    }
    for key in &counters {
        ctx.locals.insert(*key, Uniformity::Uniform);
    }
    loop {
        let mut changed = false;
        ctx.memo.clear();
        for (key, value) in &assignments {
            if ctx.locals.get(key) == Some(&Uniformity::NonUniform) {
                continue;
            }
            if ctx.classify(value) == Uniformity::NonUniform {
                ctx.locals.insert(*key, Uniformity::NonUniform);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    ctx.memo.clear();
}

fn walk(body: &[Stmt], enclosing: Uniformity, ctx: &mut Ctx, path: &mut Vec<u32>) -> Result<()> {
    for (index, stmt) in body.iter().enumerate() {
        path.push(index as u32);
        let result = walk_stmt(stmt, enclosing, ctx, path);
        path.pop();
        result?;
    }
    Ok(())
}

fn walk_stmt(
    stmt: &Stmt,
    enclosing: Uniformity,
    ctx: &mut Ctx,
    path: &mut Vec<u32>,
) -> Result<()> {
    match stmt {
        Stmt::Barrier | Stmt::StorageBarrier => {
            if enclosing == Uniformity::NonUniform {
                return Err(Error::Lower(LowerError::NonUniformBarrier(format!(
                    "barrier at {} is under a non-uniform predicate",
                    render_path(path)
                ))));
            }
            Ok(())
        }
        // A workgroup or loop reduction lowers to a staged tree with a barrier
        // between every level. Those barriers are emitted, not written, so they
        // are checked here at the statement that produces them.
        Stmt::Reduce { kind, .. } => {
            if matches!(
                kind.as_ref(),
                ReduceKind::Workgroup { .. } | ReduceKind::Loop { .. }
            ) && enclosing == Uniformity::NonUniform
            {
                return Err(Error::Lower(LowerError::NonUniformBarrier(format!(
                    "the staged reduction at {} is under a non-uniform predicate",
                    render_path(path)
                ))));
            }
            Ok(())
        }
        Stmt::If {
            condition,
            accept,
            reject,
        } => {
            let inner = enclosing.meet(ctx.classify(condition));
            walk(accept, inner, ctx, path)?;
            walk(reject, inner, ctx, path)
        }
        Stmt::Loop { count, body, .. } => {
            // A loop whose trip count differs per lane makes its body
            // divergent for barrier purposes.
            let inner = match count {
                Some(count) => enclosing.meet(ctx.classify(count)),
                None => enclosing,
            };
            walk(body, inner, ctx, path)
        }
        _ => Ok(()),
    }
}

fn render_path(path: &[u32]) -> String {
    let mut out = String::new();
    for (index, step) in path.iter().enumerate() {
        if index > 0 {
            out.push('.');
        }
        out.push_str(&step.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::TileBuilder;
    use fusor2_ir::ir::level2::{ScalarElement, WorkgroupAxis};
    use fusor2_ir::scalar::CmpOp;

    #[test]
    fn non_uniform_barrier_rejected() {
        let mut b = TileBuilder::new();
        let lane = b.builtin(Builtin::Lane);
        let sixteen = b.lit_u32(16);
        let condition = b.compare(CmpOp::Lt, lane, sixteen);
        let stmt = b.if_then_else(condition, vec![Stmt::Barrier], Vec::new());
        b.push(stmt);
        let ir = b.finish([1, 1, 1], 64, "divergent");
        match verify_uniformity(&ir) {
            Err(Error::Lower(LowerError::NonUniformBarrier(_))) => {}
            other => panic!("expected NonUniformBarrier, got {other:?}"),
        }
    }

    #[test]
    fn uniform_barrier_accepted() {
        let mut b = TileBuilder::new();
        let group = b.builtin(Builtin::ProgramId(WorkgroupAxis::X));
        let four = b.lit_u32(4);
        let condition = b.compare(CmpOp::Lt, group, four);
        let stmt = b.if_then_else(condition, vec![Stmt::Barrier], Vec::new());
        b.push(stmt);
        let ir = b.finish([1, 1, 1], 64, "uniform");
        verify_uniformity(&ir).unwrap();
    }

    #[test]
    fn a_local_holding_a_lane_value_poisons_the_predicate() {
        let mut b = TileBuilder::new();
        let local = b.alloc_local(ScalarElement::U32.element());
        let lane = b.builtin(Builtin::Lane);
        let store = b.store_local(local.clone(), lane);
        let loaded = b.load_local(local);
        let four = b.lit_u32(4);
        let condition = b.compare(CmpOp::Lt, loaded, four);
        let guarded = b.if_then_else(condition, vec![Stmt::Barrier], Vec::new());
        b.push(store);
        b.push(guarded);
        let ir = b.finish([1, 1, 1], 64, "poisoned");
        assert!(verify_uniformity(&ir).is_err());
    }

    #[test]
    fn a_local_holding_a_uniform_value_does_not() {
        let mut b = TileBuilder::new();
        let local = b.alloc_local(ScalarElement::U32.element());
        let size = b.builtin(Builtin::SubgroupSize);
        let store = b.store_local(local.clone(), size);
        let loaded = b.load_local(local);
        let four = b.lit_u32(4);
        let condition = b.compare(CmpOp::Lt, loaded, four);
        let guarded = b.if_then_else(condition, vec![Stmt::Barrier], Vec::new());
        b.push(store);
        b.push(guarded);
        let ir = b.finish([1, 1, 1], 64, "clean");
        verify_uniformity(&ir).unwrap();
    }
    /// A staged reduction emits a barrier between every tree level, so it may
    /// not sit under a divergent predicate any more than a written one may.
    /// The barriers are produced by the emitter, not written in the body, which
    /// is why the check is on the statement that produces them.
    #[test]
    fn a_staged_reduction_under_a_divergent_if_is_rejected() {
        use fusor2_ir::ir::level2::{MergeBody, ReduceKind};
        let mut b = TileBuilder::new();
        let f32e = ScalarElement::F32.element();
        let scratch = b.alloc_tile(
            f32e,
            fusor2_ir::ir::level2::TileLayout::contiguous(
                fusor2_ir::ir::level2::MemoryLevel::Workgroup,
                &[64],
            ),
        );
        let lhs = b.alloc_local(f32e);
        let rhs = b.alloc_local(f32e);
        let a = b.load_local(lhs.clone());
        let c = b.load_local(rhs.clone());
        let merged = b.binary(
            fusor2_ir::scalar::BinOp::Add,
            a,
            c,
            fusor2_ir::dtype::NumericContract::RELAXED,
        );
        let out = b.alloc_local(f32e);
        let value = b.lit_f32(1.0);
        let reduce = Stmt::Reduce {
            kind: Box::new(ReduceKind::Workgroup {
                scratch: scratch.clone(),
                group_size: 64,
            }),
            values: smallvec::smallvec![value],
            merge: Box::new(MergeBody {
                lhs: smallvec::smallvec![lhs],
                rhs: smallvec::smallvec![rhs],
                body: smallvec::smallvec![merged],
            }),
            fast: None,
            outs: smallvec::smallvec![out],
            scratch: smallvec::smallvec![scratch],
        };
        let lane = b.builtin(Builtin::Lane);
        let sixteen = b.lit_u32(16);
        let divergent = b.compare(CmpOp::Lt, lane, sixteen);
        let guarded = b.if_then_else(divergent, vec![reduce.clone()], Vec::new());
        b.push(guarded);
        let ir = b.finish([1, 1, 1], 64, "divergent-reduce");
        match verify_uniformity(&ir) {
            Err(Error::Lower(LowerError::NonUniformBarrier(_))) => {}
            other => panic!("expected NonUniformBarrier, got {other:?}"),
        }

        // The same node at the top level is fine.
        let mut b = TileBuilder::new();
        b.push(reduce);
        let ir = b.finish([1, 1, 1], 64, "plain-reduce");
        verify_uniformity(&ir).unwrap();
    }
}
