//! Uniformity analysis. A bottom-up classification of every [`TileExpr`],
//! then a statement walk carrying a predicate-uniformity stack. A `Barrier`
//! (or `StorageBarrier`) under a non-uniform predicate is
//! [`LowerError::NonUniformBarrier`][fusor_ir::ir::kernel::LowerError::NonUniformBarrier].
//!
//! The classification is conservative in the direction that fails lowering
//! rather than racing: mutable or lane-indexed memory reads, lane-indexed
//! builtins, subgroup collectives and cooperative fragments are `NonUniform`.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, BufferAccess, Builtin, KernelIr, Local, LowerError, ReduceKind, Source, Stmt,
    TileExpr, TileExprKind,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

/// Whether a value is provably identical across every invocation of the
/// group. Unknown is treated as `NonUniform`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Uniformity {
    Uniform,
    NonUniform,
}

impl Uniformity {
    /// `Uniform` only when both are.
    pub(crate) const fn meet(self, other: Self) -> Self {
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
    writable_bindings: FxHashSet<u32>,
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
                | Builtin::NumSubgroups => Uniformity::Uniform,
                // `SubgroupId` is uniform only *within* a subgroup, so at
                // workgroup scope it is not.
                Builtin::Lane | Builtin::SubgroupLane | Builtin::SubgroupId => {
                    Uniformity::NonUniform
                }
            },
            K::LoadLocal(local) => self.local(local),
            K::Load {
                src: Source::Storage(view),
                ..
            } if view.buffer.access == BufferAccess::Read
                && !self.writable_bindings.contains(&view.buffer.binding) =>
            {
                self.classify_children(expr)
            }
            K::Load { .. } | K::LoadTile { .. } | K::CoopLoad { .. } | K::CoopMma { .. } => {
                Uniformity::NonUniform
            }
            K::Reduce { kind, value, .. } => match kind.as_ref() {
                ReduceKind::Subgroup => Uniformity::NonUniform,
                ReduceKind::Workgroup { .. } => self.classify(value),
            },
            _ => self.classify_children(expr),
        }
    }

    fn classify_children(&mut self, expr: &TileExpr) -> Uniformity {
        let mut result = Uniformity::Uniform;
        expr.kind()
            .visit_children(&mut |child| result = result.meet(self.classify(child)));
        result
    }
}

/// A `Barrier` may not appear under an `If` whose predicate is non-uniform
/// over the group.
pub(crate) fn verify_uniformity(ir: &KernelIr) -> Result<()> {
    let mut ctx = Ctx {
        writable_bindings: ir
            .buffers
            .iter()
            .filter(|buffer| buffer.access == BufferAccess::ReadWrite)
            .map(|buffer| buffer.binding)
            .collect(),
        ..Ctx::default()
    };
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

/// Fixpoint: start every assigned local `Uniform` and downgrade it the moment
/// any assignment is non-uniform. Monotone, so it terminates; a loop-carried
/// local settles after at most one extra pass per dependency edge.
fn classify_locals(body: &[Stmt], ctx: &mut Ctx) {
    let mut assignments = Vec::new();
    let mut counters = Vec::new();
    collect_assignments(body, &mut assignments, &mut counters);
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

fn walk_stmt(stmt: &Stmt, enclosing: Uniformity, ctx: &mut Ctx, path: &mut Vec<u32>) -> Result<()> {
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
        // A workgroup reduction lowers to a staged tree with a barrier
        // between every level. Those barriers are emitted, not written, so they
        // are checked here at the statement that produces them.
        Stmt::Reduce { kind, .. } => {
            if matches!(kind.as_ref(), ReduceKind::Workgroup { .. })
                && enclosing == Uniformity::NonUniform
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
    use fusor_ir::ir::kernel::{
        Addr, BufferDecl, ElementType, MemoryLevel, ScalarElement, StorageView, TileLayout,
        TileLiteral,
    };

    #[test]
    fn barrier_loops_require_uniform_immutable_counts() {
        let u32_type = ElementType::Scalar(ScalarElement::U32);
        let zero = TileExpr::new(TileExprKind::Literal(TileLiteral::U32(0)), u32_type);
        let mask = TileExpr::new(
            TileExprKind::Literal(TileLiteral::Bool(true)),
            ElementType::Scalar(ScalarElement::Bool),
        );
        for (access, lane_index, writable_alias) in [
            (BufferAccess::Read, false, false),
            (BufferAccess::Read, true, false),
            (BufferAccess::ReadWrite, false, false),
            (BufferAccess::Read, false, true),
        ] {
            let buffer = Arc::new(BufferDecl {
                binding: 0,
                element: u32_type,
                layout: TileLayout::contiguous(MemoryLevel::Storage, &[32]),
                access,
            });
            let mut buffers = vec![buffer.clone()];
            if writable_alias {
                buffers.push(Arc::new(BufferDecl {
                    access: BufferAccess::ReadWrite,
                    ..(*buffer).clone()
                }));
            }
            let index = if lane_index {
                TileExpr::new(TileExprKind::Builtin(Builtin::Lane), u32_type)
            } else {
                zero.clone()
            };
            let count = TileExpr::new(
                TileExprKind::Load {
                    src: Source::Storage(StorageView {
                        buffer: buffer.clone(),
                        offset: 0,
                        layout: buffer.layout.clone(),
                    }),
                    addr: Box::new(Addr::Linear(index)),
                    mask: mask.clone(),
                    fill: zero.clone(),
                },
                u32_type,
            );
            let ir = KernelIr {
                buffers,
                grid: [1, 1, 1],
                block: 32,
                body: vec![Stmt::Loop {
                    count: Some(count),
                    index: None,
                    accumulators: Vec::new(),
                    body: vec![Stmt::Barrier],
                }],
                byte_arena: None,
                name: "uniform_count",
            };
            assert_eq!(
                verify_uniformity(&ir).is_ok(),
                access == BufferAccess::Read && !lane_index && !writable_alias
            );
        }
    }
}
