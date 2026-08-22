//! `verify_kernel` — the Kernel verifier.
//!
//! Seven clauses, in order:
//! 1. **Full expression type-check.** Every node's cached `ty` equals what
//!    inference derives, and every structural constraint inference assumes is
//!    checked explicitly.
//! 2. **Every `Load` masked or provably in range.** This discipline is the
//!    *only* thing licensing `create_shader_module_trusted`.
//! 3. **Every `Loop` accumulator declared** and written nowhere outside its
//!    loop body.
//! 4. **`cooperative_store_layout_supported` on every `CoopStore`**, plus
//!    `Addr::Rc2`. The caller's documented recovery is the per-lane store
//!    fallback: the emitters lower `CoopStore` to a masked per-lane `Store`
//!    loop when the predicate fails, so `verify_kernel` only rejects a `CoopStore`
//!    node that survived to Kernel with an unsupported layout.
//! 5. **Uniformity** — the analysis backing "guaranteed uniform" barriers.
//! 6. **`verify_arena`** against the planner's `arena_plan`.
//! 7. **`f16`/`bf16` gated on `caps`**, up front, so an f16 handle on a
//!    non-f16 adapter fails here rather than mis-lowering.

use fusor_ir::Result;
use fusor_ir::device::Caps;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, ArenaPlanner, CoopMatrixRole, CoopSrc, ElementType, KernelIr, Local,
    LowerError, ScalarElement, Source, Stmt, TileExpr, TileExprKind, TileLiteral,
    cooperative_store_layout_supported,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use crate::arena::scalar_of;
use crate::liveness::{for_each_addr_expr, for_each_child};

fn invalid(msg: impl Into<String>) -> Error {
    Error::Lower(LowerError::Validation(msg.into()))
}

/// Verify one kernel body against a device.
pub fn verify_kernel(ir: &KernelIr, caps: &Caps) -> Result<()> {
    check_dtype_caps(ir, caps)?;
    check_types(ir)?;
    check_loads(ir, caps)?;
    check_accumulators(&ir.body)?;
    check_reduce_stmts(&ir.body)?;
    check_coop_stores(&ir.body)?;
    crate::uniformity::verify_uniformity(ir)?;
    let planner = crate::planner::Planner::global();
    let plan = planner.arena_plan(ir, caps)?;
    // The plan may have bought its footprint with barrier insertions, which
    // the emitter is required to apply before emitting; the arena is verified
    // against the body the emitter will actually produce.
    if plan.barriers_inserted.is_empty() {
        planner.verify_arena(ir, &plan)
    } else {
        let emitted = crate::barrier::insert(ir, &plan.barriers_inserted)?;
        crate::verify_arena::verify_placements(
            &crate::liveness::analyze(&emitted),
            &plan.placements,
        )
    }
}

// ---------------------------------------------------------------------------
// (7) dtype capability gate
// ---------------------------------------------------------------------------

fn check_dtype_caps(ir: &KernelIr, caps: &Caps) -> Result<()> {
    let mut bad: Option<ScalarElement> = None;
    for_each_element(ir, &mut |element| {
        if bad.is_some() {
            return;
        }
        let Some(scalar) = scalar_of(element) else {
            return;
        };
        match scalar {
            ScalarElement::F16 if !caps.f16 => bad = Some(scalar),
            ScalarElement::BF16 if !caps.bf16 => bad = Some(scalar),
            _ => {}
        }
    });
    match bad {
        None => Ok(()),
        Some(scalar) => Err(Error::Legality(format!(
            "kernel {} uses {scalar:?} but the device does not support it",
            ir.name
        ))),
    }
}

/// Every element type appearing anywhere in the kernel.
fn for_each_element(ir: &KernelIr, f: &mut dyn FnMut(ElementType)) {
    for buffer in &ir.buffers {
        f(buffer.element);
    }
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    for_each_root_expr(&ir.body, &mut |expr| {
        visit_unique(expr, &mut seen, &mut |e| {
            f(e.element());
            match e.kind() {
                TileExprKind::LoadTile { tile, .. } => f(tile.element),
                TileExprKind::LoadLocal(local) => f(local.element),
                TileExprKind::Reduce { kind, .. } => match kind.as_ref() {
                    fusor_ir::ir::kernel::ReduceKind::Subgroup => {}
                    fusor_ir::ir::kernel::ReduceKind::Workgroup { scratch, .. }
                    | fusor_ir::ir::kernel::ReduceKind::Loop { scratch, .. } => f(scratch.element),
                },
                TileExprKind::CoopLoad { src, .. } => {
                    if let CoopSrc::TileRegion { tile, .. } = src.as_ref() {
                        f(tile.element);
                    }
                }
                _ => {}
            }
        });
    });
    for_each_stmt(&ir.body, &mut |stmt| match stmt {
        Stmt::StoreTile { dst, .. } | Stmt::FillTile { dst, .. } => f(dst.element),
        Stmt::CoopStoreTile { tile, .. } => f(tile.element),
        Stmt::StoreLocal { dst, .. } => f(dst.element),
        Stmt::Loop {
            index: Some(index), ..
        } => f(index.element),
        Stmt::Reduce {
            merge,
            outs,
            scratch,
            ..
        } => {
            for tile in scratch {
                f(tile.element);
            }
            for local in merge.lhs.iter().chain(&merge.rhs).chain(outs) {
                f(local.element);
            }
        }
        _ => {}
    });
}

// ---------------------------------------------------------------------------
// (1) type inference and check
// ---------------------------------------------------------------------------

/// Derive the element type of one node from its children's cached types. The
/// builder uses this to type a node; `verify_kernel` re-derives it and compares.
pub(crate) fn infer_kind(kind: &TileExprKind) -> Result<ElementType> {
    use TileExprKind as K;
    Ok(match kind {
        K::Literal(lit) => ElementType::Scalar(literal_scalar(*lit)),
        K::Builtin(_) => ElementType::Scalar(ScalarElement::U32),
        K::LoadLocal(local) => local.element,
        K::Load { src, .. } => match src {
            Source::Storage(view) => view.buffer.element,
            Source::Quantized(_) => ElementType::Scalar(ScalarElement::F32),
        },
        K::LoadTile { tile, .. } => tile.element,
        K::Unary { value, .. } => value.element(),
        K::Binary { left, right, .. } => {
            if left.element() != right.element() {
                return Err(invalid(format!(
                    "binary operands disagree: {:?} vs {:?}",
                    left.element(),
                    right.element()
                )));
            }
            left.element()
        }
        K::Compare { left, right, .. } => {
            if left.element() != right.element() {
                return Err(invalid(format!(
                    "compare operands disagree: {:?} vs {:?}",
                    left.element(),
                    right.element()
                )));
            }
            match left.element() {
                ElementType::Vector { lanes, .. } => ElementType::Vector {
                    scalar: ScalarElement::Bool,
                    lanes,
                },
                _ => ElementType::Scalar(ScalarElement::Bool),
            }
        }
        K::Round { value, .. } => value.element(),
        K::Cast { to, .. } | K::Bitcast { to, .. } => *to,
        K::Select { accept, reject, .. } => {
            if accept.element() != reject.element() {
                return Err(invalid(format!(
                    "select branches disagree: {:?} vs {:?}",
                    accept.element(),
                    reject.element()
                )));
            }
            accept.element()
        }
        K::Vec { scalar, lanes, .. } => ElementType::Vector {
            scalar: *scalar,
            lanes: *lanes,
        },
        K::VecComponent { vector, component } => match vector.element() {
            ElementType::Vector { scalar, lanes } if *component < lanes => {
                ElementType::Scalar(scalar)
            }
            other => {
                return Err(invalid(format!(
                    "vec component {component} out of range for {other:?}"
                )));
            }
        },
        K::Dot { left, right } => match (left.element(), right.element()) {
            (
                ElementType::Vector {
                    scalar: a,
                    lanes: m,
                },
                ElementType::Vector {
                    scalar: b,
                    lanes: n,
                },
            ) if a == b && m == n => ElementType::Scalar(a),
            (a, b) => {
                return Err(invalid(format!(
                    "dot operands are not equal-lane vectors: {a:?} vs {b:?}"
                )));
            }
        },
        K::Reduce { value, .. } => value.element(),
        K::CoopZero {
            role,
            scalar,
            rows,
            cols,
        }
        | K::CoopLoad {
            role,
            scalar,
            rows,
            cols,
            ..
        } => ElementType::CoopMatrix {
            scalar: *scalar,
            role: *role,
            rows: *rows,
            cols: *cols,
        },
        K::CoopMma { a, b, c } => {
            let (
                ElementType::CoopMatrix {
                    scalar: sa,
                    role: ra,
                    rows: ar,
                    cols: ac,
                },
                ElementType::CoopMatrix {
                    scalar: sb,
                    role: rb,
                    rows: br,
                    cols: bc,
                },
                ElementType::CoopMatrix {
                    scalar: sc,
                    role: rc,
                    rows: cr,
                    cols: cc,
                },
            ) = (a.element(), b.element(), c.element())
            else {
                return Err(invalid("coop mma operands must be cooperative fragments"));
            };
            if ra != CoopMatrixRole::A || rb != CoopMatrixRole::B || rc != CoopMatrixRole::C {
                return Err(invalid(format!(
                    "coop mma roles must be (A, B, C), got ({ra:?}, {rb:?}, {rc:?})"
                )));
            }
            if sa != sb {
                return Err(invalid(format!(
                    "coop mma operand scalars disagree: {sa:?} vs {sb:?}"
                )));
            }
            if ar != cr || bc != cc || ac != br {
                return Err(invalid(format!(
                    "coop mma shapes disagree: a {ar}x{ac}, b {br}x{bc}, c {cr}x{cc}"
                )));
            }
            ElementType::CoopMatrix {
                scalar: sc,
                role: CoopMatrixRole::C,
                rows: cr,
                cols: cc,
            }
        }
    })
}

const fn literal_scalar(lit: TileLiteral) -> ScalarElement {
    match lit {
        TileLiteral::F32(_) => ScalarElement::F32,
        TileLiteral::F16(_) => ScalarElement::F16,
        TileLiteral::BF16(_) => ScalarElement::BF16,
        TileLiteral::U32(_) => ScalarElement::U32,
        TileLiteral::I32(_) => ScalarElement::I32,
        TileLiteral::Bool(_) => ScalarElement::Bool,
    }
}

fn check_expr(expr: &TileExpr, seen: &mut FxHashSet<u64>) -> Result<()> {
    if !seen.insert(expr.structural_hash()) {
        return Ok(());
    }
    let mut children: Vec<TileExpr> = Vec::new();
    for_each_child(expr.kind(), &mut |child| children.push(child.clone()));
    for child in &children {
        check_expr(child, seen)?;
    }
    check_node(expr)
}

fn check_node(expr: &TileExpr) -> Result<()> {
    use TileExprKind as K;
    let kind = expr.kind();
    // Structural constraints inference assumes.
    match kind {
        K::Vec {
            scalar,
            lanes,
            parts,
        } => {
            if parts.len() as u32 != *lanes {
                return Err(invalid(format!(
                    "vec declares {lanes} lanes but carries {} parts",
                    parts.len()
                )));
            }
            for part in parts {
                if part.element() != ElementType::Scalar(*scalar) {
                    return Err(invalid(format!(
                        "vec part {:?} is not {scalar:?}",
                        part.element()
                    )));
                }
            }
        }
        K::Select { condition, .. } => {
            let ok = matches!(
                condition.element(),
                ElementType::Scalar(_)
                    | ElementType::Vector {
                        scalar: ScalarElement::Bool,
                        ..
                    }
            );
            if !ok {
                return Err(invalid(format!(
                    "select condition {:?} is neither Bool nor a scalar",
                    condition.element()
                )));
            }
        }
        K::Bitcast { value, to } => {
            if value.element().byte_size() != to.byte_size() {
                return Err(invalid(format!(
                    "bitcast {:?} -> {to:?} changes byte size",
                    value.element()
                )));
            }
        }
        K::Cast { value, to } => {
            let from = value.element();
            let legal = match (from, *to) {
                (ElementType::Scalar(_), ElementType::Scalar(_)) => true,
                (ElementType::Vector { lanes: a, .. }, ElementType::Vector { lanes: b, .. }) => {
                    a == b
                }
                _ => false,
            };
            if !legal {
                return Err(invalid(format!("illegal cast {from:?} -> {to:?}")));
            }
        }
        K::Load {
            mask, fill, src, ..
        } => {
            let element = match src {
                Source::Storage(view) => view.buffer.element,
                Source::Quantized(_) => ElementType::Scalar(ScalarElement::F32),
            };
            if fill.element() != element {
                return Err(invalid(format!(
                    "load fill {:?} does not match source element {element:?}",
                    fill.element()
                )));
            }
            check_mask(mask)?;
        }
        _ => {}
    }
    // The cached type must equal what inference derives.
    let inferred = infer_kind(kind)?;
    if inferred != expr.element() {
        return Err(invalid(format!(
            "cached element {:?} disagrees with inferred {inferred:?}",
            expr.element()
        )));
    }
    Ok(())
}

fn check_mask(mask: &TileExpr) -> Result<()> {
    match mask.element() {
        ElementType::Scalar(ScalarElement::Bool) => Ok(()),
        other => Err(invalid(format!("mask {other:?} is not Bool"))),
    }
}

fn check_types(ir: &KernelIr) -> Result<()> {
    let mut seen = FxHashSet::default();
    let mut error: Option<Error> = None;
    for_each_root_expr(&ir.body, &mut |expr| {
        if error.is_none()
            && let Err(e) = check_expr(expr, &mut seen)
        {
            error = Some(e);
        }
    });
    if let Some(error) = error {
        return Err(error);
    }
    // Statement-level type agreement.
    let mut error: Option<Error> = None;
    for_each_stmt(&ir.body, &mut |stmt| {
        if error.is_some() {
            return;
        }
        let bad = match stmt {
            Stmt::StoreTile { dst, value, .. } if dst.element != value.element() => Some(format!(
                "store into tile of {:?} from {:?}",
                dst.element,
                value.element()
            )),
            Stmt::FillTile { dst, value, .. } if dst.element != value.element() => Some(format!(
                "fill tile of {:?} from {:?}",
                dst.element,
                value.element()
            )),
            Stmt::StoreLocal { dst, value } if dst.element != value.element() => Some(format!(
                "store into local of {:?} from {:?}",
                dst.element,
                value.element()
            )),
            Stmt::Store { dst, value, .. } | Stmt::AtomicAdd { dst, value, .. }
                if dst.buffer.element != value.element() =>
            {
                Some(format!(
                    "store into buffer of {:?} from {:?}",
                    dst.buffer.element,
                    value.element()
                ))
            }
            _ => None,
        };
        if let Some(bad) = bad {
            error = Some(invalid(bad));
        }
    });
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// (2) load masking
// ---------------------------------------------------------------------------

/// Bounds available to the range prover: the launch geometry, the device's
/// subgroup width range, and the loop index locals in scope with their
/// literal trip-count bounds.
struct BoundEnv {
    grid: [u32; 3],
    block: u32,
    /// `(min, max)` subgroup width, when the device reports one. `None`
    /// leaves every subgroup builtin unbounded — a device that cannot say
    /// how wide its subgroups are cannot prove anything indexed by them.
    subgroups: Option<(u32, u32)>,
    locals: FxHashMap<usize, u64>,
}

/// Every `Load` is masked or provably in range.
///
/// The proof is contextual: `ProgramId(axis) < grid[axis]`, `Lane < block`
/// (`Builtin::Lane` lowers to `LocalInvocationIndex`, numbered within the
/// workgroup), and a `Loop`'s index local is `< count` while walking the
/// loop's accumulator updates and body. That is what lets an exactly-tiled
/// serial reduction carry a constant-true mask — the form the emitters'
/// straight-line load paths and the aligned-window sharing algebra require —
/// instead of a per-element bound check the shape has already discharged.
pub(crate) fn check_loads(ir: &KernelIr, caps: &Caps) -> Result<()> {
    let mut env = BoundEnv {
        grid: ir.grid,
        block: ir.block,
        subgroups: caps.subgroups.map(|s| (s.min, s.max)),
        locals: FxHashMap::default(),
    };
    let mut seen = FxHashSet::default();
    check_loads_in(&ir.body, &mut env, &mut seen)
}

fn check_loads_in(body: &[Stmt], env: &mut BoundEnv, seen: &mut FxHashSet<u64>) -> Result<()> {
    for stmt in body {
        match stmt {
            Stmt::Loop {
                count,
                index,
                accumulators,
                body: inner,
            } => {
                // `count` and the accumulator inits evaluate before the
                // loop, outside the index's scope.
                if let Some(count) = count {
                    check_expr_loads(count, env, seen)?;
                }
                for Accumulator { init, .. } in accumulators {
                    check_expr_loads(init, env, seen)?;
                }
                let bound = count
                    .as_ref()
                    .and_then(|c| max_value(c, env))
                    .map(|c| c.saturating_sub(1));
                let prev = match (index.as_ref(), bound) {
                    (Some(local), Some(bound)) => {
                        Some((local_key(local), env.locals.insert(local_key(local), bound)))
                    }
                    _ => None,
                };
                let mut result = Ok(());
                for Accumulator { update, .. } in accumulators {
                    if result.is_ok() {
                        result = check_expr_loads(update, env, seen);
                    }
                }
                if result.is_ok() {
                    result = check_loads_in(inner, env, seen);
                }
                if let Some((key, prev)) = prev {
                    match prev {
                        Some(prev) => env.locals.insert(key, prev),
                        None => env.locals.remove(&key),
                    };
                }
                result?;
            }
            Stmt::If {
                condition,
                accept,
                reject,
            } => {
                check_expr_loads(condition, env, seen)?;
                check_loads_in(accept, env, seen)?;
                check_loads_in(reject, env, seen)?;
            }
            other => {
                let mut result = Ok(());
                stmt_root_exprs(other, &mut |expr| {
                    if result.is_ok() {
                        result = check_expr_loads(expr, env, seen);
                    }
                });
                result?;
            }
        }
    }
    Ok(())
}

fn check_expr_loads(expr: &TileExpr, env: &BoundEnv, seen: &mut FxHashSet<u64>) -> Result<()> {
    let mut error: Option<Error> = None;
    visit_unique(expr, seen, &mut |node| {
        if error.is_some() {
            return;
        }
        let TileExprKind::Load {
            src, addr, mask, ..
        } = node.kind()
        else {
            return;
        };
        if !mask.is_constant_true() {
            return;
        }
        if load_in_range(src, addr, env) {
            return;
        }
        error = Some(Error::Lower(LowerError::UnmaskedLoad(format!(
            "load with a constant-true mask is not provably in range: {:?}",
            addr
        ))));
    });
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn load_in_range(src: &Source, addr: &Addr, env: &BoundEnv) -> bool {
    let layout = match src {
        Source::Storage(view) => &view.layout,
        Source::Quantized(view) => &view.data.layout,
    };
    match addr {
        Addr::Linear(index) => {
            max_value(index, env).is_some_and(|max| max < layout.element_count())
        }
        Addr::Rc2 { row, col } => {
            if layout.extents.len() != 2 {
                return false;
            }
            max_value(row, env).is_some_and(|max| max < u64::from(layout.extents[0]))
                && max_value(col, env).is_some_and(|max| max < u64::from(layout.extents[1]))
        }
    }
}

/// A non-negative integer literal, whichever integer type it carries.
///
/// The `I32` guard mirrors [`max_value`]'s own: a negative literal is not a
/// bound on an unsigned index, so it stays undecidable rather than wrapping.
fn literal_u64(expr: &TileExpr) -> Option<u64> {
    match expr.kind() {
        TileExprKind::Literal(TileLiteral::U32(v)) => Some(u64::from(*v)),
        TileExprKind::Literal(TileLiteral::I32(v)) if *v >= 0 => Some(*v as u64),
        _ => None,
    }
}

/// The maximum value an index expression can take, when it is decidable. A
/// literal, or an affine/monotone composition of literals; anything reading a
/// builtin or memory is unbounded and needs a real mask.
///
/// The bit operators are read as their unsigned arithmetic twins — `x >> k` is
/// `x / 2^k`, `x << k` is `x * 2^k`, `x & m` is bounded like `x % (m + 1)` when
/// `m` is `2^k - 1` — because otherwise the verifier decides `lane % 32` and
/// refuses `lane & 31`, which are the same function of `lane`. That asymmetry
/// is not neutral: it forbids a lowering from spelling an address with the
/// natural power-of-two wrap, the very form `emit::expr::mod_literal_u32`
/// rewrites the remainder into one layer down.
fn max_value(expr: &TileExpr, env: &BoundEnv) -> Option<u64> {
    use fusor_ir::ir::kernel::{Builtin, WorkgroupAxis};
    use fusor_ir::scalar::BinOp;
    match expr.kind() {
        TileExprKind::Literal(TileLiteral::U32(v)) => Some(u64::from(*v)),
        TileExprKind::Literal(TileLiteral::I32(v)) if *v >= 0 => Some(*v as u64),
        TileExprKind::Cast { value, .. } => max_value(value, env),
        TileExprKind::Select { accept, reject, .. } => {
            Some(max_value(accept, env)?.max(max_value(reject, env)?))
        }
        // `Lane` lowers to `LocalInvocationIndex`, numbered within the
        // workgroup, so it is bounded by the block size on every backend.
        TileExprKind::Builtin(Builtin::Lane) => Some(u64::from(env.block.max(1)) - 1),
        // `subgroup_invocation_id < subgroup_size`, and the widest subgroup
        // the device reports bounds that. The block is *not* a bound here: a
        // workgroup narrower than the subgroup width still numbers its lanes
        // within the hardware subgroup.
        TileExprKind::Builtin(Builtin::SubgroupLane) => {
            env.subgroups.map(|(_, max)| u64::from(max.max(1)) - 1)
        }
        TileExprKind::Builtin(Builtin::SubgroupSize) => {
            env.subgroups.map(|(_, max)| u64::from(max.max(1)))
        }
        // A workgroup of `block` lanes packs at most `ceil(block / min_width)`
        // subgroups, whichever width in the range the device picks.
        TileExprKind::Builtin(Builtin::SubgroupId) => env
            .subgroups
            .map(|(min, _)| u64::from(env.block.max(1).div_ceil(min.max(1))) - 1),
        TileExprKind::Builtin(Builtin::NumSubgroups) => env
            .subgroups
            .map(|(min, _)| u64::from(env.block.max(1).div_ceil(min.max(1)))),
        TileExprKind::Builtin(Builtin::ProgramId(axis)) => {
            let extent = match axis {
                WorkgroupAxis::X => env.grid[0],
                WorkgroupAxis::Y => env.grid[1],
                WorkgroupAxis::Z => env.grid[2],
            };
            Some(u64::from(extent.max(1)) - 1)
        }
        TileExprKind::Builtin(Builtin::NumWorkgroups(axis)) => {
            let extent = match axis {
                WorkgroupAxis::X => env.grid[0],
                WorkgroupAxis::Y => env.grid[1],
                WorkgroupAxis::Z => env.grid[2],
            };
            Some(u64::from(extent.max(1)))
        }
        // A loop index local in scope, bounded by its literal trip count.
        TileExprKind::LoadLocal(local) => env.locals.get(&local_key(local)).copied(),
        TileExprKind::Binary {
            op, left, right, ..
        } => match op {
            BinOp::Add => max_value(left, env)?.checked_add(max_value(right, env)?),
            BinOp::Mul => max_value(left, env)?.checked_mul(max_value(right, env)?),
            // Integer subtraction only lowers the bound.
            BinOp::Sub => max_value(left, env),
            // Division by a positive literal divides the bound; a dynamic
            // divisor still only lowers it.
            BinOp::Div => match literal_u64(right) {
                Some(d) if d > 0 => Some(max_value(left, env)? / d),
                _ => max_value(left, env),
            },
            BinOp::Rem => Some(max_value(right, env)?.saturating_sub(1)),
            // `x >> k == x / 2^k` at a literal count. A dynamic count still
            // only lowers the bound, which is the arm this replaces.
            BinOp::Shr => match literal_u64(right) {
                Some(k) if k < 64 => Some(max_value(left, env)? >> k),
                Some(_) => Some(0),
                None => max_value(left, env),
            },
            // `x << k == x * 2^k`, checked exactly the way `Mul` is.
            BinOp::Shl => {
                let k = literal_u64(right)?;
                if k >= 64 {
                    return None;
                }
                max_value(left, env)?.checked_mul(1u64 << k)
            }
            // `x & m <= x` and `x & m <= m` for unsigned, so *one* decidable
            // side bounds the pair: `& 0xFF` is bounded however unbounded its
            // left operand is. Stating it as a `min` also covers a mask that
            // is not `2^k - 1`.
            BinOp::BitAnd => match (max_value(left, env), max_value(right, env)) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
            // Neither `|` nor `^` sets a bit above the highest bit either side
            // can set, so the bound is that position's all-ones mask. Note
            // `a | b <= A | B` is false in general (`A = B = 2` admits
            // `1 | 2 == 3`), which is why this rounds up to the mask.
            BinOp::BitOr | BinOp::BitXor => {
                let m = max_value(left, env)?.max(max_value(right, env)?);
                Some(if m == 0 {
                    0
                } else {
                    u64::MAX >> m.leading_zeros()
                })
            }
            BinOp::Min => match (max_value(left, env), max_value(right, env)) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
            BinOp::Max => Some(max_value(left, env)?.max(max_value(right, env)?)),
            _ => None,
        },
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// (3) loop accumulators
// ---------------------------------------------------------------------------

/// Every `Loop` accumulator's `init`/`update` type matches its local, the
/// local is the accumulator of exactly one loop, and nothing outside that
/// loop's body writes it.
///
/// [`KernelIr`] carries no separate local-declaration list, so ownership is
/// established by the accumulator that introduces each local.
pub(crate) fn check_accumulators(body: &[Stmt]) -> Result<()> {
    let mut owners: FxHashMap<usize, ()> = FxHashMap::default();
    collect_accumulator_owners(body, &mut owners)?;
    let mut scope: Vec<usize> = Vec::new();
    check_accumulator_writes(body, &owners, &mut scope)
}

fn local_key(local: &Local) -> usize {
    Arc::as_ptr(local) as *const () as usize
}

// ---------------------------------------------------------------------------
// (8) the N-ary reduction
// ---------------------------------------------------------------------------

/// Every `Stmt::Reduce` is arity-consistent, element-consistent across lanes,
/// carries one scratch tile per lane where its kind needs scratch, and has a
/// `merge` reading nothing but its own formals.
///
/// The arity clause is what makes the `accs[0]` bug unrepresentable: there is no
/// single `TileReduceOp` to resolve for the whole fold, and a node whose
/// `values`, `merge.body`, `merge.lhs`, `merge.rhs` and `outs` disagree in
/// length is rejected rather than truncated.
pub(crate) fn check_reduce_stmts(body: &[Stmt]) -> Result<()> {
    let mut error: Option<Error> = None;
    for_each_stmt(body, &mut |stmt| {
        if error.is_some() {
            return;
        }
        let Stmt::Reduce {
            kind,
            values,
            merge,
            fast,
            outs,
            scratch,
        } = stmt
        else {
            return;
        };
        if let Err(e) = check_one_reduce(kind, values, merge, *fast, outs, scratch) {
            error = Some(e);
        }
    });
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn check_one_reduce(
    kind: &fusor_ir::ir::kernel::ReduceKind,
    values: &[TileExpr],
    merge: &fusor_ir::ir::kernel::MergeBody,
    fast: Option<fusor_ir::ir::kernel::TileReduceOp>,
    outs: &[Local],
    scratch: &[fusor_ir::ir::kernel::Tile],
) -> Result<()> {
    use fusor_ir::ir::kernel::ReduceKind;
    let n = values.len();
    if n == 0 {
        return Err(invalid("a reduction with no accumulator lanes"));
    }
    if !merge.is_arity_consistent() || merge.lanes() != n || outs.len() != n {
        return Err(invalid(format!(
            "reduction lane counts disagree: {n} values, {} merge lanes, \
             {} lhs, {} rhs, {} outs",
            merge.lanes(),
            merge.lhs.len(),
            merge.rhs.len(),
            outs.len()
        )));
    }
    // Lane `i`'s value, its two formals, its merged result and its output are
    // one accumulator and must be one element type.
    for i in 0..n {
        let want = values[i].element();
        for (what, got) in [
            ("lhs", merge.lhs[i].element),
            ("rhs", merge.rhs[i].element),
            ("out", outs[i].element),
            ("merge", merge.body[i].element()),
        ] {
            if got != want {
                return Err(invalid(format!(
                    "reduction lane {i} carries {want:?} but its {what} is {got:?}"
                )));
            }
        }
    }
    // `merge` reads only its own formals. A merge that reads a tile, a buffer or
    // a lane id is not a merge; cross-*lane* reads of the formals are legal and
    // required, so this checks the source of every leaf, not its index.
    let formals: FxHashSet<usize> = merge.lhs.iter().chain(&merge.rhs).map(local_key).collect();
    let mut seen = FxHashSet::default();
    for lane in &merge.body {
        let mut bad: Option<String> = None;
        visit_unique(lane, &mut seen, &mut |node| {
            if bad.is_some() {
                return;
            }
            match node.kind() {
                TileExprKind::LoadLocal(local) if !formals.contains(&local_key(local)) => {
                    bad = Some("a local that is not one of its formals".into());
                }
                TileExprKind::Builtin(b) => bad = Some(format!("{b:?}")),
                TileExprKind::Load { .. }
                | TileExprKind::LoadTile { .. }
                | TileExprKind::Reduce { .. } => {
                    bad = Some(format!("{:?}", std::mem::discriminant(node.kind())));
                }
                _ => {}
            }
        });
        if let Some(bad) = bad {
            return Err(invalid(format!("a reduction merge reads {bad}")));
        }
    }
    // Scratch: one tile per lane, and lane 0's is the one the kind names, so a
    // one-lane reduction is exactly the node it was before this form existed.
    match kind {
        ReduceKind::Subgroup => {
            if !scratch.is_empty() {
                return Err(invalid("a subgroup reduction declares scratch tiles"));
            }
        }
        ReduceKind::Workgroup { scratch: head, .. } | ReduceKind::Loop { scratch: head, .. } => {
            if scratch.len() != n {
                return Err(invalid(format!(
                    "a {n}-lane reduction declares {} scratch tiles",
                    scratch.len()
                )));
            }
            if !Arc::ptr_eq(&scratch[0], head) {
                return Err(invalid(
                    "a reduction's first scratch tile is not the one its kind names",
                ));
            }
        }
    }
    // `fast` is derived, so an author-supplied value that disagrees with `merge`
    // would take the collective path on a merge the collective cannot express.
    if let Some(op) = fast {
        if n != 1 {
            return Err(invalid(format!(
                "a {n}-lane reduction claims the {op:?} hardware collective"
            )));
        }
        if !is_plain_binary(&merge.body[0], op, &merge.lhs[0], &merge.rhs[0]) {
            return Err(invalid(format!(
                "a reduction claims {op:?} but its merge is not that binary"
            )));
        }
    }
    Ok(())
}

/// `merge.body[0] == binary(op.binary(), load(lhs), load(rhs))`, exactly.
pub(crate) fn is_plain_binary(
    body: &TileExpr,
    op: fusor_ir::ir::kernel::TileReduceOp,
    lhs: &Local,
    rhs: &Local,
) -> bool {
    let TileExprKind::Binary {
        op: b, left, right, ..
    } = body.kind()
    else {
        return false;
    };
    if *b != op.binary() {
        return false;
    }
    let reads = |e: &TileExpr, l: &Local| matches!(e.kind(), TileExprKind::LoadLocal(x) if Arc::ptr_eq(x, l));
    reads(left, lhs) && reads(right, rhs)
}

fn collect_accumulator_owners(body: &[Stmt], owners: &mut FxHashMap<usize, ()>) -> Result<()> {
    for stmt in body {
        match stmt {
            Stmt::Loop {
                accumulators, body, ..
            } => {
                for Accumulator {
                    local,
                    init,
                    update,
                } in accumulators
                {
                    if init.element() != local.element || update.element() != local.element {
                        return Err(invalid(format!(
                            "accumulator local {:?} disagrees with init {:?} / update {:?}",
                            local.element,
                            init.element(),
                            update.element()
                        )));
                    }
                    if owners.insert(local_key(local), ()).is_some() {
                        return Err(invalid(
                            "one local is the accumulator of two different loops",
                        ));
                    }
                }
                collect_accumulator_owners(body, owners)?;
            }
            Stmt::If { accept, reject, .. } => {
                collect_accumulator_owners(accept, owners)?;
                collect_accumulator_owners(reject, owners)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn check_accumulator_writes(
    body: &[Stmt],
    owners: &FxHashMap<usize, ()>,
    scope: &mut Vec<usize>,
) -> Result<()> {
    for stmt in body {
        match stmt {
            Stmt::StoreLocal { dst, .. } => {
                let key = local_key(dst);
                if owners.contains_key(&key) && !scope.contains(&key) {
                    return Err(invalid(
                        "a loop accumulator is written outside its own loop body",
                    ));
                }
            }
            Stmt::Loop {
                accumulators, body, ..
            } => {
                let pushed = accumulators.len();
                for Accumulator { local, .. } in accumulators {
                    scope.push(local_key(local));
                }
                check_accumulator_writes(body, owners, scope)?;
                scope.truncate(scope.len() - pushed);
            }
            Stmt::If { accept, reject, .. } => {
                check_accumulator_writes(accept, owners, scope)?;
                check_accumulator_writes(reject, owners, scope)?;
            }
            _ => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// (4) cooperative store layout
// ---------------------------------------------------------------------------

/// Every `CoopStore` destination is an affine rank-2 layout with a unit stride
/// on one side, addressed rank-2.
pub(crate) fn check_coop_stores(body: &[Stmt]) -> Result<()> {
    let mut error: Option<Error> = None;
    for_each_stmt(body, &mut |stmt| {
        if error.is_some() {
            return;
        }
        let Stmt::CoopStore { dst, addr, .. } = stmt else {
            return;
        };
        if !matches!(addr, Addr::Rc2 { .. }) {
            error = Some(Error::Lower(LowerError::CoopStoreLayout(
                "cooperative store needs a rank-2 address".into(),
            )));
            return;
        }
        if !cooperative_store_layout_supported(&dst.layout) {
            error = Some(Error::Lower(LowerError::CoopStoreLayout(format!(
                "destination layout {:?} is not an affine rank-2 layout with a unit stride",
                dst.layout.extents
            ))));
        }
    });
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Traversal helpers
// ---------------------------------------------------------------------------

/// Every statement in the tree, pre-order.
pub(crate) fn for_each_stmt(body: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for stmt in body {
        f(stmt);
        match stmt {
            Stmt::If { accept, reject, .. } => {
                for_each_stmt(accept, f);
                for_each_stmt(reject, f);
            }
            Stmt::Loop { body, .. } => for_each_stmt(body, f),
            _ => {}
        }
    }
}

/// Every expression appearing directly in a statement (not its children).
pub(crate) fn for_each_root_expr(body: &[Stmt], f: &mut dyn FnMut(&TileExpr)) {
    for_each_stmt(body, &mut |stmt| stmt_root_exprs(stmt, f));
}

/// The expressions of one statement, without recursing into nested bodies.
fn stmt_root_exprs(stmt: &Stmt, f: &mut dyn FnMut(&TileExpr)) {
    match stmt {
        Stmt::Store {
            addr, value, mask, ..
        }
        | Stmt::AtomicAdd {
            addr, value, mask, ..
        } => {
            for_each_addr_expr(addr, f);
            f(value);
            f(mask);
        }
        Stmt::StoreLocal { value, .. } => f(value),
        Stmt::StoreTile { index, value, .. } => {
            f(index);
            f(value);
        }
        Stmt::FillTile { value, bounds, .. } => {
            f(value);
            for bound in bounds.iter().flatten() {
                f(bound);
            }
        }
        Stmt::CoopStore { acc, addr, .. } => {
            f(acc);
            for_each_addr_expr(addr, f);
        }
        Stmt::CoopStoreTile { acc, row, col, .. } => {
            f(acc);
            f(row);
            f(col);
        }
        Stmt::If { condition, .. } => f(condition),
        Stmt::Loop {
            count,
            accumulators,
            ..
        } => {
            if let Some(count) = count {
                f(count);
            }
            for Accumulator { init, update, .. } in accumulators {
                f(init);
                f(update);
            }
        }
        Stmt::Reduce { values, merge, .. } => {
            for value in values {
                f(value);
            }
            for lane in &merge.body {
                f(lane);
            }
        }
        Stmt::Break | Stmt::Return | Stmt::Barrier | Stmt::StorageBarrier => {}
    }
}

/// Post-order over an expression DAG, visiting each distinct node once.
pub(crate) fn visit_unique(
    expr: &TileExpr,
    seen: &mut FxHashSet<u64>,
    f: &mut dyn FnMut(&TileExpr),
) {
    if !seen.insert(expr.structural_hash()) {
        return;
    }
    let mut children: Vec<TileExpr> = Vec::new();
    for_each_child(expr.kind(), &mut |child| children.push(child.clone()));
    for child in &children {
        visit_unique(child, seen, f);
    }
    f(expr);
}
