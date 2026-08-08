//! The L2 verifier.
//!
//! Checks that every node's cached `ty` equals what inference derives, that
//! every `Load` is masked or provably in range, that every `Loop` accumulator
//! is written only inside its own loop body, that every `CoopStore` has a
//! supported rank-2 layout, plus uniformity, the arena plan, and `f16`/`bf16`
//! against `caps`.
//!
//! Load masking is what licenses `create_shader_module_trusted`.

use fusor2_ir::Result;
use fusor2_ir::device::Caps;
use fusor2_ir::error::Error;
use fusor2_ir::ir::level2::{
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
pub fn verify_l2(ir: &KernelIr, caps: &Caps) -> Result<()> {
    check_dtype_caps(ir, caps)?;
    check_types(ir)?;
    check_loads(&ir.body)?;
    check_accumulators(&ir.body)?;
    check_reduce_stmts(&ir.body)?;
    check_coop_stores(&ir.body)?;
    crate::uniformity::verify_uniformity(ir)?;
    let planner = crate::planner::Planner::global();
    let plan = planner.arena_plan(ir, caps)?;
    // The plan's barrier insertions are applied before emitting, so the arena
    // is verified against the body the emitter produces.
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
                    fusor2_ir::ir::level2::ReduceKind::Subgroup => {}
                    fusor2_ir::ir::level2::ReduceKind::Workgroup { scratch, .. }
                    | fusor2_ir::ir::level2::ReduceKind::Loop { scratch, .. } => f(scratch.element),
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

/// Derive the element type of one node from its children's cached types. The
/// builder uses this to type a node; `verify_l2` re-derives it and compares.
pub fn infer_kind(kind: &TileExprKind) -> Result<ElementType> {
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
        // `Unpack2x16Float` takes a packed u32 and yields two f32 lanes; every
        // other unary is type-preserving.
        K::Unary { op, value, .. } => {
            if *op == fusor2_ir::scalar::UnOp::Unpack2x16Float {
                ElementType::Vector {
                    scalar: ScalarElement::F32,
                    lanes: 2,
                }
            } else {
                value.element()
            }
        }
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
                ElementType::Vector { scalar: a, lanes: m },
                ElementType::Vector { scalar: b, lanes: n },
            ) if a == b && m == n => ElementType::Scalar(a),
            (a, b) => {
                return Err(invalid(format!("dot operands are not equal-lane vectors: {a:?} vs {b:?}")));
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
                (
                    ElementType::Vector { lanes: a, .. },
                    ElementType::Vector { lanes: b, .. },
                ) => a == b,
                _ => false,
            };
            if !legal {
                return Err(invalid(format!("illegal cast {from:?} -> {to:?}")));
            }
        }
        K::Load { mask, fill, src, .. } => {
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

/// Every `Load` is masked or provably in range.
pub fn check_loads(body: &[Stmt]) -> Result<()> {
    let mut seen = FxHashSet::default();
    let mut error: Option<Error> = None;
    for_each_root_expr(body, &mut |expr| {
        visit_unique(expr, &mut seen, &mut |node| {
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
            if load_in_range(src, addr) {
                return;
            }
            error = Some(Error::Lower(LowerError::UnmaskedLoad(format!(
                "load with a constant-true mask is not provably in range: {:?}",
                addr
            ))));
        });
    });
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn load_in_range(src: &Source, addr: &Addr) -> bool {
    let layout = match src {
        Source::Storage(view) => &view.layout,
        Source::Quantized(view) => &view.data.layout,
    };
    match addr {
        Addr::Linear(index) => {
            max_value(index).is_some_and(|max| max < layout.element_count())
        }
        Addr::Rc2 { row, col } => {
            if layout.extents.len() != 2 {
                return false;
            }
            max_value(row).is_some_and(|max| max < u64::from(layout.extents[0]))
                && max_value(col).is_some_and(|max| max < u64::from(layout.extents[1]))
        }
    }
}

/// A non-negative integer literal, whichever integer type it carries. A
/// negative literal is not a bound on an unsigned index, so it is undecidable
/// rather than wrapping.
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
/// Bit operators are read as their unsigned arithmetic twins: `x >> k` is
/// `x / 2^k`, `x << k` is `x * 2^k`, and `x & m` is bounded like `x % (m + 1)`
/// when `m` is `2^k - 1`.
fn max_value(expr: &TileExpr) -> Option<u64> {
    use fusor2_ir::scalar::BinOp;
    match expr.kind() {
        TileExprKind::Literal(TileLiteral::U32(v)) => Some(u64::from(*v)),
        TileExprKind::Literal(TileLiteral::I32(v)) if *v >= 0 => Some(*v as u64),
        TileExprKind::Cast { value, .. } => max_value(value),
        TileExprKind::Select { accept, reject, .. } => {
            Some(max_value(accept)?.max(max_value(reject)?))
        }
        TileExprKind::Binary {
            op, left, right, ..
        } => match op {
            BinOp::Add => max_value(left)?.checked_add(max_value(right)?),
            BinOp::Mul => max_value(left)?.checked_mul(max_value(right)?),
            // Integer subtraction and division only lower the bound.
            BinOp::Sub | BinOp::Div => max_value(left),
            BinOp::Rem => Some(max_value(right)?.saturating_sub(1)),
            // `x >> k == x / 2^k` at a literal count. A dynamic count still
            // only lowers the bound.
            BinOp::Shr => match literal_u64(right) {
                Some(k) if k < 64 => Some(max_value(left)? >> k),
                Some(_) => Some(0),
                None => max_value(left),
            },
            // `x << k == x * 2^k`, checked exactly the way `Mul` is.
            BinOp::Shl => {
                let k = literal_u64(right)?;
                if k >= 64 {
                    return None;
                }
                max_value(left)?.checked_mul(1u64 << k)
            }
            // `x & m <= x` and `x & m <= m` for unsigned, so one decidable side
            // bounds the pair, mask or not a `2^k - 1`.
            BinOp::BitAnd => match (max_value(left), max_value(right)) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
            // Neither `|` nor `^` sets a bit above the highest bit either side
            // can set, so the bound is that position's all-ones mask.
            // `a | b <= A | B` is false in general (`A = B = 2` admits
            // `1 | 2 == 3`), so this rounds up to the mask.
            BinOp::BitOr | BinOp::BitXor => {
                let m = max_value(left)?.max(max_value(right)?);
                Some(if m == 0 {
                    0
                } else {
                    u64::MAX >> m.leading_zeros()
                })
            }
            BinOp::Min => match (max_value(left), max_value(right)) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
            BinOp::Max => Some(max_value(left)?.max(max_value(right)?)),
            _ => None,
        },
        _ => None,
    }
}

/// Every `Loop` accumulator's `init`/`update` type matches its local, the
/// local is the accumulator of exactly one loop, and nothing outside that
/// loop's body writes it.
///
/// [`KernelIr`] carries no local list, so membership in the builder's local
/// list is checked by [`crate::build::TileBuilder::declares_local`] at
/// construction.
pub fn check_accumulators(body: &[Stmt]) -> Result<()> {
    let mut owners: FxHashMap<usize, ()> = FxHashMap::default();
    collect_accumulator_owners(body, &mut owners)?;
    let mut scope: Vec<usize> = Vec::new();
    check_accumulator_writes(body, &owners, &mut scope)
}

fn local_key(local: &Local) -> usize {
    Arc::as_ptr(local) as *const () as usize
}

/// Every `Stmt::Reduce` is arity-consistent, element-consistent across lanes,
/// carries one scratch tile per lane where its kind needs scratch, and has a
/// `merge` reading nothing but its own formals. A node whose `values`,
/// `merge.body`, `merge.lhs`, `merge.rhs` and `outs` disagree in length is
/// rejected rather than truncated to its first slot.
pub fn check_reduce_stmts(body: &[Stmt]) -> Result<()> {
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
    kind: &fusor2_ir::ir::level2::ReduceKind,
    values: &[TileExpr],
    merge: &fusor2_ir::ir::level2::MergeBody,
    fast: Option<fusor2_ir::ir::level2::TileReduceOp>,
    outs: &[Local],
    scratch: &[fusor2_ir::ir::level2::Tile],
) -> Result<()> {
    use fusor2_ir::ir::level2::ReduceKind;
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
    // `merge` reads only its own formals: no tile, buffer or lane id.
    // Cross-lane reads of the formals are legal, so this checks the source of
    // every leaf, not its index.
    let formals: FxHashSet<usize> = merge
        .lhs
        .iter()
        .chain(&merge.rhs)
        .map(local_key)
        .collect();
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
    // Scratch: one tile per lane, and lane 0's is the one the kind names.
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
    // `fast` must agree with `merge`, or the collective path runs on a merge
    // the collective cannot express.
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
pub fn is_plain_binary(
    body: &TileExpr,
    op: fusor2_ir::ir::level2::TileReduceOp,
    lhs: &Local,
    rhs: &Local,
) -> bool {
    let TileExprKind::Binary { op: b, left, right, .. } = body.kind() else {
        return false;
    };
    if *b != op.binary() {
        return false;
    }
    let reads = |e: &TileExpr, l: &Local| {
        matches!(e.kind(), TileExprKind::LoadLocal(x) if Arc::ptr_eq(x, l))
    };
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

/// Every `CoopStore` destination is an affine rank-2 layout with a unit stride
/// on one side, addressed rank-2.
pub fn check_coop_stores(body: &[Stmt]) -> Result<()> {
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

/// Every statement in the tree, pre-order.
pub fn for_each_stmt(body: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
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
pub fn for_each_root_expr(body: &[Stmt], f: &mut dyn FnMut(&TileExpr)) {
    for_each_stmt(body, &mut |stmt| match stmt {
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
    });
}

/// Post-order over an expression DAG, visiting each distinct node once.
pub fn visit_unique(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::TileBuilder;
    use crate::build::fixtures::{caps_with, whole_buffer_view, wg_tile};
    use fusor2_ir::dtype::NumericContract;
    use fusor2_ir::ir::level2::{BufferAccess, MemoryLevel, StorageView, TileLayout};
    use fusor2_ir::scalar::BinOp;

    #[test]
    fn well_typed_kernel_verifies() {
        let mut b = TileBuilder::new();
        let tile = wg_tile(&mut b, ScalarElement::F32.element(), 64);
        let zero = b.lit_f32(0.0);
        let index = b.lit_u32(0);
        let stmt = b.store_tile(tile, index, zero);
        b.push(stmt);
        let ir = b.finish([1, 1, 1], 64, "ok");
        verify_l2(&ir, &caps_with(|_| {})).unwrap();
    }

    #[test]
    fn mistyped_binary_is_rejected() {
        let mut b = TileBuilder::new();
        let a = b.lit_f32(1.0);
        let c = b.lit_u32(1);
        let sum = b.binary(BinOp::Add, a, c, NumericContract::RELAXED);
        let local = b.alloc_local(ScalarElement::F32.element());
        let stmt = b.store_local(local, sum);
        b.push(stmt);
        let ir = b.finish([1, 1, 1], 1, "bad");
        assert!(verify_l2(&ir, &caps_with(|_| {})).is_err());
    }

    #[test]
    fn unmasked_dynamic_load_rejected() {
        let mut b = TileBuilder::new();
        let buffer = b.alloc_buffer(
            0,
            ScalarElement::F32.element(),
            TileLayout::contiguous(MemoryLevel::Storage, &[16]),
            BufferAccess::Read,
        );
        let view = whole_buffer_view(&buffer);
        let lane = b.builtin(fusor2_ir::ir::level2::Builtin::Lane);
        let mask = b.mask_true();
        let fill = b.lit_f32(0.0);
        let load = b.load(Source::Storage(view), Addr::Linear(lane), mask, fill);
        let local = b.alloc_local(ScalarElement::F32.element());
        let stmt = b.store_local(local, load);
        b.push(stmt);
        let ir = b.finish([1, 1, 1], 1, "unmasked");
        match verify_l2(&ir, &caps_with(|_| {})) {
            Err(Error::Lower(LowerError::UnmaskedLoad(_))) => {}
            other => panic!("expected UnmaskedLoad, got {other:?}"),
        }
    }

    #[test]
    fn literal_load_is_provably_in_range() {
        let mut b = TileBuilder::new();
        let buffer = b.alloc_buffer(
            0,
            ScalarElement::F32.element(),
            TileLayout::contiguous(MemoryLevel::Storage, &[16]),
            BufferAccess::Read,
        );
        let view = whole_buffer_view(&buffer);
        let index = b.lit_u32(15);
        let mask = b.mask_true();
        let fill = b.lit_f32(0.0);
        let load = b.load(Source::Storage(view), Addr::Linear(index), mask, fill);
        let local = b.alloc_local(ScalarElement::F32.element());
        let stmt = b.store_local(local, load);
        b.push(stmt);
        let ir = b.finish([1, 1, 1], 1, "in-range");
        verify_l2(&ir, &caps_with(|_| {})).unwrap();
    }

    /// `lane & 31` and `lane % 32` are the same function of `lane`, so both
    /// bound an address.
    #[test]
    fn a_power_of_two_mask_bounds_an_address_the_way_a_remainder_does() {
        for (op, rhs) in [(BinOp::BitAnd, 31u32), (BinOp::Rem, 32)] {
            let mut b = TileBuilder::new();
            let buffer = b.alloc_buffer(
                0,
                ScalarElement::F32.element(),
                TileLayout::contiguous(MemoryLevel::Storage, &[32]),
                BufferAccess::Read,
            );
            let view = whole_buffer_view(&buffer);
            let lane = b.builtin(fusor2_ir::ir::level2::Builtin::Lane);
            let k = b.lit_u32(rhs);
            let index = b.binary(op, lane, k, NumericContract::RELAXED);
            let mask = b.mask_true();
            let fill = b.lit_f32(0.0);
            let load = b.load(Source::Storage(view), Addr::Linear(index), mask, fill);
            let local = b.alloc_local(ScalarElement::F32.element());
            let stmt = b.store_local(local, load);
            b.push(stmt);
            let ir = b.finish([1, 1, 1], 32, "wrap");
            verify_l2(&ir, &caps_with(|_| {})).unwrap_or_else(|e| panic!("{op:?}: {e:?}"));
        }
    }

    /// A shift and a mask compose the way their arithmetic twins do:
    /// `(word >> 24) & 0xF` is bounded at 15 without any knowledge of `word`.
    #[test]
    fn bit_arithmetic_bounds_compose_like_their_arithmetic_twins() {
        let mut b = TileBuilder::new();
        let lane = b.builtin(fusor2_ir::ir::level2::Builtin::Lane);
        let twenty_four = b.lit_u32(24);
        let fifteen = b.lit_u32(15);
        let shifted = b.binary(BinOp::Shr, lane.clone(), twenty_four, NumericContract::RELAXED);
        let nibble = b.binary(BinOp::BitAnd, shifted, fifteen, NumericContract::RELAXED);
        assert_eq!(max_value(&nibble), Some(15));

        // `x << k` is `x * 2^k`.
        let seven = b.lit_u32(7);
        let two = b.lit_u32(2);
        let scaled = b.binary(BinOp::Shl, seven, two.clone(), NumericContract::RELAXED);
        assert_eq!(max_value(&scaled), Some(28));
        let unbounded = b.binary(BinOp::Shl, lane, two, NumericContract::RELAXED);
        assert_eq!(max_value(&unbounded), None);
    }

    #[test]
    fn undeclared_accumulator_rejected() {
        let mut b = TileBuilder::new();
        let local = b.alloc_local(ScalarElement::F32.element());
        let zero = b.lit_f32(0.0);
        let one = b.lit_f32(1.0);
        let four = b.lit_u32(4);
        // The accumulator is also written from outside the loop body.
        let outer = b.store_local(local.clone(), one.clone());
        let looped = b.loop_counted(
            Some(four),
            None,
            vec![Accumulator {
                local,
                init: zero,
                update: one,
            }],
            Vec::new(),
        );
        b.push(outer);
        b.push(looped);
        let ir = b.finish([1, 1, 1], 1, "acc");
        assert!(check_accumulators(&ir.body).is_err());
    }

    /// A rank-2 layout whose first axis needs two sub-axes, so it is not
    /// affine.
    fn non_affine_rc2(buffer: &fusor2_ir::ir::level2::Buffer) -> StorageView {
        use fusor2_ir::shape::{AxisGroup, MultiFlattenMap, SubAxis};
        let indexing = MultiFlattenMap {
            groups: smallvec::smallvec![
                AxisGroup {
                    sub_axes: smallvec::smallvec![
                        SubAxis {
                            extent: 4,
                            stride: 16
                        },
                        SubAxis {
                            extent: 2,
                            stride: 8
                        }
                    ],
                },
                AxisGroup::affine(8, 1),
            ],
        };
        StorageView {
            buffer: buffer.clone(),
            offset: 0,
            layout: TileLayout {
                extents: smallvec::smallvec![8, 8],
                indexing,
                level: MemoryLevel::Storage,
            },
        }
    }

    #[test]
    fn coop_store_non_affine_layout_rejected() {
        let mut b = TileBuilder::new();
        let buffer = b.alloc_buffer(
            0,
            ScalarElement::F32.element(),
            TileLayout::contiguous(MemoryLevel::Storage, &[8, 8]),
            BufferAccess::ReadWrite,
        );
        let view = non_affine_rc2(&buffer);
        let source = whole_buffer_view(&buffer);
        let zero = b.lit_u32(0);
        let acc = b.coop_load(
            CoopMatrixRole::C,
            ScalarElement::F32,
            8,
            8,
            CoopSrc::BroadcastCol {
                src: source,
                col: zero.clone(),
            },
        );
        let stmt = b.coop_store(
            acc,
            view,
            Addr::Rc2 {
                row: zero.clone(),
                col: zero,
            },
        );
        b.push(stmt);
        let ir = b.finish([1, 1, 1], 64, "coop");
        match check_coop_stores(&ir.body) {
            Err(Error::Lower(LowerError::CoopStoreLayout(_))) => {}
            other => panic!("expected CoopStoreLayout, got {other:?}"),
        }
    }

    #[test]
    fn coop_store_needs_a_rank_two_address() {
        let mut b = TileBuilder::new();
        let buffer = b.alloc_buffer(
            0,
            ScalarElement::F32.element(),
            TileLayout::contiguous(MemoryLevel::Storage, &[8, 8]),
            BufferAccess::ReadWrite,
        );
        let view = whole_buffer_view(&buffer);
        let zero = b.lit_u32(0);
        let acc = b.coop_load(
            CoopMatrixRole::C,
            ScalarElement::F32,
            8,
            8,
            CoopSrc::BroadcastCol {
                src: view.clone(),
                col: zero.clone(),
            },
        );
        let stmt = b.coop_store(acc, view, Addr::Linear(zero));
        b.push(stmt);
        let ir = b.finish([1, 1, 1], 64, "coop-linear");
        assert!(check_coop_stores(&ir.body).is_err());
    }

    #[test]
    fn f16_without_caps_rejected() {
        let mut b = TileBuilder::new();
        let tile = wg_tile(&mut b, ScalarElement::F16.element(), 64);
        let zero = b.zero(ScalarElement::F16.element());
        let index = b.lit_u32(0);
        let stmt = b.store_tile(tile, index, zero);
        b.push(stmt);
        let ir = b.finish([1, 1, 1], 64, "f16");
        match verify_l2(&ir, &caps_with(|c| c.f16 = false)) {
            Err(Error::Legality(_)) => {}
            other => panic!("expected Legality, got {other:?}"),
        }
        verify_l2(&ir, &caps_with(|c| c.f16 = true)).unwrap();
    }

    #[test]
    fn bf16_without_caps_rejected() {
        let mut b = TileBuilder::new();
        let tile = wg_tile(&mut b, ScalarElement::BF16.element(), 64);
        let zero = b.zero(ScalarElement::BF16.element());
        let index = b.lit_u32(0);
        let stmt = b.store_tile(tile, index, zero);
        b.push(stmt);
        let ir = b.finish([1, 1, 1], 64, "bf16");
        assert!(verify_l2(&ir, &caps_with(|c| c.bf16 = false)).is_err());
        verify_l2(&ir, &caps_with(|c| c.bf16 = true)).unwrap();
    }

    /// A two-lane `Stmt::Reduce` over `(max, sum)`-shaped scratch, built through
    /// the canonical constructor.
    fn two_lane_reduce(b: &mut TileBuilder) -> Vec<Stmt> {
        let f32e = ScalarElement::F32.element();
        let a = b.lit_f32(1.0);
        let c = b.lit_f32(2.0);
        let scratch = wg_tile(b, f32e, 64);
        let mut out = Vec::new();
        let _reads = b
            .reduce_carrier::<String>(
                fusor2_ir::ir::level2::ReduceKind::Workgroup {
                    scratch,
                    group_size: 64,
                },
                &two_slot_carrier(),
                &[a, c],
                &[64],
                &mut out,
                |b, i, lhs, rhs| {
                    Ok(b.binary(
                        if i == 0 { BinOp::Max } else { BinOp::Add },
                        lhs[i].clone(),
                        rhs[i].clone(),
                        NumericContract::RELAXED,
                    ))
                },
            )
            .unwrap();
        out
    }

    fn two_slot_carrier() -> fusor2_ir::carrier::Carrier {
        use fusor2_ir::carrier::{ArgRemap, Carrier};
        use fusor2_ir::dtype::Dtype;
        let max = Carrier::binop(
            BinOp::Max,
            Carrier::binop_identity(BinOp::Max, Dtype::F32).unwrap(),
            Dtype::F32,
        );
        let sum = Carrier::binop(
            BinOp::Add,
            Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
            Dtype::F32,
        );
        max.tuple(&sum, &ArgRemap::identity(1)).carrier
    }

    /// At one scalar binop slot the constructor returns the same
    /// `TileExprKind::Reduce` node and pushes no statement.
    #[test]
    fn a_single_slot_carrier_delegates_to_the_collective() {
        use fusor2_ir::carrier::Carrier;
        use fusor2_ir::dtype::Dtype;
        let mut b = TileBuilder::new();
        let value = b.lit_f32(3.0);
        let sum = Carrier::binop(
            BinOp::Add,
            Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
            Dtype::F32,
        );
        let mut out = Vec::new();
        let reads = b
            .reduce_carrier::<String>(
                fusor2_ir::ir::level2::ReduceKind::Subgroup,
                &sum,
                std::slice::from_ref(&value),
                &[64],
                &mut out,
                |_, _, _, _| unreachable!("the fast path never builds a merge"),
            )
            .unwrap();
        assert!(out.is_empty(), "the fast path pushes no statement");
        assert_eq!(reads.len(), 1);
        let direct = b.reduce(
            fusor2_ir::ir::level2::TileReduceOp::Sum,
            fusor2_ir::ir::level2::ReduceKind::Subgroup,
            value,
        );
        assert_eq!(reads[0], direct, "the delegated node must hash-cons together");
    }

    #[test]
    fn a_well_formed_two_lane_reduction_verifies() {
        let mut b = TileBuilder::new();
        let body = two_lane_reduce(&mut b);
        b.set_body(body);
        let ir = b.finish([1, 1, 1], 64, "two-lane");
        verify_l2(&ir, &caps_with(|_| {})).unwrap();
    }

    /// A node whose lane counts disagree is rejected rather than truncated to
    /// its first slot.
    #[test]
    fn a_reduction_with_disagreeing_lane_counts_is_rejected() {
        let mut b = TileBuilder::new();
        let mut body = two_lane_reduce(&mut b);
        let Stmt::Reduce { values, .. } = &mut body[0] else {
            panic!("expected a reduce");
        };
        values.pop();
        assert!(check_reduce_stmts(&body).is_err());

        let mut body = two_lane_reduce(&mut b);
        let Stmt::Reduce { outs, .. } = &mut body[0] else {
            panic!("expected a reduce");
        };
        outs.pop();
        assert!(check_reduce_stmts(&body).is_err());

        let mut body = two_lane_reduce(&mut b);
        let Stmt::Reduce { merge, .. } = &mut body[0] else {
            panic!("expected a reduce");
        };
        merge.lhs.pop();
        assert!(check_reduce_stmts(&body).is_err());
    }

    /// One scratch tile per lane, and lane 0's is the one the kind names.
    #[test]
    fn a_reduction_missing_a_scratch_tile_is_rejected() {
        let mut b = TileBuilder::new();
        let mut body = two_lane_reduce(&mut b);
        let Stmt::Reduce { scratch, .. } = &mut body[0] else {
            panic!("expected a reduce");
        };
        scratch.pop();
        assert!(check_reduce_stmts(&body).is_err());
    }

    /// A merge reads its formals and nothing else.
    #[test]
    fn a_merge_reading_outside_its_formals_is_rejected() {
        let mut b = TileBuilder::new();
        let lane = b.builtin(fusor2_ir::ir::level2::Builtin::Lane);
        let lane = b.cast(lane, ScalarElement::F32.element());
        let mut body = two_lane_reduce(&mut b);
        let Stmt::Reduce { merge, .. } = &mut body[0] else {
            panic!("expected a reduce");
        };
        merge.body[1] = lane;
        assert!(check_reduce_stmts(&body).is_err());

        // A foreign local is not one of the two partials being merged.
        let mut b2 = TileBuilder::new();
        let stray = b2.alloc_local(ScalarElement::F32.element());
        let read = b2.load_local(stray);
        let mut body = two_lane_reduce(&mut b2);
        let Stmt::Reduce { merge, .. } = &mut body[0] else {
            panic!("expected a reduce");
        };
        merge.body[0] = read;
        assert!(check_reduce_stmts(&body).is_err());
    }

    /// Cross-lane reads are legal: a merge that reads `lhs[0]` from lane 1
    /// verifies.
    #[test]
    fn a_merge_reading_a_sibling_lane_is_accepted() {
        let mut b = TileBuilder::new();
        let mut body = two_lane_reduce(&mut b);
        let Stmt::Reduce { merge, .. } = &mut body[0] else {
            panic!("expected a reduce");
        };
        let m = b.load_local(merge.lhs[0].clone());
        let l = b.load_local(merge.rhs[1].clone());
        merge.body[1] = b.binary(BinOp::Add, m, l, NumericContract::RELAXED);
        check_reduce_stmts(&body).unwrap();
    }

    /// A `fast` operator that disagrees with `merge` is rejected.
    #[test]
    fn a_claimed_fast_operator_must_match_the_merge() {
        use fusor2_ir::ir::level2::TileReduceOp;
        let mut b = TileBuilder::new();
        let mut body = two_lane_reduce(&mut b);
        let Stmt::Reduce { fast, .. } = &mut body[0] else {
            panic!("expected a reduce");
        };
        *fast = Some(TileReduceOp::Sum);
        assert!(
            check_reduce_stmts(&body).is_err(),
            "two lanes cannot claim one hardware operator"
        );

        // One lane whose merge is `Max` may not claim `Sum`.
        let mut b = TileBuilder::new();
        let f32e = ScalarElement::F32.element();
        let lhs = b.alloc_local(f32e);
        let rhs = b.alloc_local(f32e);
        let a = b.load_local(lhs.clone());
        let c = b.load_local(rhs.clone());
        let merged = b.binary(BinOp::Max, a, c, NumericContract::RELAXED);
        let scratch = wg_tile(&mut b, f32e, 64);
        let value = b.lit_f32(1.0);
        let out_local = b.alloc_local(f32e);
        let mk = |fast| Stmt::Reduce {
            kind: Box::new(fusor2_ir::ir::level2::ReduceKind::Workgroup {
                scratch: scratch.clone(),
                group_size: 64,
            }),
            values: smallvec::smallvec![value.clone()],
            merge: Box::new(fusor2_ir::ir::level2::MergeBody {
                lhs: smallvec::smallvec![lhs.clone()],
                rhs: smallvec::smallvec![rhs.clone()],
                body: smallvec::smallvec![merged.clone()],
            }),
            fast,
            outs: smallvec::smallvec![out_local.clone()],
            scratch: smallvec::smallvec![scratch.clone()],
        };
        assert!(check_reduce_stmts(&[mk(Some(TileReduceOp::Sum))]).is_err());
        check_reduce_stmts(&[mk(Some(TileReduceOp::Max))]).unwrap();
        check_reduce_stmts(&[mk(None)]).unwrap();
    }

    /// Element agreement across a lane: the value, both formals, the merged
    /// expression and the output are one accumulator.
    #[test]
    fn a_lane_whose_output_element_disagrees_is_rejected() {
        let mut b = TileBuilder::new();
        let wrong = b.alloc_local(ScalarElement::U32.element());
        let mut body = two_lane_reduce(&mut b);
        let Stmt::Reduce { outs, .. } = &mut body[0] else {
            panic!("expected a reduce");
        };
        outs[1] = wrong;
        assert!(check_reduce_stmts(&body).is_err());
    }

    /// Liveness sees every lane's scratch tile, so the arena sizes N tiles per
    /// reduction rather than one.
    #[test]
    fn liveness_sees_every_lane_scratch_tile() {
        let mut b = TileBuilder::new();
        let body = two_lane_reduce(&mut b);
        b.set_body(body);
        let ir = b.finish([1, 1, 1], 64, "two-lane");
        let live = crate::liveness::analyze(&ir);
        assert_eq!(
            live.order.len(),
            2,
            "a two-lane reduction owns two scratch tiles"
        );
        for key in &live.order {
            let t = &live.tiles[key];
            assert!(
                t.accesses
                    .iter()
                    .any(|a| a.kind == crate::liveness::AccessKind::ReadWrite)
            );
        }
    }
}
