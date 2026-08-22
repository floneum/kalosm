//! Launch node + `SchedPoint` -> `KernelIr` for the CPU backend.

pub(crate) mod contract;
pub(crate) mod gather_scatter;
pub(crate) mod map_fold;

use fusor_ir::Result;
use fusor_ir::device::Caps;
use fusor_ir::dtype::{Dtype, NumericContract, QLayout};
use fusor_ir::egraph::Id;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Addr, BufferAccess, BufferDecl, Builtin, ElementType, KernelIr, MemoryLevel, QuantizedView,
    ScalarElement, Source, StorageView, TileExpr, TileExprKind, TileLayout, TileLiteral,
    WorkgroupAxis,
};
use fusor_ir::ir::launch::{
    AccessPlan, AddressMap, AddressTerm, Family, Launch, Operand, SchedPoint,
};
use fusor_ir::ir::{Node, Op};
use fusor_ir::scalar::{BinOp, ScalarExpr, ScalarKind};
use fusor_ir::shape::{AxisGroup, Dim, Layout, SymId};
use fusor_ir::target::LowerCtx;
use smallvec::SmallVec;
use std::sync::Arc;

/// Lanes per workgroup for a node whose schedule point names no lane group.
/// One grid point is one workgroup; `block` lanes are walked in chunks of the
/// register width.
pub(crate) fn default_block(caps: &Caps) -> u32 {
    // A CPU "block" is an internal native loop chunk, not a GPU workgroup
    // capability. Keeping it here avoids exposing 256 GPU-style schedule
    // alternatives merely to let one Cranelift call process 256 elements.
    let _ = caps;
    256
}

pub(crate) fn lower(
    caps: &Caps,
    node: &Node,
    id: Id,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
) -> Result<KernelIr> {
    let _ = id;
    let Op::Launch(op) = &node.op else {
        return Err(Error::Legality(
            "the CPU target can only lower Launch nodes".into(),
        ));
    };
    match op {
        Launch::Map { .. } | Launch::Fold { .. } => map_fold::lower(caps, node, theta, cx),
        Launch::Contract { family, .. } => {
            if *family == Family::Coop {
                // Caps report no cooperative config, so this alternative is never selectable
                return Err(Error::Legality(
                    "Family::Coop is not lowerable on the CPU target".into(),
                ));
            }
            contract::lower(caps, node, theta, cx)
        }
        Launch::Gather { .. } | Launch::Scatter { .. } => {
            gather_scatter::lower(caps, node, theta, cx)
        }
        Launch::Region { members, .. } => compose(caps, members, theta, cx, "cpu_region"),
        Launch::Ext { def, .. } => ext::lower(*def, node, theta),
    }
}

/// One dispatch running several member kernels.
///
/// Each member is lowered through the ordinary dispatch above and the bodies
/// are concatenated over one shared grid. Each member's stores are redirected
/// to that member's own buffer; only the member standing for the composite's
/// own value keeps the root's. A member whose own grid is shorter than the
/// shared one is guarded, or it would write past its buffer. Members must
/// agree on their lane count; a mismatch is a legality error.
fn compose(
    caps: &Caps,
    members: &[Id],
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
    name: &'static str,
) -> Result<KernelIr> {
    if members.is_empty() {
        return Err(Error::Legality(
            "a composite node with no members has nothing to lower".into(),
        ));
    }
    // A composite has no register tile of its own — every member carries its
    // own tiling in its own `SchedPoint` — so the untiled point is the only
    // one this lowering can honor.
    match theta {
        SchedPoint::Point => {}
        SchedPoint::Map(t) if t.dim.is_none() && t.tm <= 1 => {}
        other => {
            return Err(Error::Legality(format!(
                "a CPU composite runs each member's own kernel at that member's own \
                 schedule point, so it has no register tile of its own to place \
                 {other:?} on"
            )));
        }
    }
    let binds = Binds::build(cx)?;
    let mut kernels = Vec::with_capacity(members.len());
    for m in members {
        let selected = cx.selected(*m);
        let node = cx.graph.node(selected);
        // Each member is scheduled at its own point, not the composite's.
        let member_theta = cx
            .plan
            .extraction
            .theta
            .get(&selected)
            .copied()
            .unwrap_or(SchedPoint::Point);
        kernels.push((*m, lower(caps, node, selected, member_theta, cx)?));
    }
    if kernels
        .iter()
        .any(|(_, kernel)| crate::gemm::ContractSpec::parse(kernel.name).is_some())
    {
        return Err(Error::Legality(
            "a platform GEMM must remain its own CPU dispatch".into(),
        ));
    }

    let block = kernels[0].1.block;
    if let Some((bad, k)) = kernels.iter().find(|(_, k)| k.block != block) {
        return Err(Error::Legality(format!(
            "composite member {bad} wants {} lanes but the first member wants \
             {block}; a CPU dispatch has one lane count, so this composite has \
             no single-kernel lowering",
            k.block
        )));
    }
    let grid = kernels.iter().fold([1u32, 1, 1], |acc, (_, k)| {
        [
            acc[0].max(k.grid[0]),
            acc[1].max(k.grid[1]),
            acc[2].max(k.grid[2]),
        ]
    });

    // Only a store aimed at the launch root is redirected: a member that
    // writes several distinct buffers keeps every one of them.
    let root_buffer = binds.of(cx.launch.root).ok();
    let mut body = Vec::new();
    for (id, kernel) in kernels {
        // A member with no buffer of its own stands for the composite's value
        // and keeps writing the launch root's buffer.
        let own = binds.of(id).ok().map(|b| StorageView {
            layout: b.layout.clone(),
            buffer: b,
            offset: 0,
        });
        let mut stmts = kernel.body;
        if let Some(view) = own {
            redirect_stores(&mut stmts, root_buffer.as_ref(), &view);
        }
        if kernel.grid[0] < grid[0] {
            let pid = TileExpr::new(
                TileExprKind::Builtin(Builtin::ProgramId(WorkgroupAxis::X)),
                u32_ty(),
            );
            stmts = vec![fusor_ir::ir::kernel::Stmt::If {
                condition: cmp(fusor_ir::scalar::CmpOp::Lt, pid, lit_u32(kernel.grid[0])),
                accept: stmts,
                reject: Vec::new(),
            }];
        }
        body.extend(stmts);
    }

    Ok(KernelIr {
        buffers: binds.buffers,
        grid,
        block,
        body,
        byte_arena: None,
        name,
    })
}

/// Point every store aimed at `from` (the launch root's buffer) at `view`
/// instead, leaving addresses, masks and values alone. With `from` absent —
/// the root owns no buffer — every store moves.
fn redirect_stores(
    stmts: &mut [fusor_ir::ir::kernel::Stmt],
    from: Option<&Arc<BufferDecl>>,
    view: &StorageView,
) {
    use fusor_ir::ir::kernel::Stmt;
    let hits = |dst: &StorageView| match from {
        Some(root) => Arc::ptr_eq(&dst.buffer, root),
        None => true,
    };
    for s in stmts {
        match s {
            Stmt::Store { dst, .. } | Stmt::AtomicAdd { dst, .. } | Stmt::CoopStore { dst, .. } => {
                if hits(dst) {
                    *dst = view.clone();
                }
            }
            Stmt::If { accept, reject, .. } => {
                redirect_stores(accept, from, view);
                redirect_stores(reject, from, view);
            }
            Stmt::Loop { body, .. } => redirect_stores(body, from, view),
            _ => {}
        }
    }
}

/// `Launch::Ext` lowering: the one escape hatch out of the closed `Logical`/`Launch` enums.
pub(crate) mod ext {
    use super::*;
    use fusor_ir::ir::{OpDefId, OpDefRegistry};
    use std::sync::RwLock;

    /// The registry `Launch::Ext` lowering resolves `OpDefId` against.
    ///
    /// The embedder installs the same registry here that it installed on the
    /// e-graph's semantics. Registration order is id order and must match.
    static DEFS: RwLock<Option<OpDefRegistry>> = RwLock::new(None);

    /// The installed registry, if the embedder installed one.
    pub(crate) fn installed() -> Option<OpDefRegistry> {
        DEFS.read()
            .expect("the OpDef registry lock is poisoned")
            .clone()
    }

    /// Lower one registered extension op through its `"cpu"` row.
    pub(crate) fn lower(def: OpDefId, node: &Node, theta: SchedPoint) -> Result<KernelIr> {
        let registry = installed().ok_or_else(|| {
            Error::Legality(format!(
                "{def:?} is an extension op, but no OpDefRegistry is installed on the \
                 CPU target; call fusor_cpu::lower::ext::install"
            ))
        })?;
        let entry = registry
            .get(def)
            .ok_or_else(|| Error::Legality(format!("no OpDef is registered as {def:?}")))?;
        let lower = entry
            .lower_per_target
            .iter()
            .find(|(target, _)| *target == "cpu")
            .map(|(_, f)| *f)
            .ok_or_else(|| {
                Error::Legality(format!(
                    "OpDef \"{}\" declares no \"cpu\" lowering; its \
                     lower_per_target names {:?}",
                    entry.name,
                    entry
                        .lower_per_target
                        .iter()
                        .map(|(t, _)| *t)
                        .collect::<Vec<_>>()
                ))
            })?;
        lower(node, &theta)
    }
}

pub(crate) fn u32_ty() -> ElementType {
    ElementType::Scalar(ScalarElement::U32)
}
pub(crate) fn bool_ty() -> ElementType {
    ElementType::Scalar(ScalarElement::Bool)
}

pub(crate) fn elem_of(d: Dtype) -> Result<ScalarElement> {
    Ok(match d {
        Dtype::F32 => ScalarElement::F32,
        Dtype::F16 => ScalarElement::F16,
        Dtype::BF16 => ScalarElement::BF16,
        Dtype::U32 => ScalarElement::U32,
        Dtype::I32 => ScalarElement::I32,
        Dtype::Q(_) => {
            return Err(Error::Legality(
                "a quantized value has no dense element type".into(),
            ));
        }
    })
}

pub(crate) fn lit_u32(v: u32) -> TileExpr {
    TileExpr::new(TileExprKind::Literal(TileLiteral::U32(v)), u32_ty())
}

pub(crate) fn lit_true() -> TileExpr {
    TileExpr::new(TileExprKind::Literal(TileLiteral::Bool(true)), bool_ty())
}

pub(crate) fn lit_f32(v: f32) -> TileExpr {
    TileExpr::new(
        TileExprKind::Literal(TileLiteral::F32(v.to_bits())),
        ElementType::Scalar(ScalarElement::F32),
    )
}

pub(crate) fn bin(
    op: fusor_ir::scalar::BinOp,
    a: TileExpr,
    b: TileExpr,
    ty: ElementType,
) -> TileExpr {
    TileExpr::new(
        TileExprKind::Binary {
            op,
            left: a,
            right: b,
            numeric: NumericContract::RELAXED,
        },
        ty,
    )
}

pub(crate) fn cmp(op: fusor_ir::scalar::CmpOp, a: TileExpr, b: TileExpr) -> TileExpr {
    TileExpr::new(
        TileExprKind::Compare {
            op,
            left: a,
            right: b,
        },
        bool_ty(),
    )
}

/// The global element index this lane owns:
/// `program_id.x * BLOCK + lane`.
pub(crate) fn global_lane(block: u32) -> TileExpr {
    let pid = TileExpr::new(
        TileExprKind::Builtin(Builtin::ProgramId(WorkgroupAxis::X)),
        u32_ty(),
    );
    let lane = TileExpr::new(TileExprKind::Builtin(Builtin::Lane), u32_ty());
    bin(
        fusor_ir::scalar::BinOp::Add,
        bin(fusor_ir::scalar::BinOp::Mul, pid, lit_u32(block), u32_ty()),
        lane,
        u32_ty(),
    )
}

/// Hand back the same `Arc` for two structurally equal buffer decls.
///
/// `emit::buffer_of` resolves a `StorageView` to a binding slot by
/// `Arc::ptr_eq`, and a `Region`'s members each build their own `Binds`;
/// interning makes identity follow content.
fn intern_decl(decl: BufferDecl) -> Arc<BufferDecl> {
    use std::sync::Mutex;
    use std::sync::OnceLock;
    static POOL: OnceLock<Mutex<Vec<Arc<BufferDecl>>>> = OnceLock::new();
    let pool = POOL.get_or_init(|| Mutex::new(Vec::new()));
    let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(hit) = pool.iter().find(|d| ***d == decl) {
        return Arc::clone(hit);
    }
    // Nothing outside the pool holds these any more, so they can never be
    // ptr-matched again; drop them rather than growing without bound.
    if pool.len() >= 512 {
        pool.retain(|d| Arc::strong_count(d) > 1);
    }
    let fresh = Arc::new(decl);
    pool.push(Arc::clone(&fresh));
    fresh
}

/// One kernel's buffer table, derived from the launch's bindings so binding
/// order and codegen cannot drift.
pub(crate) struct Binds {
    pub buffers: Vec<Arc<BufferDecl>>,
    pub by_value: Vec<(Id, usize)>,
}

impl Binds {
    /// Binding 0 is always the uniform block; the rest come straight from the
    /// plan, sorted by binding index.
    pub(crate) fn build(cx: &LowerCtx<'_>) -> Result<Self> {
        let mut bindings = cx.launch.bindings.clone();
        bindings.sort_by_key(|b| b.binding);

        let mut buffers = Vec::with_capacity(bindings.len() + 1);
        buffers.push(intern_decl(BufferDecl {
            binding: 0,
            element: u32_ty(),
            layout: TileLayout::contiguous(
                MemoryLevel::Storage,
                &[(cx.symbols.len().max(1)) as u32],
            ),
            access: BufferAccess::Read,
        }));

        let mut by_value = Vec::with_capacity(bindings.len());
        for (i, b) in bindings.iter().enumerate() {
            if b.binding == 0 {
                continue;
            }
            let facts = cx.graph.facts(b.value);
            // A quantized buffer is an opaque block stream: the decode program
            // addresses it as `u32` words, so it binds as u32 with the word
            // count of its blocks.
            let (element, extents) = match facts.dtype {
                Dtype::Q(fmt) => {
                    let layout = qlayout_of(cx, b.value).unwrap_or(QLayout::Native);
                    let extents = const_extents(cx, &facts.shape)?;
                    let elems: u64 = extents.iter().map(|e| *e as u64).product();
                    let blocks = elems.div_ceil(u64::from(fmt.block_elements()).max(1));
                    let words = (blocks * u64::from(fmt.block_bytes(layout))).div_ceil(4);
                    (u32_ty(), vec![words as u32])
                }
                d => {
                    let extents = const_extents(cx, &facts.shape)?;
                    (ElementType::Scalar(elem_of(d)?), extents)
                }
            };
            let access = match b.kind {
                fusor_ir::extract::BindKind::Read => BufferAccess::Read,
                _ => BufferAccess::ReadWrite,
            };
            // Keyed by every id in the value's class: an `Operand::src` names
            // whichever id the rule author wrote, and they all denote the same
            // buffer. `class_ids` also covers the `Union` spine nodes macro
            // ops hand their callers.
            let class = cx.graph.class_of(b.value);
            for member in cx.graph.class_ids(class) {
                by_value.push((member, i + 1));
            }
            buffers.push(intern_decl(BufferDecl {
                binding: (i + 1) as u32,
                element,
                layout: TileLayout::contiguous(MemoryLevel::Storage, &extents),
                access,
            }));
        }
        Ok(Self { buffers, by_value })
    }

    pub(crate) fn of(&self, value: Id) -> Result<Arc<BufferDecl>> {
        let idx = self
            .by_value
            .iter()
            .find(|(v, _)| *v == value)
            .map(|(_, i)| *i)
            .ok_or_else(|| {
                Error::Legality(format!("value {value} has no binding in this launch"))
            })?;
        self.buffers
            .iter()
            .find(|b| b.binding as usize == idx)
            .cloned()
            .ok_or_else(|| Error::Legality(format!("binding {idx} is missing")))
    }
}

const DERIVED_STRIDE: SymId = SymId(u32::MAX);

/// Resolve a dimension at the concrete binding this CPU artifact is compiled
/// for. The executable cache includes these values, so embedding them in the
/// native loop nest cannot reuse code for a different shape.
pub(crate) fn resolve_dim(cx: &LowerCtx<'_>, dim: Dim) -> Result<u32> {
    let value = match dim {
        Dim::Const(value) => value,
        Dim::Sym(symbol) if symbol != DERIVED_STRIDE => cx
            .dim_bindings
            .iter()
            .find_map(|(bound, value)| (*bound == symbol).then_some(*value))
            .ok_or_else(|| Error::Legality(format!("dim {symbol} is unbound at CPU lowering")))?,
        Dim::Sym(_) => {
            return Err(Error::Legality(
                "a derived row-major stride is not a standalone extent".into(),
            ));
        }
    };
    u32::try_from(value)
        .map_err(|_| Error::Legality(format!("CPU dimension {value} exceeds u32 indexing")))
}

pub(crate) fn const_extents(cx: &LowerCtx<'_>, shape: &[Dim]) -> Result<Vec<u32>> {
    shape.iter().map(|dim| resolve_dim(cx, *dim)).collect()
}

/// Concrete offset, extents and strides for the current artifact. Contiguous
/// layouts use `DERIVED_STRIDE` after a symbolic axis; derive those strides
/// from the now-concrete following extents just as session allocation does.
pub(crate) fn resolved_layout(
    cx: &LowerCtx<'_>,
    layout: &Layout,
) -> Result<(u32, Vec<u32>, Vec<u32>)> {
    let offset = resolve_dim(cx, layout.offset())?;
    let extents = const_extents(cx, layout.shape())?;
    let strides = layout
        .strides()
        .iter()
        .enumerate()
        .map(|(axis, stride)| match stride {
            Dim::Sym(symbol) if *symbol == DERIVED_STRIDE => extents[axis + 1..]
                .iter()
                .try_fold(1u32, |product, extent| product.checked_mul(*extent))
                .ok_or_else(|| Error::Legality("CPU derived stride exceeds u32 indexing".into())),
            other => resolve_dim(cx, *other),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((offset, extents, strides))
}

/// A masked load of operand `slot` at `index`.
pub(crate) fn load(buffer: Arc<BufferDecl>, index: TileExpr, mask: TileExpr) -> TileExpr {
    let element = buffer.element;
    let layout = buffer.layout.clone();
    let fill = match element {
        ElementType::Scalar(ScalarElement::U32) | ElementType::Scalar(ScalarElement::I32) => {
            lit_u32(0)
        }
        _ => lit_f32(0.0),
    };
    TileExpr::new(
        TileExprKind::Load {
            src: Source::Storage(StorageView {
                buffer,
                offset: 0,
                layout,
            }),
            addr: Box::new(Addr::Linear(index)),
            mask,
            fill,
        },
        element,
    )
}

/// One operand's value at `index`.
///
/// A `Leaf::Const` is folded into the kernel: `derive_bindings` never emits a
/// binding for one.
pub(crate) fn operand_value(
    cx: &LowerCtx<'_>,
    binds: &Binds,
    src: Id,
    index: TileExpr,
    mask: TileExpr,
) -> Result<TileExpr> {
    Ok(operand_src(cx, binds, src)?.at(index, mask))
}

/// One operand's value at the reading kernel's flat space index, mapped
/// through the edge's `layout`/`access`. [`operand_value`] is the raw form for
/// readers that have already computed a storage index themselves.
pub(crate) fn operand_at(
    cx: &LowerCtx<'_>,
    binds: &Binds,
    operand: &Operand,
    flat: TileExpr,
    space_total: u64,
    mask: TileExpr,
) -> Result<TileExpr> {
    operand_value(
        cx,
        binds,
        operand.src,
        address_of(cx, operand, flat, space_total)?,
        mask,
    )
}

/// A scatter's four extents, read off the base operand.
///
/// `Scatter::space` is the update iteration domain; the destination geometry
/// has to come from the base operand's own layout, or a scatter into a
/// 1024-row table from 300 tokens would size itself 300 rows.
pub(crate) struct ScatterGeometry {
    /// Product of the base extents before the scattered axis.
    pub outer: u32,
    /// Extent of the scattered axis in the base — the destination bins.
    pub bins: u32,
    /// Product of the base extents after the scattered axis.
    pub inner: u32,
    /// Index count.
    pub updates: u32,
}

pub(crate) fn scatter_geometry(
    cx: &LowerCtx<'_>,
    space: &fusor_ir::ir::launch::IndexSpace,
    axis: u32,
    ops: &[Operand],
) -> Result<ScatterGeometry> {
    let axis = axis as usize;
    let base = ops
        .first()
        .ok_or_else(|| Error::Legality("a scatter needs a base operand".into()))?;
    let dest = const_extents(cx, base.layout.shape())?;
    if axis >= dest.len() {
        return Err(Error::Legality(format!(
            "scatter axis {axis} is outside a rank-{} base",
            dest.len()
        )));
    }
    // The index operand is rank 1 and its element count is the update count;
    // reading it there keeps this correct whether `space` is the output space
    // or the update space.
    let idx = ops
        .get(1)
        .ok_or_else(|| Error::Legality("a scatter needs an index operand".into()))?;
    let updates = const_extents(cx, idx.layout.shape())?
        .iter()
        .product::<u32>();
    let _ = space;
    Ok(ScatterGeometry {
        outer: dest[..axis].iter().product::<u32>().max(1),
        bins: dest[axis].max(1),
        inner: dest[axis + 1..].iter().product::<u32>().max(1),
        updates: updates.max(1),
    })
}

/// `flat` run through one operand's [`AddressMap`].
pub(crate) fn address_of(
    cx: &LowerCtx<'_>,
    operand: &Operand,
    flat: TileExpr,
    space_total: u64,
) -> Result<TileExpr> {
    let map = resolved_address_map(cx, operand)?;
    if map.is_identity_over(space_total) {
        return Ok(flat);
    }
    let mut acc: Option<TileExpr> = (map.offset != 0).then(|| lit_u32(map.offset));
    for (i, t) in map.terms.iter().enumerate() {
        let mut e = flat.clone();
        if t.divisor > 1 {
            e = bin(BinOp::Div, e, lit_u32(t.divisor), u32_ty());
        }
        if map.needs_modulo(i, space_total) {
            e = bin(BinOp::Rem, e, lit_u32(t.modulus), u32_ty());
        }
        if t.stride != 1 {
            e = bin(BinOp::Mul, e, lit_u32(t.stride), u32_ty());
        }
        acc = Some(match acc {
            Some(a) => bin(BinOp::Add, a, e, u32_ty()),
            None => e,
        });
    }
    Ok(acc.unwrap_or_else(|| lit_u32(0)))
}

fn resolved_address_map(cx: &LowerCtx<'_>, operand: &Operand) -> Result<AddressMap> {
    let (offset, extents, strides) = resolved_layout(cx, &operand.layout)?;
    let groups: SmallVec<[AxisGroup; 4]> = match &operand.access {
        AccessPlan::Unflatten(map) => map.groups.clone(),
        _ => extents
            .into_iter()
            .zip(strides)
            .map(|(extent, stride)| AxisGroup::affine(extent, stride))
            .collect(),
    };

    let mut terms: SmallVec<[AddressTerm; 4]> = SmallVec::new();
    let mut div_after = 1u64;
    for group in groups.iter().rev() {
        let mut below = 1u64;
        for axis in group.sub_axes.iter().rev() {
            let divisor = div_after
                .checked_mul(below)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| Error::Legality("CPU operand divisor exceeds u32".into()))?;
            terms.push(AddressTerm {
                divisor,
                modulus: axis.extent,
                stride: axis.stride,
            });
            below = below
                .checked_mul(u64::from(axis.extent))
                .ok_or_else(|| Error::Legality("CPU operand extent product overflows".into()))?;
        }
        div_after = div_after
            .checked_mul(below)
            .ok_or_else(|| Error::Legality("CPU operand extent product overflows".into()))?;
    }
    terms.retain(|term| term.modulus > 1 && term.stride != 0);
    terms.sort_unstable_by(|left, right| right.divisor.cmp(&left.divisor));
    coalesce_address_terms(&mut terms);
    Ok(AddressMap { offset, terms })
}

fn coalesce_address_terms(terms: &mut SmallVec<[AddressTerm; 4]>) {
    let mut index = 0;
    while index + 1 < terms.len() {
        let (high, low) = (terms[index], terms[index + 1]);
        let joins = u64::from(low.divisor) * u64::from(low.modulus) == u64::from(high.divisor)
            && u64::from(low.stride) * u64::from(low.modulus) == u64::from(high.stride);
        if joins && low.modulus.checked_mul(high.modulus).is_some() {
            terms[index] = AddressTerm {
                divisor: low.divisor,
                modulus: low.modulus * high.modulus,
                stride: low.stride,
            };
            terms.remove(index + 1);
            index = index.saturating_sub(1);
        } else {
            index += 1;
        }
    }
}

/// Where one operand's elements come from: a bound buffer, or a constant the
/// kernel carries. Readers that index the same operand more than once resolve
/// it once through [`operand_src`] and call [`OperandSrc::at`] per use.
pub(crate) enum OperandSrc {
    Buffer(Arc<BufferDecl>),
    Const(TileExpr),
    /// A block-quantized operand. Reading element `i` runs the format's
    /// decode program at flat index `i`; nothing materializes the dense
    /// table.
    Quantized(QuantizedView),
}

impl OperandSrc {
    pub(crate) fn at(&self, index: TileExpr, mask: TileExpr) -> TileExpr {
        match self {
            Self::Buffer(b) => load(Arc::clone(b), index, mask),
            Self::Const(v) => v.clone(),
            Self::Quantized(view) => TileExpr::new(
                TileExprKind::Load {
                    src: Source::Quantized(view.clone()),
                    addr: Box::new(Addr::Linear(index)),
                    mask,
                    fill: lit_f32(0.0),
                },
                ElementType::Scalar(ScalarElement::F32),
            ),
        }
    }
}

pub(crate) fn operand_src(cx: &LowerCtx<'_>, binds: &Binds, src: Id) -> Result<OperandSrc> {
    if let Some(lit) = const_operand(cx, src) {
        return Ok(OperandSrc::Const(lit));
    }
    let buffer = binds.of(src)?;
    let facts = cx.graph.facts(src);
    if let Dtype::Q(fmt) = facts.dtype {
        let layout = qlayout_of(cx, src).unwrap_or(QLayout::Native);
        let data = StorageView {
            layout: buffer.layout.clone(),
            buffer,
            offset: 0,
        };
        return Ok(OperandSrc::Quantized(QuantizedView { data, fmt, layout }));
    }
    Ok(OperandSrc::Buffer(buffer))
}

/// The storage layout a quantized value carries, read off its `LeafKind`.
pub(crate) fn qlayout_of(cx: &LowerCtx<'_>, value: Id) -> Option<QLayout> {
    let class = cx.graph.class_of(value);
    cx.graph
        .class_ids(class)
        .into_iter()
        .find_map(|m| match &cx.graph.node(m).op {
            Op::Logical(fusor_ir::ir::logical::Logical::Leaf(
                fusor_ir::ir::logical::LeafKind::Quantized { layout, .. },
            )) => Some(*layout),
            _ => None,
        })
}

pub(crate) fn const_operand(cx: &LowerCtx<'_>, src: Id) -> Option<TileExpr> {
    let fusor_ir::ir::Op::Logical(fusor_ir::ir::logical::Logical::Leaf(
        fusor_ir::ir::logical::LeafKind::Const { value, .. },
    )) = &cx.graph.node(cx.selected(src)).op
    else {
        return None;
    };
    let (lit, elem) = match *value {
        fusor_ir::dtype::Splat::F32(v) => (TileLiteral::F32(v.to_bits()), ScalarElement::F32),
        fusor_ir::dtype::Splat::F16(v) => (TileLiteral::F16(v), ScalarElement::F16),
        fusor_ir::dtype::Splat::BF16(v) => (TileLiteral::BF16(v), ScalarElement::BF16),
        fusor_ir::dtype::Splat::U32(v) => (TileLiteral::U32(v), ScalarElement::U32),
        fusor_ir::dtype::Splat::I32(v) => (TileLiteral::I32(v), ScalarElement::I32),
    };
    Some(TileExpr::new(
        TileExprKind::Literal(lit),
        ElementType::Scalar(elem),
    ))
}

/// Translate one `ScalarExpr` body into Kernel, with `args[i]` supplying operand
/// `i` and `coords` supplying `IndexOf(axis)`.
pub(crate) struct Translate<'a> {
    pub args: &'a [TileExpr],
    pub coords: &'a [TileExpr],
    pub uniforms: Option<Arc<BufferDecl>>,
}

impl Translate<'_> {
    pub(crate) fn run(&self, e: &ScalarExpr) -> Result<TileExpr> {
        let ty = ElementType::Scalar(elem_of(e.dtype()).unwrap_or(ScalarElement::F32));
        Ok(match e.kind() {
            ScalarKind::Arg(i) => self
                .args
                .get(*i as usize)
                .cloned()
                .ok_or_else(|| Error::Legality(format!("Arg({i}) has no operand")))?,
            ScalarKind::Lit(l) => TileExpr::new(
                TileExprKind::Literal(match l.0 {
                    fusor_ir::dtype::Splat::F32(v) => TileLiteral::F32(v.to_bits()),
                    fusor_ir::dtype::Splat::F16(v) => TileLiteral::F16(v),
                    fusor_ir::dtype::Splat::BF16(v) => TileLiteral::BF16(v),
                    fusor_ir::dtype::Splat::U32(v) => TileLiteral::U32(v),
                    fusor_ir::dtype::Splat::I32(v) => TileLiteral::I32(v),
                }),
                ty,
            ),
            // A runtime scalar is read from the uniform block, never baked
            // into the kernel, so changing it does not recompile.
            ScalarKind::Uniform(sym) => {
                let ub = self
                    .uniforms
                    .clone()
                    .ok_or_else(|| Error::Legality("no uniform block bound".into()))?;
                let raw = load(ub, lit_u32(sym.0), lit_true());
                TileExpr::new(TileExprKind::Bitcast { value: raw, to: ty }, ty)
            }
            ScalarKind::IndexOf(axis) => self
                .coords
                .get(*axis as usize)
                .cloned()
                .ok_or_else(|| Error::Legality(format!("IndexOf({axis}) is out of range")))?,
            ScalarKind::Un { op, x } => TileExpr::new(
                TileExprKind::Unary {
                    op: *op,
                    value: self.run(x)?,
                    numeric: NumericContract::RELAXED,
                },
                ty,
            ),
            ScalarKind::Bin { op, a, b } => bin(*op, self.run(a)?, self.run(b)?, ty),
            ScalarKind::Cmp { op, a, b } => {
                // Booleans are 1.0/0.0 in the operand dtype at Logical, so a
                // comparison consumed as a value materializes here.
                let m = cmp(*op, self.run(a)?, self.run(b)?);
                TileExpr::new(
                    TileExprKind::Select {
                        condition: m,
                        accept: one_of(ty),
                        reject: zero_of(ty),
                    },
                    ty,
                )
            }
            ScalarKind::Select { c, t, f } => {
                let cond = cmp(fusor_ir::scalar::CmpOp::Ne, self.run(c)?, zero_of(ty));
                TileExpr::new(
                    TileExprKind::Select {
                        condition: cond,
                        accept: self.run(t)?,
                        reject: self.run(f)?,
                    },
                    ty,
                )
            }
            ScalarKind::Cast { to, x } => TileExpr::new(
                TileExprKind::Cast {
                    value: self.run(x)?,
                    to: ElementType::Scalar(elem_of(*to)?),
                },
                ty,
            ),
            ScalarKind::Bitcast { to, x } => TileExpr::new(
                TileExprKind::Bitcast {
                    value: self.run(x)?,
                    to: ElementType::Scalar(elem_of(*to)?),
                },
                ty,
            ),
            ScalarKind::Round { mode, x } => TileExpr::new(
                TileExprKind::Round {
                    mode: *mode,
                    value: self.run(x)?,
                },
                ty,
            ),
            ScalarKind::Dot { a, b } => TileExpr::new(
                TileExprKind::Dot {
                    left: self.run(a)?,
                    right: self.run(b)?,
                },
                ty,
            ),
            ScalarKind::Splat { lanes, x } => {
                let v = self.run(x)?;
                TileExpr::new(
                    TileExprKind::Vec {
                        scalar: elem_of(e.dtype())?,
                        lanes: *lanes,
                        parts: vec![v; *lanes as usize],
                    },
                    ElementType::Vector {
                        scalar: elem_of(e.dtype())?,
                        lanes: *lanes,
                    },
                )
            }
        })
    }
}

pub(crate) fn zero_of(ty: ElementType) -> TileExpr {
    match ty {
        ElementType::Scalar(ScalarElement::U32) | ElementType::Scalar(ScalarElement::I32) => {
            lit_u32(0)
        }
        _ => lit_f32(0.0),
    }
}

pub(crate) fn one_of(ty: ElementType) -> TileExpr {
    match ty {
        ElementType::Scalar(ScalarElement::U32) | ElementType::Scalar(ScalarElement::I32) => {
            lit_u32(1)
        }
        _ => lit_f32(1.0),
    }
}

/// Decompose a flat index into per-axis coordinates by the declared divmod
/// chain, most-significant-first.
pub(crate) fn coords_of(flat: &TileExpr, extents: &[u32]) -> Vec<TileExpr> {
    use fusor_ir::scalar::BinOp;
    let mut out = Vec::with_capacity(extents.len());
    for i in 0..extents.len() {
        let below: u32 = extents[i + 1..].iter().product::<u32>().max(1);
        let q = bin(BinOp::Div, flat.clone(), lit_u32(below), u32_ty());
        out.push(bin(BinOp::Rem, q, lit_u32(extents[i].max(1)), u32_ty()));
    }
    out
}

/// Grid extent for `n` work items at `block` lanes each.
pub(crate) fn grid_for(n: u64, block: u32) -> [u32; 3] {
    let groups = n.div_ceil(block as u64).max(1);
    [groups as u32, 1, 1]
}
