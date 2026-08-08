//! L1 node + `SchedPoint` -> `KernelIr` for the CPU backend. The same
//! `KernelIr`, a different emitter.
//!
//! Owned by W10.

pub mod contract;
pub mod gather_scatter;
pub mod map_fold;

use fusor2_ir::device::Caps;
use fusor2_ir::dtype::{Dtype, NumericContract, QLayout};
use fusor2_ir::egraph::Id;
use fusor2_ir::error::Error;
use fusor2_ir::ir::level1::{Family, L1, Operand, SchedPoint};
use fusor2_ir::ir::level2::{
    Addr, BufferAccess, BufferDecl, Builtin, ElementType, KernelIr, MemoryLevel, QuantizedView,
    ScalarElement, Source, StorageView, TileExpr, TileExprKind, TileLayout, TileLiteral,
    WorkgroupAxis,
};
use fusor2_ir::ir::{Node, Op};
use fusor2_ir::scalar::{BinOp, ScalarExpr, ScalarKind};
use fusor2_ir::shape::Dim;
use fusor2_ir::target::LowerCtx;
use fusor2_ir::Result;
use std::sync::Arc;

/// Lanes per workgroup for a node whose schedule point names no lane group.
///
/// Not a written-in tile: `fusor2_tile::domains::emitted_block` is the one
/// place the width is decided, shared with the fold domain that prices it and
/// with the GPU backend. One grid point is one workgroup; `block` lanes are
/// walked in chunks of the register width.
pub fn default_block(caps: &Caps) -> u32 {
    fusor2_tile::domains::emitted_block(1, caps)
}

pub fn lower(
    caps: &Caps,
    node: &Node,
    id: Id,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
) -> Result<KernelIr> {
    let _ = id;
    let Op::L1(op) = &node.op else {
        return Err(Error::Legality(
            "the CPU target can only lower L1 nodes".into(),
        ));
    };
    match op {
        L1::KMap { .. } | L1::KFold { .. } => map_fold::lower(caps, node, theta, cx),
        L1::KContract { family, .. } => {
            if *family == Family::Coop {
                // Caps report no cooperative config, so this alternative is
                // never selectable; refusing it is a legality answer, not a
                // fallback.
                return Err(Error::Legality(
                    "Family::Coop is not lowerable on the CPU target".into(),
                ));
            }
            contract::lower(caps, node, theta, cx)
        }
        L1::KGather { .. } | L1::KScatter { .. } => gather_scatter::lower(caps, node, theta, cx),
        L1::KRegion { members, .. } => compose(caps, members, theta, cx, "cpu_region"),
        L1::KMerged(wave) => compose(caps, wave.segments(), theta, cx, "cpu_merged"),
        L1::Ext { def, .. } => ext::lower(*def, node, theta),
    }
}

// ---------------------------------------------------------------------------
// Composite nodes: KRegion and KMerged
// ---------------------------------------------------------------------------

/// One dispatch running several member kernels.
///
/// `KRegion` and `KMerged` are the same shape to a backend: a list of L1 nodes
/// the plan asks for in **one** launch. Each member is lowered through the
/// ordinary dispatch above — so a merged wave of contractions is still the
/// contraction lowering, not a second copy of it — and the bodies are then
/// concatenated over one shared grid. That is exactly what a horizontal merge
/// is: one dispatch, several bodies, the same workgroup index.
///
/// Two things make the concatenation sound rather than a splice:
///
/// * **Each member's stores are redirected to that member's own buffer.** The
///   single-node lowerings all write `binds.of(cx.launch.root)`, because a
///   launch normally *is* one node. In a composite the root names the whole
///   launch, so a member that the plan bound a buffer for writes that buffer
///   and only the member standing for the composite's own value keeps the
///   root's.
/// * **A member whose own grid is shorter than the shared one is guarded.**
///   `map_fold::lower_fold` addresses its output by raw workgroup id with no
///   upper bound, because its grid is exactly its row count; running it over a
///   longer grid without the guard would write past the buffer.
///
/// Members must agree on their lane count. They do whenever the merge rules
/// mint the node — every segment of a wave shares a `MergeKey`, so every
/// segment lowers to the same `block` — and when they do not (a `KRegion` over
/// an elementwise producer and a fold, whose lane counts are 256 and the
/// reduced extent) this refuses with a legality answer rather than emitting a
/// kernel whose reduce scratch is shorter than its lane loop.
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
    // **The composite's own point.** `L1::KRegion` and `L1::KMerged` carry the
    // linear `MapDomain` of their members' shared index space, which describes
    // the *GPU* body — one guarded pass over a linearized index, `tm` outputs
    // per lane. This backend does not emit that body: it lowers each member
    // through the ordinary dispatch and concatenates, so a register tile at
    // the composite level has nowhere to land and every member already carries
    // its own tiling in its own `SchedPoint`.
    //
    // The untiled point is therefore the only one this lowering can honor, and
    // it is a genuine member of the node's domain rather than a geometry from
    // outside it. Nothing selects another: `tm` moves exactly one term of the
    // cost model — `realize::geometry` divides the workgroup count by it — and
    // fewer resident lanes never scores better than more, so a wider tile is
    // weakly dominated at every shape and ties break to domain index 0.
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
        // **Each member's own point, not the composite's.** A member is a
        // selected node in its own right and extraction scheduled it as one;
        // handing it the wave's point would give a `KFold` a `Map` tiling and
        // a `KContract` a point from a domain it never declared.
        let member_theta = cx
            .plan
            .extraction
            .theta
            .get(&selected)
            .copied()
            .unwrap_or(SchedPoint::Point);
        kernels.push((*m, lower(caps, node, selected, member_theta, cx)?));
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
    // already writes several distinct buffers (a scatter's base, a region's
    // extra output) keeps every one of them.
    let root_buffer = binds.of(cx.launch.root).ok();
    let mut body = Vec::new();
    for (id, kernel) in kernels {
        // The member that owns no buffer of its own is the one standing for
        // the composite's value; it keeps writing the launch root's buffer.
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
            stmts = vec![fusor2_ir::ir::level2::Stmt::If {
                condition: cmp(
                    fusor2_ir::scalar::CmpOp::Lt,
                    pid,
                    lit_u32(kernel.grid[0]),
                ),
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
    stmts: &mut [fusor2_ir::ir::level2::Stmt],
    from: Option<&Arc<BufferDecl>>,
    view: &StorageView,
) {
    use fusor2_ir::ir::level2::Stmt;
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

// ---------------------------------------------------------------------------
// The extension seam
// ---------------------------------------------------------------------------

/// `L1::Ext` lowering: the one escape hatch out of the closed `L0`/`L1` enums.
pub mod ext {
    use super::*;
    use fusor2_ir::ir::{OpDefId, OpDefRegistry};
    use std::sync::RwLock;

    /// The registry `L1::Ext` lowering resolves `OpDefId` against.
    ///
    /// [`LowerCtx`] carries the plan, the launch, the graph and the symbol
    /// list — not the [`OpDefRegistry`] the graph was built with — and
    /// `Semantics`, which *does* hold one, exposes no accessor for it. So a
    /// target handed one selected `L1::Ext { def }` has no way to reach that
    /// def's `lower_per_target` row, and `OpDef::lower_per_target` has been
    /// dead for exactly that reason.
    ///
    /// Until `LowerCtx` grows the field, the embedder installs the same
    /// registry here that it installed on the e-graph's semantics. Registration
    /// order is id order and must match, which is the same contract
    /// `CoreSemantics::with_registry` already imposes.
    static DEFS: RwLock<Option<OpDefRegistry>> = RwLock::new(None);

    /// Install the extension registry this process lowers against. Idempotent
    /// and last-write-wins; a second install with a differently ordered
    /// registry would silently rename every `OpDefId`, so callers pass the
    /// registry the graph was built with, unchanged.
    pub fn install(registry: OpDefRegistry) {
        *DEFS.write().expect("the OpDef registry lock is poisoned") = Some(registry);
    }

    /// The installed registry, if the embedder installed one.
    pub fn installed() -> Option<OpDefRegistry> {
        DEFS.read()
            .expect("the OpDef registry lock is poisoned")
            .clone()
    }

    /// Lower one registered extension op through its `"cpu"` row.
    pub fn lower(def: OpDefId, node: &Node, theta: SchedPoint) -> Result<KernelIr> {
        let registry = installed().ok_or_else(|| {
            Error::Legality(format!(
                "{def:?} is an extension op, but no OpDefRegistry is installed on the \
                 CPU target; call fusor2_cpu::lower::ext::install"
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

// ---------------------------------------------------------------------------
// Shared construction helpers
// ---------------------------------------------------------------------------

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
    op: fusor2_ir::scalar::BinOp,
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

pub(crate) fn cmp(op: fusor2_ir::scalar::CmpOp, a: TileExpr, b: TileExpr) -> TileExpr {
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
        fusor2_ir::scalar::BinOp::Add,
        bin(
            fusor2_ir::scalar::BinOp::Mul,
            pid,
            lit_u32(block),
            u32_ty(),
        ),
        lane,
        u32_ty(),
    )
}

/// Hand back **the same `Arc`** for two structurally equal buffer decls.
///
/// `emit::buffer_of` resolves a `StorageView` to a binding slot by
/// `Arc::ptr_eq`, so a view built from one `Binds` cannot be emitted into a
/// `KernelIr` whose table was built by a second, structurally identical
/// `Binds`. That is exactly what happens the moment one launch holds more than
/// one node — a `KRegion` or a `KMerged` wave, where every member lowers
/// through its own `Binds::build(cx)`. Interning makes identity follow
/// content, which is the property the emitter is really asking about.
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
    pub fn build(cx: &LowerCtx<'_>) -> Result<Self> {
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
            // count of its blocks rather than an element count it has no
            // dense element type for.
            let (element, extents) = match facts.dtype {
                Dtype::Q(fmt) => {
                    let layout = qlayout_of(cx, b.value).unwrap_or(QLayout::Native);
                    let elems: u64 = const_extents(&facts.shape)?
                        .iter()
                        .map(|e| *e as u64)
                        .product();
                    let blocks = elems.div_ceil(u64::from(fmt.block_elements()).max(1));
                    let words = (blocks * u64::from(fmt.block_bytes(layout))).div_ceil(4);
                    (u32_ty(), vec![words as u32])
                }
                d => (
                    ElementType::Scalar(elem_of(d)?),
                    const_extents(&facts.shape)?,
                ),
            };
            let access = match b.kind {
                fusor2_ir::extract::BindKind::Read => BufferAccess::Read,
                _ => BufferAccess::ReadWrite,
            };
            // Keyed by every id in the value's class, not only by the
            // selected one: an `Operand::src` names whichever id the rule
            // author wrote, and they all denote the same buffer. `class_ids`
            // rather than `chain`, because `chain` is the *selectable* set and
            // drops the `Union` spine — and a macro op hands its caller the
            // spine node, so `rope`, `attention` and every adjoint over them
            // name their operands by one.
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

    pub fn of(&self, value: Id) -> Result<Arc<BufferDecl>> {
        let idx = self
            .by_value
            .iter()
            .find(|(v, _)| *v == value)
            .map(|(_, i)| *i)
            .ok_or_else(|| Error::Legality(format!("value {value} has no binding in this launch")))?;
        self.buffers
            .iter()
            .find(|b| b.binding as usize == idx)
            .cloned()
            .ok_or_else(|| Error::Legality(format!("binding {idx} is missing")))
    }
}

pub(crate) fn const_extents(shape: &[Dim]) -> Result<Vec<u32>> {
    shape
        .iter()
        .map(|d| {
            d.as_const().map(|v| v as u32).ok_or_else(|| {
                Error::Legality(
                    "the CPU lowering path needs concrete extents; a symbolic dim must be \
                     specialized or bound through the uniform block first"
                        .into(),
                )
            })
        })
        .collect()
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
/// A `Leaf::Const` is **folded into the kernel**: no buffer, no binding, no
/// traffic — which is exactly what `LeafRole::Free` means in the plan, so
/// `derive_bindings` never emits a binding for one and loading it would look
/// up a key that deliberately does not exist.
pub(crate) fn operand_value(
    cx: &LowerCtx<'_>,
    binds: &Binds,
    src: Id,
    index: TileExpr,
    mask: TileExpr,
) -> Result<TileExpr> {
    Ok(operand_src(cx, binds, src)?.at(index, mask))
}

/// One operand's value at the reading kernel's **flat space index**.
///
/// The edge's `layout`/`access` is what says which storage element that is:
/// a stride-0 broadcast axis, a transposed view, a narrowed slice and a conv
/// window all disagree with the bare flat index. [`operand_value`] is the raw
/// form for readers that have already computed a storage index themselves
/// (gather, scatter, the contraction nests).
pub(crate) fn operand_at(
    cx: &LowerCtx<'_>,
    binds: &Binds,
    operand: &Operand,
    flat: TileExpr,
    space_total: u64,
    mask: TileExpr,
) -> Result<TileExpr> {
    operand_value(cx, binds, operand.src, address_of(operand, flat, space_total)?, mask)
}

/// A scatter's four extents, read off the *base operand* rather than off the
/// index space.
///
/// `KScatter::space` is the **update** iteration domain: rank of the base, but
/// the scattered axis carries the index count. The destination geometry has to
/// come from the base operand's own layout, or a scatter into a 1024-row table
/// from 300 tokens would size itself 300 rows.
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
    space: &fusor2_ir::ir::level1::IndexSpace,
    axis: u32,
    ops: &[Operand],
) -> Result<ScatterGeometry> {
    let axis = axis as usize;
    let base = ops
        .first()
        .ok_or_else(|| Error::Legality("a scatter needs a base operand".into()))?;
    // The operand's layout is the edge's view of the base; its shape is the
    // destination shape whatever access the edge carries.
    let dest = const_extents(base.layout.shape())?;
    if axis >= dest.len() {
        return Err(Error::Legality(format!(
            "scatter axis {axis} is outside a rank-{} base",
            dest.len()
        )));
    }
    // The index operand is rank 1 and its element count *is* the update
    // count. Reading it there rather than off `space` is what keeps this
    // correct under both minting conventions: `rules::lower_floor` hands the
    // output space and `fusor2_tile::rules::scatter` hands the update space.
    let idx = ops
        .get(1)
        .ok_or_else(|| Error::Legality("a scatter needs an index operand".into()))?;
    let updates = const_extents(idx.layout.shape())?.iter().product::<u32>();
    let _ = (cx, space);
    Ok(ScatterGeometry {
        outer: dest[..axis].iter().product::<u32>().max(1),
        bins: dest[axis].max(1),
        inner: dest[axis + 1..].iter().product::<u32>().max(1),
        updates: updates.max(1),
    })
}

/// `flat` run through one operand's [`AddressMap`].
pub(crate) fn address_of(operand: &Operand, flat: TileExpr, space_total: u64) -> Result<TileExpr> {
    let map = operand.address_map().ok_or_else(|| {
        Error::Legality(
            "the CPU lowering path needs a decidable operand index map; a symbolic \
             stride must be specialized or bound through the uniform block first"
                .into(),
        )
    })?;
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
///
/// Layout is an operand attribute the extractor prices, never a device
/// branch, so it is recovered from the leaf rather than assumed.
pub(crate) fn qlayout_of(cx: &LowerCtx<'_>, value: Id) -> Option<QLayout> {
    let class = cx.graph.class_of(value);
    cx.graph.class_ids(class).into_iter().find_map(|m| {
        match &cx.graph.node(m).op {
            Op::L0(fusor2_ir::ir::level0::L0::Leaf(
                fusor2_ir::ir::level0::LeafKind::Quantized { layout, .. },
            )) => Some(*layout),
            _ => None,
        }
    })
}

pub(crate) fn const_operand(cx: &LowerCtx<'_>, src: Id) -> Option<TileExpr> {
    let fusor2_ir::ir::Op::L0(fusor2_ir::ir::level0::L0::Leaf(
        fusor2_ir::ir::level0::LeafKind::Const { value, .. },
    )) = &cx.graph.node(cx.selected(src)).op
    else {
        return None;
    };
    let (lit, elem) = match *value {
        fusor2_ir::dtype::Splat::F32(v) => (TileLiteral::F32(v.to_bits()), ScalarElement::F32),
        fusor2_ir::dtype::Splat::F16(v) => (TileLiteral::F16(v), ScalarElement::F16),
        fusor2_ir::dtype::Splat::BF16(v) => (TileLiteral::BF16(v), ScalarElement::BF16),
        fusor2_ir::dtype::Splat::U32(v) => (TileLiteral::U32(v), ScalarElement::U32),
        fusor2_ir::dtype::Splat::I32(v) => (TileLiteral::I32(v), ScalarElement::I32),
    };
    Some(TileExpr::new(
        TileExprKind::Literal(lit),
        ElementType::Scalar(elem),
    ))
}

/// Translate one `ScalarExpr` body into L2, with `args[i]` supplying operand
/// `i` and `coords` supplying `IndexOf(axis)`.
pub(crate) struct Translate<'a> {
    pub args: &'a [TileExpr],
    pub coords: &'a [TileExpr],
    pub uniforms: Option<Arc<BufferDecl>>,
}

impl Translate<'_> {
    pub fn run(&self, e: &ScalarExpr) -> Result<TileExpr> {
        let ty = ElementType::Scalar(elem_of(e.dtype()).unwrap_or(ScalarElement::F32));
        Ok(match e.kind() {
            ScalarKind::Arg(i) => self
                .args
                .get(*i as usize)
                .cloned()
                .ok_or_else(|| Error::Legality(format!("Arg({i}) has no operand")))?,
            ScalarKind::Lit(l) => TileExpr::new(
                TileExprKind::Literal(match l.0 {
                    fusor2_ir::dtype::Splat::F32(v) => TileLiteral::F32(v.to_bits()),
                    fusor2_ir::dtype::Splat::F16(v) => TileLiteral::F16(v),
                    fusor2_ir::dtype::Splat::BF16(v) => TileLiteral::BF16(v),
                    fusor2_ir::dtype::Splat::U32(v) => TileLiteral::U32(v),
                    fusor2_ir::dtype::Splat::I32(v) => TileLiteral::I32(v),
                }),
                ty,
            ),
            // A runtime scalar is read from the uniform block, never baked
            // into the kernel: that is what deletes the trainer's `[1]`-tensor
            // scalars and its per-step recompiles.
            ScalarKind::Uniform(sym) => {
                let ub = self
                    .uniforms
                    .clone()
                    .ok_or_else(|| Error::Legality("no uniform block bound".into()))?;
                let raw = load(ub, lit_u32(sym.0), lit_true());
                TileExpr::new(
                    TileExprKind::Bitcast { value: raw, to: ty },
                    ty,
                )
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
                // Booleans are 1.0/0.0 in the operand dtype at L0, so a
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
                let cond = cmp(
                    fusor2_ir::scalar::CmpOp::Ne,
                    self.run(c)?,
                    zero_of(ty),
                );
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
    use fusor2_ir::scalar::BinOp;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coords_decompose_a_flat_index() {
        // Structural check: three axes yield three divmod expressions.
        let flat = lit_u32(0);
        let c = coords_of(&flat, &[2, 3, 4]);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn grid_covers_every_element() {
        assert_eq!(grid_for(1, 256), [1, 1, 1]);
        assert_eq!(grid_for(256, 256), [1, 1, 1]);
        assert_eq!(grid_for(257, 256), [2, 1, 1]);
        assert_eq!(grid_for(0, 256), [1, 1, 1]);
    }

    #[test]
    fn a_symbolic_extent_is_a_legality_answer_not_a_panic() {
        let e = const_extents(&[Dim::Sym(fusor2_ir::shape::SymId(0))]);
        assert!(matches!(e, Err(Error::Legality(_))));
    }
}

/// End-to-end cover for the three node kinds this target used to refuse:
/// `L1::Ext`, `KRegion` and `KMerged`. Every case lowers, compiles, runs on
/// the worker pool and asserts the bytes that came back.
#[cfg(test)]
mod composite_tests {
    use super::*;
    use fusor2_ir::device::Caps;
    use fusor2_ir::dtype::Persistence;
    use fusor2_ir::egraph::EGraph;
    use fusor2_ir::extract::{BindKind, BindingPlan, Extraction, Launch, Plan, PlanHash};
    use fusor2_ir::facts::{ValueFacts, Work};
    use fusor2_ir::ir::level0::{BufferId, L0, LeafKind};
    use fusor2_ir::ir::level1::{
        AccessPlan, Effect, IndexSpace, KMerged, MergeKey, MergeSegment, ScheduleDomain, WaveCat,
    };
    use fusor2_ir::ir::level2::Stmt;
    use fusor2_ir::ir::{AttrId, OpDef, OpDefId, OpDefRegistry, OpTag, VerifyCtx};
    use fusor2_ir::scalar::{BinOp, CmpOp};
    use fusor2_ir::semantics::{CoreSemantics, SumArenaPlanner};
    use fusor2_ir::shape::Layout;
    use fusor2_ir::target::{Buf, Target};

    /// The width these hand-built fixtures use. Production reads it from
    /// `default_block(caps)`; a fixture only needs a legal number.
    const BLOCK: u32 = 256;

    use crate::alloc::AlignedBuf;
    use crate::target::CpuTarget;

    fn f32_ty() -> ElementType {
        ElementType::Scalar(ScalarElement::F32)
    }

    fn graph() -> EGraph {
        EGraph::new(CoreSemantics::new(Arc::new(SumArenaPlanner)))
    }

    fn buffer(g: &mut EGraph, n: u64) -> Id {
        let next = g.len() as u32;
        g.add(Op::L0(L0::Leaf(LeafKind::Buffer {
            name: BufferId(next),
            dtype: Dtype::F32,
            shape: smallvec::smallvec![Dim::Const(n)],
        })))
        .unwrap()
    }

    fn alias(g: &EGraph, src: Id) -> Operand {
        Operand {
            src,
            layout: Layout::contiguous(&g.facts(src).shape),
            access: AccessPlan::Alias,
        }
    }

    fn kmap(g: &mut EGraph, n: u64, body: ScalarExpr, x: Id) -> Id {
        let ops = vec![alias(g, x)];
        g.add(Op::L1(L1::KMap {
            space: IndexSpace::new([Dim::Const(n)]),
            body,
            ops,
            sched: ScheduleDomain::Point,
        }))
        .unwrap()
    }

    fn plan_for(root: Id, bindings: Vec<BindingPlan>) -> Plan {
        Plan {
            extraction: Extraction::default(),
            launches: vec![Launch {
                root,
                members: smallvec::smallvec![root],
                bindings,
                grid: [1, 1, 1],
                block: BLOCK,
            }],
            buffers: Vec::new(),
            symbols: Vec::new(),
            hash: PlanHash(0),
            cost: fusor2_ir::cost::Picoseconds(0),
        }
    }

    fn read(target: &CpuTarget, bytes: u64, data: &[f32]) -> Buf {
        let buf = target.alloc(bytes, Persistence::Step).unwrap();
        {
            let raw = buf.downcast_ref::<AlignedBuf>().unwrap();
            // SAFETY: nothing else holds this buffer yet; the pool handed it
            // back because its refcount was one.
            let slice = unsafe {
                std::slice::from_raw_parts_mut(raw.as_mut_ptr(), raw.len())
            };
            slice.fill(0);
            for (i, v) in data.iter().enumerate() {
                slice[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    fn back(buf: &Buf, n: usize) -> Vec<f32> {
        let raw = buf.downcast_ref::<AlignedBuf>().unwrap();
        raw.as_slice()[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn run(target: &CpuTarget, ir: &KernelIr, binds: &[Buf]) {
        let artifact = target.emit(ir).unwrap();
        target
            .launch(&artifact, ir.grid, binds, &Default::default())
            .unwrap();
    }

    // -- L1::Ext ----------------------------------------------------------

    /// A registered extension op's own `"cpu"` lowering: `y = 3 * x`.
    ///
    /// It builds its whole `KernelIr` — buffer decls included — from the node,
    /// which is the entire contract `OpDef::lower_per_target` offers. Nothing
    /// in `fusor2-cpu` knows what "triple" means.
    ///
    /// `theta` is checked rather than ignored: `L1::schedule()` returns `None`
    /// for `Ext` — fusor2 cannot enumerate geometries for a lowering it did
    /// not write — so `SchedPoint::Point` is the only point an extension op
    /// can be handed, and anything else means extraction scheduled a node
    /// against a domain that does not exist.
    fn lower_triple(node: &Node, theta: &SchedPoint) -> Result<KernelIr> {
        if !matches!(theta, SchedPoint::Point) {
            return Err(Error::Legality(format!(
                "an L1::Ext node declares no schedule domain, so {theta:?} names \
                 a geometry nothing could have selected"
            )));
        }
        let Op::L1(L1::Ext { ops, .. }) = &node.op else {
            return Err(Error::Legality("triple got a foreign node".into()));
        };
        let shape = ops
            .first()
            .ok_or_else(|| Error::Legality("triple needs an operand".into()))?
            .layout
            .shape();
        let n: u32 = shape
            .iter()
            .map(|d| d.as_const().unwrap_or(1) as u32)
            .product();
        let decl = |binding: u32, element: ElementType, len: u32, access| {
            Arc::new(BufferDecl {
                binding,
                element,
                layout: TileLayout::contiguous(MemoryLevel::Storage, &[len]),
                access,
            })
        };
        let uniforms = decl(0, u32_ty(), 1, BufferAccess::Read);
        let input = decl(1, f32_ty(), n, BufferAccess::Read);
        let output = decl(2, f32_ty(), n, BufferAccess::ReadWrite);

        let flat = global_lane(BLOCK);
        let mask = cmp(CmpOp::Lt, flat.clone(), lit_u32(n));
        let x = load(Arc::clone(&input), flat.clone(), mask.clone());
        let value = bin(BinOp::Mul, x, lit_f32(3.0), f32_ty());
        let body = vec![Stmt::Store {
            dst: StorageView {
                layout: output.layout.clone(),
                buffer: Arc::clone(&output),
                offset: 0,
            },
            addr: Addr::Linear(flat),
            value,
            mask,
        }];
        Ok(KernelIr {
            buffers: vec![uniforms, input, output],
            grid: grid_for(n as u64, BLOCK),
            block: BLOCK,
            body,
            byte_arena: None,
            name: "triple",
        })
    }

    fn infer_first(ins: &[ValueFacts]) -> fusor2_ir::Result<ValueFacts> {
        ins.first()
            .cloned()
            .ok_or_else(|| Error::Shape("an extension op needs an operand".into()))
    }
    fn work_per_element(_ins: &[ValueFacts], out: &ValueFacts) -> Work {
        let n = out.elements().unwrap_or(1);
        Work {
            macs: n,
            transcendentals: 0,
            index_ops: n,
            wg_bytes: 0,
        }
    }
    fn verify_ok(_cx: &VerifyCtx<'_>) -> fusor2_ir::Result<()> {
        Ok(())
    }

    /// Id 0 lowers on the CPU; id 1 names only another target.
    fn triple_registry() -> OpDefRegistry {
        let mut registry = OpDefRegistry::new();
        registry.register(OpDef {
            name: "triple",
            tag: OpTag::Ext,
            verify: verify_ok,
            infer: infer_first,
            work: work_per_element,
            adjoint: None,
            lower_per_target: &[("cpu", lower_triple)],
            effect: Effect::Pure,
        });
        registry.register(OpDef {
            name: "gpu_only",
            tag: OpTag::Ext,
            verify: verify_ok,
            infer: infer_first,
            work: work_per_element,
            adjoint: None,
            lower_per_target: &[],
            effect: Effect::Pure,
        });
        registry
    }

    #[test]
    fn a_registered_op_def_lowers_and_runs() {
        let registry = triple_registry();
        ext::install(registry.clone());
        let mut g = EGraph::new(CoreSemantics::with_registry(
            Arc::new(SumArenaPlanner),
            registry,
        ));
        let x = buffer(&mut g, 8);
        let ops = vec![alias(&g, x)];
        let e = g
            .add(Op::L1(L1::Ext {
                def: OpDefId(0),
                ops,
                attrs: AttrId(0),
            }))
            .unwrap();

        let plan = plan_for(
            e,
            vec![
                BindingPlan {
                    binding: 1,
                    value: x,
                    kind: BindKind::Read,
                },
                BindingPlan {
                    binding: 2,
                    value: e,
                    kind: BindKind::Write,
                },
            ],
        );
        let cx = LowerCtx {
            plan: &plan,
            launch: &plan.launches[0],
            graph: &g,
            symbols: &[],
        };
        let caps = Caps::clone(crate::caps::cpu_caps());
        let ir = lower(&caps, g.node(e), e, SchedPoint::Point, &cx).unwrap();
        assert_eq!(ir.name, "triple", "the OpDef's own lowering must run");

        let target = CpuTarget::new().unwrap();
        let input = read(&target, 32, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let output = read(&target, 32, &[0.0; 8]);
        run(&target, &ir, &[input, output.clone()]);
        assert_eq!(
            back(&output, 8),
            vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]
        );

        // And the negative half, in the same test because the registry is one
        // process-global: an op that names no `"cpu"` row is a legality answer
        // naming the op, never a panic and never a silent fallback.
        let other = Node {
            op: Op::L1(L1::Ext {
                def: OpDefId(1),
                ops: vec![alias(&g, x)],
                attrs: AttrId(0),
            }),
            level: fusor2_ir::ir::Level::L1,
            children: smallvec::smallvec![x],
        };
        let err = lower(&caps, &other, x, SchedPoint::Point, &cx).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("gpu_only") && msg.contains("cpu"), "{msg}");
    }

    // -- KRegion and KMerged ----------------------------------------------

    /// Two elementwise members, two live outputs, one dispatch. This is the
    /// multi-output region the architecture calls "the same rewrite as
    /// producer inlining, differing only in that it emits an extra buffer".
    /// The domain a composite landing `member`'s value carries — the same
    /// call `rules::merge` mints it with and `verify_l1` checks it against.
    fn region_sched(g: &EGraph, member: Id) -> ScheduleDomain {
        ScheduleDomain::Map(fusor2_ir::ir::level1::MapDomain::linear_over(
            crate::caps::cpu_caps(),
            &g.facts(member).shape,
        ))
    }

    fn two_member_graph(g: &mut EGraph) -> (Id, Id, Id) {
        let x = buffer(g, 8);
        let doubled = kmap(
            g,
            8,
            ScalarExpr::bin(
                BinOp::Mul,
                ScalarExpr::arg(0, Dtype::F32),
                ScalarExpr::lit(fusor2_ir::dtype::Splat::F32(2.0)),
            ),
            x,
        );
        let plus_one = kmap(
            g,
            8,
            ScalarExpr::bin(
                BinOp::Add,
                ScalarExpr::arg(0, Dtype::F32),
                ScalarExpr::lit(fusor2_ir::dtype::Splat::F32(1.0)),
            ),
            x,
        );
        (x, doubled, plus_one)
    }

    fn composite_bindings(x: Id, root: Id, a: Id, b: Id) -> Vec<BindingPlan> {
        vec![
            BindingPlan {
                binding: 1,
                value: x,
                kind: BindKind::Read,
            },
            BindingPlan {
                binding: 2,
                value: root,
                kind: BindKind::Write,
            },
            BindingPlan {
                binding: 3,
                value: a,
                kind: BindKind::Write,
            },
            BindingPlan {
                binding: 4,
                value: b,
                kind: BindKind::Write,
            },
        ]
    }

    fn run_composite(g: &EGraph, root: Id, x: Id, a: Id, b: Id) -> (Vec<f32>, Vec<f32>, usize) {
        let plan = plan_for(root, composite_bindings(x, root, a, b));
        let cx = LowerCtx {
            plan: &plan,
            launch: &plan.launches[0],
            graph: g,
            symbols: &[],
        };
        let caps = Caps::clone(crate::caps::cpu_caps());
        let ir = lower(&caps, g.node(root), root, SchedPoint::Point, &cx).unwrap();
        let stores = count_stores(&ir.body);

        let target = CpuTarget::new().unwrap();
        let input = read(&target, 32, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let root_buf = read(&target, 32, &[0.0; 8]);
        let a_buf = read(&target, 32, &[0.0; 8]);
        let b_buf = read(&target, 32, &[0.0; 8]);
        run(
            &target,
            &ir,
            &[input, root_buf, a_buf.clone(), b_buf.clone()],
        );
        (back(&a_buf, 8), back(&b_buf, 8), stores)
    }

    fn count_stores(stmts: &[Stmt]) -> usize {
        stmts
            .iter()
            .map(|s| match s {
                Stmt::Store { .. } => 1,
                Stmt::If { accept, reject, .. } => count_stores(accept) + count_stores(reject),
                Stmt::Loop { body, .. } => count_stores(body),
                _ => 0,
            })
            .sum()
    }

    #[test]
    fn a_kregion_runs_every_member_into_its_own_buffer_in_one_dispatch() {
        let mut g = graph();
        let (x, doubled, plus_one) = two_member_graph(&mut g);
        let region = g
            .add(Op::L1(L1::KRegion {
                members: smallvec::smallvec![doubled, plus_one],
                live_outs: smallvec::smallvec![0, 1],
                sched: region_sched(&g, doubled),
            }))
            .unwrap();
        let (a, b, stores) = run_composite(&g, region, x, doubled, plus_one);
        assert_eq!(a, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        assert_eq!(b, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(stores, 2, "one region, two live outputs, one dispatch");
    }

    #[test]
    fn a_kmerged_wave_runs_every_segment_in_one_dispatch() {
        let mut g = graph();
        let (x, doubled, plus_one) = two_member_graph(&mut g);
        let key = MergeKey {
            m: Dim::Const(8),
            n: Dim::Const(1),
            k: Dim::Const(1),
            batch: Dim::Const(1),
            splits: 1,
            dtype: Dtype::F32,
            family: fusor2_ir::ir::level1::Family::Sgemv,
        };
        let wave = KMerged::new(
            WaveCat::Row,
            [
                MergeSegment {
                    id: doubled,
                    key,
                    has_epilogue: false,
                },
                MergeSegment {
                    id: plus_one,
                    key,
                    has_epilogue: false,
                },
            ],
            region_sched(&g, doubled),
        )
        .unwrap();
        let merged = g.add(Op::L1(L1::KMerged(wave))).unwrap();
        let (a, b, stores) = run_composite(&g, merged, x, doubled, plus_one);
        assert_eq!(a, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        assert_eq!(b, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(stores, 2, "two segments, one dispatch");
    }

    /// A composite whose members disagree on their lane count has no
    /// single-kernel CPU lowering, and says so instead of emitting a kernel
    /// whose reduce scratch is shorter than its lane loop.
    #[test]
    fn mismatched_lane_counts_are_a_legality_answer() {
        let mut g = graph();
        let x = buffer(&mut g, 8);
        let m = kmap(&mut g, 8, ScalarExpr::arg(0, Dtype::F32), x);
        let ops = vec![alias(&g, x)];
        let f = g
            .add(Op::L1(L1::KFold {
                space: IndexSpace::new([Dim::Const(8)]),
                axis: 0,
                vec_axes: smallvec::smallvec![],
                carrier: fusor2_ir::carrier::Carrier::binop(
                    fusor2_ir::scalar::BinOp::Add,
                    fusor2_ir::dtype::Splat::F32(0.0),
                    Dtype::F32,
                ),
                acc: Dtype::F32,
                post: smallvec::smallvec![ScalarExpr::arg(0, Dtype::F32)],
                ops,
                sched: ScheduleDomain::Point,
            }))
            .unwrap();
        let region = g
            .add(Op::L1(L1::KRegion {
                members: smallvec::smallvec![m, f],
                live_outs: smallvec::smallvec![0],
                sched: region_sched(&g, m),
            }))
            .unwrap();
        let plan = plan_for(region, composite_bindings(x, region, m, f));
        let cx = LowerCtx {
            plan: &plan,
            launch: &plan.launches[0],
            graph: &g,
            symbols: &[],
        };
        let caps = Caps::clone(crate::caps::cpu_caps());
        let err = lower(&caps, g.node(region), region, SchedPoint::Point, &cx).unwrap_err();
        assert!(
            matches!(err, Error::Legality(ref m) if m.contains("lane count")),
            "{err}"
        );
    }
}
