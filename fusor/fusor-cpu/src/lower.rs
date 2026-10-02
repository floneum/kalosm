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
    ScalarElement, Source, Stmt, StorageView, TileExpr, TileExprKind, TileLayout, WorkgroupAxis,
};
use fusor_ir::ir::launch::{AddressMap, Family, Launch, Operand, SchedPoint};
use fusor_ir::ir::{Node, Op};
use fusor_ir::scalar::{ScalarExpr, ScalarKind};
use fusor_ir::shape::{Dim, Layout};
use fusor_ir::target::LowerCtx;
use fusor_tile::build::{
    Kernel, const_splat, qlayout_of, quantized_words, scalar_element, splat_literal,
};
use std::sync::Arc;

/// Lanes per workgroup for a node whose schedule point names no lane group.
/// One grid point is one workgroup; `block` lanes are walked in chunks of the
/// register width. A CPU "block" is an internal native loop chunk, not a GPU
/// workgroup capability, so no schedule alternative spans it.
pub(crate) const DEFAULT_BLOCK: u32 = 256;

pub(crate) fn lower(
    caps: &Caps,
    node: &Node,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
) -> Result<KernelIr> {
    let Op::Launch(op) = &node.op else {
        return Err(Error::Legality(
            "the CPU target can only lower Launch nodes".into(),
        ));
    };
    match op {
        Launch::Map { .. } | Launch::Fold { .. } | Launch::StreamFold { .. } => {
            map_fold::lower(caps, node, theta, cx)
        }
        Launch::Contract { family, .. } => {
            if *family == Family::Coop {
                // Caps report no cooperative config, so this alternative is never selectable
                return Err(Error::Legality(
                    "Family::Coop is not lowerable on the CPU target".into(),
                ));
            }
            contract::lower(node, cx)
        }
        Launch::Gather { .. } | Launch::Scatter { .. } => gather_scatter::lower(node, theta, cx),
        // Members are read by id: the last one shares the slab's class, and
        // selecting it would lower the slab again.
        Launch::Slab { members, .. } => compose(caps, members, theta, cx, "cpu_slab"),
        Launch::Group { members, .. } => compose(caps, members, theta, cx, "cpu_group"),
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
        let selected = *m;
        let node = cx.graph.node(selected);
        // Each member is scheduled at its own point, not the composite's.
        let member_theta = cx
            .plan
            .extraction
            .theta
            .get(&selected)
            .copied()
            .unwrap_or(SchedPoint::Point);
        kernels.push((*m, lower(caps, node, member_theta, cx)?));
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
    let b = Kernel::new();
    let mut body = Vec::new();
    for (id, kernel) in kernels {
        // A member with no buffer of its own stands for the composite's value
        // and keeps writing the launch root's buffer.
        let own = binds.of(id).ok().map(|buffer| view(&buffer));
        let mut stmts = kernel.body;
        if let Some(view) = own {
            redirect_stores(&mut stmts, root_buffer.as_ref(), &view);
        }
        if kernel.grid[0] < grid[0] {
            let pid = b.builtin(Builtin::ProgramId(WorkgroupAxis::X));
            stmts = vec![Stmt::If {
                condition: b.lt(pid, b.u32(kernel.grid[0])),
                accept: stmts,
                reject: Vec::new(),
            }];
        }
        body.extend(stmts);
    }

    Ok(binds.finish(name, grid, block, body))
}

/// Point every store aimed at `from` (the launch root's buffer) at `view`
/// instead, leaving addresses, masks and values alone. With `from` absent —
/// the root owns no buffer — every store moves.
fn redirect_stores(stmts: &mut [Stmt], from: Option<&Arc<BufferDecl>>, view: &StorageView) {
    Stmt::walk_mut(stmts, &mut |s| {
        if let Stmt::Store { dst, .. } | Stmt::AtomicAdd { dst, .. } | Stmt::CoopStore { dst, .. } =
            s
            && from.is_none_or(|root| Arc::ptr_eq(&dst.buffer, root))
        {
            *dst = view.clone();
        }
    });
}

/// A storage view of a whole bound buffer.
pub(crate) fn view(buffer: &Arc<BufferDecl>) -> StorageView {
    StorageView {
        buffer: Arc::clone(buffer),
        offset: 0,
        layout: buffer.layout.clone(),
    }
}

/// A logical dtype's dense element; a quantized value has none.
pub(crate) fn elem_of(d: Dtype) -> Result<ScalarElement> {
    match d {
        Dtype::Q(_) => Err(Error::Legality(
            "a quantized value has no dense element type".into(),
        )),
        d => Ok(scalar_element(d)),
    }
}

/// The global element index this lane owns:
/// `program_id.x * BLOCK + lane`.
pub(crate) fn global_lane(b: &Kernel, block: u32) -> TileExpr {
    let pid = b.builtin(Builtin::ProgramId(WorkgroupAxis::X));
    b.add(b.mul(pid, b.u32(block)), b.builtin(Builtin::Lane))
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
            element: ScalarElement::U32.element(),
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
                    let words = quantized_words(fmt, layout, elems);
                    (ScalarElement::U32.element(), vec![words as u32])
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

    /// The kernel over this buffer table.
    pub(crate) fn finish(
        self,
        name: &'static str,
        grid: [u32; 3],
        block: u32,
        body: Vec<Stmt>,
    ) -> KernelIr {
        KernelIr {
            buffers: self.buffers,
            grid,
            block,
            body,
            byte_arena: None,
            name,
        }
    }

    /// [`Translate`] `e` with this table's uniform block (binding 0).
    pub(crate) fn translate(
        &self,
        b: &Kernel,
        args: &[TileExpr],
        coords: &[TileExpr],
        e: &ScalarExpr,
    ) -> Result<TileExpr> {
        Translate {
            b,
            args,
            coords,
            uniforms: self.buffers.first().cloned(),
        }
        .run(e)
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

/// Resolve a dimension at the concrete binding this CPU artifact is compiled
/// for. The executable cache includes these values, so embedding them in the
/// native loop nest cannot reuse code for a different shape.
pub(crate) fn resolve_dim(cx: &LowerCtx<'_>, dim: Dim) -> Result<u32> {
    let value = dim
        .evaluate(&mut |symbol| {
            cx.dim_bindings
                .iter()
                .find_map(|(bound, value)| (*bound == symbol).then_some(*value))
        })
        .ok_or_else(|| Error::Legality(format!("dim {dim} is unbound at CPU lowering")))?;
    u32::try_from(value)
        .map_err(|_| Error::Legality(format!("CPU dimension {value} exceeds u32 indexing")))
}

pub(crate) fn const_extents(cx: &LowerCtx<'_>, shape: &[Dim]) -> Result<Vec<u32>> {
    shape.iter().map(|dim| resolve_dim(cx, *dim)).collect()
}

/// Concrete offset, extents and strides for the current artifact.
pub(crate) fn resolved_layout(
    cx: &LowerCtx<'_>,
    layout: &Layout,
) -> Result<(u32, Vec<u32>, Vec<u32>)> {
    let offset = resolve_dim(cx, layout.offset())?;
    let extents = const_extents(cx, layout.shape())?;
    let strides = const_extents(cx, layout.strides())?;
    Ok((offset, extents, strides))
}

/// A masked load of a whole bound buffer at `index`.
pub(crate) fn load(
    b: &Kernel,
    buffer: Arc<BufferDecl>,
    index: TileExpr,
    mask: TileExpr,
) -> TileExpr {
    let fill = match buffer.element {
        ElementType::Scalar(ScalarElement::U32) | ElementType::Scalar(ScalarElement::I32) => {
            b.u32(0)
        }
        _ => b.f32(0.0),
    };
    b.load(
        Source::Storage(view(&buffer)),
        Addr::Linear(index),
        mask,
        fill,
    )
}

/// One operand's value at the reading kernel's flat space index, mapped
/// through the edge's `layout`/`access`.
pub(crate) fn operand_at(
    b: &Kernel,
    cx: &LowerCtx<'_>,
    binds: &Binds,
    operand: &Operand,
    flat: TileExpr,
    space_total: u64,
    mask: TileExpr,
) -> Result<TileExpr> {
    let index = address_of(b, cx, operand, flat, space_total)?;
    Ok(operand_src(b, cx, binds, operand.src)?.at(b, index, mask))
}

/// `flat` run through one operand's address map, at this artifact's
/// concrete binding.
pub(crate) fn address_of(
    b: &Kernel,
    cx: &LowerCtx<'_>,
    operand: &Operand,
    flat: TileExpr,
    space_total: u64,
) -> Result<TileExpr> {
    Ok(b.address(&resolved_address_map(cx, operand)?, flat, space_total))
}

/// The edge's address map at this artifact's concrete binding.
fn resolved_address_map(cx: &LowerCtx<'_>, operand: &Operand) -> Result<AddressMap> {
    let (offset, extents, strides) = resolved_layout(cx, &operand.layout)?;
    let dims = |v: Vec<u32>| {
        v.into_iter()
            .map(|e| Dim::Const(u64::from(e)))
            .collect::<Vec<_>>()
    };
    let layout = Layout::from_parts(
        Dim::Const(u64::from(offset)),
        &dims(extents),
        &dims(strides),
    )?;
    Operand {
        src: operand.src,
        layout,
        access: operand.access.clone(),
    }
    .address_map()
    .ok_or_else(|| Error::Legality("CPU operand address exceeds u32 indexing".into()))
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
    pub(crate) fn at(&self, b: &Kernel, index: TileExpr, mask: TileExpr) -> TileExpr {
        match self {
            Self::Buffer(buffer) => load(b, Arc::clone(buffer), index, mask),
            Self::Const(v) => v.clone(),
            Self::Quantized(view) => b.load(
                Source::Quantized(view.clone()),
                Addr::Linear(index),
                mask,
                b.f32(0.0),
            ),
        }
    }
}

pub(crate) fn operand_src(
    b: &Kernel,
    cx: &LowerCtx<'_>,
    binds: &Binds,
    src: Id,
) -> Result<OperandSrc> {
    if let Some(splat) = const_splat(cx, src) {
        return Ok(OperandSrc::Const(b.lit(splat_literal(splat))));
    }
    let buffer = binds.of(src)?;
    let facts = cx.graph.facts(src);
    if let Dtype::Q(fmt) = facts.dtype {
        let layout = qlayout_of(cx, src).unwrap_or(QLayout::Native);
        let data = view(&buffer);
        return Ok(OperandSrc::Quantized(QuantizedView { data, fmt, layout }));
    }
    Ok(OperandSrc::Buffer(buffer))
}

/// Translate one `ScalarExpr` body into Kernel, with `args[i]` supplying operand
/// `i` and `coords` supplying `IndexOf(axis)`. Every node takes the body's own
/// dtype, and a comparison consumed as a value selects 1/0 in it.
pub(crate) struct Translate<'a> {
    pub b: &'a Kernel,
    pub args: &'a [TileExpr],
    pub coords: &'a [TileExpr],
    pub uniforms: Option<Arc<BufferDecl>>,
}

impl Translate<'_> {
    pub(crate) fn run(&self, e: &ScalarExpr) -> Result<TileExpr> {
        let b = self.b;
        let ty = ElementType::Scalar(elem_of(e.dtype()).unwrap_or(ScalarElement::F32));
        let node = |kind| TileExpr::new(kind, ty);
        Ok(match e.kind() {
            ScalarKind::Arg(i) => self
                .args
                .get(*i as usize)
                .cloned()
                .ok_or_else(|| Error::Legality(format!("Arg({i}) has no operand")))?,
            ScalarKind::Lit(l) => node(TileExprKind::Literal(splat_literal(l.0))),
            // A runtime scalar is read from the uniform block, never baked
            // into the kernel, so changing it does not recompile.
            ScalarKind::Uniform(sym) => {
                let ub = self
                    .uniforms
                    .clone()
                    .ok_or_else(|| Error::Legality("no uniform block bound".into()))?;
                let raw = load(b, ub, b.u32(sym.0), b.bool(true));
                node(TileExprKind::Bitcast { value: raw, to: ty })
            }
            ScalarKind::IndexOf(axis) => self
                .coords
                .get(*axis as usize)
                .cloned()
                .ok_or_else(|| Error::Legality(format!("IndexOf({axis}) is out of range")))?,
            ScalarKind::Un { op, x } => node(TileExprKind::Unary {
                op: *op,
                value: self.run(x)?,
                numeric: NumericContract::RELAXED,
            }),
            ScalarKind::Bin { op, a, b: r } => node(TileExprKind::Binary {
                op: *op,
                left: self.run(a)?,
                right: self.run(r)?,
                numeric: NumericContract::RELAXED,
            }),
            // Booleans are 1.0/0.0 in the operand dtype at Logical, so a
            // comparison consumed as a value materializes here.
            ScalarKind::Cmp { op, a, b: r } => node(TileExprKind::Select {
                condition: b.compare(*op, self.run(a)?, self.run(r)?),
                accept: self.literal(ty, 1),
                reject: self.literal(ty, 0),
            }),
            ScalarKind::Select { c, t, f } => {
                let zero = self.literal(ty, 0);
                node(TileExprKind::Select {
                    condition: b.compare(fusor_ir::scalar::CmpOp::Ne, self.run(c)?, zero),
                    accept: self.run(t)?,
                    reject: self.run(f)?,
                })
            }
            ScalarKind::Cast { to, x } => node(TileExprKind::Cast {
                value: self.run(x)?,
                to: ElementType::Scalar(elem_of(*to)?),
            }),
            ScalarKind::Bitcast { to, x } => node(TileExprKind::Bitcast {
                value: self.run(x)?,
                to: ElementType::Scalar(elem_of(*to)?),
            }),
            ScalarKind::Round { mode, x } => node(TileExprKind::Round {
                mode: *mode,
                value: self.run(x)?,
            }),
            ScalarKind::Dot { a, b: r } => node(TileExprKind::Dot {
                left: self.run(a)?,
                right: self.run(r)?,
            }),
            ScalarKind::Splat { lanes, x } => {
                let v = self.run(x)?;
                b.vector(elem_of(e.dtype())?, vec![v; *lanes as usize])
            }
        })
    }

    /// `v` as an integer literal for an integer `ty`, else as an f32.
    fn literal(&self, ty: ElementType, v: u8) -> TileExpr {
        match ty {
            ElementType::Scalar(ScalarElement::U32 | ScalarElement::I32) => {
                self.b.u32(u32::from(v))
            }
            _ => self.b.f32(f32::from(v)),
        }
    }
}

/// Decompose a flat index into per-axis coordinates by the declared divmod
/// chain, most-significant-first.
pub(crate) fn coords_of(b: &Kernel, flat: &TileExpr, extents: &[u32]) -> Vec<TileExpr> {
    (0..extents.len())
        .map(|i| {
            let below: u32 = extents[i + 1..].iter().product::<u32>().max(1);
            b.rem(b.div(flat.clone(), b.u32(below)), b.u32(extents[i].max(1)))
        })
        .collect()
}

/// Grid extent for `n` work items at `block` lanes each.
pub(crate) fn grid_for(n: u64, block: u32) -> [u32; 3] {
    let groups = n.div_ceil(block as u64).max(1);
    [groups as u32, 1, 1]
}
