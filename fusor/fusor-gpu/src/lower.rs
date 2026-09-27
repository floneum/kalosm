//! Launch node + `SchedPoint` -> `KernelIr`, one module per node family,
//! and the [`Ctx`] they share. Operand layouts come from `Plan::buffers`,
//! never re-derived; a mismatch is a broken plan ([`Error::Plan`]).

pub(crate) mod contract;
pub(crate) mod gather_scatter;
pub(crate) mod group;
pub(crate) mod map_fold;
pub(crate) mod slab;

use fusor_cost::realize::distribute_workgroups;
use fusor_ir::Result;
use fusor_ir::device::{Caps, Limits};
use fusor_ir::dtype::{Dtype, NumericContract, QLayout};
use fusor_ir::egraph::Id;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Addr, Buffer, BufferAccess, BufferDecl, Builtin, ElementType, KernelIr, MemoryLevel,
    ScalarElement, Source, Stmt, TileCompareOp, TileExpr, TileLayout,
};
use fusor_ir::ir::launch::{ContractSide, IndexSpace, Launch, Operand, SchedPoint};
use fusor_ir::ir::{Node, Op};
use fusor_ir::shape::{AxisGroup, Dim, Layout, MultiFlattenMap, SubAxis, SymId};
use fusor_ir::target::LowerCtx;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

use crate::uniforms::UniformPack;

/// Binding index of the always-present uniform block.
pub(crate) const UNIFORM_BINDING: u32 = 0;

/// A contraction input, with general indexing for non-affine padded views.
#[derive(Clone)]
pub(crate) enum StagedSource {
    Mem(Source),
    Const(TileExpr),
    Indexed {
        operand: Box<Operand>,
        cols: Dim,
        elements: u64,
        // Proven axis boundaries of the independent batch, row and column coordinates.
        axes: Option<(usize, usize)>,
        rows_per_batch: Dim,
    },
}

/// The plan's layout and dtype for `value`; a leaf is dense over its facts.
pub(crate) fn bound_layout(cx: &LowerCtx<'_>, value: Id) -> (Layout, Dtype) {
    let value = cx.selected(value);
    match cx.plan.buffers.iter().find(|b| b.value == value) {
        Some(b) => (b.layout.clone(), b.dtype),
        None => {
            let facts = cx.graph.facts(value);
            (Layout::contiguous(&facts.shape), facts.dtype)
        }
    }
}
/// The step-invariant decl extent `shape[0] * strides[0]` (padding lives in
/// the strides), symbolic dims counting as 1: nothing reads it for a symbolic
/// buffer, and resolving it would bake the sequence length into the kernel.
fn decl_elements(layout: &Layout) -> u64 {
    let (Some(first), Some(stride0)) = (layout.shape().first(), layout.strides().first()) else {
        return 1;
    };
    let outer = first.as_const().unwrap_or(1);
    let stride0 = stride0.as_const().unwrap_or_else(|| {
        layout.shape()[1..]
            .iter()
            .map(|d| d.as_const().unwrap_or(1))
            .product()
    });
    outer.saturating_mul(stride0).max(1)
}

/// Runtime extents for the plan's symbols: the grid reads these, the kernel
/// body reads binding 0. Every read is recorded so the artifact cache keys a
/// kernel on exactly the symbols it consulted.
#[derive(Clone, Debug, Default)]
pub(crate) struct DimBinding {
    values: FxHashMap<SymId, u64>,
    consulted: std::sync::Arc<parking_lot::Mutex<rustc_hash::FxHashSet<SymId>>>,
    /// Symbols read only to fold the dispatch grid: they leave the module
    /// byte-identical, so the cache replays the grid instead of rebuilding.
    grid: std::sync::Arc<parking_lot::Mutex<GridReads>>,
}

/// What a lowering read to fold its dispatch grid.
#[derive(Clone, Debug, Default)]
struct GridReads {
    symbols: rustc_hash::FxHashSet<SymId>,
    /// Every grid requested; more than one distinct makes none replayable.
    specs: Vec<GridSpec>,
}

/// The index space, innermost-axis tile and workgroup width of a dispatch:
/// all of its grid's dependence on the binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GridSpec {
    pub space: IndexSpace,
    pub block: u32,
    pub inner_tile: u32,
}

impl DimBinding {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn from_pairs(pairs: impl IntoIterator<Item = (SymId, u64)>) -> Self {
        Self {
            values: pairs.into_iter().collect(),
            grid: Default::default(),
            consulted: Default::default(),
        }
    }

    pub(crate) fn get(&self, sym: SymId) -> Option<u64> {
        if sym.is_derived() {
            // A derived symbol consults the symbols its expression reaches.
            return Dim::Sym(sym).evaluate(&mut |s| self.get(s));
        }
        let hit = self.values.get(&sym).copied();
        if hit.is_some() {
            self.consulted.lock().insert(sym);
        }
        hit
    }

    /// Concrete extent of a dim, or `None` when the symbol is unbound.
    pub(crate) fn resolve(&self, dim: Dim) -> Option<u64> {
        match dim {
            Dim::Const(v) => Some(v),
            Dim::Sym(s) => self.get(s),
        }
    }

    /// Concrete extent, or `Error::Plan` for an unbound symbol.
    pub(crate) fn require(&self, dim: Dim) -> Result<u64> {
        self.resolve(dim)
            .ok_or_else(|| Error::Plan(format!("dim {dim} is unbound at dispatch")))
    }

    /// Concrete extent for a grid fold, recorded as a grid read only.
    fn require_for_grid(&self, dim: Dim) -> Result<u64> {
        let value = dim.evaluate(&mut |s| {
            let hit = self.values.get(&s).copied();
            if hit.is_some() {
                self.grid.lock().symbols.insert(s);
            }
            hit
        });
        value.ok_or_else(|| Error::Plan(format!("dim {dim} is unbound at dispatch")))
    }

    /// The single [`grid_for`] call whose replay reproduces `grid`, if the
    /// lowering made exactly one.
    pub(crate) fn grid_derivation(&self, grid: [u32; 3], limits: &Limits) -> Option<GridSpec> {
        let spec = {
            let g = self.grid.lock();
            let mut specs = g.specs.iter();
            let first = specs.next()?.clone();
            if specs.any(|s| *s != first) {
                return None;
            }
            first
        };
        (spec.grid(self, limits).ok()? == grid).then_some(spec)
    }

    /// Every symbol the emitted module can depend on; grid-only reads count
    /// unless the grid is replayable.
    pub(crate) fn body_consulted(&self, replayable: bool) -> Vec<SymId> {
        let mut out: rustc_hash::FxHashSet<SymId> = self.consulted.lock().clone();
        if !replayable {
            out.extend(self.grid.lock().symbols.iter().copied());
        }
        let mut out: Vec<SymId> = out.into_iter().collect();
        out.sort_unstable();
        out
    }
}

/// The dispatch grid for an index space at a given workgroup width.
pub(crate) fn grid_for(
    space: &IndexSpace,
    block: u32,
    binding: &DimBinding,
    limits: &Limits,
) -> Result<[u32; 3]> {
    tiled_grid_for(space, block, 1, binding, limits)
}

pub(crate) fn tiled_grid_for(
    space: &IndexSpace,
    block: u32,
    inner_tile: u32,
    binding: &DimBinding,
    limits: &Limits,
) -> Result<[u32; 3]> {
    let spec = GridSpec {
        space: space.clone(),
        block,
        inner_tile,
    };
    binding.grid.lock().specs.push(spec.clone());
    spec.grid(binding, limits)
}

impl GridSpec {
    pub(crate) fn grid(&self, binding: &DimBinding, limits: &Limits) -> Result<[u32; 3]> {
        let mut elements: u64 = 1;
        for (axis, dim) in self.space.dims.iter().enumerate() {
            let mut extent = binding.require_for_grid(*dim)?;
            if axis + 1 == self.space.rank() {
                extent = extent.div_ceil(u64::from(self.inner_tile.max(1)));
            }
            elements = elements
                .checked_mul(extent)
                .ok_or_else(|| Error::Plan("index space overflows a u64".into()))?;
        }
        let groups = elements.div_ceil(u64::from(self.block.max(1)));
        let groups = u32::try_from(groups)
            .map_err(|_| Error::Plan(format!("{groups} workgroups exceeds a u32")))?;
        Ok(distribute_workgroups(
            groups,
            limits.max_compute_workgroups_per_dimension,
        ))
    }
}

/// An N-D strided operand seen as a 2-D matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MatrixView {
    pub rows: u32,
    pub cols: u32,
    pub offset: u32,
    pub layout: TileLayout,
}

/// Flatten a strided layout into a 2-D matrix view: `shape[..row_dims]` to
/// rows, the rest to columns. A side that does not merge affinely becomes a
/// [`MultiFlattenMap`] divmodded per load; an empty side is a single index 0.
pub(crate) fn flatten_matrix_layout_split(
    layout: &Layout,
    row_dims: usize,
    binding: &DimBinding,
) -> Result<MatrixView> {
    let rank = layout.rank();
    if row_dims > rank {
        return Err(Error::Plan(format!(
            "matrix split at {row_dims} is outside rank {rank}"
        )));
    }

    let mut shape = SmallVec::<[u64; 6]>::new();
    for d in layout.shape() {
        shape.push(binding.require(*d)?);
    }
    let mut strides = SmallVec::<[u64; 6]>::new();
    for stride in layout.strides() {
        strides.push(binding.require(*stride)?);
    }

    let rows: u64 = shape[..row_dims].iter().product();
    let cols: u64 = shape[row_dims..].iter().product();
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| Error::Plan(format!("{rows} rows exceeds a u32 coordinate")))?;
    let cols_u32 = u32::try_from(cols)
        .map_err(|_| Error::Plan(format!("{cols} cols exceeds a u32 coordinate")))?;
    let offset = u32::try_from(binding.require(layout.offset())?)
        .map_err(|_| Error::Plan("layout offset exceeds a u32".into()))?;

    let side_is_affine = |lo: usize, hi: usize| -> bool {
        (lo..hi)
            .zip(lo + 1..hi)
            .all(|(axis, next)| strides[axis] == strides[next].saturating_mul(shape[next]))
    };

    // An empty side is a single index of 0; stride 0 keeps it out of the
    // address rather than reaching past the end of `strides`.
    let innermost = |lo: usize, hi: usize| -> u64 { if lo == hi { 0 } else { strides[hi - 1] } };
    let tile_layout = if side_is_affine(0, row_dims) && side_is_affine(row_dims, rank) {
        let row_stride = u32::try_from(innermost(0, row_dims))
            .map_err(|_| Error::Plan("row stride exceeds a u32".into()))?;
        let col_stride = u32::try_from(innermost(row_dims, rank))
            .map_err(|_| Error::Plan("col stride exceeds a u32".into()))?;
        TileLayout {
            extents: smallvec::smallvec![rows_u32, cols_u32],
            indexing: MultiFlattenMap::affine(&[rows_u32, cols_u32], &[row_stride, col_stride]),
            level: MemoryLevel::Storage,
        }
    } else {
        let group = |lo: usize, hi: usize| -> Result<AxisGroup> {
            let mut sub_axes: SmallVec<[SubAxis; 2]> = SmallVec::new();
            for axis in lo..hi {
                // Extent-1 axes would only cost a divmod per load.
                if shape[axis] == 1 {
                    continue;
                }
                sub_axes.push(SubAxis {
                    extent: u32::try_from(shape[axis])
                        .map_err(|_| Error::Plan("sub-axis extent exceeds a u32".into()))?,
                    stride: u32::try_from(strides[axis])
                        .map_err(|_| Error::Plan("sub-axis stride exceeds a u32".into()))?,
                });
            }
            if sub_axes.is_empty() {
                sub_axes.push(SubAxis {
                    extent: 1,
                    stride: 0,
                });
            }
            Ok(AxisGroup { sub_axes })
        };
        TileLayout {
            extents: smallvec::smallvec![rows_u32, cols_u32],
            indexing: MultiFlattenMap {
                groups: smallvec::smallvec![group(0, row_dims)?, group(row_dims, rank)?],
            },
            level: MemoryLevel::Storage,
        }
    };

    Ok(MatrixView {
        rows: rows_u32,
        cols: cols_u32,
        offset,
        layout: tile_layout,
    })
}

/// The axis split presenting `layout` as `rows` by `cols`: canonical operands
/// are `[batch.., m.., k..]`, so the split is where the prefix product is
/// `rows`; the longest such prefix wins ties from extent-1 axes.
pub(crate) fn matrix_split_for(
    layout: &Layout,
    binding: &DimBinding,
    rows: u64,
    cols: u64,
) -> Result<usize> {
    let rank = layout.rank();
    let mut extents = SmallVec::<[u64; 6]>::new();
    for d in layout.shape() {
        extents.push(binding.require(*d)?);
    }
    (0..=rank)
        .rev()
        .find(|split| {
            extents[..*split].iter().product::<u64>() == rows
                && extents[*split..].iter().product::<u64>() == cols
        })
        .ok_or_else(|| {
            Error::Plan(format!(
                "no axis split of {extents:?} yields {rows} rows by {cols} columns"
            ))
        })
}

pub(crate) use fusor_tile::build::{Kernel, qlayout_of, quantized_words, scalar_element};
use fusor_tile::build::{const_splat, finite_literal};

/// Per-kernel lowering state: the buffer table in binding order, the uniform
/// word layout, and the Kernel builder.
pub(crate) struct Ctx<'a> {
    pub caps: &'a Caps,
    pub cx: &'a LowerCtx<'a>,
    pub b: Kernel,
    pub binding: DimBinding,
    /// Binding order. Index 0 is always the uniform block.
    pub buffers: Vec<Buffer>,
    /// `Plan` value -> index into [`Self::buffers`].
    slot_of: FxHashMap<Id, usize>,
    /// Element offset of each arena value within the arena binding.
    arena_offset: FxHashMap<Id, u32>,
    pub(crate) pack: std::sync::Arc<UniformPack>,
    /// A group member's linear workgroup index within its own range, in
    /// place of the dispatch's builtins.
    pub workgroup: Option<TileExpr>,
    /// The fewest lanes a member may lower at: a group runs every member at
    /// its widest member's block.
    pub block_floor: u32,
}

impl<'a> Ctx<'a> {
    /// Build the buffer table for one launch over the plan's binding-0 pack.
    pub(crate) fn with_pack(
        caps: &'a Caps,
        cx: &'a LowerCtx<'a>,
        binding: DimBinding,
        pack: std::sync::Arc<UniformPack>,
    ) -> Result<Self> {
        // A relower mints the same decl ids, so the body-hash dedup hits.
        fusor_ir::ir::kernel::reset_decl_ids();
        Self::with_pack_in(caps, cx, binding, pack)
    }

    /// [`Self::with_pack`] continuing decl numbering, for a group member.
    pub(crate) fn with_pack_in(
        caps: &'a Caps,
        cx: &'a LowerCtx<'a>,
        binding: DimBinding,
        pack: std::sync::Arc<UniformPack>,
    ) -> Result<Self> {
        let decl = |binding, dtype, extent: u32, access| {
            Arc::new(BufferDecl {
                binding,
                element: ElementType::Scalar(scalar_element(dtype)),
                layout: TileLayout::contiguous(MemoryLevel::Storage, &[extent.max(1)]),
                access,
            })
        };
        let uniform_words = (pack.byte_len() / 4).max(1) as u32;
        let mut buffers: Vec<Buffer> = vec![decl(
            UNIFORM_BINDING,
            Dtype::U32,
            uniform_words,
            BufferAccess::Read,
        )];

        let mut ordered: Vec<_> = cx.launch.bindings.iter().collect();
        ordered.sort_by_key(|b| b.binding);

        let mut slot_of = FxHashMap::default();
        let mut arena_offset: FxHashMap<Id, u32> = FxHashMap::default();
        let mut arena_views = FxHashMap::default();
        for plan_binding in ordered.iter() {
            let (layout, dtype) = bound_layout(cx, plan_binding.value);
            let class = cx.graph.class_of(plan_binding.value);
            // Typed views share one physical arena binding.
            if plan_binding.arena {
                let bytes = cx
                    .plan
                    .buffers
                    .iter()
                    .find(|b| b.value == plan_binding.value)
                    .and_then(|b| b.arena)
                    .ok_or_else(|| {
                        Error::Plan(format!("arena value {} has no offset", plan_binding.value))
                    })?;
                let elem = dtype.byte_size().max(1);
                let off = u32::try_from(bytes / elem)
                    .map_err(|_| Error::Plan("arena offset exceeds a u32".into()))?;
                let slot = if let Some(slot) = arena_views.get(&dtype) {
                    *slot
                } else {
                    let extent = u32::try_from(cx.plan.arena_bytes / elem)
                        .map_err(|_| Error::Plan("arena element count exceeds a u32".into()))?;
                    buffers.push(decl(
                        plan_binding.binding,
                        dtype,
                        extent,
                        BufferAccess::ReadWrite,
                    ));
                    let slot = buffers.len() - 1;
                    arena_views.insert(dtype, slot);
                    slot
                };
                for member in cx.graph.class_ids(class) {
                    slot_of.insert(member, slot);
                    arena_offset.insert(member, off);
                }
                continue;
            }
            let elements = decl_elements(&layout);
            // A quantized buffer binds as its `u32` block word stream.
            let elements = match dtype {
                Dtype::Q(fmt) => {
                    let qlayout = qlayout_of(cx, plan_binding.value).unwrap_or(QLayout::Native);
                    quantized_words(fmt, qlayout, elements)
                }
                _ => elements,
            };
            let extent = u32::try_from(elements)
                .map_err(|_| Error::Plan("buffer element count exceeds a u32".into()))?;
            let access = match plan_binding.kind {
                fusor_ir::extract::BindKind::Read => BufferAccess::Read,
                _ => BufferAccess::ReadWrite,
            };
            // Keyed by every id in the class (`Union` spine included): an
            // `Operand::src` may name any of them.
            for member in cx.graph.class_ids(class) {
                slot_of.insert(member, buffers.len());
            }
            buffers.push(decl(plan_binding.binding, dtype, extent, access));
        }

        Ok(Self {
            caps,
            cx,
            b: Kernel::new(),
            binding,
            buffers,
            slot_of,
            arena_offset,
            pack,
            workgroup: None,
            block_floor: 0,
        })
    }

    /// Lanes for a lowering that wants `want`: never below a group's block.
    pub(crate) fn block(&self, want: u32) -> u32 {
        want.max(self.block_floor)
    }

    /// Element offset of a value inside its binding: its arena slot, or 0.
    pub(crate) fn offset_of(&self, value: Id) -> u32 {
        self.arena_offset.get(&value).copied().unwrap_or(0)
    }

    /// This workgroup's linear index (within its range, for a group member).
    pub(crate) fn linear_workgroup(&self) -> TileExpr {
        use fusor_ir::ir::kernel::WorkgroupAxis;
        if let Some(w) = &self.workgroup {
            return w.clone();
        }
        let b = &self.b;
        let id = |axis| b.builtin(Builtin::ProgramId(axis));
        // gx + gy*X + gz*X*Y, with X and Y read from `num_workgroups`.
        let x = b.builtin(Builtin::NumWorkgroups(WorkgroupAxis::X));
        let y = b.builtin(Builtin::NumWorkgroups(WorkgroupAxis::Y));
        b.add(
            b.add(id(WorkgroupAxis::X), b.mul(id(WorkgroupAxis::Y), x.clone())),
            b.mul(id(WorkgroupAxis::Z), b.mul(x, y)),
        )
    }

    /// Whether this launch binds a buffer for `value`; a slab member kept in
    /// workgroup memory has none.
    pub(crate) fn has_buffer(&self, value: Id) -> bool {
        self.slot_of.contains_key(&value)
    }

    /// The bound buffer for a plan value.
    pub(crate) fn buffer(&self, value: Id) -> Result<Buffer> {
        let slot = self
            .slot_of
            .get(&value)
            .ok_or_else(|| Error::Plan(format!("value {value} is not bound by this launch")))?;
        Ok(self.buffers[*slot].clone())
    }

    pub(crate) fn plan_dtype(&self, value: Id) -> Result<Dtype> {
        Ok(bound_layout(self.cx, value).1)
    }

    /// A flat rank-1 view of a value's buffer, for elementwise access.
    pub(crate) fn linear_view(&self, value: Id) -> Result<fusor_ir::ir::kernel::StorageView> {
        let buffer = self.buffer(value)?;
        let layout = buffer.layout.clone();
        Ok(fusor_ir::ir::kernel::StorageView {
            buffer,
            offset: self.offset_of(value),
            layout,
        })
    }

    /// A 2-D matrix view of an operand split at `row_dims`.
    pub(crate) fn matrix_view(
        &self,
        operand: &Operand,
        row_dims: usize,
    ) -> Result<Option<fusor_ir::ir::kernel::StorageView>> {
        let Some(layout) = self.repad_operand_layout(operand)? else {
            return Ok(None);
        };
        let view = flatten_matrix_layout_split(&layout, row_dims, &self.binding)?;
        let buffer = self.buffer(operand.src)?;
        Ok(Some(fusor_ir::ir::kernel::StorageView {
            buffer,
            offset: view.offset + self.offset_of(operand.src),
            layout: view.layout,
        }))
    }

    /// Restate an affine operand over its producer's padded allocation;
    /// `None` when an axis spans padding.
    fn repad_operand_layout(&self, operand: &Operand) -> Result<Option<Layout>> {
        let selected = self.cx.selected(operand.src);
        let Some(plan) = self.cx.plan.buffers.iter().find(|b| b.value == selected) else {
            return Ok(Some(operand.layout.clone()));
        };
        let logical = &self.cx.graph.facts(selected).shape;
        let dense = Layout::row_major_strides(logical);
        if plan.layout == Layout::contiguous(logical) || logical.is_empty() {
            return Ok(Some(operand.layout.clone()));
        }
        if !plan.layout.offset().known_eq(Dim::Const(0))
            || !operand.layout.offset().known_eq(Dim::Const(0))
        {
            return Ok(None);
        }
        let mut strides = Vec::with_capacity(operand.layout.rank());
        for (&extent, &stride) in operand.layout.shape().iter().zip(operand.layout.strides()) {
            if extent.as_const().is_some_and(|e| e <= 1) || stride.known_eq(Dim::Const(0)) {
                strides.push(stride);
                continue;
            }
            let mut mapped = None;
            for (axis, &dense_stride) in dense.iter().enumerate() {
                if stride.known_eq(dense_stride) && extent.known_eq(logical[axis]) {
                    mapped = Some(plan.layout.strides()[axis]);
                    break;
                }
                let (Some(stride), Some(extent), Some(dense_stride), Some(logical_extent)) = (
                    stride.as_const(),
                    extent.as_const(),
                    dense_stride.as_const(),
                    logical[axis].as_const(),
                ) else {
                    continue;
                };
                if dense_stride == 0 || stride % dense_stride != 0 {
                    continue;
                }
                let step = stride / dense_stride;
                if step >= 1 && step.saturating_mul(extent - 1) < logical_extent {
                    mapped = Some(Dim::Const(step) * plan.layout.strides()[axis]);
                    break;
                }
            }
            let Some(mapped) = mapped else {
                return Ok(None);
            };
            strides.push(mapped);
        }
        Layout::from_parts(operand.layout.offset(), operand.layout.shape(), &strides).map(Some)
    }

    /// The [`Source`] a contraction stages one operand from: storage, or a
    /// quantized decode run at the staging fill's `(row, col)`.
    pub(crate) fn contract_stage_source(
        &self,
        operand: &Operand,
        view: &fusor_ir::ir::kernel::StorageView,
    ) -> Result<Source> {
        let Dtype::Q(fmt) = self.plan_dtype(operand.src)? else {
            return Ok(Source::Storage(view.clone()));
        };
        let qlayout = qlayout_of(self.cx, operand.src).unwrap_or(QLayout::Native);
        Ok(Source::Quantized(fusor_ir::ir::kernel::QuantizedView {
            data: view.clone(),
            fmt,
            layout: qlayout,
        }))
    }

    /// Every buffer one contraction side reads, as a staging source apiece:
    /// an absorbed producer (a GGUF block decode's planes and scales) brings
    /// several edges, all loaded at the side's `(rows, cols)` before `pre`.
    pub(crate) fn contract_side_sources(
        &self,
        side: &ContractSide,
        batch: u32,
        rows_per_batch: Dim,
        cols: Dim,
    ) -> Result<Vec<StagedSource>> {
        let rows = Dim::Const(u64::from(batch)) * rows_per_batch;
        side.ops
            .iter()
            .map(|o| {
                // A `Const` leaf folds into the kernel, as in `load_operand`.
                if let Some(lit) = self.const_operand(o.src) {
                    return Ok(StagedSource::Const(lit));
                }
                let view = match (rows.as_const(), cols.as_const()) {
                    (Some(rows), Some(cols))
                        if o.layout.shape().iter().all(|d| d.as_const().is_some())
                            && o.layout.strides().iter().all(|d| d.as_const().is_some())
                            && o.layout.offset().as_const().is_some() =>
                    {
                        let extent = |value| {
                            u32::try_from(value).map_err(|_| {
                                Error::Plan("contraction matrix extent exceeds a u32".into())
                            })
                        };
                        self.contract_operand_view(o, extent(rows)?, extent(cols)?)?
                    }
                    _ => None,
                };
                match view {
                    Some(view) => Ok(StagedSource::Mem(self.contract_stage_source(o, &view)?)),
                    None => Ok(StagedSource::Indexed {
                        operand: Box::new(o.clone()),
                        cols,
                        rows_per_batch,
                        axes: if matches!(o.access, fusor_ir::ir::launch::AccessPlan::Unflatten(_))
                        {
                            None
                        } else {
                            let shape = o.layout.shape();
                            let product = |dims: &[Dim]| {
                                let value = dims.iter().copied().fold(Dim::ONE, |a, b| a * b);
                                (value != Dim::Sym(fusor_ir::shape::OPAQUE_SYM)).then_some(value)
                            };
                            (0..=shape.len()).rev().find_map(|column| {
                                if product(&shape[column..]) != Some(cols) {
                                    return None;
                                }
                                (0..=column)
                                    .rev()
                                    .find(|&row| {
                                        product(&shape[..row]) == Some(Dim::Const(u64::from(batch)))
                                            && product(&shape[row..column]) == Some(rows_per_batch)
                                    })
                                    .map(|row| (row, column))
                            })
                        },
                        elements: o
                            .layout
                            .shape()
                            .iter()
                            .try_fold(1u64, |n, d| n.checked_mul(d.as_const()?))
                            .unwrap_or(u64::MAX),
                    }),
                }
            })
            .collect()
    }

    pub(crate) fn contract_operand_view(
        &self,
        operand: &Operand,
        rows: u32,
        cols: u32,
    ) -> Result<Option<fusor_ir::ir::kernel::StorageView>> {
        let split = matrix_split_for(
            &operand.layout,
            &self.binding,
            u64::from(rows),
            u64::from(cols),
        )?;
        self.matrix_view(operand, split)
    }

    /// Read a `u32` word out of binding 0.
    pub(crate) fn uniform_word(&self, slot: u32) -> TileExpr {
        let view = fusor_ir::ir::kernel::StorageView {
            buffer: self.buffers[0].clone(),
            offset: 0,
            layout: self.buffers[0].layout.clone(),
        };
        let b = &self.b;
        b.load(
            Source::Storage(view),
            Addr::Linear(b.u32(slot)),
            b.bool(true),
            b.u32(0),
        )
    }

    /// A `u32` expression for a dim: a literal, or a binding-0 word when
    /// symbolic so a length change recompiles nothing.
    pub(crate) fn dim_expr(&self, dim: Dim) -> Result<TileExpr> {
        match dim {
            Dim::Const(v) => {
                let v = u32::try_from(v)
                    .map_err(|_| Error::Plan(format!("extent {v} exceeds a u32")))?;
                Ok(self.b.u32(v))
            }
            Dim::Sym(s) => {
                let slot = self
                    .pack
                    .dim_slot(s)
                    .ok_or_else(|| Error::Plan(format!("symbol {s} has no uniform slot")))?;
                Ok(self.uniform_word(slot))
            }
        }
    }

    /// `1 * d0 * d1 * ...`, the element count of `dims` as a `u32` expression.
    pub(crate) fn extent_product(&self, dims: &[Dim]) -> Result<TileExpr> {
        dims.iter().try_fold(self.b.u32(1), |acc, d| {
            Ok(self.b.mul(acc, self.dim_expr(*d)?))
        })
    }

    /// An `f32` expression for a runtime scalar, read from binding 0.
    pub(crate) fn scalar_expr(&self, sym: SymId) -> Result<TileExpr> {
        let slot = self
            .pack
            .scalar_slot(sym)
            .ok_or_else(|| Error::Plan(format!("scalar {sym} has no uniform slot")))?;
        Ok(self
            .b
            .bitcast(self.uniform_word(slot), ScalarElement::F32.element()))
    }

    /// The global linear element index: workgroup * `block` + lane.
    pub(crate) fn global_index(&self, block: u32) -> TileExpr {
        let lane = self.b.builtin(Builtin::Lane);
        self.b
            .add(self.b.mul(self.linear_workgroup(), self.b.u32(block)), lane)
    }

    /// The value this launch writes: the launch root when it is bound for
    /// writing, else the first writable binding.
    pub(crate) fn output(&self) -> Result<Id> {
        let root = self.cx.launch.root;
        if self
            .cx
            .launch
            .bindings
            .iter()
            .any(|b| b.value == root && b.kind != fusor_ir::extract::BindKind::Read)
        {
            return Ok(root);
        }
        self.cx
            .launch
            .bindings
            .iter()
            .find(|b| b.kind != fusor_ir::extract::BindKind::Read)
            .map(|b| b.value)
            .ok_or_else(|| Error::Plan("launch binds nothing writable".into()))
    }

    /// Per-axis coordinates of a flat index over `space`, most-significant
    /// first: one divmod per axis past the innermost.
    pub(crate) fn coords_from_linear(
        &self,
        linear: TileExpr,
        space: &IndexSpace,
    ) -> Result<Vec<TileExpr>> {
        let rank = space.rank();
        let mut coords = vec![linear.clone(); rank];
        let mut rest = linear;
        for axis in (0..rank).rev() {
            let extent = self.dim_expr(space.dims[axis])?;
            if axis == 0 {
                coords[0] = rest;
                break;
            }
            coords[axis] = self.b.rem(rest.clone(), extent.clone());
            rest = self.b.div(rest, extent);
        }
        Ok(coords)
    }

    /// Translate a [`fusor_ir::scalar::ScalarExpr`] body into Kernel over
    /// loaded `args` and `IndexOf` `coords`. Comparisons yield 1/0 in the
    /// operand's dtype, as at Logical.
    pub(crate) fn eval_scalar(
        &self,
        expr: &fusor_ir::scalar::ScalarExpr,
        args: &[TileExpr],
        coords: &[TileExpr],
    ) -> Result<TileExpr> {
        use fusor_ir::scalar::ScalarKind as K;
        let relaxed = NumericContract::RELAXED;
        let ev = |e| self.eval_scalar(e, args, coords);
        Ok(match expr.kind() {
            K::Arg(i) => args.get(*i as usize).cloned().ok_or_else(|| {
                Error::Plan(format!("body reads Arg({i}) with {} operands", args.len()))
            })?,
            K::Lit(l) => self.b.lit(finite_literal(l.0)),
            K::Uniform(sym) if expr.dtype() == Dtype::U32 && self.pack.dim_slot(*sym).is_some() => {
                self.dim_expr(Dim::Sym(*sym))?
            }
            K::Uniform(sym) => self.scalar_expr(*sym)?,
            K::IndexOf(axis) => {
                let c = coords.get(*axis as usize).cloned().ok_or_else(|| {
                    Error::Plan(format!(
                        "body reads IndexOf({axis}) outside the index space"
                    ))
                })?;
                self.b.cast(c, ElementType::Scalar(ScalarElement::U32))
            }
            K::Un { op, x } => self.b.unary(*op, ev(x)?, relaxed),
            K::Bin { op, a, b } => {
                let l = ev(a)?;
                self.b.binary(*op, l, ev(b)?, relaxed)
            }
            K::Cmp { op, a, b } => {
                let l = ev(a)?;
                let elem = l.element();
                let c = self.b.compare(*op, l, ev(b)?);
                // `f32(cmp)`, never `select(0, 1, cmp)`: WARP's DXIL JIT
                // removes the device on the select (tests/warp_probe.rs).
                self.b.cast(c, elem)
            }
            K::Select { c, t, f } => {
                let cv = ev(c)?;
                let (tv, fv) = (ev(t)?, ev(f)?);
                let zero = self.b.zero_of(cv.element());
                let nonzero = self.b.compare(TileCompareOp::Ne, cv, zero);
                self.b.select(nonzero, tv, fv)
            }
            K::Cast { to, x } => self
                .b
                .cast(ev(x)?, ElementType::Scalar(scalar_element(*to))),
            K::Bitcast { to, x } => self
                .b
                .bitcast(ev(x)?, ElementType::Scalar(scalar_element(*to))),
            // Its own node, so fast math cannot fold QAT's rounding away.
            K::Round { mode, x } => self.b.round(*mode, ev(x)?),
            K::Dot { a, b } => {
                let l = ev(a)?;
                self.b.dot(l, ev(b)?)
            }
            K::Splat { lanes, x } => {
                let v = ev(x)?;
                let scalar = match v.element() {
                    ElementType::Scalar(s) => s,
                    ElementType::Vector { scalar, .. } => scalar,
                    ElementType::CoopMatrix { scalar, .. } => scalar,
                };
                self.b.vector(scalar, vec![v; *lanes as usize])
            }
        })
    }

    /// Load one operand at the reading kernel's flat space index, through
    /// the edge's address map; [`Ctx::load_operand`] takes a storage index.
    pub(crate) fn load_mapped(
        &self,
        operand: &Operand,
        flat: TileExpr,
        space_total: u64,
    ) -> Result<TileExpr> {
        let addr = self.operand_address(operand, flat, space_total)?;
        self.load_operand(operand, addr)
    }

    /// `flat` run through one operand's index map.
    pub(crate) fn operand_address(
        &self,
        operand: &Operand,
        flat: TileExpr,
        space_total: u64,
    ) -> Result<TileExpr> {
        let Some(map) = operand.address_map() else {
            // Symbolic: computed from binding-0 words instead.
            return self.symbolic_operand_address(operand, flat);
        };
        Ok(self.b.address(&map, flat, space_total))
    }

    /// [`Ctx::operand_address`] for a symbolic layout: `offset + Σ ((flat /
    /// Π right extents) % extent) * stride`, the leading axis skipping its `%`
    /// since `flat` is masked below the space total.
    fn symbolic_operand_address(&self, operand: &Operand, flat: TileExpr) -> Result<TileExpr> {
        if matches!(
            operand.access,
            fusor_ir::ir::launch::AccessPlan::Unflatten(_)
        ) {
            return Err(Error::Plan(format!(
                "a symbolic Unflatten window is not lowerable; operand {} laid out {:?}",
                operand.src, operand.layout
            )));
        }
        let layout = &operand.layout;
        let offset = self.dim_expr(layout.offset())?;
        let relative = self.strided_address(flat, layout.shape(), layout.strides())?;
        Ok(self.b.add(offset, relative))
    }

    fn strided_address(&self, flat: TileExpr, shape: &[Dim], strides: &[Dim]) -> Result<TileExpr> {
        if shape.iter().all(|extent| extent.known_eq(Dim::ONE)) {
            return Ok(self.b.u32(0));
        }
        if strides == Layout::row_major_strides(shape).as_slice() {
            return Ok(flat);
        }
        let mut acc = None;
        let mut div: Option<TileExpr> = None;
        for axis in (0..shape.len()).rev() {
            let (extent, stride) = (shape[axis], strides[axis]);
            if !stride.known_eq(Dim::Const(0)) && !extent.known_eq(Dim::ONE) {
                let modulus = (axis != 0).then(|| self.dim_expr(extent)).transpose()?;
                let stride = (!stride.known_eq(Dim::ONE))
                    .then(|| self.dim_expr(stride))
                    .transpose()?;
                let term = self.b.term(flat.clone(), div.clone(), modulus, stride);
                self.b.accumulate(&mut acc, term);
            }
            if !extent.known_eq(Dim::ONE) {
                let m = self.dim_expr(extent)?;
                div = Some(match div {
                    Some(d) => self.b.mul(d, m),
                    None => m,
                });
            }
        }
        Ok(acc.unwrap_or_else(|| self.b.u32(0)))
    }

    /// Re-address a logical dense element index of `src` into the buffer the
    /// plan laid out (a `Coop` output padded to whole blocks); identity for an
    /// unpadded value.
    fn repad_index(&self, src: Id, index: TileExpr) -> Result<TileExpr> {
        let selected = self.cx.selected(src);
        let Some(plan) = self
            .cx
            .plan
            .buffers
            .iter()
            .find(|b| b.value == selected)
            .cloned()
        else {
            return Ok(index);
        };
        let logical = self.cx.graph.facts(selected).shape.clone();
        if plan.layout.rank() != logical.len() || logical.is_empty() {
            return Ok(index);
        }
        let strides = plan.layout.strides().to_vec();
        let shape = plan.layout.shape().to_vec();
        let dense = Layout::row_major_strides(&logical);
        let unpadded = plan.layout.offset().known_eq(Dim::Const(0))
            && shape.iter().zip(&logical).all(|(p, l)| p.known_eq(*l))
            && strides.iter().zip(&dense).all(|(s, w)| s.known_eq(*w));
        if unpadded {
            return Ok(index);
        }
        let offset = plan.layout.offset();
        let mut acc = (!offset.known_eq(Dim::Const(0)))
            .then(|| self.dim_expr(offset))
            .transpose()?;
        for axis in 0..logical.len() {
            let (extent, stride) = (logical[axis], strides[axis]);
            if extent.as_const().is_some_and(|e| e <= 1) || stride.known_eq(Dim::Const(0)) {
                continue;
            }
            let div = (!dense[axis].known_eq(Dim::Const(1)))
                .then(|| self.dim_expr(dense[axis]))
                .transpose()?;
            let modulus = (axis > 0).then(|| self.dim_expr(extent)).transpose()?;
            let stride = (!stride.known_eq(Dim::Const(1)))
                .then(|| self.dim_expr(stride))
                .transpose()?;
            let term = self.b.term(index.clone(), div, modulus, stride);
            self.b.accumulate(&mut acc, term);
        }
        Ok(acc.unwrap_or_else(|| self.b.u32(0)))
    }

    /// Load one operand at an already-computed logical element index, masked
    /// by the buffer's extent.
    pub(crate) fn load_operand(&self, operand: &Operand, index: TileExpr) -> Result<TileExpr> {
        // A `Leaf::Const` folds into the kernel and has no binding.
        if let Some(lit) = self.const_operand(operand.src) {
            return Ok(lit);
        }
        let index = self.repad_index(operand.src, index)?;
        // A quantized operand runs its decode program at flat index `i`.
        if let Dtype::Q(fmt) = self.plan_dtype(operand.src)? {
            let qlayout = qlayout_of(self.cx, operand.src).unwrap_or(QLayout::Native);
            let facts = self.cx.graph.facts(self.cx.selected(operand.src));
            let cols = facts
                .shape
                .last()
                .map(|d| self.binding.require(*d))
                .transpose()?
                .unwrap_or(0);
            let mut rows: u64 = 1;
            for d in &facts.shape[..facts.shape.len().saturating_sub(1)] {
                rows = rows.saturating_mul(self.binding.require(*d)?);
            }
            let bound = self
                .b
                .u32(u32::try_from(rows.saturating_mul(cols)).unwrap_or(u32::MAX));
            let view = fusor_ir::ir::kernel::QuantizedView {
                data: self.linear_view(operand.src)?,
                fmt,
                layout: qlayout,
            };
            return Ok(self.b.load(
                Source::Quantized(view),
                Addr::Linear(index.clone()),
                self.b.lt(index, bound),
                self.b.f32(0.0),
            ));
        }
        // The bound is the plan layout's `shape[0] * strides[0]` over `Dim`s
        // (padding lives in the strides), never a resolved decl extent.
        let view = self.linear_view(operand.src)?;
        let elem = view.buffer.element;
        let (plan_layout, _) = bound_layout(self.cx, operand.src);
        let bound = match (plan_layout.shape().first(), plan_layout.strides().first()) {
            (Some(&outer), Some(&stride0)) => {
                let outer_e = if outer.known_eq(Dim::Const(1)) {
                    None
                } else {
                    Some(self.dim_expr(outer)?)
                };
                let stride_e = match stride0 {
                    s if s.known_eq(Dim::Const(1)) || s.known_eq(Dim::Const(0)) => None,
                    s => Some(self.dim_expr(s)?),
                };
                match (outer_e, stride_e) {
                    (Some(o), Some(s)) => self.b.mul(o, s),
                    (Some(o), None) => o,
                    (None, Some(s)) => s,
                    (None, None) => self.b.u32(1),
                }
            }
            _ => self.b.u32(1),
        };
        Ok(self.b.load(
            Source::Storage(view),
            Addr::Linear(index.clone()),
            self.b.lt(index, bound),
            self.b.zero_of(elem),
        ))
    }

    /// The literal a `Leaf::Const` operand folds to, clamped finite for WGSL.
    pub(crate) fn const_operand(&self, src: Id) -> Option<TileExpr> {
        const_splat(self.cx, src).map(|s| self.b.lit(finite_literal(s)))
    }

    /// Finish a kernel body into a [`KernelIr`].
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
            byte_arena: if self.caps.workgroup_alias {
                Some(fusor_ir::ir::kernel::ByteArenaToken)
            } else {
                None
            },
            name,
        }
    }
}

/// Lower one selected Launch node at one schedule point.
pub(crate) fn lower_node(
    caps: &Caps,
    node: &Node,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
    binding: DimBinding,
    pack: std::sync::Arc<UniformPack>,
) -> Result<KernelIr> {
    lower_launch(Ctx::with_pack(caps, cx, binding, pack)?, node, theta, false)
}

/// [`lower_node`] for a group member: continued decl numbering, workgroup
/// index `workgroup`, at least `block_floor` lanes, the group's buffer table.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_member(
    caps: &Caps,
    node: &Node,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
    binding: DimBinding,
    pack: std::sync::Arc<UniformPack>,
    workgroup: TileExpr,
    block_floor: u32,
    buffers: &[Buffer],
) -> Result<KernelIr> {
    let mut ctx = Ctx::with_pack_in(caps, cx, binding, pack)?;
    ctx.buffers = buffers.to_vec();
    ctx.workgroup = Some(workgroup);
    ctx.block_floor = block_floor;
    lower_launch(ctx, node, theta, true)
}

fn lower_launch(ctx: Ctx<'_>, node: &Node, theta: SchedPoint, member: bool) -> Result<KernelIr> {
    let Op::Launch(op) = &node.op else {
        return Err(Error::Plan(format!(
            "lowering was handed a {:?} node, but only Launch nodes are selectable",
            node.level
        )));
    };
    match op {
        Launch::Map { .. } => map_fold::lower_kmap(ctx, op, theta),
        Launch::Fold { .. } | Launch::StreamFold { .. } => map_fold::lower_kfold(ctx, op, theta),
        Launch::Contract { family, .. } => contract::lower_contract(ctx, op, *family, theta),
        Launch::Gather { .. } => gather_scatter::lower_kgather(ctx, op, theta),
        Launch::Slab { .. } => slab::lower_kslab(ctx, op, theta),
        Launch::Scatter { .. } | Launch::Group { .. } if member => Err(Error::Plan(format!(
            "a {:?} cannot be a group member",
            op.tag()
        ))),
        Launch::Scatter { .. } => gather_scatter::lower_kscatter(ctx, op, theta),
        Launch::Group { .. } => group::lower_kgroup(ctx, op, theta),
    }
}

/// [`lower_node`] at a fresh binding and the plan's own uniform pack.
pub(crate) fn lower(
    caps: &Caps,
    node: &Node,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
) -> Result<KernelIr> {
    let pack = std::sync::Arc::new(UniformPack::new(cx.plan));
    lower_node(caps, node, theta, cx, DimBinding::new(), pack)
}
