//! L1 node + `SchedPoint` -> `KernelIr`, one module per node family.
//!
//! Everything shared by the six family lowerings lives here: the grid fold,
//! the 2-D matrix flattening of an N-D strided operand, the hash-consing L2
//! term builder, and the [`Ctx`] that turns `Plan`-carried buffer layouts into
//! L2 storage views.
//!
//! Operand layouts are never re-derived: every layout comes from
//! `Plan::buffers[..].layout`. A mismatch here is [`Error::Plan`], not a
//! routing decision.

pub mod contract;
pub mod gather_scatter;
pub mod map_fold;
pub mod merged;

use fusor2_ir::Result;
use fusor2_ir::device::{Caps, Limits};
use fusor2_ir::dtype::{Dtype, NumericContract, QLayout};
use fusor2_ir::egraph::Id;
use fusor2_ir::error::Error;
use fusor2_ir::ir::level1::{ContractSide, IndexSpace, L1, Operand, SchedPoint};
use fusor2_ir::ir::level2::{
    Addr, Buffer, BufferAccess, BufferDecl, Builtin, ElementType, KernelIr, MemoryLevel,
    ScalarElement, Source, Stmt, TileBinaryOp, TileCompareOp, TileExpr, TileLayout, TileLiteral,
};
use fusor2_ir::ir::{Node, Op};
use fusor2_ir::shape::{AxisGroup, Dim, Layout, MultiFlattenMap, SubAxis, SymId};
use fusor2_ir::target::LowerCtx;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

use crate::uniforms::UniformPack;

/// Binding index of the always-present uniform block.
pub const UNIFORM_BINDING: u32 = 0;

/// One staged input of a contraction side: a memory source, or a `Const`
/// leaf already folded to its literal.
#[derive(Clone)]
pub enum StagedSource {
    Mem(Source),
    Const(TileExpr),
}

/// The storage layout and dtype one launch binding reads or writes.
///
/// `Plan::buffers` covers only what the plan produces; an external leaf has no
/// `BufferPlan`. Where a `BufferPlan` exists it is authoritative, being the
/// padded stride set the extractor committed to; otherwise the value is a leaf
/// and its own facts describe it.
pub fn bound_layout(cx: &LowerCtx<'_>, value: Id) -> (Layout, Dtype) {
    let value = cx.selected(value);
    match cx.plan.buffers.iter().find(|b| b.value == value) {
        Some(b) => (b.layout.clone(), b.dtype),
        None => {
            let facts = cx.graph.facts(value);
            (Layout::contiguous(&facts.shape), facts.dtype)
        }
    }
}

/// Element count of a layout under the dispatch bindings. Resolved per axis,
/// since `BufferPlan::elements` is the "unknown" sentinel as soon as one
/// extent is symbolic.
fn layout_elements(binding: &DimBinding, layout: &Layout) -> Result<u64> {
    let mut acc: u64 = 1;
    for d in layout.shape() {
        acc = acc.saturating_mul(binding.require(*d)?);
    }
    Ok(acc)
}

/// Runtime extents for the plan's symbols. A plan is compiled once for a whole
/// shape family: the grid reads this, and the kernel body reads binding 0.
#[derive(Clone, Debug, Default)]
pub struct DimBinding {
    values: FxHashMap<SymId, u64>,
}

impl DimBinding {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_pairs(pairs: impl IntoIterator<Item = (SymId, u64)>) -> Self {
        Self {
            values: pairs.into_iter().collect(),
        }
    }

    pub fn bind(&mut self, sym: SymId, value: u64) {
        self.values.insert(sym, value);
    }

    pub fn get(&self, sym: SymId) -> Option<u64> {
        self.values.get(&sym).copied()
    }

    pub fn values(&self) -> &FxHashMap<SymId, u64> {
        &self.values
    }

    /// Concrete extent of a dim, or `None` when the symbol is unbound.
    pub fn resolve(&self, dim: Dim) -> Option<u64> {
        match dim {
            Dim::Const(v) => Some(v),
            Dim::Sym(s) => self.get(s),
        }
    }

    /// Concrete extent, or `Error::Plan` when the symbol is unbound.
    pub fn require(&self, dim: Dim) -> Result<u64> {
        self.resolve(dim)
            .ok_or_else(|| Error::Plan(format!("dim {dim} is unbound at dispatch")))
    }
}

/// Fold a 1-D workgroup count onto the 3-D dispatch grid.
///
/// The slab count is picked first and `x` sized to the slab, so no slab is
/// left nearly empty; an over-launched workgroup still runs the kernel
/// prologue and the in-range compares.
pub fn distribute_workgroups(total: u32, max_per_dim: u32) -> [u32; 3] {
    let max_per_dim = max_per_dim.max(1);
    if total <= max_per_dim {
        return [total, 1, 1];
    }
    let y = total.div_ceil(max_per_dim).min(max_per_dim);
    let x = total.div_ceil(y).min(max_per_dim);
    let z = total.div_ceil(x.saturating_mul(y)).max(1);
    [x, y, z]
}

/// The dispatch grid for an index space at a given workgroup width.
pub fn grid_for(
    space: &IndexSpace,
    block: u32,
    binding: &DimBinding,
    limits: &Limits,
) -> Result<[u32; 3]> {
    let mut elements: u64 = 1;
    for dim in &space.dims {
        elements = elements
            .checked_mul(binding.require(*dim)?)
            .ok_or_else(|| Error::Plan("index space overflows a u64".into()))?;
    }
    let block = u64::from(block.max(1));
    let groups = elements.div_ceil(block);
    let groups = u32::try_from(groups)
        .map_err(|_| Error::Plan(format!("{groups} workgroups exceeds a u32")))?;
    Ok(distribute_workgroups(
        groups,
        limits.max_compute_workgroups_per_dimension,
    ))
}

/// An N-D strided operand seen as a 2-D matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixView {
    pub rows: u32,
    pub cols: u32,
    pub offset: u32,
    pub layout: TileLayout,
}

/// Flatten a strided layout into a 2-D matrix view: `shape[..row_dims]`
/// flattens to rows, `shape[row_dims..]` to columns.
///
/// Sides whose dims merge affinely use a plain strided layout; anything else
/// (a conv im2col window, a non-affine batch prefix) becomes a
/// [`MultiFlattenMap`] whose sub-axes divmod the flat coordinate apart per
/// load. Extent-1 axes are dropped, saving a divmod per load. A failure is
/// [`Error::Plan`]. `row_dims` may be `0` or `rank`, and an empty side
/// contributes a single index of 0, so its stride never enters an address.
pub fn flatten_matrix_layout_split(
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
    for (axis, s) in layout.strides().iter().enumerate() {
        // A `row_major_strides` placeholder means the plan carried a stride it
        // never derived. Recompute it from the (now concrete) shape rather
        // than emitting the placeholder.
        let v = match s {
            Dim::Sym(sym) if *sym == crate::uniforms::DERIVED_STRIDE => {
                shape.iter().skip(axis + 1).product::<u64>()
            }
            other => binding.require(*other)?,
        };
        strides.push(v);
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
    let innermost = |lo: usize, hi: usize| -> u64 {
        if lo == hi { 0 } else { strides[hi - 1] }
    };
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
                // Extent-1 axes contribute nothing to the flat coordinate
                // decomposition; dropping them saves a divmod per load.
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

/// The axis split that presents `layout` as exactly `rows` by `cols`
/// elements.
///
/// [`L1::KContract`](fusor2_ir::ir::level1::L1::KContract) records the `m`,
/// `n`, `k` and `batch` extents, not the label partition, so the axis count on
/// each side is recovered here: `canonical_for_mnk` admits only
/// `a = [batch.., m.., k..]` and `b = [batch.., k.., n..]`, so the split is the
/// position whose prefix multiplies to `rows` and whose suffix multiplies to
/// `cols`. The longest qualifying prefix is taken, pinning the choice when an
/// extent-1 axis makes two positions equivalent.
pub fn matrix_split_for(
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

/// `u32` words a block-quantized value of `elements` elements occupies.
pub fn quantized_words(fmt: fusor2_ir::dtype::QFmt, layout: QLayout, elements: u64) -> u64 {
    let blocks = elements.div_ceil(u64::from(fmt.block_elements()).max(1));
    (blocks * u64::from(fmt.block_bytes(layout))).div_ceil(4)
}

pub use fusor2_tile::lower::{qlayout_of, scalar_element};

/// Hash-consing L2 term builder: [`fusor2_tile::build::TileBuilder`] owns the
/// memo and every typed constructor, so two identical subtrees built
/// separately return the same `Arc`. Only the finite-literal clamps below are
/// GPU-side, as naga obligations.
pub type L2 = fusor2_tile::build::TileBuilder;

/// Clamp an infinite literal to the largest finite value WGSL can spell.
///
/// WGSL has no infinite literal and naga rejects a module holding one, while a
/// causal mask (`select(kv <= q, scale*s, -inf)`) and the `Fold{Max}` identity
/// both carry one. The sentinel is `-3.40282e38`, the same one `emit::expr`'s
/// reduce identities use; `exp(x - m)` underflows to zero against it.
pub fn finite_f32(v: f32) -> f32 {
    if v.is_infinite() {
        if v.is_sign_negative() {
            -crate::emit::expr::WGSL_SAFE_F32_MAX
        } else {
            crate::emit::expr::WGSL_SAFE_F32_MAX
        }
    } else if v.is_nan() {
        0.0
    } else {
        v
    }
}

/// [`finite_f32`] on the f16 bit pattern: 0x7C00 / 0xFC00 are the
/// infinities, 65504 the largest finite magnitude.
pub fn finite_f16(bits: u16) -> u16 {
    let v = half::f16::from_bits(bits);
    if v.is_infinite() || v.is_nan() {
        half::f16::from_f32(if v.is_sign_negative() && v.is_infinite() {
            -65504.0
        } else if v.is_infinite() {
            65504.0
        } else {
            0.0
        })
        .to_bits()
    } else {
        bits
    }
}

/// [`finite_f32`] on the bf16 bit pattern.
pub fn finite_bf16(bits: u16) -> u16 {
    let v = half::bf16::from_bits(bits);
    if v.is_nan() {
        return half::bf16::ZERO.to_bits();
    }
    if !v.is_infinite() {
        return bits;
    }
    // `from_f32(-3.40282e38)` rounds *back* to -inf in bf16, so take bf16's
    // own finite extreme rather than round-tripping the f32 sentinel.
    if v.is_sign_negative() {
        half::bf16::MIN.to_bits()
    } else {
        half::bf16::MAX.to_bits()
    }
}

/// Read a `u32` word out of the uniform block through `b`.
fn uniform_word_in(b: &mut L2, uniform: &Buffer, slot: u32) -> TileExpr {
    let view = fusor2_ir::ir::level2::StorageView {
        buffer: uniform.clone(),
        offset: 0,
        layout: uniform.layout.clone(),
    };
    let index = b.lit_u32(slot);
    let mask = b.lit_bool(true);
    let fill = b.lit_u32(0);
    b.load(Source::Storage(view), Addr::Linear(index), mask, fill)
}

/// The GPU's [`fusor2_tile::lower::ScalarEnv`]: a runtime scalar reads a
/// binding-0 word, and every literal is clamped finite — naga rejects a
/// module holding an infinite literal.
struct GpuScalarEnv<'a> {
    pack: &'a UniformPack,
    uniform: Buffer,
}

impl fusor2_tile::lower::ScalarEnv for GpuScalarEnv<'_> {
    fn uniform(
        &mut self,
        b: &mut L2,
        sym: SymId,
        _dtype: Dtype,
    ) -> Result<TileExpr> {
        let slot = self
            .pack
            .scalar_slot(sym)
            .ok_or_else(|| Error::Plan(format!("scalar {sym} has no uniform slot")))?;
        let word = uniform_word_in(b, &self.uniform, slot);
        Ok(b.bitcast(word, ElementType::Scalar(ScalarElement::F32)))
    }

    fn literal(&mut self, b: &mut L2, value: fusor2_ir::dtype::Splat) -> TileExpr {
        use fusor2_ir::dtype::Splat;
        match value {
            Splat::F32(v) => b.lit_f32(finite_f32(v)),
            Splat::F16(v) => b.lit(TileLiteral::F16(finite_f16(v))),
            Splat::BF16(v) => b.lit(TileLiteral::BF16(finite_bf16(v))),
            Splat::U32(v) => b.lit_u32(v),
            Splat::I32(v) => b.lit_i32(v),
        }
    }
}

/// Per-kernel lowering state: the buffer table in binding order, the uniform
/// word layout, and the L2 builder.
pub struct Ctx<'a> {
    pub caps: &'a Caps,
    pub cx: &'a LowerCtx<'a>,
    pub b: L2,
    pub binding: DimBinding,
    /// Binding order. Index 0 is always the uniform block.
    pub buffers: Vec<Buffer>,
    /// `Plan` value -> index into [`Self::buffers`].
    slot_of: FxHashMap<Id, usize>,
    pack: UniformPack,
}

impl<'a> Ctx<'a> {
    /// Build the buffer table for one launch.
    ///
    /// Binding 0 is the uniform block; every plan binding follows in
    /// `BindingPlan::binding` order at `1 + position`. A kernel reads a
    /// symbolic extent from binding 0, so the extent stays out of its identity.
    pub fn new(caps: &'a Caps, cx: &'a LowerCtx<'a>, binding: DimBinding) -> Result<Self> {
        let pack = UniformPack::new(cx.plan);
        let uniform_words = (pack.byte_len() / 4).max(1) as u32;
        let mut buffers: Vec<Buffer> = vec![Arc::new(BufferDecl {
            binding: UNIFORM_BINDING,
            element: ElementType::Scalar(ScalarElement::U32),
            layout: TileLayout::contiguous(MemoryLevel::Storage, &[uniform_words]),
            access: BufferAccess::Read,
        })];

        let mut ordered: Vec<_> = cx.launch.bindings.iter().collect();
        ordered.sort_by_key(|b| b.binding);

        let mut slot_of = FxHashMap::default();
        for (position, plan_binding) in ordered.iter().enumerate() {
            let (layout, dtype) = bound_layout(cx, plan_binding.value);
            let elements = layout_elements(&binding, &layout)?;
            // A quantized buffer holds blocks, not elements: it binds as the
            // `u32` word stream the decode program addresses.
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
                fusor2_ir::extract::BindKind::Read => BufferAccess::Read,
                _ => BufferAccess::ReadWrite,
            };
            // Keyed by every id in the value's class: an `Operand::src` may
            // name any of them and they all denote the same buffer.
            // `class_ids` rather than `chain`, which drops the `Union` spine
            // that a macro op hands back to its caller.
            let class = cx.graph.class_of(plan_binding.value);
            for member in cx.graph.class_ids(class) {
                slot_of.insert(member, buffers.len());
            }
            buffers.push(Arc::new(BufferDecl {
                binding: 1 + position as u32,
                element: ElementType::Scalar(scalar_element(dtype)),
                layout: TileLayout::contiguous(MemoryLevel::Storage, &[extent.max(1)]),
                access,
            }));
        }

        Ok(Self {
            caps,
            cx,
            b: L2::new(),
            binding,
            buffers,
            slot_of,
            pack,
        })
    }

    /// The bound buffer for a plan value.
    pub fn buffer(&self, value: Id) -> Result<Buffer> {
        let slot = self
            .slot_of
            .get(&value)
            .ok_or_else(|| Error::Plan(format!("value {value} is not bound by this launch")))?;
        Ok(self.buffers[*slot].clone())
    }

    /// The `BufferPlan` layout for a value, never re-derived where the plan
    /// has one. See [`bound_layout`] for the leaf case.
    pub fn plan_layout(&self, value: Id) -> Result<Layout> {
        Ok(bound_layout(self.cx, value).0)
    }

    pub fn plan_dtype(&self, value: Id) -> Result<Dtype> {
        Ok(bound_layout(self.cx, value).1)
    }

    /// A flat rank-1 view of a value's buffer, for elementwise access.
    pub fn linear_view(&self, value: Id) -> Result<fusor2_ir::ir::level2::StorageView> {
        let buffer = self.buffer(value)?;
        let layout = buffer.layout.clone();
        Ok(fusor2_ir::ir::level2::StorageView {
            buffer,
            offset: 0,
            layout,
        })
    }

    /// A 2-D matrix view of an operand, split at `row_dims`, built from the
    /// plan's layout.
    pub fn matrix_view(
        &self,
        operand: &Operand,
        row_dims: usize,
    ) -> Result<fusor2_ir::ir::level2::StorageView> {
        let layout = self.repad_operand_layout(operand)?;
        let view = flatten_matrix_layout_split(&layout, row_dims, &self.binding)?;
        let buffer = self.buffer(operand.src)?;
        Ok(fusor2_ir::ir::level2::StorageView {
            buffer,
            offset: view.offset,
            layout: view.layout,
        })
    }

    /// An operand's layout restated over the producer's *plan* buffer.
    ///
    /// The operand's strides address the producer's logical dense element
    /// space; the buffer holds what the plan laid out, and the two differ when
    /// the producer's schedule point padded it. Every operand axis must walk
    /// one producer axis, and the restatement substitutes the padded stride
    /// for the dense one axis by axis. An operand whose stride is no producer
    /// axis's own is an error, never a silent dense read.
    fn repad_operand_layout(&self, operand: &Operand) -> Result<Layout> {
        let selected = self.cx.selected(operand.src);
        let Some(plan) = self
            .cx
            .plan
            .buffers
            .iter()
            .find(|b| b.value == selected)
        else {
            return Ok(operand.layout.clone());
        };
        let logical = self.cx.graph.facts(selected).shape.clone();
        if plan.layout.rank() != logical.len() || logical.is_empty() {
            return Ok(operand.layout.clone());
        }
        let dense = Layout::row_major_strides(&logical);
        let unpadded = plan.layout.offset().known_eq(Dim::Const(0))
            && plan
                .layout
                .shape()
                .iter()
                .zip(&logical)
                .all(|(p, l)| p.known_eq(*l))
            && plan
                .layout
                .strides()
                .iter()
                .zip(&dense)
                .all(|(s, w)| s.known_eq(*w));
        if unpadded {
            return Ok(operand.layout.clone());
        }
        if !plan.layout.offset().known_eq(Dim::Const(0)) {
            return Err(Error::Plan(format!(
                "operand {} reads a buffer at offset {}; the contraction path \
                 cannot restate an offset layout",
                operand.src,
                plan.layout.offset()
            )));
        }
        // A reshaped spelling of the producer (a `[2, 2, 3, 4]` read of a
        // `[4, 3, 4]` contract) walks `k` steps of one producer axis: its
        // stride is `k * dense[i]` and it stays inside that axis
        // (`k * (ext - 1) < logical[i]`), so `k * padded[i]` restates it.
        let padded = plan.layout.strides();
        let remap = |ext: Dim, s: Dim| -> Result<Dim> {
            // Unobservable axes keep whatever they said.
            if ext.known_eq(Dim::Const(1)) || s.known_eq(Dim::Const(0)) {
                return Ok(s);
            }
            let (Some(sv), Some(ev)) = (s.as_const(), ext.as_const()) else {
                return Err(Error::Plan(format!(
                    "operand {} reads a padded buffer through symbolic stride {s}",
                    operand.src
                )));
            };
            for (i, d) in dense.iter().enumerate() {
                let (Some(dv), Some(lv)) = (d.as_const(), logical[i].as_const()) else {
                    continue;
                };
                if dv == 0 || sv % dv != 0 {
                    continue;
                }
                let k = sv / dv;
                if k >= 1 && k.saturating_mul(ev - 1) < lv {
                    let pv = padded[i].as_const().ok_or_else(|| {
                        Error::Plan(format!(
                            "operand {} reads a buffer with symbolic padded stride",
                            operand.src
                        ))
                    })?;
                    return Ok(Dim::Const(k * pv));
                }
            }
            Err(Error::Plan(format!(
                "operand {} reads a padded buffer through stride {s}, which walks \
                 no single axis of the producer's dense layout {dense:?}",
                operand.src
            )))
        };
        let strides: Vec<Dim> = operand
            .layout
            .shape()
            .iter()
            .zip(operand.layout.strides())
            .map(|(ext, s)| remap(*ext, *s))
            .collect::<Result<_>>()?;
        Layout::from_parts(operand.layout.offset(), operand.layout.shape(), &strides)
    }

    /// The [`Source`] a contraction stages one operand from.
    ///
    /// Dense operands read storage. A block-quantized operand reads
    /// [`Source::Quantized`], whose decode program the L2 emitter runs at the
    /// `(row, col)` the staging fill already computes — so a quantized weight
    /// costs the decode math on the way into shared memory and nothing else.
    /// The staging tile, the fragments, the MMA and the arena footprint are the
    /// dense ones.
    pub fn contract_stage_source(
        &mut self,
        operand: &Operand,
        view: &fusor2_ir::ir::level2::StorageView,
    ) -> Result<Source> {
        let Dtype::Q(fmt) = self.plan_dtype(operand.src)? else {
            return Ok(Source::Storage(view.clone()));
        };
        let qlayout = qlayout_of(self.cx, operand.src).unwrap_or(QLayout::Native);
        Ok(Source::Quantized(fusor2_ir::ir::level2::QuantizedView {
            data: view.clone(),
            fmt,
            layout: qlayout,
        }))
    }

    /// Every buffer one contraction side reads, as a staging source apiece.
    ///
    /// A side is a list because an absorbed producer brings its own edges: a
    /// GGUF block decode arrives with the quant plane, the block scale, the
    /// block minimum and the group scales, each a `Restride` of the same block
    /// stream at its own offset. They share the side's `(rows, cols)` index and
    /// differ only in strides, so each gets its own view and all are loaded at
    /// the same coordinate before the side's `pre` runs over the results.
    pub fn contract_side_sources(
        &mut self,
        side: &ContractSide,
        rows: u32,
        cols: u32,
    ) -> Result<Vec<StagedSource>> {
        side.ops
            .iter()
            .map(|o| {
                // A `Const` leaf is folded into the kernel: no buffer, no
                // binding. Absorbed producers bring these — a layer norm's
                // `1/N`, an epsilon.
                if let Some(lit) = self.const_operand(o.src) {
                    return Ok(StagedSource::Const(lit));
                }
                let view = self.contract_operand_view(o, rows, cols)?;
                Ok(StagedSource::Mem(self.contract_stage_source(o, &view)?))
            })
            .collect()
    }

    pub fn contract_operand_view(
        &self,
        operand: &Operand,
        rows: u32,
        cols: u32,
    ) -> Result<fusor2_ir::ir::level2::StorageView> {
        let split =
            matrix_split_for(&operand.layout, &self.binding, u64::from(rows), u64::from(cols))?;
        self.matrix_view(operand, split)
    }

    /// Read a `u32` word out of binding 0.
    pub fn uniform_word(&mut self, slot: u32) -> TileExpr {
        uniform_word_in(&mut self.b, &self.buffers[0], slot)
    }

    /// A `u32` expression for a dim: a literal when constant, a binding-0 word
    /// when symbolic. A sequence length is a word, never a baked constant.
    pub fn dim_expr(&mut self, dim: Dim) -> Result<TileExpr> {
        match dim {
            Dim::Const(v) => {
                let v = u32::try_from(v)
                    .map_err(|_| Error::Plan(format!("extent {v} exceeds a u32")))?;
                Ok(self.b.lit_u32(v))
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

    /// An `f32` expression for a runtime scalar: `m * lr` reads a word, so a
    /// learning-rate change recompiles nothing.
    pub fn scalar_expr(&mut self, sym: SymId) -> Result<TileExpr> {
        let slot = self
            .pack
            .scalar_slot(sym)
            .ok_or_else(|| Error::Plan(format!("scalar {sym} has no uniform slot")))?;
        let word = self.uniform_word(slot);
        Ok(self
            .b
            .bitcast(word, ElementType::Scalar(ScalarElement::F32)))
    }

    /// The global linear element index this invocation owns, linearized
    /// against this launch's grid.
    ///
    /// `grid` must be the same `[x, y, z]` the kernel is dispatched with, the
    /// one handed to [`Ctx::finish`], not
    /// `max_compute_workgroups_per_dimension`: `x` is sized to the slab, so
    /// the limit as a stride puts every workgroup past the first slab at an
    /// out-of-range index where it masks itself out.
    pub fn global_index(&mut self, block: u32, grid: [u32; 3]) -> TileExpr {
        let lane = self.b.builtin(Builtin::Lane);
        let gx = self.b.builtin(Builtin::ProgramId(
            fusor2_ir::ir::level2::WorkgroupAxis::X,
        ));
        let gy = self.b.builtin(Builtin::ProgramId(
            fusor2_ir::ir::level2::WorkgroupAxis::Y,
        ));
        let gz = self.b.builtin(Builtin::ProgramId(
            fusor2_ir::ir::level2::WorkgroupAxis::Z,
        ));
        // group = gx + gy*X + gz*X*Y, exactly as the grid fold laid it out.
        let x_e = self.b.lit_u32(grid[0].max(1));
        let xy_e = self.b.lit_u32(grid[0].max(1).saturating_mul(grid[1].max(1)));
        let yx = self.b.mul(gy, x_e);
        let zxy = self.b.mul(gz, xy_e);
        let group = self.b.add(gx, yx);
        let group = self.b.add(group, zxy);
        let block_e = self.b.lit_u32(block);
        let base = self.b.mul(group, block_e);
        self.b.add(base, lane)
    }

    /// The value this launch writes: the launch root when it is bound for
    /// writing, else the first writable binding.
    pub fn output(&self) -> Result<Id> {
        let root = self.cx.launch.root;
        if self
            .cx
            .launch
            .bindings
            .iter()
            .any(|b| b.value == root && b.kind != fusor2_ir::extract::BindKind::Read)
        {
            return Ok(root);
        }
        self.cx
            .launch
            .bindings
            .iter()
            .find(|b| b.kind != fusor2_ir::extract::BindKind::Read)
            .map(|b| b.value)
            .ok_or_else(|| Error::Plan("launch binds nothing writable".into()))
    }

    /// Per-axis coordinates of a flat index over `space`, most-significant
    /// axis first. One divmod per axis past the innermost, exactly as the
    /// index-op cost term prices.
    pub fn coords_from_linear(
        &mut self,
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
            coords[axis] =
                self.b
                    .binary(TileBinaryOp::Rem, rest.clone(), extent.clone(), NumericContract::RELAXED);
            rest = self
                .b
                .binary(TileBinaryOp::Div, rest, extent, NumericContract::RELAXED);
        }
        Ok(coords)
    }

    /// Translate a [`fusor2_ir::scalar::ScalarExpr`] body into L2, through
    /// the shared walker with this backend's uniform access and finite
    /// literal clamps plugged in.
    pub fn eval_scalar(
        &mut self,
        expr: &fusor2_ir::scalar::ScalarExpr,
        args: &[TileExpr],
        coords: &[TileExpr],
    ) -> Result<TileExpr> {
        let mut env = GpuScalarEnv {
            pack: &self.pack,
            uniform: self.buffers[0].clone(),
        };
        fusor2_tile::lower::eval_scalar(&mut self.b, &mut env, expr, args, coords)
    }

    fn zero_of(&mut self, elem: ElementType) -> TileExpr {
        fusor2_tile::lower::zero_of(&mut self.b, elem)
    }

    /// Load one operand at the reading kernel's flat space index, running it
    /// through the edge's [`fusor2_ir::ir::level1::AddressMap`] first.
    ///
    /// [`Ctx::load_operand`] is the raw form, for readers that computed a
    /// storage index themselves (gather, scatter, the contraction nests).
    /// Everything whose index is the space coordinate comes through here: a
    /// stride-0 broadcast axis, a transposed view, a narrowed slice and a conv
    /// window all disagree with the bare flat index.
    pub fn load_mapped(
        &mut self,
        operand: &Operand,
        flat: TileExpr,
        space_total: u64,
    ) -> Result<TileExpr> {
        let addr = self.operand_address(operand, flat, space_total)?;
        self.load_operand(operand, addr)
    }

    /// `flat` run through one operand's index map.
    pub fn operand_address(
        &mut self,
        operand: &Operand,
        flat: TileExpr,
        space_total: u64,
    ) -> Result<TileExpr> {
        let map = operand.address_map().ok_or_else(|| {
            Error::Plan(
                "the GPU lowering path needs a decidable operand index map; a symbolic \
                 stride must be specialized or bound through the uniform block first"
                    .into(),
            )
        })?;
        Ok(fusor2_tile::lower::map_address(
            &mut self.b,
            &map,
            flat,
            space_total,
        ))
    }

    /// Re-address a logical dense element index of `src` into the buffer the
    /// plan laid out for it.
    ///
    /// `fusor2_cost::plan::buffer_layout_for` pads a `Coop` contraction's
    /// output to whole `bm x bn` blocks, while every reader of that value names
    /// its elements densely, so a `[16, 1]` contraction padded to `[16, 16]`
    /// would otherwise be read as the first sixteen elements of row 0.
    ///
    /// Emits nothing when the plan's layout is the logical dense one.
    fn repad_index(&mut self, src: Id, index: TileExpr) -> Result<TileExpr> {
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
        // The delinearize needs every extent decidable; otherwise keep the
        // dense address, which the rest of the launch agrees on.
        let Ok(extents) = logical
            .iter()
            .map(|d| self.binding.require(*d))
            .collect::<Result<Vec<u64>>>()
        else {
            return Ok(index);
        };
        let Ok(logical_strides) = dense
            .iter()
            .map(|d| self.binding.require(*d))
            .collect::<Result<Vec<u64>>>()
        else {
            return Ok(index);
        };
        let Ok(padded_strides) = strides
            .iter()
            .map(|d| self.binding.require(*d))
            .collect::<Result<Vec<u64>>>()
        else {
            return Ok(index);
        };
        let offset = plan.layout.offset().as_const().unwrap_or(0);

        let mut acc: Option<TileExpr> = (offset != 0).then(|| {
            let o = u32::try_from(offset).unwrap_or(u32::MAX);
            self.b.lit_u32(o)
        });
        for axis in 0..logical.len() {
            let extent = extents[axis];
            let stride = padded_strides[axis];
            if extent <= 1 || stride == 0 {
                continue;
            }
            let mut e = index.clone();
            let div = logical_strides[axis];
            if div > 1 {
                let d = self.b.lit_u32(u32::try_from(div).unwrap_or(u32::MAX));
                e = self
                    .b
                    .binary(TileBinaryOp::Div, e, d, NumericContract::RELAXED);
            }
            // The most significant axis needs no `%`: `flat` is already below
            // its bound for every live lane, and an overhang lane is masked.
            if axis > 0 {
                let m = self.b.lit_u32(u32::try_from(extent).unwrap_or(u32::MAX));
                e = self
                    .b
                    .binary(TileBinaryOp::Rem, e, m, NumericContract::RELAXED);
            }
            if stride != 1 {
                let s = self.b.lit_u32(u32::try_from(stride).unwrap_or(u32::MAX));
                e = self.b.mul(e, s);
            }
            acc = Some(match acc {
                Some(a) => self.b.add(a, e),
                None => e,
            });
        }
        Ok(match acc {
            Some(a) => a,
            None => self.b.lit_u32(0),
        })
    }

    /// Load one operand at an already-computed **storage** element index. The
    /// mask is the plan's runtime bounds obligation; a load is never emitted
    /// unmasked unless the extent is a compile-time multiple of the block.
    pub fn load_operand(&mut self, operand: &Operand, index: TileExpr) -> Result<TileExpr> {
        // A `Leaf::Const` is folded into the kernel: `LeafRole::Free` in the
        // plan, so `derive_bindings` emits no binding for it.
        if let Some(lit) = self.const_operand(operand.src) {
            return Ok(lit);
        }
        // An `Operand`'s index arithmetic is stated over the producer's
        // logical dense element space, and the buffer holds what the plan laid
        // out. A `Coop` contraction pads `m` to `bm` and `n` to `bn` so its
        // subgroup-collective store needs no mask.
        let index = self.repad_index(operand.src, index)?;
        let buffer = self.buffer(operand.src)?;
        // A block-quantized operand has no dense element to load: reading
        // element `i` runs the format's decode program at flat index `i`.
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
            let bound = self.b.lit_u32(u32::try_from(rows.saturating_mul(cols)).unwrap_or(u32::MAX));
            let mask = self.b.compare(TileCompareOp::Lt, index.clone(), bound);
            let fill = self.b.lit_f32(0.0);
            let layout = buffer.layout.clone();
            let view = fusor2_ir::ir::level2::QuantizedView {
                data: fusor2_ir::ir::level2::StorageView {
                    buffer,
                    offset: 0,
                    layout,
                },
                fmt,
                layout: qlayout,
            };
            return Ok(self
                .b
                .load(Source::Quantized(view), Addr::Linear(index), mask, fill));
        }
        let elem = buffer.element;
        // `index` is a storage element index and `Addr::Linear` addresses the
        // buffer directly, so the bound is the buffer's own extent; an
        // `Unflatten` map describes the reading space and `operand_address`
        // has already consumed it.
        let layout = buffer.layout.clone();
        let extent: u32 = layout.extents.iter().product::<u32>().max(1);
        let view = fusor2_ir::ir::level2::StorageView {
            buffer,
            offset: 0,
            layout,
        };
        let bound = self.b.lit_u32(extent);
        let mask = self.b.compare(TileCompareOp::Lt, index.clone(), bound);
        let fill = self.zero_of(elem);
        Ok(self
            .b
            .load(Source::Storage(view), Addr::Linear(index), mask, fill))
    }

    /// The literal a `Leaf::Const` operand folds to, if it is one.
    pub(crate) fn const_operand(&mut self, src: Id) -> Option<TileExpr> {
        fusor2_tile::lower::const_operand(&mut self.b, self.cx, src)
    }

    /// Finish a kernel body into a [`KernelIr`].
    pub fn finish(self, name: &'static str, grid: [u32; 3], block: u32, body: Vec<Stmt>) -> KernelIr {
        KernelIr {
            buffers: self.buffers,
            grid,
            block,
            body,
            byte_arena: if self.caps.workgroup_alias {
                Some(fusor2_ir::ir::level2::ByteArenaToken)
            } else {
                None
            },
            name,
        }
    }
}

/// Lower one selected L1 node at one schedule point.
///
/// One match over `L1` into the eight submodule entry points. Every arm has a
/// real body; there is no "unsupported, fall back" path.
pub fn lower_node(
    caps: &Caps,
    node: &Node,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
    binding: DimBinding,
) -> Result<Vec<KernelIr>> {
    let Op::L1(op) = &node.op else {
        return Err(Error::Plan(format!(
            "lowering was handed a {:?} node, but only L1 nodes are selectable",
            node.level
        )));
    };
    let ctx = Ctx::new(caps, cx, binding)?;
    match op {
        L1::KMap { .. } => map_fold::lower_kmap(ctx, op, theta).map(|k| vec![k]),
        L1::KFold { .. } => map_fold::lower_kfold(ctx, op, theta).map(|k| vec![k]),
        L1::KContract { family, .. } => contract::lower_contract(ctx, op, *family, theta),
        L1::KGather { .. } => gather_scatter::lower_kgather(ctx, op, theta).map(|k| vec![k]),
        L1::KScatter { .. } => gather_scatter::lower_kscatter(ctx, op, theta),
        L1::KMerged(m) => merged::lower_kmerged(ctx, m, theta).map(|k| vec![k]),
        L1::KRegion { .. } => merged::lower_kregion(ctx, op, theta).map(|k| vec![k]),
        L1::Ext { def, .. } => ext::lower(*def, node, theta).map(|k| vec![k]),
    }
}

/// `L1::Ext` lowering: the one escape hatch out of the closed `L0`/`L1` enums.
/// The registry itself lives in [`fusor2_ir::target::ext`], shared with every
/// other target and keyed by the target's name.
pub mod ext {
    use super::*;
    use fusor2_ir::ir::OpDefId;

    pub use fusor2_ir::target::ext::{install, installed};

    /// Lower one registered extension op through its `"gpu"` row.
    pub fn lower(def: OpDefId, node: &Node, theta: SchedPoint) -> Result<KernelIr> {
        fusor2_ir::target::ext::lower("gpu", def, node, theta)
    }
}

/// Dispatch one selected L1 node to its family lowering.
///
/// The [`fusor2_ir::target::Target`] contract returns exactly one `KernelIr`
/// per node; families that need two launches (split-K, sort-then-segment) are
/// driven through [`lower_node`] by the executor, which binds their scratch.
pub fn lower(
    caps: &Caps,
    node: &Node,
    id: Id,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
) -> Result<KernelIr> {
    let _ = id;
    let binding = DimBinding::new();
    let mut kernels = lower_node(caps, node, theta, cx, binding)?;
    if kernels.len() == 1 {
        Ok(kernels.remove(0))
    } else {
        Err(Error::Plan(format!(
            "{} lowers to {} kernels; use lower_node",
            node.op.tag() as u32,
            kernels.len()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slack is strictly under one slab, and `122_880` yields
    /// `[61440, 2, 1]`, not `[65535, 2, 1]`.
    #[test]
    fn distribute_workgroups_slack_under_one_slab() {
        const MAX: u32 = 65535;
        for total in [1u32, 65535, 65536, 122_880, 4_000_000] {
            let [x, y, z] = distribute_workgroups(total, MAX);
            assert!(x <= MAX && y <= MAX && z <= MAX, "{total} exceeds the limit");
            let launched = u64::from(x) * u64::from(y) * u64::from(z);
            assert!(launched >= u64::from(total), "{total} is not covered");
            assert!(
                launched - u64::from(total) < u64::from(x.max(1)),
                "{total} launches {launched} = {x}x{y}x{z}, slack is a full slab"
            );
        }
        assert_eq!(distribute_workgroups(122_880, MAX), [61440, 2, 1]);
        assert_ne!(distribute_workgroups(122_880, MAX), [65535, 2, 1]);
    }

    /// The linearization of `@builtin(workgroup_id)` is `gx + gy*X + gz*X*Y`
    /// over the dispatched grid; `max_compute_workgroups_per_dimension` is not
    /// `X`, since `x` is sized to the slab.
    #[test]
    fn the_workgroup_linearization_uses_the_grid_not_the_limit() {
        const MAX: u32 = 65535;
        for total in [1u32, 6, 65535, 65536, 122_880, 200_000, 4_000_000] {
            let [x, y, z] = distribute_workgroups(total, MAX);
            // Every group in `0..total` is hit exactly once by the grid walk.
            let mut seen = vec![false; total as usize];
            for gz in 0..z {
                for gy in 0..y {
                    for gx in 0..x {
                        let id = u64::from(gx)
                            + u64::from(gy) * u64::from(x)
                            + u64::from(gz) * u64::from(x) * u64::from(y);
                        if let Some(slot) = seen.get_mut(id as usize) {
                            assert!(!*slot, "{total}: group {id} visited twice");
                            *slot = true;
                        }
                    }
                }
            }
            assert!(seen.iter().all(|s| *s), "{total}: a group was never visited");
        }

        // At 6 groups the grid is [3, 2, 1], so a `gy * MAX` stride would send
        // the whole second row out of range.
        let grid = distribute_workgroups(6, 4);
        assert_eq!(grid, [3, 2, 1]);
        assert_ne!(grid[0], 4, "the slab width is not the per-dimension limit");
    }

    #[test]
    fn distribute_workgroups_covers_a_wide_sweep() {
        const MAX: u32 = 65535;
        for total in (0..3_000_000).step_by(1409).chain([0, 1, u32::MAX]) {
            let [x, y, z] = distribute_workgroups(total, MAX);
            let launched = u64::from(x) * u64::from(y) * u64::from(z);
            assert!(launched >= u64::from(total), "{total} is not covered");
            assert!(x <= MAX && y <= MAX && z <= MAX);
        }
    }

    /// WGSL has no infinite literal, so every `-inf` the frontend hands down
    /// arrives as the largest finite magnitude.
    #[test]
    fn an_infinite_literal_is_clamped_to_something_wgsl_can_spell() {
        assert!(finite_f32(f32::NEG_INFINITY).is_finite());
        assert!(finite_f32(f32::INFINITY).is_finite());
        assert!(finite_f32(f32::NEG_INFINITY) < -3.0e38);
        assert!(finite_f32(f32::INFINITY) > 3.0e38);
        assert_eq!(finite_f32(-2.5), -2.5);
        assert_eq!(finite_f32(f32::NAN), 0.0);

        let neg = half::f16::from_bits(finite_f16(half::f16::NEG_INFINITY.to_bits()));
        assert!(neg.is_finite() && neg.to_f32() <= -65504.0);
        let bneg = half::bf16::from_bits(finite_bf16(half::bf16::NEG_INFINITY.to_bits()));
        assert!(bneg.is_finite() && bneg.to_f32() < -3.0e38);

        // The max-carrier sentinel agrees with `emit::expr`'s reduce identity,
        // so a max started by hand and one started by a `Reduce` match.
        let mut b = L2::new();
        let sentinel = b.neg_inf(ScalarElement::F32);
        let expected = b.lit_f32(-crate::emit::expr::WGSL_SAFE_F32_MAX);
        assert_eq!(sentinel, expected);
    }

    #[test]
    fn hash_consing_merges_separately_built_subtrees() {
        let mut b = L2::new();
        let a1 = {
            let x = b.lit_f32(2.0);
            let y = b.lit_f32(3.0);
            b.add(x, y)
        };
        let a2 = {
            let x = b.lit_f32(2.0);
            let y = b.lit_f32(3.0);
            b.add(x, y)
        };
        assert_eq!(a1, a2);
        assert_eq!(a1.structural_hash(), a2.structural_hash());
    }

    #[test]
    fn affine_flatten_uses_plain_strides() {
        let binding = DimBinding::new();
        let layout = Layout::contiguous(&[Dim::Const(4), Dim::Const(8), Dim::Const(16)]);
        let view = flatten_matrix_layout_split(&layout, 2, &binding).unwrap();
        assert_eq!(view.rows, 32);
        assert_eq!(view.cols, 16);
        assert!(view.layout.is_affine(), "an affine side must stay affine");
        assert_eq!(view.layout.indexing.groups[0].sub_axes[0].stride, 16);
        assert_eq!(view.layout.indexing.groups[1].sub_axes[0].stride, 1);
    }

    #[test]
    fn non_affine_flatten_emits_sub_axes_and_drops_extent_one() {
        let binding = DimBinding::new();
        // A transposed batch prefix: axis 0 does not merge with axis 1.
        let layout = Layout::from_parts(
            Dim::Const(0),
            &[Dim::Const(3), Dim::Const(1), Dim::Const(5), Dim::Const(7)],
            &[
                Dim::Const(1),
                Dim::Const(0),
                Dim::Const(21),
                Dim::Const(3),
            ],
        )
        .unwrap();
        let view = flatten_matrix_layout_split(&layout, 2, &binding).unwrap();
        assert_eq!(view.rows, 3);
        assert_eq!(view.cols, 35);
        // The extent-1 axis is dropped, saving a divmod per load.
        assert_eq!(view.layout.indexing.groups[0].sub_axes.len(), 1);
        assert_eq!(view.layout.indexing.groups[0].sub_axes[0].extent, 3);
    }

    #[test]
    fn a_split_past_the_rank_is_a_plan_error() {
        let binding = DimBinding::new();
        let layout = Layout::contiguous(&[Dim::Const(4), Dim::Const(8)]);
        assert!(matches!(
            flatten_matrix_layout_split(&layout, 3, &binding),
            Err(Error::Plan(_))
        ));
    }

    /// A contraction whose `n` extent is 1 has no `n` axes, so its B operand
    /// is a one-column matrix: the split lands *on* the rank, not inside it.
    /// The empty side must contribute nothing to the address.
    #[test]
    fn a_degenerate_split_is_a_one_row_or_one_column_matrix() {
        let binding = DimBinding::new();
        let layout = Layout::contiguous(&[Dim::Const(4), Dim::Const(8)]);

        let cols = flatten_matrix_layout_split(&layout, 2, &binding).unwrap();
        assert_eq!((cols.rows, cols.cols), (32, 1));
        assert!(cols.layout.is_affine());
        // Row `r` is element `r`; the empty column side has stride 0.
        assert_eq!(cols.layout.indexing.groups[0].sub_axes[0].stride, 1);
        assert_eq!(cols.layout.indexing.groups[1].sub_axes[0].stride, 0);

        let rows = flatten_matrix_layout_split(&layout, 0, &binding).unwrap();
        assert_eq!((rows.rows, rows.cols), (1, 32));
        assert_eq!(rows.layout.indexing.groups[0].sub_axes[0].stride, 0);
        assert_eq!(rows.layout.indexing.groups[1].sub_axes[0].stride, 1);
    }

    /// The split a contraction operand needs comes from its `(rows, cols)`
    /// element counts, not from its rank: a batched `[b, m, k]` A splits at 2,
    /// and a `[b, k]` B of an `n = 1` contraction splits at 2 as well.
    #[test]
    fn matrix_split_comes_from_the_extents_not_the_rank() {
        let binding = DimBinding::new();
        let a = Layout::contiguous(&[Dim::Const(3), Dim::Const(4), Dim::Const(5)]);
        assert_eq!(matrix_split_for(&a, &binding, 12, 5).unwrap(), 2);
        assert_eq!(matrix_split_for(&a, &binding, 3, 20).unwrap(), 1);
        assert_eq!(matrix_split_for(&a, &binding, 60, 1).unwrap(), 3);
        assert_eq!(matrix_split_for(&a, &binding, 1, 60).unwrap(), 0);

        // `sum(x * x, 1)` over `[3, 5]`: batch 3, m 1, n 1, k 5. A is
        // `[batch * m, k] = [3, 5]` and B is `[batch * k, n] = [15, 1]`.
        let x = Layout::contiguous(&[Dim::Const(3), Dim::Const(5)]);
        assert_eq!(matrix_split_for(&x, &binding, 3, 5).unwrap(), 1);
        assert_eq!(matrix_split_for(&x, &binding, 15, 1).unwrap(), 2);

        assert!(matches!(
            matrix_split_for(&x, &binding, 7, 2),
            Err(Error::Plan(_))
        ));
    }

    #[test]
    fn grid_for_respects_the_per_dimension_limit() {
        let limits = Limits::default();
        let space = IndexSpace::new([Dim::Const(1024), Dim::Const(1024)]);
        let grid = grid_for(&space, 256, &DimBinding::new(), &limits).unwrap();
        let launched = u64::from(grid[0]) * u64::from(grid[1]) * u64::from(grid[2]);
        assert!(launched >= (1024 * 1024) / 256);
        assert!(grid.iter().all(|d| *d <= limits.max_compute_workgroups_per_dimension));
    }

    #[test]
    fn grid_for_needs_every_symbol_bound() {
        let limits = Limits::default();
        let space = IndexSpace::new([Dim::Sym(SymId(0))]);
        assert!(grid_for(&space, 64, &DimBinding::new(), &limits).is_err());
        let bound = DimBinding::from_pairs([(SymId(0), 512)]);
        assert_eq!(grid_for(&space, 64, &bound, &limits).unwrap(), [8, 1, 1]);
    }

    /// The same plan at three sequence lengths produces three grids and one
    /// kernel identity, because the extent never enters the body.
    #[test]
    fn sequence_length_only_moves_the_grid() {
        let limits = Limits::default();
        let space = IndexSpace::new([Dim::Sym(SymId(0)), Dim::Const(64)]);
        let grids: Vec<_> = [256u64, 512, 768]
            .into_iter()
            .map(|n| {
                grid_for(
                    &space,
                    256,
                    &DimBinding::from_pairs([(SymId(0), n)]),
                    &limits,
                )
                .unwrap()
            })
            .collect();
        assert_eq!(grids, vec![[64, 1, 1], [128, 1, 1], [192, 1, 1]]);
    }
}

/// End-to-end cover for the extension seam: a registered `OpDef` lowering to
/// real WGSL, dispatching on the adapter, and returning the numbers it
/// computed.
#[cfg(test)]
mod ext_tests {
    use super::*;
    use fusor2_ir::dtype::Persistence;
    use fusor2_ir::egraph::EGraph;
    use fusor2_ir::extract::{
        BindKind, BindingPlan, Extraction, Launch, Plan, PlanHash,
    };
    use fusor2_ir::facts::{ValueFacts, Work};
    use fusor2_ir::ir::level0::{BufferId, L0, LeafKind};
    use fusor2_ir::ir::level1::{AccessPlan, Effect};
    use fusor2_ir::ir::level2::{
        Addr, BufferAccess, BufferDecl, MemoryLevel, Source, StorageView, TileLayout,
    };
    use fusor2_ir::ir::{AttrId, OpDef, OpDefId, OpDefRegistry, OpTag, VerifyCtx};
    use fusor2_ir::semantics::{CoreSemantics, SumArenaPlanner};
    use fusor2_ir::shape::Layout;
    use fusor2_ir::ir::level2::WorkgroupAxis;
    use fusor2_ir::target::Target;
    use std::sync::Arc;

    const N: u32 = 8;

    /// The registered op's own `"gpu"` lowering: `y = 3 * x`, building the
    /// whole `KernelIr` from the node.
    ///
    /// `L1::schedule` returns `None` for `Ext`, so extraction owes this seam
    /// `SchedPoint::Point`; any other point is an error.
    fn lower_triple(node: &Node, theta: &SchedPoint) -> Result<KernelIr> {
        let SchedPoint::Point = theta else {
            return Err(Error::Plan(
                format!("Ext lowerings take SchedPoint::Point; got {theta:?}").into(),
            ));
        };
        let Op::L1(L1::Ext { ops, .. }) = &node.op else {
            return Err(Error::Plan("triple got a foreign node".into()));
        };
        let n: u32 = ops
            .first()
            .ok_or_else(|| Error::Plan("triple needs an operand".into()))?
            .layout
            .shape()
            .iter()
            .map(|d| d.as_const().unwrap_or(1) as u32)
            .product();
        let f32e = ElementType::Scalar(ScalarElement::F32);
        let decl = |binding, element, len: u32, access| {
            Arc::new(BufferDecl {
                binding,
                element,
                layout: TileLayout::contiguous(MemoryLevel::Storage, &[len]),
                access,
            })
        };
        let uniforms = decl(
            0,
            ElementType::Scalar(ScalarElement::U32),
            1,
            BufferAccess::Read,
        );
        let input = decl(1, f32e, n, BufferAccess::Read);
        let output = decl(2, f32e, n, BufferAccess::ReadWrite);

        let mut b = L2::new();
        let block = 64u32;
        let gid = b.builtin(Builtin::ProgramId(WorkgroupAxis::X));
        let lane = b.builtin(Builtin::Lane);
        let stride = b.lit_u32(block);
        let base = b.mul(gid, stride);
        let index = b.add(base, lane);
        let bound = b.lit_u32(n);
        let mask = b.compare(TileCompareOp::Lt, index.clone(), bound);
        let fill = b.zero_scalar(ScalarElement::F32);
        let view = |d: &Arc<BufferDecl>| StorageView {
            layout: d.layout.clone(),
            buffer: Arc::clone(d),
            offset: 0,
        };
        let x = b.load(
            Source::Storage(view(&input)),
            Addr::Linear(index.clone()),
            mask.clone(),
            fill,
        );
        let three = b.lit_f32(3.0);
        let value = b.mul(x, three);
        Ok(KernelIr {
            buffers: vec![uniforms, input, output.clone()],
            grid: [n.div_ceil(block).max(1), 1, 1],
            block,
            body: vec![Stmt::Store {
                dst: view(&output),
                addr: Addr::Linear(index),
                value,
                mask,
            }],
            byte_arena: None,
            name: "triple",
        })
    }

    fn registry() -> OpDefRegistry {
        fn infer(ins: &[ValueFacts]) -> Result<ValueFacts> {
            ins.first()
                .cloned()
                .ok_or_else(|| Error::Shape("triple needs an operand".into()))
        }
        fn work(_: &[ValueFacts], out: &ValueFacts) -> Work {
            let n = out.elements().unwrap_or(1);
            Work {
                macs: n,
                transcendentals: 0,
                index_ops: n,
                wg_bytes: 0,
            }
        }
        fn verify(_: &VerifyCtx<'_>) -> Result<()> {
            Ok(())
        }
        let mut r = OpDefRegistry::new();
        r.register(OpDef {
            name: "triple",
            tag: OpTag::Ext,
            verify,
            infer,
            work,
            adjoint: None,
            lower_per_target: &[("gpu", lower_triple)],
            effect: Effect::Pure,
        });
        r
    }

    #[test]
    fn a_registered_op_def_lowers_emits_and_dispatches() {
        let reg = registry();
        ext::install(reg.clone());
        let mut g = EGraph::new(CoreSemantics::with_registry(
            Arc::new(SumArenaPlanner),
            reg,
        ));
        let x = g
            .add(Op::L0(L0::Leaf(LeafKind::Buffer {
                name: BufferId(0),
                dtype: fusor2_ir::dtype::Dtype::F32,
                shape: smallvec::smallvec![Dim::Const(N as u64)],
            })))
            .unwrap();
        let ops = vec![Operand {
            src: x,
            layout: Layout::contiguous(&g.facts(x).shape),
            access: AccessPlan::Alias,
        }];
        let e = g
            .add(Op::L1(L1::Ext {
                def: OpDefId(0),
                ops,
                attrs: AttrId(0),
            }))
            .unwrap();

        let plan = Plan {
            extraction: Extraction::default(),
            launches: vec![Launch {
                root: e,
                members: smallvec::smallvec![e],
                bindings: vec![
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
                grid: [1, 1, 1],
                block: 64,
            }],
            buffers: Vec::new(),
            symbols: Vec::new(),
            hash: PlanHash(0),
            cost: fusor2_ir::cost::Picoseconds(0),
        };
        let cx = LowerCtx {
            plan: &plan,
            launch: &plan.launches[0],
            graph: &g,
            symbols: &[],
        };

        let target = crate::target::GpuTarget::new_blocking().unwrap();
        let ir = lower(target.caps(), g.node(e), e, SchedPoint::Point, &cx).unwrap();
        assert_eq!(ir.name, "triple", "the OpDef's own lowering must run");

        let artifact = target.emit(&ir).unwrap();
        let data: Vec<f32> = (1..=N).map(|v| v as f32).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input = target
            .pool()
            .create_buffer_init(&bytes, crate::pool::TENSOR_USAGE)
            .unwrap();
        let out = target
            .alloc((N as u64) * 4, Persistence::Step)
            .unwrap();
        let uniform = target.alloc(4, Persistence::Step).unwrap();
        target
            .launch(
                &artifact,
                ir.grid,
                &[uniform, input, out.clone()],
                &Default::default(),
            )
            .unwrap();
        target.wait().unwrap();
        let back = target
            .launcher()
            .readback(target.pool(), &out, (N as u64) * 4)
            .unwrap();
        let got: Vec<f32> = back[..(N as usize) * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]);

        // And an op with no `"gpu"` row is a plan answer naming the op, never
        // a panic and never a silent fallback.
        let mut empty = OpDefRegistry::new();
        empty.register(OpDef {
            name: "cpu_only",
            tag: OpTag::Ext,
            verify: |_| Ok(()),
            infer: |ins: &[ValueFacts]| {
                ins.first()
                    .cloned()
                    .ok_or_else(|| Error::Shape("no operand".into()))
            },
            work: |_, out: &ValueFacts| Work {
                macs: out.elements().unwrap_or(1),
                ..Work::default()
            },
            adjoint: None,
            lower_per_target: &[],
            effect: Effect::Pure,
        });
        ext::install(empty);
        let err = ext::lower(OpDefId(0), g.node(e), SchedPoint::Point).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("cpu_only") && msg.contains("gpu"), "{msg}");
    }
}
