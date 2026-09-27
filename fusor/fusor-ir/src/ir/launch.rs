//! Launch nests: index spaces, kernels, launches — where fusion, tiling and
//! kernel family are expressed. Buffers are derived from the extracted plan.

use crate::carrier::Carrier;
use crate::dtype::Dtype;
use crate::egraph::Id;
use crate::ir::OpTag;
use crate::scalar::ScalarExpr;
use crate::shape::{Dim, Dims, Layout, MultiFlattenMap, SlidingWindow};
use smallvec::SmallVec;
use std::sync::Arc;

/// The Launch op family.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Launch {
    Map {
        space: IndexSpace,
        body: ScalarExpr,
        ops: Vec<Operand>,
        sched: ScheduleDomain,
    },

    /// A reduction nest over a [`Carrier`], which owns the element
    /// expression, per-slot identities and merge. `space` is
    /// `free.. ++ vec.. ++ [reduced]`.
    Fold {
        space: IndexSpace,
        axis: u32,
        /// Free axes in the accumulator's data space: a contiguous block
        /// before `axis`. Operands address the full `space`; every
        /// [`ScalarExpr`] here is written against [`Launch::iter_space`].
        vec_axes: SmallVec<[u32; 2]>,
        carrier: Carrier,
        acc: Dtype,
        /// One per slot, over `Arg(0..width)`; cross-slot reads are legal.
        post: SmallVec<[ScalarExpr; 4]>,
        ops: Vec<Operand>,
        sched: ScheduleDomain,
    },

    /// A scalar producer Fold inlined at one operand of another Fold. The
    /// replaced operand's `src` is canonicalized to `Id(0)`; its layout maps
    /// consumer coordinates into the producer's dense output.
    StreamFold {
        producer: Box<Launch>,
        fold: Box<Launch>,
        operand: u32,
        sched: ScheduleDomain,
    },

    /// Dense contraction. `family` is this node's lowering (all families
    /// coexist as alternatives); `acc` is independent of operand dtype.
    Contract {
        /// Logical output axes; matrix dimensions flatten adjacent axis groups.
        output: IndexSpace,
        m: Dim,
        n: Dim,
        k: Dim,
        batch: Dim,
        family: Family,
        post: ScalarExpr,
        acc: Dtype,
        a: ContractSide,
        b: ContractSide,
        sched: ScheduleDomain,
    },

    Gather {
        space: IndexSpace,
        axis: u32,
        mode: GatherMode,
        ops: Vec<Operand>,
        sched: ScheduleDomain,
    },

    /// Scatter. Both lowerings coexist and compete on cost.
    Scatter {
        space: IndexSpace,
        axis: u32,
        mode: ScatterMode,
        combine: crate::ir::logical::ScatterCombine,
        ops: Vec<Operand>,
        sched: ScheduleDomain,
    },

    /// A pipeline of launches run by one dispatch, one workgroup per slab:
    /// workgroup `s` computes slab `s` of every member in dependency order,
    /// with a barrier between stages. The last member's value is this
    /// node's; children are the members themselves, by id.
    Slab {
        slabs: u32,
        members: SmallVec<[Id; 8]>,
        sched: ScheduleDomain,
    },

    /// Independent launches run by one dispatch, each on its own range of
    /// workgroups. The last member's value is this node's; children are the
    /// members by id.
    Group {
        members: SmallVec<[Id; 8]>,
        sched: ScheduleDomain,
    },
}
impl Launch {
    /// Stream `producer` into `fold`'s `operand`, when the read addresses the
    /// scalar producer's dense output.
    pub fn stream_fold(producer: Self, mut fold: Self, operand: u32) -> Option<Self> {
        let Self::Fold { sched, ops, .. } = &mut fold else {
            return None;
        };
        let sched = sched.clone();
        ops.get_mut(operand as usize)?.src = Id(0);
        let op = Self::StreamFold {
            producer: Box::new(producer),
            fold: Box::new(fold),
            operand,
            sched,
        };
        op.stream_compatible().then_some(op)
    }

    pub fn stream_compatible(&self) -> bool {
        let Self::StreamFold {
            producer,
            fold,
            operand,
            ..
        } = self
        else {
            return false;
        };
        let Self::Fold {
            space: source,
            axis,
            carrier,
            vec_axes,
            post,
            ..
        } = producer.as_ref()
        else {
            return false;
        };
        let Self::Fold { space, ops, .. } = fold.as_ref() else {
            return false;
        };
        if carrier.slots.as_slice() != [crate::carrier::SlotTy::Scalar]
            || !vec_axes.is_empty()
            || post.len() != 1
            || !carrier.associative
            || *axis as usize >= source.rank()
        {
            return false;
        }
        let mut supported = true;
        post[0].walk(&mut |e| {
            if matches!(e.kind(), crate::scalar::ScalarKind::IndexOf(i) if *i > 0) {
                supported = false;
            }
        });
        for merge in &carrier.merge {
            merge.walk(&mut |e| {
                if matches!(e.kind(), crate::scalar::ScalarKind::IndexOf(_)) {
                    supported = false;
                }
            });
        }
        if !supported {
            return false;
        }
        let Some(read) = ops.get(*operand as usize) else {
            return false;
        };
        if !matches!(read.access, AccessPlan::Alias)
            || !read.layout.offset().known_eq(Dim::Const(0))
            || read.layout.rank() != space.rank()
            || !read
                .layout
                .shape()
                .iter()
                .zip(&space.dims)
                .all(|(a, b)| a.known_eq(*b) || a.known_eq(Dim::ONE))
        {
            return false;
        }
        let output: Vec<Dim> = source
            .dims
            .iter()
            .enumerate()
            .filter_map(|(i, d)| (i != *axis as usize).then_some(*d))
            .collect();
        if output.iter().any(|d| d.known_eq(Dim::Const(0))) {
            return false;
        }
        let count = crate::shape::const_elements(&output);
        let last = read
            .layout
            .shape()
            .iter()
            .zip(read.layout.strides())
            .try_fold(0u64, |n, (d, s)| {
                n.checked_add(d.as_const()?.saturating_sub(1).checked_mul(s.as_const()?)?)
            });
        if let (Some(count), Some(last)) = (count, last) {
            return last < count;
        }
        let strides = Layout::row_major_strides(&output);
        let mut used = vec![false; output.len()];
        for (extent, stride) in read.layout.shape().iter().zip(read.layout.strides()) {
            if extent.known_eq(Dim::ONE) || stride.known_eq(Dim::Const(0)) {
                continue;
            }
            let Some(i) = output
                .iter()
                .zip(&strides)
                .enumerate()
                .position(|(i, (d, s))| {
                    !used[i]
                        && extent.known_eq(*d)
                        && stride.known_eq(*s)
                        && *s != Dim::Sym(crate::shape::OPAQUE_SYM)
                })
            else {
                return false;
            };
            used[i] = true;
        }
        true
    }

    /// The GPU fold strategy and block width before a composite widens its block.
    pub fn fold_schedule(
        &self,
        theta: Option<SchedPoint>,
        caps: &crate::device::Caps,
    ) -> Option<FoldSchedule> {
        if let Self::StreamFold { fold, .. } = self {
            return fold.fold_schedule(theta, caps);
        }
        let Self::Fold {
            carrier, vec_axes, ..
        } = self
        else {
            return None;
        };
        if caps.kind != crate::device::DeviceKind::Gpu {
            return None;
        }
        let fast = vec_axes.is_empty() && super::kernel::fast_reduce_op(carrier).is_some();
        let default = emitted_block(1, caps);
        let strat = match theta {
            Some(SchedPoint::Fold(s)) => s,
            _ if fast && caps.subgroups.is_some() => FoldStrat::Subgroup,
            _ => FoldStrat::WgTree {
                lane_group: default,
            },
        };
        let lanes = strat.lane_group(caps.subgroup_width()).max(1);
        let block = if fast {
            match strat {
                FoldStrat::Subgroup => lanes.min(caps.limits.max_compute_invocations_per_workgroup),
                _ => emitted_block(lanes, caps),
            }
        } else {
            lanes.max(default)
        };
        let scratch = if lanes <= 1 || (fast && strat == FoldStrat::Subgroup) {
            0
        } else {
            block
        };
        Some(FoldSchedule {
            strategy: strat,
            block,
            scratch,
        })
    }

    pub const fn tag(&self) -> OpTag {
        match self {
            Self::Map { .. } => OpTag::LaunchMap,
            Self::Fold { .. } => OpTag::LaunchFold,
            Self::StreamFold { .. } => OpTag::LaunchStreamFold,
            Self::Contract { .. } => OpTag::LaunchContract,
            Self::Gather { .. } => OpTag::LaunchGather,
            Self::Scatter { .. } => OpTag::LaunchScatter,
            Self::Slab { .. } => OpTag::LaunchSlab,
            Self::Group { .. } => OpTag::LaunchGroup,
        }
    }

    /// The domain this node's own expressions are written against: `space`
    /// minus a promoted `Fold`'s accumulator-resident axes.
    pub fn iter_space(&self) -> IndexSpace {
        match self {
            Self::StreamFold { fold, .. } => fold.iter_space(),
            Self::Fold {
                space, vec_axes, ..
            } if !vec_axes.is_empty() => space.iterated(vec_axes),
            _ => self.space().cloned().unwrap_or_default(),
        }
    }

    /// The index space operand layouts are stated against.
    pub fn space(&self) -> Option<&IndexSpace> {
        match self {
            Self::Map { space, .. }
            | Self::Fold { space, .. }
            | Self::Gather { space, .. }
            | Self::Scatter { space, .. } => Some(space),
            _ => None,
        }
    }

    /// The operand lists this node reads directly: `ops`, or a contraction's
    /// two sides.
    fn operand_lists(&self) -> [&[Operand]; 2] {
        match self {
            Self::Map { ops, .. }
            | Self::Fold { ops, .. }
            | Self::Gather { ops, .. }
            | Self::Scatter { ops, .. } => [ops, &[]],
            Self::Contract { a, b, .. } => [&a.ops, &b.ops],
            Self::StreamFold { .. } | Self::Slab { .. } | Self::Group { .. } => [&[], &[]],
        }
    }

    /// Every operand this node reads, in `children_of` order; a streamed
    /// fold's generated operand is skipped and a composite reads none.
    pub fn operands(&self) -> impl Iterator<Item = &Operand> {
        let ([a, b], [c, d], skip) = match self {
            Self::StreamFold {
                producer,
                fold,
                operand,
                ..
            } => {
                let first = producer.operand_lists();
                let skip = first[0].len() + first[1].len() + *operand as usize;
                (first, fold.operand_lists(), skip)
            }
            _ => (self.operand_lists(), [&[][..], &[]], usize::MAX),
        };
        [a, b, c, d]
            .into_iter()
            .flatten()
            .enumerate()
            .filter(move |(i, _)| *i != skip)
            .map(|(_, o)| o)
    }

    /// The operands a rule may re-spell in place: `ops`, or a contraction's A
    /// side then its B side.
    pub fn operands_mut(&mut self) -> impl Iterator<Item = &mut Operand> {
        let [a, b]: [&mut [Operand]; 2] = match self {
            Self::Map { ops, .. }
            | Self::Fold { ops, .. }
            | Self::Gather { ops, .. }
            | Self::Scatter { ops, .. } => [ops, &mut []],
            Self::Contract { a, b, .. } => [&mut a.ops, &mut b.ops],
            Self::StreamFold { .. } | Self::Slab { .. } | Self::Group { .. } => [&mut [], &mut []],
        };
        a.iter_mut().chain(b.iter_mut())
    }

    /// This node's enumerable schedule space.
    pub fn schedule(&self) -> Option<&ScheduleDomain> {
        match self {
            Self::Map { sched, .. }
            | Self::Fold { sched, .. }
            | Self::StreamFold { sched, .. }
            | Self::Contract { sched, .. }
            | Self::Gather { sched, .. }
            | Self::Scatter { sched, .. }
            | Self::Slab { sched, .. }
            | Self::Group { sched, .. } => Some(sched),
        }
    }
}
/// A kernel's iteration domain.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct IndexSpace {
    pub dims: SmallVec<[Dim; 6]>,
}

impl IndexSpace {
    pub fn new(dims: impl IntoIterator<Item = Dim>) -> Self {
        Self {
            dims: dims.into_iter().collect(),
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// The legality side of `map_into_fold`: a producer may be inlined only
    /// into a consumer whose space covers it.
    pub fn covers(&self, other: &IndexSpace) -> bool {
        other.dims.len() <= self.dims.len()
            && other
                .dims
                .iter()
                .zip(self.dims.iter())
                .all(|(a, b)| a.known_eq(*b))
    }

    pub fn iterations(&self) -> Option<u64> {
        crate::shape::const_elements(&self.dims)
    }

    /// A fold's iteration domain: this space minus its promoted axes.
    pub fn iterated(&self, vec_axes: &[u32]) -> IndexSpace {
        IndexSpace {
            dims: self.dims_except(|i| vec_axes.contains(&i)),
        }
    }

    /// A fold's output dims before its carrier axis: this space minus the
    /// reduced axis and every promoted axis.
    pub fn fold_out_dims(&self, axis: u32, vec_axes: &[u32]) -> Dims {
        self.dims_except(|i| i == axis || vec_axes.contains(&i))
    }

    /// A fold's output shape, spelled as inference spells it: `None` under a
    /// multi-slot carrier with a symbolic `Vector` extent.
    pub fn fold_shape(&self, axis: u32, vec_axes: &[u32], carrier: &Carrier) -> Option<Dims> {
        let mut shape = self.fold_out_dims(axis, vec_axes);
        shape.extend(carrier.out_dim()?);
        Some(shape)
    }

    fn dims_except(&self, drop: impl Fn(u32) -> bool) -> Dims {
        self.dims
            .iter()
            .enumerate()
            .filter(|(i, _)| !drop(*i as u32))
            .map(|(_, d)| *d)
            .collect()
    }
}

/// One kernel operand. Access is an attribute of the edge, not of the
/// producer: one consumer may alias a slice another packs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Operand {
    pub src: Id,
    pub layout: Layout,
    pub access: AccessPlan,
}

/// How one operand is read.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AccessPlan {
    Alias,
    Gather,
    Pack {
        into: Layout,
    },
    /// Non-affine index map; plain per-axis strides cannot express a conv
    /// window operand.
    Unflatten(MultiFlattenMap),
}

impl AccessPlan {
    /// Index-arithmetic ops per element this access costs.
    pub fn index_ops(&self) -> u64 {
        match self {
            Self::Alias => 0,
            Self::Gather => 1,
            Self::Pack { .. } => 1,
            Self::Unflatten(map) => map.divmod_ops(),
        }
    }
}

/// One side of a [`Launch::Contract`]: the non-empty operand list it reads
/// (multi-edge producers such as a GGUF block decode) and the elementwise
/// `pre` over `Arg(0..ops.len())`, numbered within this side. Every operand
/// maps the same index triple, so geometry may be read off [`Self::primary`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ContractSide {
    pub pre: ScalarExpr,
    pub ops: SmallVec<[Operand; 2]>,
}

impl ContractSide {
    /// The single-operand side every contraction is born with.
    pub fn one(pre: ScalarExpr, op: Operand) -> Self {
        Self {
            pre,
            ops: smallvec::smallvec![op],
        }
    }

    pub fn new(pre: ScalarExpr, ops: impl IntoIterator<Item = Operand>) -> Self {
        Self {
            pre,
            ops: ops.into_iter().collect(),
        }
    }

    /// The operand this side's geometry is read off; reachability predicates
    /// must still range over all of [`Self::ops`].
    pub fn primary(&self) -> &Operand {
        &self.ops[0]
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// One divmod term of an operand's index map:
/// `((flat / divisor) % modulus) * stride`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AddressTerm {
    pub divisor: u32,
    pub modulus: u32,
    pub stride: u32,
}

/// How one operand turns the reading kernel's flat space index into a
/// storage element index: `offset + sum(term)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AddressMap {
    pub offset: u32,
    pub terms: SmallVec<[AddressTerm; 4]>,
}

impl AddressMap {
    /// Whether reading with the bare flat index is already this map, for a
    /// space of `space_total` elements.
    pub fn is_identity_over(&self, space_total: u64) -> bool {
        if self.offset != 0 {
            return false;
        }
        match self.terms.as_slice() {
            [] => space_total <= 1,
            [t] => t.divisor == 1 && t.stride == 1 && u64::from(t.modulus) >= space_total,
            _ => false,
        }
    }

    /// Whether term `i` still needs its `%`: the most significant term's
    /// quotient is already below its modulus, so the mask does that work.
    pub fn needs_modulo(&self, i: usize, space_total: u64) -> bool {
        let t = self.terms[i];
        u64::from(t.divisor) * u64::from(t.modulus) < space_total
    }
}

impl Operand {
    /// The flat-index-to-storage-index map this edge declares, or `None` when
    /// a dim is symbolic or overflows `u32`. `Alias`, `Gather` and `Pack` read
    /// the layout; `Unflatten` carries its own map over the layout's offset.
    pub fn address_map(&self) -> Option<AddressMap> {
        let offset = u32::try_from(self.layout.offset().as_const()?).ok()?;
        let groups = match &self.access {
            AccessPlan::Unflatten(map) => map.groups.clone(),
            _ => self.layout.affine_groups()?,
        };

        let mut terms: SmallVec<[AddressTerm; 4]> = SmallVec::new();
        let mut div_after = 1u64;
        for g in groups.iter().rev() {
            let mut below = 1u64;
            for sub in g.sub_axes.iter().rev() {
                let divisor = div_after.checked_mul(below)?;
                terms.push(AddressTerm {
                    divisor: u32::try_from(divisor).ok()?,
                    modulus: sub.extent,
                    stride: sub.stride,
                });
                below = below.checked_mul(u64::from(sub.extent))?;
            }
            div_after = div_after.checked_mul(below)?;
        }
        // One-wide and stride-0 axes contribute zero.
        terms.retain(|t| t.modulus > 1 && t.stride != 0);
        terms.sort_unstable_by_key(|t| std::cmp::Reverse(t.divisor));
        coalesce(&mut terms);
        Some(AddressMap { offset, terms })
    }

    /// Whether this read moves as `axis`'s coordinate advances over a
    /// row-major walk of `space`: some address term with a stride overlaps
    /// the axis's flat-index window. `None` when undecidable.
    pub fn varies_along(&self, space: &IndexSpace, axis: u32) -> Option<bool> {
        let a = axis as usize;
        let lo = crate::shape::const_elements(space.dims.get(a + 1..)?)?;
        let hi = lo.checked_mul(space.dims.get(a)?.as_const()?)?;
        let map = self.address_map()?;
        Some(map.terms.iter().any(|t| {
            let t_lo = u64::from(t.divisor);
            let t_hi = t_lo.saturating_mul(u64::from(t.modulus));
            t.stride != 0 && t_lo < hi && lo < t_hi
        }))
    }

    /// Re-spell this edge under another [`AccessPlan`], or decline when it
    /// would read different elements: leaving an `Unflatten` (whose map may
    /// be independent of the layout) needs equal address maps. Every
    /// access-plan rewrite must come through here.
    pub fn respell(&self, access: AccessPlan) -> Option<Operand> {
        let out = Operand {
            src: self.src,
            layout: self.layout.clone(),
            access,
        };
        match &self.access {
            AccessPlan::Unflatten(_) => (out.address_map()? == self.address_map()?).then_some(out),
            _ => Some(out),
        }
    }
}

/// Merge adjacent terms that are contiguous in both the logical and the
/// storage order, so a dense operand collapses to the bare flat index.
fn coalesce(terms: &mut SmallVec<[AddressTerm; 4]>) {
    let mut i = 0;
    while i + 1 < terms.len() {
        let (hi, lo) = (terms[i], terms[i + 1]);
        let joins = u64::from(lo.divisor) * u64::from(lo.modulus) == u64::from(hi.divisor)
            && u64::from(lo.stride) * u64::from(lo.modulus) == u64::from(hi.stride);
        if joins && lo.modulus.checked_mul(hi.modulus).is_some() {
            terms[i] = AddressTerm {
                divisor: lo.divisor,
                modulus: lo.modulus * hi.modulus,
                stride: lo.stride,
            };
            terms.remove(i + 1);
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
}

/// Dense-contraction kernel family.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Coop,
    Sgemm,
    Sgemv,
}

/// Gather lowering.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GatherMode {
    RowPerGroup,
    /// The source is the quantized leaf itself, decoded per gathered row.
    /// Minted only from a `Gather`-of-`Dequant` pair, so the node is float.
    QuantizedRows,
}

/// Scatter lowering. Both coexist as alternatives and compete on cost.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScatterMode {
    /// Guarded on `Caps::atomic_f32`.
    Atomic,
    SortSegment,
}

/// Attention mask shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MaskKind {
    None,
    QkMask,
    BatchKeyMask,
    Causal,
}

/// Whether a node mutates state. A selected [`Effect::InPlace`] node is
/// pinned in the materialized set, so its effect applies once.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Effect {
    Pure,
    InPlace(BufferRole),
}

/// Which operand an in-place node writes through.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferRole(pub u32);

/// The enumerable schedule-parameter space of one node, resolved as a move
/// in the global search rather than minted as e-nodes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScheduleDomain {
    Point,
    Coop(Arc<CoopDomain>),
    Sgemm(Arc<SgemmDomain>),
    Sgemv(Arc<SgemvDomain>),
    Fold(Arc<FoldDomain>),
    Map(Arc<MapDomain>),
}

impl ScheduleDomain {
    pub fn len(&self) -> usize {
        match self {
            Self::Point => 1,
            Self::Coop(d) => d.len(),
            Self::Sgemm(d) => d.params.len(),
            Self::Sgemv(d) => d.params.len(),
            Self::Fold(d) => d.strategies.len(),
            Self::Map(d) => d.tilings.len(),
        }
    }
    /// True when no supported point exists. Constructors must not attach an
    /// empty domain to a graph node.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn point(&self, index: usize) -> Option<SchedPoint> {
        match self {
            Self::Point => (index == 0).then_some(SchedPoint::Point),
            Self::Coop(d) => d.point(index),
            Self::Sgemm(d) => d.params.get(index).copied().map(SchedPoint::Sgemm),
            Self::Sgemv(d) => d.params.get(index).copied().map(SchedPoint::Sgemv),
            Self::Fold(d) => d.strategies.get(index).copied().map(SchedPoint::Fold),
            Self::Map(d) => d.tilings.get(index).copied().map(SchedPoint::Map),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = SchedPoint> + '_ {
        (0..self.len()).filter_map(|i| self.point(i))
    }
}

/// One resolved schedule.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SchedPoint {
    Point,
    Coop { geom: CoopGeom, staging: u8 },
    Sgemm(SgemmParams),
    Sgemv(SgemvParams),
    Fold(FoldStrat),
    Map(MapTiling),
}

/// Cooperative-matrix tile geometry.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CoopGeom {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub n_passes: u32,
    pub subgroups: u32,
    pub rg: u32,
    pub cg: u32,
}

impl CoopGeom {
    /// Fragment side; every per-subgroup fragment grid counts whole
    /// `COOP_DIM x COOP_DIM` fragments.
    pub const COOP_DIM: u32 = 8;

    /// Minimize threadgroup fragment loads `cg*bm + rg*bn_pass` subject to
    /// both fragment sides staying whole multiples of [`Self::COOP_DIM`];
    /// ties keep the smaller `rg`.
    pub const fn subgroup_split(
        bm: u32,
        bn: u32,
        n_passes: u32,
        subgroups: u32,
    ) -> Option<(u32, u32)> {
        let bn_pass = bn / n_passes;
        let mut best_rg = 0;
        let mut best_loads = 0;
        let mut rg = 1;
        while rg <= subgroups {
            let cg = subgroups / rg;
            if subgroups.is_multiple_of(rg)
                && bm.is_multiple_of(Self::COOP_DIM * rg)
                && bn_pass.is_multiple_of(Self::COOP_DIM * cg)
            {
                let loads = cg * bm + rg * bn_pass;
                if best_rg == 0 || loads < best_loads {
                    best_rg = rg;
                    best_loads = loads;
                }
            }
            rg += 1;
        }
        if best_rg == 0 {
            None
        } else {
            Some((best_rg, subgroups / best_rg))
        }
    }

    pub const fn lanes(&self, subgroup_width: u32) -> u32 {
        self.rg * self.cg * subgroup_width
    }

    /// Structural legality, independent of workgroup-memory footprint.
    pub const fn legal(&self, subgroup_width: u32, max_wg_lanes: u32) -> bool {
        self.n_passes != 0
            && self.rg != 0
            && self.cg != 0
            && self.lanes(subgroup_width) <= max_wg_lanes
            && self.bm.is_multiple_of(Self::COOP_DIM * self.rg)
            && (self.bn / self.n_passes).is_multiple_of(Self::COOP_DIM * self.cg)
    }
}

/// One supported geometry and staging depth. These axes are coupled by the
/// workgroup memory limit, so a domain contains pairs, not their cross product.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CoopSchedule {
    pub geom: CoopGeom,
    pub staging: u8,
}

/// The supported cooperative schedules of one contraction. Split-K is a
/// graph rewrite with an explicit combine, never a single-kernel schedule.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CoopDomain {
    pub schedules: SmallVec<[CoopSchedule; 16]>,
}

impl CoopDomain {
    pub fn len(&self) -> usize {
        self.schedules.len()
    }
    pub fn is_empty(&self) -> bool {
        self.schedules.is_empty()
    }
    pub fn point(&self, index: usize) -> Option<SchedPoint> {
        let CoopSchedule { geom, staging } = *self.schedules.get(index)?;
        Some(SchedPoint::Coop { geom, staging })
    }
}

/// SGEMM block and thread tiling.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SgemmParams {
    pub double_buffer: bool,
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub tm: u32,
    pub tn: u32,
}

impl SgemmParams {
    /// `tm | bm`, `tn | bn`, 32..=max lanes, staged footprint within the
    /// workgroup-storage limit.
    pub const fn legal(&self, elem_bytes: u32, max_wg_storage: u32, max_lanes: u32) -> bool {
        if self.tm == 0
            || self.tn == 0
            || !self.bm.is_multiple_of(self.tm)
            || !self.bn.is_multiple_of(self.tn)
        {
            return false;
        }
        let lanes = (self.bm / self.tm) * (self.bn / self.tn);
        let depth = if self.double_buffer { 2 } else { 1 };
        let bytes = (self.bm + self.bn) * self.bk * elem_bytes * depth;
        lanes >= 32 && lanes <= max_lanes && bytes <= max_wg_storage
    }
}

/// Every legal SGEMM tiling.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct SgemmDomain {
    pub params: SmallVec<[SgemmParams; 16]>,
}

/// SGEMV vectorization and workgroup structure.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SgemvParams {
    pub vector: u32,
    pub subgroups: u32,
    /// Output columns per workgroup. `1` reduces across the workgroup;
    /// `cols > 1` (a multiple of `subgroups`, fixed subgroup width) gives each
    /// subgroup `cols / subgroups` columns reduced within it.
    pub cols: u32,
    /// Runs the lane's k window is split into (`1` with `gap == 0` is
    /// canonical): run `r` sits at `r * gap`, so one lane revisits a packed
    /// word at several k offsets. Legal only with `cols > 1`.
    pub parts: u32,
    /// K distance between a split window's runs. `0` when `parts == 1`.
    pub gap: u32,
}

impl SgemvParams {
    /// Consecutive elements per run of the lane's k window.
    pub const fn run(&self) -> u32 {
        if self.parts <= 1 {
            self.vector
        } else {
            self.vector / self.parts
        }
    }
}

/// Every legal SGEMV parameterization.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct SgemvDomain {
    pub params: SmallVec<[SgemvParams; 16]>,
}

/// How a fold reduces across lanes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FoldStrat {
    Subgroup,
    WgTree { lane_group: u32 },
    LoopThenTree { iterations: u32, lane_group: u32 },
}

/// The GPU fold's strategy, thread block, and scratch elements per carrier lane.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FoldSchedule {
    pub strategy: FoldStrat,
    pub block: u32,
    pub scratch: u32,
}

impl FoldStrat {
    /// The lane group this strategy closes over. `Subgroup` closes over the
    /// device's subgroup, so the caller supplies that width.
    pub fn lane_group(&self, subgroup_width: u32) -> u32 {
        match self {
            Self::Subgroup => subgroup_width,
            Self::WgTree { lane_group } | Self::LoopThenTree { lane_group, .. } => *lane_group,
        }
    }
}

/// The workgroup width both emitters allocate scratch over — the single
/// source verification, domain generation and emission share.
pub fn emitted_block(lane_group: u32, caps: &crate::device::Caps) -> u32 {
    const DEFAULT_BLOCK: u32 = 256;
    lane_group
        .max(DEFAULT_BLOCK.min(caps.limits.max_compute_invocations_per_workgroup))
        .min(caps.limits.max_compute_invocations_per_workgroup.max(1))
        .max(1)
}

/// The block a slab lowers at: a power of two covering `per_slab` (the most
/// iterations any stage runs per slab), between one subgroup and the default.
pub fn slab_block(per_slab: u64, caps: &crate::device::Caps) -> u32 {
    let top = emitted_block(1, caps);
    let floor = caps.subgroup_width().clamp(1, top);
    u32::try_from(per_slab.max(1).next_power_of_two())
        .unwrap_or(u32::MAX)
        .clamp(floor, top)
}

/// Lanes a slab's fold stage gives each output row.
pub fn slab_lanes_per_row(block: u32, rows_per_slab: u64, k: u64) -> u32 {
    let rows = u32::try_from(rows_per_slab.max(1).next_power_of_two()).unwrap_or(u32::MAX);
    let k = u32::try_from(k.max(1).next_power_of_two()).unwrap_or(u32::MAX);
    (block / rows.min(block)).min(k).max(1)
}

/// Subgroup width used by a slab fold when the workgroup has full subgroups.
pub fn slab_subgroup_width(
    block: u32,
    rows_per_slab: u64,
    k: u64,
    carrier: &Carrier,
    caps: &crate::device::Caps,
) -> Option<u32> {
    let width = caps.subgroups.filter(|s| s.is_fixed())?.assumed();
    (width > 0
        && super::kernel::fast_reduce_op(carrier).is_some()
        && block.is_multiple_of(width)
        && slab_lanes_per_row(block, rows_per_slab, k) >= width)
        .then_some(width)
}

/// Workgroup bytes one fold strategy's cross-lane close needs: one
/// [`emitted_block`] tile per accumulator lane.
pub fn fold_scratch_bytes(
    strat: &FoldStrat,
    lanes: u64,
    acc_bytes: u64,
    subgroup_width: u32,
    caps: &crate::device::Caps,
) -> u64 {
    let lane_group = strat.lane_group(subgroup_width);
    // A one-lane group reduces a whole row per invocation and stages
    // nothing, which keeps wide promoted carriers schedulable.
    if lane_group <= 1 {
        return 0;
    }
    let block = u64::from(emitted_block(lane_group, caps));
    lanes.saturating_mul(block).saturating_mul(acc_bytes.max(1))
}

/// Reduction strategies and lane-group widths worth scoring together.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct FoldDomain {
    pub strategies: SmallVec<[FoldStrat; 8]>,
}

impl ScheduleDomain {
    /// Construct schedules for a fold with a new carrier. Promotion changes
    /// scratch requirements, so it cannot copy the old domain verbatim.
    pub fn with_fold_carrier(
        &self,
        lanes: u64,
        acc_bytes: u64,
        caps: &crate::device::Caps,
    ) -> Option<Self> {
        let fits = |s: &FoldStrat| {
            fold_scratch_bytes(s, lanes, acc_bytes, caps.subgroup_width(), caps)
                <= u64::from(caps.limits.max_compute_workgroup_storage_size)
        };
        let strategies = match self {
            Self::Fold(domain) => domain.strategies.iter().copied().filter(fits).collect(),
            Self::Point => {
                let default = FoldStrat::WgTree {
                    lane_group: emitted_block(1, caps),
                };
                if fits(&default) {
                    return Some(Self::Point);
                }
                // A row per lane reduces privately and needs no scratch.
                smallvec::smallvec![FoldStrat::WgTree { lane_group: 1 }]
            }
            _ => return None,
        };
        if strategies.is_empty() {
            return None;
        }
        Some(Self::Fold(FoldDomain { strategies }.into()))
    }
}

/// Elementwise register-reuse tiling. `vector` is the SIMD width on the CPU
/// backend and 1 on GPU.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MapTiling {
    pub dim: Option<u32>,
    pub tm: u32,
    pub vector: u32,
}

/// Candidate tilings: one per eligible dim, plus untiled.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct MapDomain {
    pub tilings: SmallVec<[MapTiling; 8]>,
}

/// Window geometry a structural adjoint reads: non-overlapping windows give
/// a mask-and-broadcast, overlapping ones `Scatter{Add}`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WindowAdjoint {
    pub window: SlidingWindow,
    pub is_mask: bool,
}

impl WindowAdjoint {
    pub const fn of(window: SlidingWindow) -> Self {
        Self {
            window,
            is_mask: window.is_non_overlapping(),
        }
    }
}

/// Scratch budget for a cooperative schedule, as lowering declares it,
/// including the output staging tile.
pub fn coop_tiles(
    geom: CoopGeom,
    elem: super::kernel::ScalarElement,
    staging: u8,
) -> super::kernel::Tiles {
    use super::kernel::{ElementType, MemoryLevel, ScalarElement, TileDecl, TileLayout, Tiles};
    let depth = u32::from(staging);
    let bn_pass = geom.bn / geom.n_passes;
    let tile = |name, elem, shape: &[u32]| {
        std::sync::Arc::new(TileDecl::new(
            ElementType::Scalar(elem),
            TileLayout::contiguous(MemoryLevel::Workgroup, shape),
            name,
        ))
    };
    Tiles {
        decls: smallvec::smallvec![
            tile("coop_a", elem, &[depth * geom.bm, geom.bk]),
            tile("coop_b", elem, &[depth * geom.bk, bn_pass]),
            tile("coop_acc", ScalarElement::F32, &[geom.bm, bn_pass]),
        ],
    }
}
