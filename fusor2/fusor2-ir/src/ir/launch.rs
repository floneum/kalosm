//! Launch `nest` — index spaces, kernels, launches. Only Launch can express fusion,
//! tiling, split-K, layout alias-vs-gather, kernel family, horizontal merging,
//! register tiling and rematerialization. **Allocation is not described at
//! Launch**: buffers are derived from the extracted plan.

use crate::carrier::Carrier;
use crate::dtype::Dtype;
use crate::egraph::Id;
use crate::ir::{AttrId, OpDefId, OpTag};
use crate::scalar::ScalarExpr;
use crate::shape::{Dim, Layout, MultiFlattenMap, SlidingWindow};
use smallvec::SmallVec;

/// The Launch op family.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Launch {
    Map {
        space: IndexSpace,
        body: ScalarExpr,
        ops: Vec<Operand>,
        sched: ScheduleDomain,
    },

    /// A reduction nest over a [`Carrier`]. `space` is
    /// `free.. ++ vec.. ++ [reduced]`; the carrier owns the element
    /// expression (`lift`, which is where the old `pre` went), the per-slot
    /// identities and the merge, so a multi-slot accumulator is expressible
    /// and there is no `TileReduceOp` to resolve for a whole fold.
    Fold {
        space: IndexSpace,
        axis: u32,
        /// Free axes living in the **accumulator's data space** rather than
        /// the iteration domain — a contiguous block immediately before
        /// `axis`, which is what makes the output shape identical before and
        /// after promotion. Operand address maps are stated against the full
        /// `space` and are never rewritten; every [`ScalarExpr`] on this node
        /// is written against [`Launch::iter_space`].
        vec_axes: SmallVec<[u32; 2]>,
        carrier: Carrier,
        acc: Dtype,
        /// One per slot, over `Arg(0..width)`. Cross-slot reads are legal —
        /// flash's normalized output reads the running sum. Shape-preserving:
        /// a post never changes the appended carrier axis.
        post: SmallVec<[ScalarExpr; 4]>,
        ops: Vec<Operand>,
        sched: ScheduleDomain,
    },

    /// Dense contraction. **`family` is a property of this node's lowering,
    /// never a decision stored on an Logical op**: all four families coexist in
    /// one chain, so a gemv-shaped contraction cannot pick Coop, have the
    /// tile scorer decline, and silently run a third path. `acc` is
    /// independent of operand dtype, which is what makes
    /// `contract{acc: F32}(F16, F16) -> F16` one node.
    Contract {
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

    /// A multi-output region: the same rewrite as producer inlining,
    /// differing only in that it emits an extra buffer.
    ///
    /// `sched` is the members' shared index space walked as one linearized
    /// body — see [`MapDomain::linear_over`]. Without it the one node family
    /// the architecture calls its own fusion primitive would be the one whose
    /// geometry is not a selection.
    Region {
        members: SmallVec<[Id; 8]>,
        live_outs: SmallVec<[u32; 4]>,
        sched: ScheduleDomain,
    },

    /// The one open extension point.
    Ext {
        def: OpDefId,
        ops: Vec<Operand>,
        attrs: AttrId,
    },
}

impl Launch {
    pub const fn tag(&self) -> OpTag {
        match self {
            Self::Map { .. } => OpTag::LaunchMap,
            Self::Fold { .. } => OpTag::LaunchFold,
            Self::Contract { .. } => OpTag::LaunchContract,
            Self::Gather { .. } => OpTag::LaunchGather,
            Self::Scatter { .. } => OpTag::LaunchScatter,
            Self::Region { .. } => OpTag::LaunchRegion,
            Self::Ext { .. } => OpTag::Ext,
        }
    }

    /// The domain this node's own expressions are written against: `space`
    /// minus any accumulator-resident axes. Equal to `space` for every node
    /// but a promoted `Fold`.
    pub fn iter_space(&self) -> IndexSpace {
        match self {
            Self::Fold {
                space, vec_axes, ..
            } if !vec_axes.is_empty() => IndexSpace::new(
                space
                    .dims
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !vec_axes.contains(&(*i as u32)))
                    .map(|(_, d)| *d),
            ),
            Self::Map { space, .. }
            | Self::Fold { space, .. }
            | Self::Gather { space, .. }
            | Self::Scatter { space, .. } => space.clone(),
            _ => IndexSpace::default(),
        }
    }

    /// This node's enumerable schedule space, or `None` when it has none.
    ///
    /// `Ext` is the only `None`: the open extension point carries an
    /// `OpDef`-supplied lowering that fusor2 cannot enumerate geometries for,
    /// so its lowering is handed `SchedPoint::Point` and nothing else.
    pub fn schedule(&self) -> Option<&ScheduleDomain> {
        match self {
            Self::Map { sched, .. }
            | Self::Fold { sched, .. }
            | Self::Contract { sched, .. }
            | Self::Gather { sched, .. }
            | Self::Scatter { sched, .. }
            | Self::Region { sched, .. } => Some(sched),
            Self::Ext { .. } => None,
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
        self.dims
            .iter()
            .try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))
    }
}

/// One kernel operand. **Access is an attribute of the edge, not of the
/// producing node** — one consumer may alias a strided parameter slice
/// while another packs it, which is the flat-parameter/gradient-concat case
/// and the im2col operand case coexisting in one graph.
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

/// One side of a [`Launch::Contract`]: the buffers it reads and the elementwise
/// chain run per loaded element.
///
/// # Why a side is a list
///
/// A side carries multiple operands so producers that read more than one
/// buffer can be absorbed. The GGUF block decode reads one block stream through
/// several `Restride` views at once — the quant plane, the block scale, the
/// block minimum, the group scales — so it is irreducibly multi-edge and
/// benefits from a side as a list of operands.
///
/// # Numbering
///
/// `pre` is written over `Arg(0..ops.len())` numbered **within this side**,
/// not across the node. Splicing a producer into `a` therefore never
/// renumbers `b`'s body, which is what keeps [`map_into_contract`] a local
/// rewrite on the side it fires for.
///
/// [`map_into_contract`]: crate::rules::fusion::map_into_contract
///
/// # Shape
///
/// Every operand of a side maps the same index triple to its own address —
/// `(batch, m, k)` on `a`, `(batch, k, n)` on `b` — so they agree on shape
/// and differ only in buffer, stride and access. Geometry may be read off
/// [`Self::primary`]; predicates about *reachability* (addressability, dtype
/// admissibility, traffic) must range over all of [`Self::ops`].
///
/// `ops` is non-empty. A side with no operand would make `pre` a constant,
/// which is a `Map` and not a contraction; `verify_launch` rejects it.
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

    /// The operand this side's geometry is read off. See the type's docs for
    /// why any operand would do and why predicates still may not use it.
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

/// How one operand turns the reading kernel's **flat space index** into a
/// **storage element index**: `offset + sum(term)`.
///
/// This is the one place the edge's `layout`/`access` becomes arithmetic. An
/// emitter that indexes an operand with the raw flat index is only correct
/// when [`AddressMap::is_identity_over`] holds — a stride-0 broadcast axis, a
/// transposed view, a narrowed slice and a conv window all disagree with it.
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
    /// The flat-index-to-storage-index map this **edge** declares, or `None`
    /// when a dim is symbolic or overflows `u32`.
    ///
    /// `Alias`, `Gather` and `Pack` all name the operand's own `layout`: they
    /// are competing *spellings* of one access (see `rules::layout`, which
    /// declines to mint a `Gather` over a contiguous layout precisely because
    /// it would be the same index map twice), so they differ in how the read
    /// is emitted, never in which element it reads. `Unflatten` carries its
    /// own map because a conv window's sub-axis strides collide and plain
    /// per-axis strides cannot express that; the layout still supplies the
    /// base offset, which `MultiFlattenMap` has nowhere to put.
    pub fn address_map(&self) -> Option<AddressMap> {
        let offset = u32::try_from(self.layout.offset().as_const()?).ok()?;
        let groups: SmallVec<[crate::shape::AxisGroup; 4]> = match &self.access {
            AccessPlan::Unflatten(map) => map.groups.clone(),
            _ => self
                .layout
                .shape()
                .iter()
                .zip(self.layout.strides())
                .map(|(d, s)| {
                    Some(crate::shape::AxisGroup::affine(
                        u32::try_from(d.as_const()?).ok()?,
                        u32::try_from(s.as_const()?).ok()?,
                    ))
                })
                .collect::<Option<_>>()?,
        };

        // Row-major over the logical axes, then most-significant-first within
        // each axis group — the order `MultiFlattenMap` declares.
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
        // A one-wide axis and a stride-0 axis both contribute exactly zero.
        // Dropping them is what makes a broadcast read its single row.
        terms.retain(|t| t.modulus > 1 && t.stride != 0);
        terms.sort_unstable_by(|a, b| b.divisor.cmp(&a.divisor));
        coalesce(&mut terms);
        Some(AddressMap { offset, terms })
    }

    /// Re-spell this edge under another [`AccessPlan`], or decline when the
    /// re-spelling would read different elements.
    ///
    /// `Alias`, `Gather` and `Pack` all derive every address from `layout`,
    /// so moving between them is sound whatever the extents are — including
    /// symbolic ones, where the map is undecidable but the *reason* the two
    /// agree does not read an extent. An `Unflatten` map is the one plan that
    /// may have been stated **independently** of the layout
    /// (`rules::sink::fold_operand_views` mints exactly that), so leaving one
    /// is sound only when the candidate's address map compares equal, and an
    /// undecidable extent declines.
    ///
    /// `rules::layout::operand_alias` pinned this hazard for the `Alias`
    /// spelling — 29 conformance cases on wrong values when a co-selection
    /// pass first made the member reachable. The `Gather` and `Pack`
    /// re-spellings (and the cpu backend's access rules) carried the
    /// identical hazard; only the extraction budget kept those members
    /// unselected. Every access-plan rewrite must come through here.
    pub fn respell(&self, access: AccessPlan) -> Option<Operand> {
        let out = Operand {
            src: self.src,
            layout: self.layout.clone(),
            access,
        };
        match &self.access {
            AccessPlan::Unflatten(_) => {
                (out.address_map()? == self.address_map()?).then_some(out)
            }
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
///
/// `Vectorized` — one lane moving four elements instead of one — was offered
/// 554 times over the suite and every model and selected zero times, so it is
/// deleted; the two survivors are the row nest and the quantized-row nest.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GatherMode {
    RowPerGroup,
    /// The gather's source operand is the *quantized leaf itself*, addressed
    /// in its dense logical element space; both backends' operand loaders run
    /// the format's decode program at the flat index, so only the gathered
    /// rows ever decode and no dense table is materialized. Minted only from
    /// a `Gather`-of-`Dequant` pair (`GATHER_QUANTIZED_ROWS`), so the node is
    /// float-typed — minting it straight over the leaf would give the class
    /// the source's `Q(fmt)` dtype and the consuming `Dequant` would decode
    /// twice.
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

/// Whether a node mutates state. A selected node with [`Effect::InPlace`]
/// is **pinned in the materialized set**: without that, toggling a
/// two-consumer atomic scatter out of `M` applies its atomics twice.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Effect {
    Pure,
    InPlace(BufferRole),
}

/// Which operand an in-place node writes through.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferRole(pub u32);

// ---------------------------------------------------------------------------
// Schedule domains
// ---------------------------------------------------------------------------

/// The enumerable schedule-parameter space of one node. **It is not
/// e-nodes**: minting every point blows the graph up; minting a
/// locally-Pareto top-k lets a cheap heuristic gate the real cost model;
/// and a nested argmin inside the node's cost is circular, because the
/// geometry it picks determines the output's padded strides and therefore
/// every consumer's read traffic. Carrying the complete domain and
/// resolving it as a move in the global search is the only formulation that
/// is simultaneously small, complete and non-circular.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScheduleDomain {
    Point,
    Coop(CoopDomain),
    Sgemm(SgemmDomain),
    Sgemv(SgemvDomain),
    Fold(FoldDomain),
    Map(MapDomain),
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
    /// True when no legal point exists — the node is unselectable.
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
    Coop {
        geom: CoopGeom,
        splits: u32,
        staging: u8,
    },
    Sgemm(SgemmParams),
    Sgemv(SgemvParams),
    Fold(FoldStrat),
    Map(MapTiling),
}

/// Cooperative-matrix tile geometry. [`Self::subgroup_split`] is the
/// template every other geometry factorization follows: a closed-form
/// objective with an explicit feasibility predicate.
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
            if subgroups % rg == 0
                && bm % (Self::COOP_DIM * rg) == 0
                && bn_pass % (Self::COOP_DIM * cg) == 0
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
            && self.bm % (Self::COOP_DIM * self.rg) == 0
            && (self.bn / self.n_passes) % (Self::COOP_DIM * self.cg) == 0
    }
}

/// The complete legal cooperative schedule space of one contraction.
/// `geoms` is filtered by lane limits and the exact
/// `ArenaPlan::total_bytes`; `splits` is never-split plus every divisor of
/// the K loop leaving two iterations per workgroup; `staging` is 1 or 2.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CoopDomain {
    pub geoms: SmallVec<[CoopGeom; 16]>,
    pub splits: SmallVec<[u32; 8]>,
    pub staging: SmallVec<[u8; 2]>,
}

impl CoopDomain {
    pub fn len(&self) -> usize {
        self.geoms.len() * self.splits.len() * self.staging.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn point(&self, index: usize) -> Option<SchedPoint> {
        let ns = self.splits.len();
        let nd = self.staging.len();
        if ns == 0 || nd == 0 {
            return None;
        }
        let geom = *self.geoms.get(index / (ns * nd))?;
        let rem = index % (ns * nd);
        Some(SchedPoint::Coop {
            geom,
            splits: self.splits[rem / nd],
            staging: self.staging[rem % nd],
        })
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
        if self.tm == 0 || self.tn == 0 || self.bm % self.tm != 0 || self.bn % self.tn != 0 {
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
    /// Output columns per workgroup.
    ///
    /// `1` is the whole-workgroup structure: every lane of all `subgroups`
    /// subgroups cooperates on a single output element and the reduction
    /// crosses the workgroup. `cols > 1` requires `cols % subgroups == 0`
    /// and a fixed subgroup width: each subgroup owns `cols / subgroups`
    /// columns end-to-end, all `width` lanes cooperate on each of them, the
    /// k window of one pass is shared across the subgroup's columns, and the
    /// reduction never leaves the subgroup.
    pub cols: u32,
    /// Runs the lane's k window is split into.
    ///
    /// `1` (with `gap == 0`, the canonical spelling) keeps the window as
    /// `vector` consecutive k elements. `parts > 1` — legal only on the
    /// multi-column structure — lays the window out as `parts` runs of
    /// `vector / parts` consecutive elements, run `r` at offset `r * gap`
    /// from the lane's base, with `gap / run` adjacent lanes' runs packing
    /// each gap before the window's own runs interleave. The subgroup's pass
    /// still covers exactly `width * vector` consecutive k; only which lane
    /// owns which element changes. A split window lets one lane revisit the
    /// same packed word of a bit-packed operand at several k offsets, so the
    /// word loads hash-cons to a single evaluation — purely a schedule
    /// choice, discovered by measurement like every other axis.
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
    WgTree {
        lane_group: u32,
    },
    LoopThenTree {
        iterations: u32,
        lane_group: u32,
    },
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

/// The workgroup width both emitters actually allocate scratch over — **the
/// single source of it**.
///
/// Only `lane_group` rides on [`FoldStrat`]; the block is the default width
/// floored by the lane group and clamped to the device, so a footprint filter
/// that reads `lane_group` alone under-counts by up to 256x. `DEFAULT_BLOCK`
/// is a *policy* constant, but it lives here rather than in `fusor2-tile`
/// because `verify_launch` has to admit against the same number the domain
/// generator filters on and the emitters allocate — and it is `fusor2-ir`
/// that verifies.
pub fn emitted_block(lane_group: u32, caps: &crate::device::Caps) -> u32 {
    const DEFAULT_BLOCK: u32 = 256;
    lane_group
        .max(DEFAULT_BLOCK.min(caps.limits.max_compute_invocations_per_workgroup))
        .min(caps.limits.max_compute_invocations_per_workgroup.max(1))
        .max(1)
}

/// Workgroup bytes one fold strategy's cross-lane close needs, for a carrier
/// of `lanes` accumulator lanes at `acc_bytes` each.
///
/// Both emitters allocate one scratch tile of [`emitted_block`] elements *per
/// accumulator lane*. This is the quantity `verify_launch` admits against and the
/// quantity the fold domain generator filters on; they are one function so
/// they cannot drift.
pub fn fold_scratch_bytes(
    strat: &FoldStrat,
    lanes: u64,
    acc_bytes: u64,
    subgroup_width: u32,
    caps: &crate::device::Caps,
) -> u64 {
    let lane_group = strat.lane_group(subgroup_width);
    // **A one-lane group stages nothing.** Both emitters map invocation
    // `group` to `row = group / lane_group` and `lane = group % lane_group`,
    // and loop `(axis_extent + lane_group - 1) / lane_group` times. At
    // `lane_group == 1` every invocation owns a whole output row and reduces
    // the entire axis into its own accumulator, so the cross-lane merge is
    // over a group of one — an identity with nothing to stage.
    //
    // This is what makes a WIDE promoted carrier schedulable at all. The
    // footprint of a cross-lane close is `lanes * emitted_block * acc_bytes`
    // and `emitted_block` is floored at the default block regardless of the
    // lane group, so a 24-lane Welford carrier wants 24 KiB and a 64-lane
    // `TN` register tile wants 64 KiB — over any device's limit at *every*
    // lane group. Without this clause the fold domain of such a carrier is
    // empty, `PROMOTE` mints a nest nothing can schedule, and §4.2 turns that
    // into a hard `verify_plan` failure instead of a slow plan. With it, the
    // row-per-lane schedule the emitters already lower correctly is also the
    // one the generator can offer, and register tiling stays derivable.
    if lane_group <= 1 {
        return 0;
    }
    let block = u64::from(emitted_block(lane_group, caps));
    lanes
        .saturating_mul(block)
        .saturating_mul(acc_bytes.max(1))
}

/// Reduction strategies and lane-group widths worth scoring. Workgroup
/// width, lane-group width and staging depth are *coupled*, so they are one
/// domain scored together rather than three greedy formulas.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct FoldDomain {
    pub strategies: SmallVec<[FoldStrat; 8]>,
}

/// Elementwise register-reuse tiling. `vector` is the SIMD width on the CPU
/// backend and 1 on GPU.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MapTiling {
    pub dim: Option<u32>,
    pub tm: u32,
    pub vector: u32,
}

/// Candidate tilings: one per eligible dim, plus untiled. Replaces the
/// strict LLC-watermark cliff and the argmax-invariant-bytes selection.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct MapDomain {
    pub tilings: SmallVec<[MapTiling; 8]>,
}

/// Outputs per lane worth scoring for a linearized body. A *policy* constant,
/// and it belongs beside the generator that reads it rather than in a
/// lowering — same rule `fusor2_tile::domains` follows for `BLOCK_CHOICES`.
const LINEAR_TM_CHOICES: [u32; 3] = [2, 4, 8];

impl MapDomain {
    /// Every tiling worth scoring for a body that walks **one linearized
    /// index** over `elements` outputs — the shape [`Launch::Region`] takes on
    /// either backend.
    ///
    /// `dim` is always `None` because there is no axis to name: `tm` is the
    /// register tile along the linear index and `vector` is the SIMD width,
    /// which a composite body does not choose. A tiling survives only when it
    /// leaves at least one full subgroup of work, so a body too small to tile
    /// reports one point and says so, rather than offering a point that would
    /// launch empty lanes.
    ///
    /// This needs `Caps` and an element count and nothing else — no arena
    /// plan — which is why it lives here and both the rule that mints the
    /// node and the verifier that checks it call the same function.
    pub fn linear(caps: &crate::device::Caps, elements: u64) -> Self {
        let sgw = u64::from(caps.subgroup_width().max(1));
        let mut tilings: SmallVec<[MapTiling; 8]> = SmallVec::new();
        tilings.push(MapTiling {
            dim: None,
            tm: 1,
            vector: 1,
        });
        for tm in LINEAR_TM_CHOICES {
            if elements >= u64::from(tm).saturating_mul(sgw) {
                tilings.push(MapTiling {
                    dim: None,
                    tm,
                    vector: 1,
                });
            }
        }
        Self { tilings }
    }

    /// [`Self::linear`] over a value's shape. A symbolic extent prices as 1,
    /// the same convention `semantics::work` uses, so a shape-family node
    /// gets the conservative domain rather than a tiling its smallest legal
    /// binding cannot fill.
    pub fn linear_over(caps: &crate::device::Caps, shape: &[Dim]) -> Self {
        let elements = shape
            .iter()
            .map(|d| d.as_const().unwrap_or(1))
            .fold(1u64, |a, b| a.saturating_mul(b));
        Self::linear(caps, elements)
    }
}

/// Window geometry a structural adjoint reads. Two integers decide the
/// whole thing: `step >= window` proves the adjoint is an elementwise
/// mask-and-broadcast; overlapping windows give `Scatter{Add}`.
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

#[cfg(test)]
mod schedule_tests {
    use super::*;
    use crate::device::{Caps, DeviceKind, Limits, SubgroupWidths};

    fn caps(subgroup: u32) -> Caps {
        Caps {
            kind: DeviceKind::Gpu,
            name: "test".into(),
            limits: Limits::default(),
            subgroups: Some(SubgroupWidths {
                min: subgroup,
                max: subgroup,
            }),
            f16: true,
            bf16: false,
            coop: Default::default(),
            atomic_f32: true,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: Default::default(),
            threads: 1,
        }
    }

    fn tms(d: &MapDomain) -> Vec<u32> {
        d.tilings
            .iter()
            .map(|t| {
                assert_eq!(t.dim, None, "a linearized body has no axis to name");
                assert_eq!(t.vector, 1);
                t.tm
            })
            .collect()
    }

    /// A real composite has something to decide, and a tiny one says it has
    /// nothing — rather than offering a point that would launch empty lanes.
    #[test]
    fn the_linear_domain_follows_the_work_and_the_device() {
        assert_eq!(tms(&MapDomain::linear(&caps(32), 8192)), vec![1, 2, 4, 8]);
        assert_eq!(tms(&MapDomain::linear(&caps(32), 4)), vec![1]);
        // Twice the subgroup width needs twice the work for the same tile.
        assert_eq!(tms(&MapDomain::linear(&caps(64), 256)), vec![1, 2, 4]);
        assert_eq!(tms(&MapDomain::linear(&caps(32), 256)), vec![1, 2, 4, 8]);
        // The untiled point is always a member, so a composite is always
        // lowerable at the floor its lowering falls back to.
        assert!(MapDomain::linear(&caps(32), 0).tilings.contains(&MapTiling {
            dim: None,
            tm: 1,
            vector: 1
        }));
    }

    /// A symbolic extent prices as 1, so a shape-family composite gets the
    /// conservative domain rather than a tile its smallest binding cannot
    /// fill.
    #[test]
    fn a_symbolic_extent_prices_conservatively() {
        let sym = [Dim::Const(8), Dim::Sym(crate::shape::SymId(0))];
        assert_eq!(tms(&MapDomain::linear_over(&caps(32), &sym)), vec![1]);
        assert_eq!(
            MapDomain::linear_over(&caps(32), &[Dim::Const(128), Dim::Const(64)]),
            MapDomain::linear(&caps(32), 8192)
        );
    }

    /// **The gate.** Both composite forms carry a schedule domain, so
    /// extraction resolves their geometry like every other node's. `Ext` is
    /// the only `None` left: fusor2 cannot enumerate geometries for a
    /// lowering it did not write.
    #[test]
    fn every_node_but_ext_declares_a_schedule_domain() {
        let d = ScheduleDomain::Map(MapDomain::linear(&caps(32), 8192));
        let region = Launch::Region {
            members: smallvec::smallvec![Id(1), Id(2)],
            live_outs: smallvec::smallvec![0],
            sched: d.clone(),
        };
        assert_eq!(region.schedule(), Some(&d));
        assert!(region.schedule().unwrap().len() > 1);

        let ext = Launch::Ext {
            def: crate::ir::OpDefId(0),
            ops: Vec::new(),
            attrs: crate::ir::AttrId(0),
        };
        assert_eq!(ext.schedule(), None);
    }
}

#[cfg(test)]
mod address_tests {
    use super::*;
    use crate::shape::{AxisGroup, SubAxis};

    fn dims(v: &[u64]) -> Vec<Dim> {
        v.iter().map(|d| Dim::Const(*d)).collect()
    }

    fn alias(shape: &[u64], strides: &[u64]) -> Operand {
        Operand {
            src: Id(0),
            layout: Layout::from_parts(Dim::Const(0), &dims(shape), &dims(strides)).unwrap(),
            access: AccessPlan::Alias,
        }
    }

    /// Evaluate the map the way an emitter must.
    fn at(map: &AddressMap, flat: u32) -> u32 {
        let mut acc = map.offset;
        for t in &map.terms {
            acc = acc.wrapping_add(((flat / t.divisor) % t.modulus).wrapping_mul(t.stride));
        }
        acc
    }

    #[test]
    fn a_dense_operand_is_the_bare_flat_index() {
        let m = alias(&[3, 5], &[5, 1]).address_map().unwrap();
        assert!(m.is_identity_over(15));
        for f in 0..15 {
            assert_eq!(at(&m, f), f);
        }
    }

    /// The `rms_norm_no_weight` [3,5] case: the second operand broadcasts a
    /// per-row scalar across the row, so element 7 must read `inv[1]`.
    #[test]
    fn a_stride_zero_axis_reads_one_element_per_row() {
        let m = alias(&[3, 5], &[1, 0]).address_map().unwrap();
        assert!(!m.is_identity_over(15));
        for f in 0..15u32 {
            assert_eq!(at(&m, f), f / 5, "flat {f}");
        }
    }

    #[test]
    fn a_transpose_swaps_the_divisors() {
        // [4,3] view of a [3,4] row-major buffer.
        let m = alias(&[4, 3], &[1, 4]).address_map().unwrap();
        for r in 0..4u32 {
            for c in 0..3u32 {
                assert_eq!(at(&m, r * 3 + c), c * 4 + r);
            }
        }
    }

    #[test]
    fn an_offset_narrow_starts_where_the_slice_does() {
        let o = Operand {
            src: Id(0),
            layout: Layout::from_parts(Dim::Const(2), &dims(&[3]), &dims(&[1])).unwrap(),
            access: AccessPlan::Alias,
        };
        let m = o.address_map().unwrap();
        assert!(!m.is_identity_over(3));
        assert_eq!((at(&m, 0), at(&m, 1), at(&m, 2)), (2, 3, 4));
    }

    /// A conv window operand: the offset sub-axis collides with the position
    /// sub-axis, which per-axis strides cannot express.
    #[test]
    fn a_window_group_divmods_most_significant_first() {
        let map = MultiFlattenMap {
            groups: smallvec::smallvec![AxisGroup {
                sub_axes: smallvec::smallvec![
                    SubAxis {
                        extent: 3,
                        stride: 2
                    },
                    SubAxis {
                        extent: 2,
                        stride: 1
                    },
                ],
            }],
        };
        let o = Operand {
            src: Id(0),
            layout: Layout::contiguous(&dims(&[6])),
            access: AccessPlan::Unflatten(map),
        };
        let m = o.address_map().unwrap();
        // coord = pos*2 + off, storage = pos*2 + off*1
        for pos in 0..3u32 {
            for off in 0..2u32 {
                assert_eq!(at(&m, pos * 2 + off), pos * 2 + off);
            }
        }
    }

    #[test]
    fn a_scalar_broadcast_collapses_to_the_offset() {
        let m = alias(&[4], &[0]).address_map().unwrap();
        assert!(m.terms.is_empty());
        assert!(!m.is_identity_over(4));
        assert_eq!(at(&m, 3), 0);
    }

    #[test]
    fn the_leading_modulo_is_elided_only_when_the_space_is_covered() {
        let m = alias(&[3, 5], &[1, 0]).address_map().unwrap();
        assert_eq!(m.terms.len(), 1);
        assert!(!m.needs_modulo(0, 15));
        // A larger reading space would run past the operand, so the `%` stays.
        assert!(m.needs_modulo(0, 30));
    }
}
