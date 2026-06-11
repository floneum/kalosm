//! Tiled lowering for fused map-reduce operations.
//!
//! When the fused producer's index structure shows 2D reuse — every
//! reduce-axis-dependent input misses at least one of two parallel dims — the
//! reduction lowers as a workgroup-tiled kernel: each k-tile of every staged
//! input is loaded cooperatively into workgroup memory once and reused by all
//! lanes, and each thread accumulates a TM×TN register tile. A composed
//! matmul is exactly this shape (`a[.., m, k]` misses `n`, `b[.., k, n]`
//! misses `m`), so the tiled matmul kernel falls out of the caching the tiled
//! loads provide — as does any other contraction the tensor API composes,
//! whatever its dim order, pre-scaling, or surrounding expression.

use fusor_tile_ir as tile_ir;
use std::hash::Hash;
use tile_ir::tile::{Mask, Tile};

use crate::{
    access_analysis::InputAccesses,
    mir::{
        inputs::MirValue, kernel_backend, kernel_backend::DirectKernel, operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_direct::{
        TensorMeta, ValueTile, apply_unary_function_chain, declare_value,
        eval_nary_expr_on_value_tiles, layout_index, linear_group, tile_u32, zero_literal,
    },
    nary_wise::{NaryExpr, NaryOp, NaryScalar},
    reduce::{ReduceOp, ReduceOperation},
    reduce_direct::{ReduceKernelInputs, tile_literal_for, tile_reduce_op},
    tensor::DataTypeEnum,
    visit_tiled::{MaybeQData, distribute_workgroups},
};

/// Workgroup tile geometry. One fixed shape keeps plan selection
/// deterministic; the gates below reject shapes it cannot cover profitably.
const BM: u32 = 32;
const BN: u32 = 32;
const BK: u32 = 8;
const TM: u32 = 4;
const TN: u32 = 4;
const LANES: u32 = (BM / TM) * (BN / TN);

/// Metadata for dependence analysis. Block-quantized matrices address by
/// (row, col) with a zero in the kernel meta's column slot; the *analysis*
/// must still see the column as a real dependence, so it gets full row-major
/// strides here.
pub(crate) fn analysis_meta(value: &MaybeQData) -> Option<TensorMeta> {
    match value {
        MaybeQData::Tensor(tensor) => TensorMeta::new(tensor),
        MaybeQData::QMatrix(matrix) => {
            let mut meta = TensorMeta::for_matrix(matrix)?;
            let mut acc = 1u32;
            for dim in (0..meta.shape.len()).rev() {
                meta.strides[dim] = acc;
                acc = acc.checked_mul(meta.shape[dim])?;
            }
            Some(meta)
        }
    }
}

/// Load one input value at `coords`, dequantizing block-quantized inputs
/// through the format-aware per-element path.
fn load_input_value(
    program: &mut tile_ir::tile::TileBlock<'_>,
    storage: &crate::nary_direct::Storage2,
    meta: &TensorMeta,
    coords: &[Tile],
    mask: Mask,
) -> ValueTile {
    if let crate::nary_direct::Storage2::Quantized(matrix) = storage {
        let along_row = coords.last().cloned().unwrap_or_else(|| tile_u32(0));
        let which_row = layout_index(meta, coords);
        return ValueTile::F32(program.load_quantized(matrix, along_row, which_row, mask, 0.0));
    }
    storage.load(program, layout_index(meta, coords), mask)
}

/// The allocation footprint one input re-streams per redundant read.
pub(crate) fn input_allocation_bytes(meta: &TensorMeta, value: &MaybeQData) -> u64 {
    let element_bytes = match value {
        MaybeQData::Tensor(tensor) => match tensor.datatype() {
            DataTypeEnum::F16 => 2,
            DataTypeEnum::F32 | DataTypeEnum::U32 => 4,
        },
        MaybeQData::QMatrix(_) => 4,
    };
    meta.allocation_len as u64 * element_bytes
}

struct ReduceTiledKernelVariant;

/// How a staged input's workgroup tile is addressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum StagedKind {
    /// References the row dim and the reduce axis: `[BM, BK]` tile.
    RowK,
    /// References the col dim and the reduce axis: `[BK, BN]` tile.
    KCol,
    /// References only the reduce axis (plus grid dims): `[BK]` tile.
    KOnly,
}

impl StagedKind {
    fn tile_elements(self) -> u32 {
        match self {
            StagedKind::RowK => BM * BK,
            StagedKind::KCol => BK * BN,
            StagedKind::KOnly => BK,
        }
    }
}

#[derive(Clone, Debug)]
enum InputRole {
    /// Reduce-axis-dependent: staged through workgroup memory each k-tile.
    Staged(StagedKind),
    /// Independent of the reduce axis: loaded once per thread, outside the
    /// k loop.
    Direct,
}

#[derive(Clone, Debug)]
struct PlannedInput {
    /// Index-space dim read by each input dimension (pure `DimIndex` only).
    dims: Vec<usize>,
    role: InputRole,
}

/// Where staged inputs are cached between reuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Staging {
    /// Full tiles big enough to amortize barriers: k-tiles staged
    /// cooperatively through workgroup memory.
    Workgroup,
    /// Partial or skinny tiles: each thread keeps its TM row values and TN
    /// column values in registers per k step, so the reuse across its
    /// register tile survives without barriers.
    Register,
}

/// Per-workgroup output-tile geometry. Always 64 lanes:
/// `(bm / tm) * (bn / tn) == LANES`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TileGeom {
    bm: u32,
    bn: u32,
    tm: u32,
    tn: u32,
}

impl TileGeom {
    /// The 2D contraction tile: matches the dense matmul geometry.
    const PAIR: Self = Self {
        bm: BM,
        bn: BN,
        tm: TM,
        tn: TN,
    };
    /// The 1D tile: one parallel dim, each thread covering `tm` outputs so
    /// row-missing inputs load once per k step instead of `tm` times.
    const SINGLE: Self = Self {
        bm: 256,
        bn: 1,
        tm: 4,
        tn: 1,
    };

    fn outs(self) -> u32 {
        self.tm * self.tn
    }
}

struct TiledReducePlan {
    /// The parallel dim(s) tiled across the workgroup. `col_dim` is `None`
    /// for 1D tiling: a single reuse dim with no second axis to pair it
    /// with.
    row_dim: usize,
    col_dim: Option<usize>,
    geom: TileGeom,
    /// Every other non-reduce dim: one coordinate per workgroup.
    grid_dims: Vec<usize>,
    inputs: Vec<PlannedInput>,
    staging: Staging,
    /// Out-of-bounds index-space slots evaluate to the reduction identity
    /// with zero-filled tiles (a bare multiply chain summed), so the inner
    /// loop can skip per-element masking — the tiled matmul fast path.
    fill_neutral: bool,
}

/// Out-of-bounds slots collapse to the reduction identity without masking iff
/// the expression is a flat multiply folded with `Sum` whose factors map a
/// zero-filled load to zero (and stay finite): every staged tile zero-fills
/// its out-of-bounds slots, every out-of-bounds index-space slot is out of
/// bounds for at least one staged input, and one zero factor kills the
/// product. Factors may wrap their load in zero-preserving unary ops (casts,
/// negation, finite constant scales) — the dense matmul's f16→f32 accumulate
/// casts and scale epilogues keep the fast path.
fn is_fill_neutral(expr: &NaryExpr, op: ReduceOp) -> bool {
    if op != ReduceOp::Sum {
        return false;
    }
    fn scalar_is_finite(scalar: NaryScalar) -> bool {
        match scalar {
            NaryScalar::F32(value) => value.is_finite(),
            NaryScalar::F16(value) => value.is_finite(),
            NaryScalar::U32(_) => true,
        }
    }
    fn factor_is_zero_preserving(expr: &NaryExpr) -> bool {
        match expr {
            NaryExpr::IndexedInput { .. } => true,
            NaryExpr::Scalar(scalar) => scalar_is_finite(*scalar),
            NaryExpr::Op { children, function } if children.len() == 1 => {
                let zero_preserving = matches!(
                    function.op,
                    NaryOp::Cast | NaryOp::Neg | NaryOp::Abs
                ) || matches!(function.op, NaryOp::MulConst(scalar) if scalar_is_finite(scalar));
                zero_preserving && factor_is_zero_preserving(&children[0])
            }
            _ => false,
        }
    }
    fn flat_mul_factors_collapse(expr: &NaryExpr) -> bool {
        match expr {
            NaryExpr::Op { children, function }
                if function.op == NaryOp::Mul && children.len() == 2 =>
            {
                children.iter().all(flat_mul_factors_collapse)
            }
            other => factor_is_zero_preserving(other),
        }
    }
    matches!(expr, NaryExpr::Op { function, .. } if function.op == NaryOp::Mul)
        && flat_mul_factors_collapse(expr)
}

fn plan_tiled_reduce(
    operation: &ReduceOperation,
    values: &[MaybeQData],
    device: &crate::Device,
) -> Option<TiledReducePlan> {
    let shape = &operation.shape;
    let axis = operation.axis;
    let rank = shape.len();
    // One parallel dim + the reduce axis is enough for 1D tiling; the 2D
    // pair search simply finds nothing below rank 3.
    if rank < 2 {
        return None;
    }
    let metas: Vec<TensorMeta> = values.iter().map(analysis_meta).collect::<Option<_>>()?;
    let access = InputAccesses::collect(&operation.expression, operation.inputs.len(), &metas)?;

    // Pick the (row, col) pair: both must be fed by at least one staged
    // input, and no staged input may reference both (its tile would need all
    // three tiled dims). Among valid pairs prefer the most square, largest
    // coverage; ties resolve to the lowest dim indices for determinism.
    let k_dep: Vec<bool> = (0..access.effective.len())
        .map(|i| access.depends_on(i, axis))
        .collect();
    if !k_dep.iter().any(|dep| *dep) {
        return None;
    }
    let mut best: Option<((usize, usize), (usize, usize))> = None;
    for row_dim in 0..rank {
        for col_dim in 0..rank {
            if row_dim == col_dim || row_dim == axis || col_dim == axis {
                continue;
            }
            let mut row_fed = false;
            let mut col_fed = false;
            let mut valid = true;
            for (i, k_dep) in k_dep.iter().enumerate() {
                if !k_dep {
                    continue;
                }
                let has_row = access.depends_on(i, row_dim);
                let has_col = access.depends_on(i, col_dim);
                if has_row && has_col {
                    valid = false;
                    break;
                }
                row_fed |= has_row;
                col_fed |= has_col;
            }
            if !valid || !row_fed || !col_fed {
                continue;
            }
            let (m, n) = (shape[row_dim], shape[col_dim]);
            let score = (m.min(n), m * n);
            if best
                .as_ref()
                .is_none_or(|(best_score, _)| score > *best_score)
            {
                best = Some((score, (row_dim, col_dim)));
            }
        }
    }
    // No pair: fall back to a single reuse dim — a parallel dim some
    // k-dependent input misses. Each thread then covers `tm` outputs along
    // it, so the missing input streams once per thread instead of `tm`
    // times. Trading `tm`× thread count for that reuse only pays when the
    // missed inputs are too large for cache to absorb the re-streams
    // (cache-resident reuse is free, and the occupancy loss is not), so the
    // missed inputs must clear the cache-thrash threshold.
    let (row_dim, col_dim) = match best {
        Some((_, (row_dim, col_dim))) => (row_dim, Some(col_dim)),
        None => {
            let mut best_single: Option<((u64, usize), usize)> = None;
            for d in 0..rank {
                if d == axis || shape[d] < TileGeom::SINGLE.bm as usize {
                    continue;
                }
                let missed_bytes: u64 = (0..k_dep.len())
                    .filter(|&i| k_dep[i] && !access.depends_on(i, d))
                    .map(|i| input_allocation_bytes(&metas[i], &values[i]))
                    .sum();
                if missed_bytes < device.last_level_cache_bytes() {
                    continue;
                }
                let score = (missed_bytes, shape[d]);
                if best_single
                    .as_ref()
                    .is_none_or(|(best_score, _)| score > *best_score)
                {
                    best_single = Some((score, d));
                }
            }
            let (_, d) = best_single?;
            (d, None)
        }
    };
    let geom = match col_dim {
        Some(_) => TileGeom::PAIR,
        None => TileGeom::SINGLE,
    };

    let m: u32 = shape[row_dim].try_into().ok()?;
    let n: u32 = match col_dim {
        Some(col_dim) => shape[col_dim].try_into().ok()?,
        None => 1,
    };
    let k: u32 = shape[axis].try_into().ok()?;
    let limits = device.limits();
    if LANES > limits.max_compute_workgroup_size_x
        || LANES > limits.max_compute_invocations_per_workgroup
    {
        return None;
    }
    let grid_dims: Vec<usize> = (0..rank)
        .filter(|&d| d != axis && d != row_dim && Some(d) != col_dim)
        .collect();
    let batch: u32 = grid_dims.iter().try_fold(1u32, |acc, &d| {
        acc.checked_mul(u32::try_from(shape[d]).ok()?)
    })?;
    let tiles_m = m.div_ceil(geom.bm);
    let tiles_n = n.div_ceil(geom.bn);
    let workgroups = batch.checked_mul(tiles_m)?.checked_mul(tiles_n)?;
    let actual_outputs = batch.checked_mul(m)?.checked_mul(n)?;
    let covered_outputs = workgroups.checked_mul(geom.bm * geom.bn)?;

    let mut workgroup_bytes = 0u32;
    let mut inputs = Vec::with_capacity(access.dims.len());
    for (i, (dims, k_dep)) in access.dims.iter().zip(&k_dep).enumerate() {
        let value = &values[i];
        let role = if *k_dep {
            let has_col = col_dim.is_some_and(|col_dim| access.depends_on(i, col_dim));
            let kind = match (access.depends_on(i, row_dim), has_col) {
                (true, false) => StagedKind::RowK,
                (false, true) => StagedKind::KCol,
                (false, false) => StagedKind::KOnly,
                (true, true) => unreachable!("pair selection rejects row+col staged inputs"),
            };
            let element_bytes = match value {
                MaybeQData::Tensor(tensor) => match tensor.datatype() {
                    DataTypeEnum::F16 => 2,
                    DataTypeEnum::F32 | DataTypeEnum::U32 => 4,
                },
                MaybeQData::QMatrix(_) => 4,
            };
            workgroup_bytes =
                workgroup_bytes.checked_add(kind.tile_elements().checked_mul(element_bytes)?)?;
            InputRole::Staged(kind)
        } else {
            InputRole::Direct
        };
        inputs.push(PlannedInput {
            dims: dims.clone(),
            role,
        });
    }

    // Workgroup staging needs full-enough 2D tiles to amortize its barriers:
    // the same shape and 75%-utilization gates as the dedicated dense matmul
    // tile selection. Anything else keeps the register tile, whose reuse
    // needs no barriers and tolerates partial coverage.
    let workgroup_eligible = col_dim.is_some()
        && m >= BM
        && n >= BN
        && k >= BK
        && (actual_outputs as u64) * 4 >= (covered_outputs as u64) * 3
        && workgroup_bytes <= limits.max_compute_workgroup_storage_size;
    let staging = if workgroup_eligible {
        Staging::Workgroup
    } else {
        Staging::Register
    };

    let fill_neutral = is_fill_neutral(&operation.expression, operation.function.op);

    Some(TiledReducePlan {
        row_dim,
        col_dim,
        geom,
        grid_dims,
        inputs,
        staging,
        fill_neutral,
    })
}

/// The full index-space coordinate vector for one (batch, row, col, k)
/// position, in dim order.
fn full_coords(
    plan: &TiledReducePlan,
    rank: usize,
    axis: usize,
    batch_coords: &[Tile],
    row: &Tile,
    col: &Tile,
    k: &Tile,
) -> Vec<Tile> {
    (0..rank)
        .map(|d| {
            if d == axis {
                k.clone()
            } else if d == plan.row_dim {
                row.clone()
            } else if Some(d) == plan.col_dim {
                col.clone()
            } else {
                let slot = plan.grid_dims.iter().position(|&g| g == d).unwrap();
                batch_coords[slot].clone()
            }
        })
        .collect()
}

fn input_coords(input: &PlannedInput, coords: &[Tile]) -> Vec<Tile> {
    input.dims.iter().map(|&d| coords[d].clone()).collect()
}

pub(crate) fn datatype_scalar(datatype: DataTypeEnum) -> tile_ir::ScalarElement {
    match datatype {
        DataTypeEnum::F32 => tile_ir::ScalarElement::F32,
        DataTypeEnum::F16 => tile_ir::ScalarElement::F16,
        DataTypeEnum::U32 => tile_ir::ScalarElement::U32,
    }
}

fn value_tile(datatype: DataTypeEnum, value: Tile) -> ValueTile {
    match datatype {
        DataTypeEnum::F32 => ValueTile::F32(value),
        DataTypeEnum::F16 => ValueTile::F16(value),
        DataTypeEnum::U32 => ValueTile::U32(value),
    }
}

pub(crate) fn build_reduce_tiled_kernel(
    operation: &ReduceOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
) -> Option<DirectKernel> {
    let parsed = ReduceKernelInputs::parse(operation, graph, inputs)?;
    let device = graph.device();
    let plan = plan_tiled_reduce(operation, &parsed.values, &device)?;
    if std::env::var_os("FUSOR_TRACE_REDUCE_TILED").is_some() {
        eprintln!(
            "reduce_tiled row={} col={:?} axis={} staging={:?} fill_neutral={} staged={} name={}",
            plan.row_dim,
            plan.col_dim,
            operation.axis,
            plan.staging,
            plan.fill_neutral,
            plan.inputs
                .iter()
                .filter(|input| matches!(input.role, InputRole::Staged(_)))
                .count(),
            operation.name(),
        );
    }

    let rank = operation.shape.len();
    let axis = operation.axis;
    let geom = plan.geom;
    let m: u32 = operation.shape[plan.row_dim].try_into().ok()?;
    let n: u32 = match plan.col_dim {
        Some(col_dim) => operation.shape[col_dim].try_into().ok()?,
        None => 1,
    };
    let k: u32 = operation.shape[axis].try_into().ok()?;
    let batch_sizes: Vec<u32> = plan
        .grid_dims
        .iter()
        .map(|&d| u32::try_from(operation.shape[d]).ok())
        .collect::<Option<_>>()?;
    let batch: u32 = batch_sizes.iter().product();
    let tiles_m = m.div_ceil(geom.bm);
    let tiles_n = n.div_ceil(geom.bn);
    let total_tiles = batch * tiles_m * tiles_n;
    let k_tiles = k.div_ceil(BK);

    let dispatch_size = distribute_workgroups(
        total_tiles,
        device.limits().max_compute_workgroups_per_dimension,
    );
    let variant =
        kernel_backend::KernelVariantKey::with_payload::<ReduceTiledKernelVariant>(|state| {
            plan.row_dim.hash(state);
            plan.col_dim.hash(state);
            plan.geom.hash(state);
            plan.staging.hash(state);
            plan.fill_neutral.hash(state);
            for input in &plan.inputs {
                input.dims.hash(state);
                match input.role {
                    InputRole::Staged(kind) => kind.hash(state),
                    InputRole::Direct => "direct".hash(state),
                }
            }
        });
    let cache_key = operation.kernel_cache_key_with_dispatch(
        variant,
        Some(workgroup_shape),
        dispatch_size,
        inputs,
    );

    let reduce_dtype = operation.function.datatype();
    let reduce_op = tile_reduce_op(operation.function.op);
    let initial = operation.function.initial_value;
    let expression = operation.expression.clone();
    let post_chain = operation.post_element_wise.clone();
    let output = inputs.last()?.as_tensor()?.clone();
    let output_dtype = output.datatype();
    let output_value = MaybeQData::Tensor(output);
    let values = parsed.values;
    let plan = std::sync::Arc::new(plan);

    kernel_backend::run_kernel(
        device.kernel_cache(),
        format!("{}_tiled", operation.name()),
        cache_key,
        dispatch_size,
        move |kb| {
            let mut storages = Vec::with_capacity(values.len());
            let mut metas = Vec::with_capacity(values.len());
            for value in &values {
                let (storage, meta) = declare_value(kb, value, false)?;
                storages.push(storage);
                metas.push(meta);
            }
            let (output_storage, output_meta) = declare_value(kb, &output_value, true)?;

            let phase = kb.program();
            let staged_tiles: Vec<Option<tile_ir::tile::WorkgroupTile>> = plan
                .inputs
                .iter()
                .zip(&metas)
                .map(|(input, meta)| match (plan.staging, &input.role) {
                    (Staging::Workgroup, InputRole::Staged(kind)) => {
                        let element = datatype_scalar(meta.datatype);
                        Some(match kind {
                            StagedKind::RowK => phase.alloc_workgroup_tile(element, BM, BK),
                            StagedKind::KCol => phase.alloc_workgroup_tile(element, BK, BN),
                            StagedKind::KOnly => phase.alloc_workgroup_array(element, BK),
                        })
                    }
                    _ => None,
                })
                .collect();

            phase.program_grid(LANES, dispatch_size, |program| {
                let tile_id = linear_group(program, dispatch_size);
                let tile_active = tile_id.clone().lt(total_tiles);
                let batch_id = tile_id.clone() / (tiles_m * tiles_n);
                let local_tile = tile_id % (tiles_m * tiles_n);
                let m_tile = local_tile.clone() / tiles_n;
                let n_tile = local_tile % tiles_n;

                // Mixed-radix batch decomposition over the grid dims, in dim
                // order (earlier dims vary slowest).
                let batch_coords: Vec<Tile> = (0..batch_sizes.len())
                    .map(|i| {
                        let size = batch_sizes[i];
                        if size == 1 {
                            return tile_u32(0);
                        }
                        let divisor: u32 = batch_sizes[i + 1..].iter().product();
                        let quotient = if divisor == 1 {
                            batch_id.clone()
                        } else {
                            batch_id.clone() / divisor
                        };
                        quotient % size
                    })
                    .collect();

                let lane = program.lane();
                let lane_row = program.bind(lane.clone() / (geom.bn / geom.tn));
                let lane_col = program.bind(lane % (geom.bn / geom.tn));
                let m_tile_base = program.bind(m_tile * geom.bm);
                let n_tile_base = program.bind(n_tile * geom.bn);
                let row_base = program.bind(m_tile_base.clone() + lane_row.clone() * geom.tm);
                let col_base = program.bind(n_tile_base.clone() + lane_col.clone() * geom.tn);

                let identity = || Tile::literal(tile_literal_for(initial, reduce_dtype));

                // Per-thread output-slot bounds, hoisted out of the k loop.
                let rc_masks: Vec<Mask> = (0..geom.outs())
                    .map(|idx| {
                        let (r, c) = (idx / geom.tn, idx % geom.tn);
                        let row = row_base.clone() + r;
                        let col = col_base.clone() + c;
                        tile_active.clone().and(row.lt(m)).and(col.lt(n))
                    })
                    .collect();

                // Direct (k-independent) inputs: one bound load per output
                // slot, reused across the whole k loop.
                let direct_values: Vec<Option<Vec<(ValueTile, DataTypeEnum)>>> = plan
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(i, input)| {
                        if !matches!(input.role, InputRole::Direct) {
                            return None;
                        }
                        Some(
                            (0..geom.outs())
                                .map(|idx| {
                                    let (r, c) = (idx / geom.tn, idx % geom.tn);
                                    let row = row_base.clone() + r;
                                    let col = col_base.clone() + c;
                                    let coords = full_coords(
                                        &plan,
                                        rank,
                                        axis,
                                        &batch_coords,
                                        &row,
                                        &col,
                                        &tile_u32(0),
                                    );
                                    let index =
                                        layout_index(&metas[i], &input_coords(input, &coords));
                                    let loaded = storages[i].load(
                                        program,
                                        index,
                                        rc_masks[idx as usize].clone(),
                                    );
                                    // Bind to a local so the k loop reuses
                                    // the value instead of re-issuing the
                                    // storage load every iteration.
                                    let native = match loaded {
                                        ValueTile::F32(v)
                                        | ValueTile::F16(v)
                                        | ValueTile::U32(v) => v,
                                        ValueTile::Bool(_) => {
                                            unreachable!("tensor inputs are f32/f16/u32")
                                        }
                                    };
                                    (
                                        value_tile(metas[i].datatype, program.bind(native)),
                                        metas[i].datatype,
                                    )
                                })
                                .collect(),
                        )
                    })
                    .collect();

                let initial_accs: Vec<Tile> = (0..geom.outs()).map(|_| identity()).collect();
                let sums = match plan.staging {
                    Staging::Register => program.fold_vec(
                        tile_ir::tile::range(k),
                        initial_accs,
                        |program, k_index, accs| {
                            // Per staged input, the register slice this thread
                            // reuses across its TM×TN tile: TM row values, TN
                            // column values, or one k-only value.
                            let reg_values: Vec<Option<Vec<(ValueTile, DataTypeEnum)>>> = plan
                                .inputs
                                .iter()
                                .enumerate()
                                .map(|(i, input)| {
                                    let InputRole::Staged(kind) = input.role else {
                                        return None;
                                    };
                                    let count = match kind {
                                        StagedKind::RowK => geom.tm,
                                        StagedKind::KCol => geom.tn,
                                        StagedKind::KOnly => 1,
                                    };
                                    Some(
                                        (0..count)
                                            .map(|j| {
                                                let (row, col, mask) = match kind {
                                                    StagedKind::RowK => {
                                                        let row = row_base.clone() + j;
                                                        let mask = tile_active
                                                            .clone()
                                                            .and(row.clone().lt(m));
                                                        (row, tile_u32(0), mask)
                                                    }
                                                    StagedKind::KCol => {
                                                        let col = col_base.clone() + j;
                                                        let mask = tile_active
                                                            .clone()
                                                            .and(col.clone().lt(n));
                                                        (tile_u32(0), col, mask)
                                                    }
                                                    StagedKind::KOnly => (
                                                        tile_u32(0),
                                                        tile_u32(0),
                                                        tile_active.clone(),
                                                    ),
                                                };
                                                let coords = full_coords(
                                                    &plan,
                                                    rank,
                                                    axis,
                                                    &batch_coords,
                                                    &row,
                                                    &col,
                                                    &k_index,
                                                );
                                                let index = layout_index(
                                                    &metas[i],
                                                    &input_coords(input, &coords),
                                                );
                                                let loaded =
                                                    storages[i].load(program, index, mask.clone());
                                                let value = match loaded {
                                                    ValueTile::F32(v)
                                                    | ValueTile::F16(v)
                                                    | ValueTile::U32(v) => Tile::select(
                                                        mask,
                                                        v,
                                                        Tile::literal(zero_literal(
                                                            metas[i].datatype,
                                                        )),
                                                    ),
                                                    ValueTile::Bool(_) => unreachable!(
                                                        "tensor inputs are f32/f16/u32"
                                                    ),
                                                };
                                                (
                                                    value_tile(metas[i].datatype, value),
                                                    metas[i].datatype,
                                                )
                                            })
                                            .collect(),
                                    )
                                })
                                .collect();

                            // Every k step is in range and out-of-bounds
                            // output slots never store, so no accumulate
                            // masking is needed in register mode.
                            accs.into_iter()
                                .enumerate()
                                .map(|(idx, acc)| {
                                    let (r, c) = (idx as u32 / geom.tn, idx as u32 % geom.tn);
                                    let slot_values: Vec<(ValueTile, DataTypeEnum)> = plan
                                        .inputs
                                        .iter()
                                        .enumerate()
                                        .map(|(i, input)| match input.role {
                                            InputRole::Staged(kind) => {
                                                let j = match kind {
                                                    StagedKind::RowK => r,
                                                    StagedKind::KCol => c,
                                                    StagedKind::KOnly => 0,
                                                };
                                                reg_values[i].as_ref().unwrap()[j as usize].clone()
                                            }
                                            InputRole::Direct => {
                                                direct_values[i].as_ref().unwrap()[idx].clone()
                                            }
                                        })
                                        .collect();
                                    let (value, _) =
                                        eval_nary_expr_on_value_tiles(&expression, &slot_values);
                                    let value = match value.cast_to(reduce_dtype) {
                                        ValueTile::F32(v)
                                        | ValueTile::F16(v)
                                        | ValueTile::U32(v) => v,
                                        ValueTile::Bool(_) => {
                                            unreachable!("reduce dtype is f32/f16/u32")
                                        }
                                    };
                                    acc.binary(reduce_op.binary(), value)
                                })
                                .collect()
                        },
                    ),
                    Staging::Workgroup => program.fold_vec(
                        tile_ir::tile::range(k_tiles),
                        initial_accs,
                        |program, k_tile, accs| {
                            let k_base = program.bind(k_tile * BK);

                            // Cooperative staging: every lane copies its share of
                            // each staged input's k-tile into workgroup memory.
                            for (i, input) in plan.inputs.iter().enumerate() {
                                let InputRole::Staged(kind) = input.role else {
                                    continue;
                                };
                                let tile = staged_tiles[i].as_ref().unwrap();
                                let elements = kind.tile_elements();
                                for pass in 0..elements.div_ceil(LANES) {
                                    let flat = program.lane() + pass * LANES;
                                    let (row, col, k_index, flat_ok) = match kind {
                                        StagedKind::RowK => {
                                            let local_row = flat.clone() / BK;
                                            let local_k = flat.clone() % BK;
                                            (
                                                m_tile_base.clone() + local_row,
                                                tile_u32(0),
                                                k_base.clone() + local_k,
                                                Mask::all(),
                                            )
                                        }
                                        StagedKind::KCol => {
                                            let local_k = flat.clone() / BN;
                                            let local_col = flat.clone() % BN;
                                            (
                                                tile_u32(0),
                                                n_tile_base.clone() + local_col,
                                                k_base.clone() + local_k,
                                                Mask::all(),
                                            )
                                        }
                                        StagedKind::KOnly => (
                                            tile_u32(0),
                                            tile_u32(0),
                                            k_base.clone() + flat.clone(),
                                            flat.clone().lt(elements),
                                        ),
                                    };
                                    let mut in_bounds =
                                        tile_active.clone().and(flat_ok).and(k_index.clone().lt(k));
                                    if matches!(kind, StagedKind::RowK) {
                                        in_bounds = in_bounds.and(row.clone().lt(m));
                                    }
                                    if matches!(kind, StagedKind::KCol) {
                                        in_bounds = in_bounds.and(col.clone().lt(n));
                                    }
                                    let coords = full_coords(
                                        &plan,
                                        rank,
                                        axis,
                                        &batch_coords,
                                        &row,
                                        &col,
                                        &k_index,
                                    );
                                    let loaded = load_input_value(
                                        program,
                                        &storages[i],
                                        &metas[i],
                                        &input_coords(input, &coords),
                                        in_bounds.clone(),
                                    );
                                    // Zero-fill out-of-bounds slots: in the
                                    // fill-neutral fast path that zero is what
                                    // collapses the product, elsewhere the inner
                                    // mask discards the slot anyway.
                                    let value = match loaded.cast_to(metas[i].datatype) {
                                        ValueTile::F32(v)
                                        | ValueTile::F16(v)
                                        | ValueTile::U32(v) => Tile::select(
                                            in_bounds,
                                            v,
                                            Tile::literal(zero_literal(metas[i].datatype)),
                                        ),
                                        ValueTile::Bool(_) => {
                                            unreachable!("tensor inputs are f32/f16/u32")
                                        }
                                    };
                                    program.store_workgroup(tile, flat, value);
                                }
                            }
                            program.workgroup_barrier();

                            let chunk_sums: Vec<Tile> = (0..geom.outs())
                                .map(|idx| {
                                    let (r, c) = (idx / geom.tn, idx % geom.tn);
                                    let local_row = lane_row.clone() * geom.tm + r;
                                    let local_col = lane_col.clone() * geom.tn + c;
                                    let mut chunk = identity();
                                    for kk in 0..BK {
                                        let slot_values: Vec<(ValueTile, DataTypeEnum)> = plan
                                            .inputs
                                            .iter()
                                            .enumerate()
                                            .map(|(i, input)| match input.role {
                                                InputRole::Staged(kind) => {
                                                    let tile = staged_tiles[i].as_ref().unwrap();
                                                    let flat = match kind {
                                                        StagedKind::RowK => {
                                                            local_row.clone() * BK + kk
                                                        }
                                                        StagedKind::KCol => {
                                                            local_col.clone() + kk * BN
                                                        }
                                                        StagedKind::KOnly => tile_u32(kk),
                                                    };
                                                    (
                                                        value_tile(
                                                            metas[i].datatype,
                                                            program.load_workgroup(tile, flat),
                                                        ),
                                                        metas[i].datatype,
                                                    )
                                                }
                                                InputRole::Direct => {
                                                    direct_values[i].as_ref().unwrap()[idx as usize]
                                                        .clone()
                                                }
                                            })
                                            .collect();
                                        let (value, _) = eval_nary_expr_on_value_tiles(
                                            &expression,
                                            &slot_values,
                                        );
                                        let value = match value.cast_to(reduce_dtype) {
                                            ValueTile::F32(v)
                                            | ValueTile::F16(v)
                                            | ValueTile::U32(v) => v,
                                            ValueTile::Bool(_) => {
                                                unreachable!("reduce dtype is f32/f16/u32")
                                            }
                                        };
                                        let value = if plan.fill_neutral {
                                            value
                                        } else {
                                            let valid = rc_masks[idx as usize]
                                                .clone()
                                                .and((k_base.clone() + kk).lt(k));
                                            Tile::select(valid, value, identity())
                                        };
                                        chunk = chunk.binary(reduce_op.binary(), value);
                                    }
                                    chunk
                                })
                                .collect();
                            // Bind every chunk before the trailing barrier: the
                            // next iteration's staging overwrites the tiles these
                            // reads source from.
                            let chunk_sums: Vec<Tile> = chunk_sums
                                .into_iter()
                                .map(|chunk| program.bind(chunk))
                                .collect();
                            program.workgroup_barrier();
                            accs.into_iter()
                                .zip(chunk_sums)
                                .map(|(acc, chunk)| acc.binary(reduce_op.binary(), chunk))
                                .collect()
                        },
                    ),
                };

                for (idx, sum) in sums.into_iter().enumerate() {
                    let idx = idx as u32;
                    let (r, c) = (idx / geom.tn, idx % geom.tn);
                    let row = row_base.clone() + r;
                    let col = col_base.clone() + c;
                    let (value, value_ty) = apply_unary_function_chain(
                        value_tile(reduce_dtype, sum).into_f32(),
                        reduce_dtype,
                        &post_chain,
                    )
                    .expect("validated reduce post_element_wise chain");
                    let value = ValueTile::F32(value)
                        .cast_to(value_ty)
                        .cast_to(output_dtype);
                    let coords =
                        full_coords(&plan, rank, axis, &batch_coords, &row, &col, &tile_u32(0));
                    let out_coords: Vec<Tile> = (0..rank)
                        .filter(|&d| d != axis)
                        .map(|d| coords[d].clone())
                        .collect();
                    let output_index = layout_index(&output_meta, &out_coords);
                    output_storage.store(
                        program,
                        output_index,
                        value,
                        rc_masks[idx as usize].clone(),
                    );
                }
            });
            Some(())
        },
    )
}
