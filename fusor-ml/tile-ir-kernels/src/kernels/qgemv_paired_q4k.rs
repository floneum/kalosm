//! Q4K paired-epilogue GEMV program kernels.

use fusor_tile_ir::tile::{range, Program, Storage, Tile};
use fusor_tile_ir::{QuantizedMatrix, WorkgroupAxis};

use crate::grid::dot4_sum;
use crate::types::{matrix_shape, PairedEpilogue};

/// Q4K block dequantizes into 8 contiguous f32 weights per lane (one
/// `Dequantize` shared node), so the K-loop advances 8 elements per lane.
const VALUES_PER_LANE: u32 = 8;
/// Subgroup width assumed by the Q4K paired tiling.
const SUBGROUP_SIZE: u32 = 32;

/// Tile shape for `qgemv_q4k_paired_ggml`. The kernel only takes `block` (the
/// workgroup-size literal) as a runtime arg; `subgroups` and `pairs_per_subgroup`
/// ride along as runtime args, and `dots_per_subgroup` always equals
/// `pairs_per_subgroup * 2`.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Q4KPairedShape {
    pub subgroups: u32,
    pub pairs_per_subgroup: u32,
    pub block: u32,
}

impl Q4KPairedShape {
    pub const fn new(subgroups: u32, pairs_per_subgroup: u32, block: u32) -> Self {
        Self {
            subgroups,
            pairs_per_subgroup,
            block,
        }
    }

    pub const fn pairs_per_workgroup(self) -> u32 {
        self.subgroups * self.pairs_per_subgroup
    }
}

const Q4K_PAIRED_TILES: &[(&str, Q4KPairedShape)] = &[
    ("ggml_2x1", Q4KPairedShape::new(2, 1, 64)),
    ("ggml_2x2", Q4KPairedShape::new(2, 2, 64)),
    ("ggml_2x4", Q4KPairedShape::new(2, 4, 64)),
    ("ggml_4x1", Q4KPairedShape::new(4, 1, 128)),
    ("ggml_4x2", Q4KPairedShape::new(4, 2, 128)),
    ("ggml_4x4", Q4KPairedShape::new(4, 4, 128)),
    ("ggml_8x1", Q4KPairedShape::new(8, 1, 256)),
    ("ggml_8x2", Q4KPairedShape::new(8, 2, 256)),
];

fn q4k_paired_shape() -> Q4KPairedShape {
    const DEFAULT: Q4KPairedShape = Q4KPairedShape::new(4, 4, 128);
    let Ok(value) = std::env::var("FUSOR_Q4K_PAIRED_TILE") else {
        return DEFAULT;
    };
    Q4K_PAIRED_TILES
        .iter()
        .find(|(name, _)| *name == value)
        .map(|(_, shape)| *shape)
        .unwrap_or(DEFAULT)
}

/// Compute launch geometry for the paired Q4K GEMV kernel.
pub fn qgemv_q4k_paired_dispatch(
    pair_cols: u32,
    m_rows: u32,
    max_workgroups_per_dimension: u32,
) -> Option<([u32; 3], u32, Q4KPairedShape)> {
    let shape = q4k_paired_shape();
    let cols_workgroups = pair_cols.div_ceil(shape.pairs_per_workgroup());
    let total_workgroups = cols_workgroups.checked_mul(m_rows.max(1))?;
    let workgroups_x = total_workgroups.min(max_workgroups_per_dimension).max(1);
    let dispatch_size = [workgroups_x, total_workgroups.div_ceil(workgroups_x), 1];
    dispatch_size
        .iter()
        .all(|dim| *dim <= max_workgroups_per_dimension)
        .then_some((dispatch_size, workgroups_x, shape))
}

/// Inputs and launch geometry for the Q4K paired-epilogue GEMV kernels.
///
/// These kernels consume a Q4K matrix whose columns are laid out as
/// `[gate columns, up columns]`. Each kernel computes both halves for a
/// column pair, applies `epilogue` in-register, and writes the paired result.
///
/// ```no_run
/// # use fusor_tile_ir::{tile, GgmlQuantFormat, ScalarElement, Shape};
/// # use fusor_tile_ir_kernels::{
/// #     PairedEpilogue, Q4KPairedGgml, Q4KPairedShape, qgemv_q4k_paired, quantized_matrix,
/// # };
/// let epilogue =
///     PairedEpilogue::with_extras("mul", 0, |tiles| tiles[0].clone() * tiles[1].clone());
/// let ir = tile::build(|program| {
///     let f32 = ScalarElement::F32.element();
///     let a = program.storage_read(f32, Shape::new([1, 4096]));
///     let b = quantized_matrix(program, GgmlQuantFormat::Q4K, 4096, 8192);
///     let y = program.storage_write(f32, Shape::new([1, 4096]));
///     qgemv_q4k_paired(
///         program,
///         Q4KPairedGgml {
///             a: &a,
///             b: &b,
///             y: &y,
///             pair_cols: 4096,
///             m_rows: 1,
///             workgroups_x: 1,
///             shape: Q4KPairedShape::new(8, 2, 256),
///             epilogue: &epilogue,
///             extras: &[],
///         },
///     );
/// });
/// ```
pub struct Q4KPairedGgml<'a> {
    /// Single-row or batched activation matrix.
    pub a: &'a Storage,
    /// Q4K matrix with `pair_cols * 2` columns.
    pub b: &'a QuantizedMatrix,
    /// Output matrix with `pair_cols` columns.
    pub y: &'a Storage,
    /// Number of gate/up pairs in `b`.
    pub pair_cols: u32,
    /// Number of rows from `a` and `y` covered by the launch.
    pub m_rows: u32,
    /// Preferred dispatch width on X. Clamped to the kernel's total workgroup count.
    pub workgroups_x: u32,
    /// Workgroup/subgroup decomposition for each paired output tile.
    pub shape: Q4KPairedShape,
    /// Register-level operation applied to each `(gate, up)` pair.
    pub epilogue: &'a PairedEpilogue,
    /// One-dimensional extra tensors consumed by `epilogue`.
    pub extras: &'a [Storage],
}

/// Q4K paired-epilogue qgemv body. The kernel reduces the gate and up halves
/// of a `[gate; up]` matmul output and applies the supplied `PairedEpilogue`
/// in-register before the single output store.
///
/// Runtime-typed (ARBOR_DESIGN.md §2/§5): `shape.block` is threaded into
/// `program_grid` as a runtime workgroup-size literal — no `program_grid::<BLOCK>`
/// monomorphization and no `match block` fan-out. Each Q4K block is dequantized
/// to 8 f32 weights through **one** `Dequantize` shared node
/// (`load_quantized_block_vec`), then composed with the activations through the
/// shared `dot4_sum` helper.
pub fn qgemv_q4k_paired(program: &mut Program, spec: Q4KPairedGgml<'_>) {
    let Q4KPairedGgml {
        a,
        b,
        y,
        pair_cols,
        m_rows,
        workgroups_x,
        shape,
        epilogue,
        extras,
    } = spec;
    let subgroups = shape.subgroups;
    let pairs_per_subgroup = shape.pairs_per_subgroup;
    let pairs_per_subgroup_usize = pairs_per_subgroup as usize;
    let dots_per_subgroup_usize = pairs_per_subgroup_usize * 2;
    debug_assert_eq!(subgroups * SUBGROUP_SIZE, shape.block);
    debug_assert_eq!(
        extras.len(),
        epilogue.extras_count(),
        "kernel extras count must match epilogue arity"
    );
    debug_assert!(b.format.is_q4k_family());
    debug_assert_eq!(b.cols, pair_cols * 2);

    let [_, k] = matrix_shape(a.layout());
    let cols_per_workgroup = subgroups * pairs_per_subgroup;
    let cols_workgroups = pair_cols.div_ceil(cols_per_workgroup);
    let m_rows = m_rows.max(1);
    let total_workgroups = cols_workgroups * m_rows;
    let workgroups_x = workgroups_x.min(total_workgroups.max(1));
    let dispatch_y = total_workgroups.div_ceil(workgroups_x);
    let k_per_iter = SUBGROUP_SIZE * VALUES_PER_LANE;
    let k_iterations = k.div_ceil(k_per_iter);
    let full_k_iterations = k.is_multiple_of(k_per_iter);
    let full_cols = pair_cols.is_multiple_of(cols_per_workgroup);
    let b_cloned = b.clone();

    program.program_grid(shape.block, [workgroups_x, dispatch_y, 1], |program| {
        let workgroup_idx = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * workgroups_x;
        let row = workgroup_idx.clone() / cols_workgroups;
        let col_workgroup = workgroup_idx % cols_workgroups;
        let row_in_bounds = row.clone().lt(m_rows);
        let col_group_base = col_workgroup * cols_per_workgroup;
        let subgroup_col_base = program.subgroup_id() * pairs_per_subgroup;
        let col0 = col_group_base + subgroup_col_base;
        let lane = program.subgroup_lane();

        let sums: Vec<Tile> = program.fold_vec(
            range(k_iterations),
            vec![program.f32(0.0); dots_per_subgroup_usize],
            |program, loop_index, accs| {
                let k_base = loop_index * k_per_iter + lane.clone() * VALUES_PER_LANE;
                let in_bounds_k = if full_k_iterations {
                    row_in_bounds.clone()
                } else {
                    row_in_bounds.clone().and(k_base.lt(k))
                };

                // Activations are shared across every gate/up dot in this
                // subgroup pass: 8 contiguous f32 loads at `k_base + i`.
                let acts: Vec<Tile> = (0..VALUES_PER_LANE)
                    .map(|i| {
                        let scalar = program.load(
                            a.at((row.clone(), k_base.clone() + i)),
                            in_bounds_k.clone(),
                            0.0,
                        );
                        program.bind(scalar)
                    })
                    .collect();

                accs.into_iter()
                    .enumerate()
                    .map(|(idx, acc)| {
                        let offset = idx % pairs_per_subgroup_usize;
                        let gate = col0.clone() + offset as u32;
                        let col = if idx < pairs_per_subgroup_usize {
                            gate.clone()
                        } else {
                            gate.clone() + pair_cols
                        };
                        let mask = if full_cols {
                            in_bounds_k.clone()
                        } else {
                            in_bounds_k.clone().and(gate.lt(pair_cols))
                        };
                        let bs = program.load_quantized_block_vec(
                            VALUES_PER_LANE,
                            &b_cloned,
                            &k_base,
                            &col,
                            mask,
                            0.0,
                        );
                        acc + dot4_sum(program, &acts, &bs)
                    })
                    .collect()
            },
        );

        for offset in 0..pairs_per_subgroup_usize {
            let col = col0.clone() + offset as u32;
            let gate = program.subgroup_reduce_sum(sums[offset].clone());
            let up = program.subgroup_reduce_sum(sums[offset + pairs_per_subgroup_usize].clone());
            let store_lane = if full_cols {
                lane.eq(0u32)
            } else {
                lane.eq(0u32).and(col.lt(pair_cols))
            };
            let mask = store_lane.and(row_in_bounds.clone());
            // Load any per-column extras (e.g. bias vectors) at the current
            // output column. Indexing is `extras[k][col]` — extras are 1D
            // tensors of length `pair_cols`.
            let extra_tiles: Vec<Tile> = extras
                .iter()
                .map(|extra| program.load(extra.at(col.clone()), mask.clone(), 0.0))
                .collect();
            let value = epilogue.apply(gate, up, &extra_tiles);
            program.store(y.at((row.clone(), col)), value, mask);
        }
    });
}
