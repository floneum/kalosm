use std::{fmt::Write, sync::OnceLock};

use crate::MatMulOperation;
use crate::mir::globals::KernelGlobalSpace;
use crate::mir::inputs::TensorInput;
use crate::{Device, mir::kernel::GenericKernel};

/// Parameters for cooperative matrix matmul.
///
/// Uses hardware 8×8 matrix multiply instructions via `coopMultiplyAdd`.
/// Matches egging's cooperative-tiled matmul layout for optimal Apple Silicon performance.
#[derive(Debug, Clone, PartialEq)]
pub struct CoopGemmParams {
    pub block_m: u32,    // M-dimension tile per workgroup
    pub block_n: u32,    // Total N-dimension tile per workgroup (across all n_passes)
    pub block_k: u32,    // K-dimension tile for shared memory blocking
    pub n_passes: u32,   // Number of passes over the N dimension
    pub mma_size: u32,   // Cooperative matrix size (8 for 8×8)
    pub wg_threads: u32, // Threads per workgroup
}

impl Default for CoopGemmParams {
    fn default() -> Self {
        Self {
            block_m: 128,
            block_n: 64,
            block_k: 16,
            n_passes: 4,
            mma_size: 8,
            wg_threads: 256,
        }
    }
}

pub(super) fn workgroup_shape_constraints(
    _: &MatMulOperation,
    _: &Device,
    params: &CoopGemmParams,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::Equals(params.wg_threads),
    );
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}

pub(super) fn dispatch_size(
    last_dim_size: usize,
    second_to_last_dim_size: usize,
    batch_size: usize,
    workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
    params: &CoopGemmParams,
) -> [u32; 3] {
    [
        (second_to_last_dim_size as u32).div_ceil(params.block_m),
        (last_dim_size as u32).div_ceil(params.block_n),
        (batch_size as u32).div_ceil(workgroup_shape.z()),
    ]
}

pub(super) fn build_kernel(
    matmul: &MatMulOperation,
    _: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
    inputs: &[crate::mir::inputs::MirValue],
    kernel: &mut GenericKernel,
    params: &CoopGemmParams,
) {
    let [input_a, input_b, _] = inputs else {
        panic!("MatMulOperation requires 3 inputs");
    };
    let input_a_tensor = input_a.as_tensor().unwrap();
    let input_b_tensor = input_b.as_tensor().unwrap();
    let use_direct_store = matmul.post_element_wise.functions.is_empty();
    let post_element_wise_functions = OnceLock::new();

    kernel.enable_cooperative_matrix();

    let block_m = params.block_m;
    let block_n = params.block_n;
    let block_k = params.block_k;
    let n_passes = params.n_passes;
    let mma_size = params.mma_size;
    let wg_threads = params.wg_threads;
    let sg_size: u32 = 32;
    let num_sgs = wg_threads / sg_size;
    let bn_pass = block_n / n_passes;
    let block_m_per_sg = block_m / num_sgs;
    let m_subtiles_per_sg = block_m_per_sg / mma_size;
    let n_subtiles = bn_pass / mma_size;
    let k_substeps = block_k / mma_size;

    // Shared memory layout — matching the repaired egging cooperative path:
    //
    // On Apple, coopLoad<A/B> on these 8x8 tiles behaves like a row-major load from
    // shared memory. Keeping both tiles in row-major form avoids the extra transpose
    // fusor was doing in the hot path.
    //
    // Tile A: row-major MxK, tile_a[m * stride_a + k]  (stride_a = block_k + 1)
    //   coopLoad<A>(&tile_a[(sg_m_base + sm*mma) * stride_a + k_sub*mma], stride_a)
    //
    // Tile B: row-major KxN, tile_b[k * stride_b + n]  (stride_b = bn_pass + padding via K-stride)
    //   coopLoad<B>(&tile_b[k_sub*mma*stride_b + sn*mma], stride_b)
    //
    // MMA: D = A * B + C  →  D_math[M, N] = Σ_K A_math[M, K] * B_math[K, N]
    let stride_a = block_k + 1; // row-major MxK with K-padding
    let stride_b = block_k + 1; // row-major KxN with N packed on the inner axis
    let tile_a_buf_size = block_m * stride_a;
    let tile_b_buf_size = bn_pass * stride_b;

    let datatype = matmul.matmul_dtype();

    let input_a = kernel.add_tensor_input(matmul.rank(), false, input_a_tensor.datatype());
    let input_b = kernel.add_tensor_input(matmul.rank(), false, input_b_tensor.datatype());
    let output =
        kernel.add_tensor_input(matmul.rank(), true, matmul.post_element_wise.out_datatype());

    // Separate double-buffer arrays (matching egging: 4 separate workgroup arrays)
    let tile_a0 = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        datatype,
        tile_a_buf_size.to_string(),
    );
    let tile_a1 = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        datatype,
        tile_a_buf_size.to_string(),
    );
    let tile_b0 = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        datatype,
        tile_b_buf_size.to_string(),
    );
    let tile_b1 = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        datatype,
        tile_b_buf_size.to_string(),
    );
    // Plain matmuls can reuse tile_a0 for accumulator zero-init and edge-tile fallback
    // writeback, which matches egging's direct-store fast path and trims workgroup usage.
    let scratch_size = num_sgs * mma_size * mma_size;
    let scratch = if use_direct_store {
        tile_a0.clone()
    } else {
        kernel.add_global_array(
            KernelGlobalSpace::Workgroup,
            datatype,
            scratch_size.to_string(),
        )
    };

    assert_eq!(workgroup_shape.x(), wg_threads, "workgroup size mismatch");

    let workgroup_index = kernel.workgroup_index();
    let workgroup_local_index = kernel.workgroup_local_index();

    let k_size = input_a.shape_binding(matmul.rank() - 1);
    let m_size = input_a.shape_binding(matmul.rank() - 2);
    let n_size = input_b.shape_binding(matmul.rank() - 1);

    writeln!(kernel, "let wg_m = {workgroup_index}.x * {block_m}u;").unwrap();
    writeln!(kernel, "let wg_n = {workgroup_index}.y * {block_n}u;").unwrap();
    writeln!(kernel, "var block_batch = {workgroup_index}.z;").unwrap();

    for dim in (0..matmul.rank()).rev().skip(2) {
        let shape = input_a.shape_binding(dim);
        writeln!(kernel, "let block_batch_{dim} = block_batch % {shape};").unwrap();
        writeln!(kernel, "block_batch /= {shape};").unwrap();
    }

    for (name, tensor) in [("a", &input_a), ("b", &input_b), ("c", &output)] {
        write!(
            kernel,
            "let {name}_start_index = {}",
            tensor.offset_binding()
        )
        .unwrap();
        for dim in (0..matmul.rank()).rev().skip(2) {
            let stride = tensor.stride_binding(dim);
            write!(kernel, " + block_batch_{dim}*{stride}").unwrap();
        }
        writeln!(kernel, ";").unwrap();
    }

    writeln!(kernel, "let lid = {workgroup_local_index};").unwrap();
    writeln!(kernel, "let sg_id = lid / {sg_size}u;").unwrap();
    writeln!(kernel, "let sg_m_base = sg_id * {block_m_per_sg}u;").unwrap();

    let a_row_stride = input_a.stride_binding(matmul.rank() - 2);
    let a_col_stride = input_a.stride_binding(matmul.rank() - 1);
    let b_row_stride = input_b.stride_binding(matmul.rank() - 2);
    let b_col_stride = input_b.stride_binding(matmul.rank() - 1);
    let out_row_stride = output.stride_binding(matmul.rank() - 2);
    let out_col_stride = output.stride_binding(matmul.rank() - 1);

    // Number of K-tile pairs (egging processes 2 k-tiles per loop iteration for double buffering)
    writeln!(
        kernel,
        "let num_k_tiles = ({k_size} + {block_k}u - 1u) / {block_k}u;"
    )
    .unwrap();
    writeln!(kernel, "let num_k_tile_pairs = (num_k_tiles + 1u) / 2u;").unwrap();

    // ========= N-pass outer loop =========
    writeln!(
        kernel,
        "for (var n_pass = 0u; n_pass < {n_passes}u; n_pass++) {{"
    )
    .unwrap();
    writeln!(kernel, "  let n_base = wg_n + n_pass * {bn_pass}u;").unwrap();

    // Zero the scratch buffer and load zero accumulators (must be done each n_pass)
    let zero_elems = mma_size * mma_size;
    writeln!(
        kernel,
        "  if (lid < {zero_elems}u) {{ {scratch}[lid] = 0.0; }}"
    )
    .unwrap();
    writeln!(kernel, "  workgroupBarrier();").unwrap();
    for sm in 0..m_subtiles_per_sg {
        for sn in 0..n_subtiles {
            writeln!(
                kernel,
                "  var acc_{sm}_{sn} = coopLoad<coop_mat8x8<f32, C>>(&{scratch}[0u], {mma_size}u);"
            )
            .unwrap();
        }
    }

    // ========= K-loop: process 2 k-tiles per iteration (double buffering) =========
    // Both k-tiles are always loaded and MMA'd (uniform control flow required for coop ops).
    // Out-of-bounds tile loads write 0, so MMA with zero tiles is a no-op for accumulators.
    writeln!(
        kernel,
        "  for (var t_pair = 0u; t_pair < num_k_tile_pairs; t_pair++) {{"
    )
    .unwrap();

    // === First K-tile of the pair (into buffer 0) ===
    writeln!(kernel, "    let k_base_0 = t_pair * 2u * {block_k}u;").unwrap();
    emit_tile_a_load_row_major(
        kernel,
        &tile_a0,
        &input_a,
        "a_start_index",
        &a_row_stride,
        &a_col_stride,
        &m_size,
        &k_size,
        block_m,
        block_k,
        stride_a,
        wg_threads,
        "k_base_0",
    );
    emit_tile_b_load(
        kernel,
        &tile_b0,
        &input_b,
        "b_start_index",
        &b_row_stride,
        &b_col_stride,
        &k_size,
        &n_size,
        block_k,
        bn_pass,
        stride_b,
        wg_threads,
        "k_base_0",
        "n_base",
    );
    writeln!(kernel, "    workgroupBarrier();").unwrap();

    // MMA for first k-tile
    emit_mma_block(
        kernel,
        &tile_a0,
        &tile_b0,
        stride_a,
        stride_b,
        m_subtiles_per_sg,
        n_subtiles,
        k_substeps,
        mma_size,
        "p0",
    );

    writeln!(kernel, "    workgroupBarrier();").unwrap();

    // === Second K-tile of the pair (into buffer 1) ===
    writeln!(
        kernel,
        "    let k_base_1 = (t_pair * 2u + 1u) * {block_k}u;"
    )
    .unwrap();
    emit_tile_a_load_row_major(
        kernel,
        &tile_a1,
        &input_a,
        "a_start_index",
        &a_row_stride,
        &a_col_stride,
        &m_size,
        &k_size,
        block_m,
        block_k,
        stride_a,
        wg_threads,
        "k_base_1",
    );
    emit_tile_b_load(
        kernel,
        &tile_b1,
        &input_b,
        "b_start_index",
        &b_row_stride,
        &b_col_stride,
        &k_size,
        &n_size,
        block_k,
        bn_pass,
        stride_b,
        wg_threads,
        "k_base_1",
        "n_base",
    );
    writeln!(kernel, "    workgroupBarrier();").unwrap();

    // MMA for second k-tile
    emit_mma_block(
        kernel,
        &tile_a1,
        &tile_b1,
        stride_a,
        stride_b,
        m_subtiles_per_sg,
        n_subtiles,
        k_substeps,
        mma_size,
        "p1",
    );

    writeln!(kernel, "    workgroupBarrier();").unwrap();

    writeln!(kernel, "  }}").unwrap(); // end K-pair loop

    // ========= Writeback =========
    // Plain matmuls can store accumulator tiles directly to the output buffer.
    // Anything that needs an epilogue, has a partial tile, or writes to a non-unit
    // output column stride falls back to scratch + scalar replay.
    {
        let sg_scratch_offset = mma_size * mma_size; // 64 elements per subgroup
        for sm in 0..m_subtiles_per_sg {
            for sn in 0..n_subtiles {
                let m_off = sm * mma_size;
                let n_off = sn * mma_size;
                let total_elems = mma_size * mma_size; // 64
                let elems_per_thread = total_elems.div_ceil(sg_size);
                writeln!(
                    kernel,
                    "  let tile_m_base_{sm}_{sn} = wg_m + sg_m_base + {m_off}u;"
                )
                .unwrap();
                writeln!(kernel, "  let tile_n_base_{sm}_{sn} = n_base + {n_off}u;").unwrap();
                writeln!(
                    kernel,
                    "  let tile_in_bounds_{sm}_{sn} = tile_m_base_{sm}_{sn} + {mma_size}u - 1u < {m_size} && tile_n_base_{sm}_{sn} + {mma_size}u - 1u < {n_size};"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "  if ({use_direct_store} && {out_col_stride} == 1u && tile_in_bounds_{sm}_{sn}) {{"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "    coopStore(acc_{sm}_{sn}, &{output}[c_start_index + tile_m_base_{sm}_{sn} * {out_row_stride} + tile_n_base_{sm}_{sn}], {out_row_stride});"
                )
                .unwrap();
                writeln!(kernel, "  }} else {{").unwrap();
                writeln!(
                    kernel,
                    "    coopStore(acc_{sm}_{sn}, &{scratch}[sg_id * {sg_scratch_offset}u], {mma_size}u);"
                )
                .unwrap();
                writeln!(kernel, "    workgroupBarrier();").unwrap();
                writeln!(kernel, "    {{ // writeback acc_{sm}_{sn}").unwrap();
                writeln!(kernel, "      let sg_local = lid % {sg_size}u;").unwrap();
                for li in 0..elems_per_thread {
                    let base = li * sg_size;
                    writeln!(kernel, "      if (sg_local + {base}u < {total_elems}u) {{").unwrap();
                    writeln!(kernel, "        let idx = sg_local + {base}u;").unwrap();
                    // coopStore writes the 8x8 tile row-major when we spill to scratch,
                    // so replay edge tiles with N on the inner axis.
                    let local_m = format!("(idx / {mma_size}u)");
                    let local_n = format!("(idx % {mma_size}u)");
                    writeln!(
                        kernel,
                        "        let global_m = tile_m_base_{sm}_{sn} + {local_m};"
                    )
                    .unwrap();
                    writeln!(
                        kernel,
                        "        let global_n = tile_n_base_{sm}_{sn} + {local_n};"
                    )
                    .unwrap();
                    writeln!(
                        kernel,
                        "        if (global_m < {m_size} && global_n < {n_size}) {{"
                    )
                    .unwrap();
                    writeln!(
                        kernel,
                        "          let matmul_value = {scratch}[sg_id * {sg_scratch_offset}u + idx];"
                    )
                    .unwrap();
                    write!(
                        kernel,
                        "          {output}[c_start_index + global_m * {out_row_stride} + global_n * {out_col_stride}] = "
                    )
                    .unwrap();
                    let post_element_wise_functions = post_element_wise_functions
                        .get_or_init(|| matmul.post_element_wise.add_functions(kernel));
                    let result = post_element_wise_functions
                        .iter()
                        .fold("matmul_value".to_string(), |acc, f| f.call(vec![acc]));
                    writeln!(kernel, "{result};").unwrap();
                    writeln!(kernel, "        }}").unwrap();
                    writeln!(kernel, "      }}").unwrap();
                }
                writeln!(kernel, "    }}").unwrap();
                writeln!(kernel, "    workgroupBarrier();").unwrap();
                writeln!(kernel, "  }}").unwrap();
            }
        }
    }

    writeln!(kernel, "}}").unwrap(); // end N-pass loop
}

/// Emit MMA inner loop: process all k-substeps within a k-tile
fn emit_mma_block(
    kernel: &mut GenericKernel,
    tile_a: &crate::mir::globals::KernelGlobal,
    tile_b: &crate::mir::globals::KernelGlobal,
    stride_a: u32,
    stride_b: u32,
    m_subtiles_per_sg: u32,
    n_subtiles: u32,
    k_substeps: u32,
    mma_size: u32,
    prefix: &str,
) {
    for k_sub in 0..k_substeps {
        // Load A subtiles from row-major MxK shared storage.
        for sm in 0..m_subtiles_per_sg {
            let a_offset_m = sm * mma_size;
            let a_offset_k = k_sub * mma_size;
            writeln!(kernel, "    let {prefix}_a_k{k_sub}_m{sm} = coopLoad<coop_mat8x8<f32, A>>(&{tile_a}[(sg_m_base + {a_offset_m}u) * {stride_a}u + {a_offset_k}u], {stride_a}u);").unwrap();
        }
        // Load B subtiles from row-major KxN shared storage.
        for sn in 0..n_subtiles {
            let b_offset_k = k_sub * mma_size;
            let b_offset_n = sn * mma_size;
            writeln!(kernel, "    let {prefix}_b_k{k_sub}_n{sn} = coopLoad<coop_mat8x8<f32, B>>(&{tile_b}[{b_offset_k}u * {stride_b}u + {b_offset_n}u], {stride_b}u);").unwrap();
        }
        // Multiply-accumulate
        for sm in 0..m_subtiles_per_sg {
            for sn in 0..n_subtiles {
                writeln!(kernel, "    acc_{sm}_{sn} = coopMultiplyAdd({prefix}_a_k{k_sub}_m{sm}, {prefix}_b_k{k_sub}_n{sn}, acc_{sm}_{sn});").unwrap();
            }
        }
    }
}

/// Emit tile A load from global → shared memory.
/// Row-major layout: tile_a[m * stride_a + k] where stride_a = block_k + 1
/// Vec4 loads: each thread reads 4 consecutive K values from global memory
/// Thread assignment: lid % block_m → m_in_tile, lid / block_m → k_group
fn emit_tile_a_load_row_major(
    kernel: &mut GenericKernel,
    tile: &crate::mir::globals::KernelGlobal,
    input: &TensorInput,
    start_index: &str,
    row_stride: &dyn std::fmt::Display, // M stride in global memory
    col_stride: &dyn std::fmt::Display, // K stride in global memory
    m_size: &dyn std::fmt::Display,
    k_size: &dyn std::fmt::Display,
    block_m: u32,
    block_k: u32,
    tile_stride: u32, // = block_k + 1
    wg_threads: u32,
    k_base: &str,
) {
    // Each thread loads 4 K values (vec4). Total coverage: wg_threads threads × 4 K values each
    // threads_per_m_row = wg_threads, but we tile as: m = lid % block_m, k_group = lid / block_m
    // With 256 threads and block_m=128: k_group ∈ {0, 1} for first pass, need block_k/4 = 4 k-groups total
    let k_groups_per_pass = wg_threads / block_m;
    let total_k_groups = block_k / 4; // 4 K values per vec4 load
    let num_passes = total_k_groups.div_ceil(k_groups_per_pass);

    writeln!(
        kernel,
        "    {{ // Load tile A [m][k] (row-major, vec4 in K)"
    )
    .unwrap();
    for pass in 0..num_passes {
        let thread_offset = pass * wg_threads;
        writeln!(
            kernel,
            "      if (lid + {thread_offset}u < {block_m}u * ({block_k}u / 4u)) {{"
        )
        .unwrap();
        writeln!(
            kernel,
            "        let a_m = (lid + {thread_offset}u) % {block_m}u;"
        )
        .unwrap();
        writeln!(
            kernel,
            "        let a_k_group = (lid + {thread_offset}u) / {block_m}u;"
        )
        .unwrap();
        writeln!(kernel, "        let global_m = wg_m + a_m;").unwrap();
        writeln!(kernel, "        let global_k = {k_base} + a_k_group * 4u;").unwrap();
        // Read 4 consecutive K values from global memory, then scatter them into the
        // row-major shared tile. Keeping the loads grouped mirrors egging's fast path.
        writeln!(kernel, "        let a_addr = {start_index} + min(global_m, {m_size} - 1u) * {row_stride} + min(global_k, {k_size} - 1u) * {col_stride};").unwrap();
        writeln!(kernel, "        let m_in_bounds = global_m < {m_size};").unwrap();
        writeln!(
            kernel,
            "        let a_values = vec4<f32>({input}[a_addr + 0u * {col_stride}], {input}[a_addr + 1u * {col_stride}], {input}[a_addr + 2u * {col_stride}], {input}[a_addr + 3u * {col_stride}]);"
        )
        .unwrap();
        writeln!(kernel, "        {tile}[a_m * {tile_stride}u + (a_k_group * 4u + 0u)] = select(0.0, a_values.x, m_in_bounds && (global_k + 0u) < {k_size});").unwrap();
        writeln!(kernel, "        {tile}[a_m * {tile_stride}u + (a_k_group * 4u + 1u)] = select(0.0, a_values.y, m_in_bounds && (global_k + 1u) < {k_size});").unwrap();
        writeln!(kernel, "        {tile}[a_m * {tile_stride}u + (a_k_group * 4u + 2u)] = select(0.0, a_values.z, m_in_bounds && (global_k + 2u) < {k_size});").unwrap();
        writeln!(kernel, "        {tile}[a_m * {tile_stride}u + (a_k_group * 4u + 3u)] = select(0.0, a_values.w, m_in_bounds && (global_k + 3u) < {k_size});").unwrap();
        writeln!(kernel, "      }}").unwrap();
    }
    writeln!(kernel, "    }}").unwrap();
}

/// Emit tile B load from global → shared memory.
/// Row-major layout: tile_b[k * stride_b + n] where stride_b = block_k + 1
fn emit_tile_b_load(
    kernel: &mut GenericKernel,
    tile: &crate::mir::globals::KernelGlobal,
    input: &TensorInput,
    start_index: &str,
    row_stride: &dyn std::fmt::Display,
    col_stride: &dyn std::fmt::Display,
    k_size: &dyn std::fmt::Display,
    n_size: &dyn std::fmt::Display,
    block_k: u32,
    bn_pass: u32,
    tile_stride: u32,
    wg_threads: u32,
    k_base: &str,
    n_base: &str,
) {
    if bn_pass % 4 == 0 {
        let total_vec4_groups = block_k * (bn_pass / 4);
        let vec4_groups_per_thread = total_vec4_groups.div_ceil(wg_threads);

        writeln!(
            kernel,
            "    {{ // Load tile B [k][n] (row-major, vec4 in N)"
        )
        .unwrap();
        for li in 0..vec4_groups_per_thread {
            let base = li * wg_threads;
            writeln!(kernel, "      if (lid + {base}u < {total_vec4_groups}u) {{").unwrap();
            writeln!(kernel, "        let idx_b = lid + {base}u;").unwrap();
            writeln!(kernel, "        let tile_k = idx_b % {block_k}u;").unwrap();
            writeln!(kernel, "        let tile_n_group = idx_b / {block_k}u;").unwrap();
            writeln!(kernel, "        let global_k = {k_base} + tile_k;").unwrap();
            writeln!(
                kernel,
                "        let global_n = {n_base} + tile_n_group * 4u;"
            )
            .unwrap();
            writeln!(kernel, "        let b_addr = {start_index} + min(global_k, {k_size} - 1u) * {row_stride} + min(global_n, {n_size} - 1u) * {col_stride};").unwrap();
            writeln!(kernel, "        let k_in_bounds = global_k < {k_size};").unwrap();
            writeln!(
                kernel,
                "        let b_values = vec4<f32>({input}[b_addr + 0u * {col_stride}], {input}[b_addr + 1u * {col_stride}], {input}[b_addr + 2u * {col_stride}], {input}[b_addr + 3u * {col_stride}]);"
            )
            .unwrap();
            writeln!(kernel, "        {tile}[tile_k * {tile_stride}u + (tile_n_group * 4u + 0u)] = select(0.0, b_values.x, k_in_bounds && (global_n + 0u) < {n_size});").unwrap();
            writeln!(kernel, "        {tile}[tile_k * {tile_stride}u + (tile_n_group * 4u + 1u)] = select(0.0, b_values.y, k_in_bounds && (global_n + 1u) < {n_size});").unwrap();
            writeln!(kernel, "        {tile}[tile_k * {tile_stride}u + (tile_n_group * 4u + 2u)] = select(0.0, b_values.z, k_in_bounds && (global_n + 2u) < {n_size});").unwrap();
            writeln!(kernel, "        {tile}[tile_k * {tile_stride}u + (tile_n_group * 4u + 3u)] = select(0.0, b_values.w, k_in_bounds && (global_n + 3u) < {n_size});").unwrap();
            writeln!(kernel, "      }}").unwrap();
        }
    } else {
        let total_elems = block_k * bn_pass;
        let elems_per_thread = total_elems.div_ceil(wg_threads);

        writeln!(kernel, "    {{ // Load tile B [k][n] (row-major)").unwrap();
        for li in 0..elems_per_thread {
            let base = li * wg_threads;
            writeln!(kernel, "      if (lid + {base}u < {total_elems}u) {{").unwrap();
            writeln!(kernel, "        let idx_b = lid + {base}u;").unwrap();
            writeln!(kernel, "        let tile_k = idx_b % {block_k}u;").unwrap();
            writeln!(kernel, "        let tile_n = idx_b / {block_k}u;").unwrap();
            writeln!(kernel, "        let global_k = {k_base} + tile_k;").unwrap();
            writeln!(kernel, "        let global_n = {n_base} + tile_n;").unwrap();
            writeln!(
                kernel,
                "        let in_bounds_b = global_k < {k_size} && global_n < {n_size};"
            )
            .unwrap();
            writeln!(kernel, "        let b_val = select(0.0, {input}[{start_index} + min(global_k, {k_size} - 1u) * {row_stride} + min(global_n, {n_size} - 1u) * {col_stride}], in_bounds_b);").unwrap();
            writeln!(
                kernel,
                "        {tile}[tile_k * {tile_stride}u + tile_n] = b_val;"
            )
            .unwrap();
            writeln!(kernel, "      }}").unwrap();
        }
    }
    writeln!(kernel, "    }}").unwrap();
}
