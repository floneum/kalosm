use crate::{
    Layout, Tensor, compute_graph::NodeIndex,
    nary_wise::UnaryFunctionChain, tensor::DataTypeEnum,
};

mod cost;
mod kernel;
pub mod sgemm;
mod sgemm_params;
pub mod sgemv;
mod sgemv_params;
mod variants;

pub(crate) use kernel::{MatmulMergeKey, build_merged_matmul_kernel};
pub(crate) use variants::CoopTile;

#[derive(Debug, Clone, PartialEq, Hash)]
pub(crate) enum MatMulParams {
    Vector(sgemv::SgemvParams),
    MatMul(sgemm::SgemmParams),
    /// The cooperative-matrix family. Neither geometry nor the matrix
    /// kind is a parameter: the scored tile selection derives geometry per
    /// kernel build and the kind follows from the datatype, so dispatch,
    /// allocation, and kernel agree by construction.
    CoopMatMul,
}

/// An affine relayout between an operand's dims and its node's logical
/// space: conv's sliding windows. The kernels concretize it lazily — the
/// coop path composes it with the node's runtime buffer layout, the generic
/// reduce substitutes it into the load coordinates.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OperandBaseMap {
    pub(crate) layout: Layout,
    pub(crate) base_shape: Box<[usize]>,
}

impl std::hash::Hash for OperandBaseMap {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.layout.offset().hash(state);
        self.layout.shape().hash(state);
        self.layout.strides().hash(state);
        self.base_shape.hash(state);
    }
}

/// One matmul operand's dim grouping: the operand's dims split into
/// `batch_dims` leading batch dims, `row_dims` row dims, and column dims
/// for the rest. The row and column groups flatten to the two matrix axes.
/// A plain `[batch.., rows, cols]` operand has one dim per group; conv's
/// im2col operand keeps the windowed view's dims (mapped onto the node by
/// `base_map`), and the kernels divmod the flat matrix coordinates back
/// apart per load.
#[derive(Debug, Clone, PartialEq, Hash)]
pub(crate) struct MatrixOperand {
    pub(crate) shape: Box<[usize]>,
    pub(crate) batch_dims: usize,
    pub(crate) row_dims: usize,
    pub(crate) base_map: Option<OperandBaseMap>,
}

impl MatrixOperand {
    pub(crate) fn plain(shape: &[usize]) -> Self {
        assert!(shape.len() >= 2, "matrix operands are at least rank 2");
        Self {
            shape: shape.into(),
            batch_dims: shape.len() - 2,
            row_dims: 1,
            base_map: None,
        }
    }

    /// One dim per group, reading the node's own dims directly: the
    /// operand's shape is the logical matmul shape.
    pub(crate) fn is_plain(&self) -> bool {
        self.row_dims == 1 && self.batch_dims + 2 == self.shape.len() && self.base_map.is_none()
    }

    pub(crate) fn batch_shape(&self) -> &[usize] {
        &self.shape[..self.batch_dims]
    }

    pub(crate) fn row_shape(&self) -> &[usize] {
        &self.shape[self.batch_dims..self.batch_dims + self.row_dims]
    }

    pub(crate) fn col_shape(&self) -> &[usize] {
        &self.shape[self.batch_dims + self.row_dims..]
    }

    pub(crate) fn rows(&self) -> usize {
        self.row_shape().iter().product()
    }

    pub(crate) fn cols(&self) -> usize {
        self.col_shape().iter().product()
    }

    /// Leading dim count flattening to the 2-D matrix row axis (batch
    /// included): the split for [`flatten_matrix_layout_split`].
    ///
    /// [`flatten_matrix_layout_split`]: crate::mir::tile_direct::flatten_matrix_layout_split
    pub(crate) fn split(&self) -> usize {
        self.batch_dims + self.row_dims
    }

    /// Index expressions reading the operand at the contraction coordinates:
    /// batch dims index through, the flat row/column coordinates decompose
    /// over the row/column groups (the identity for single-dim groups, so
    /// plain operands load with bare `DimIndex` coordinates), and the
    /// operand coordinates map through `base_map` to the node's own dims.
    pub(crate) fn index_expressions(
        &self,
        row_dim: usize,
        col_dim: usize,
    ) -> Vec<crate::nary_wise::NaryExpr> {
        use crate::nary_wise::NaryExpr;
        let mut indices: Vec<NaryExpr> = (0..self.batch_dims).map(NaryExpr::DimIndex).collect();
        indices.extend(
            crate::view::row_major_indices_from_flat(NaryExpr::DimIndex(row_dim), self.row_shape())
                .expect("operand dims fit u32"),
        );
        indices.extend(
            crate::view::row_major_indices_from_flat(NaryExpr::DimIndex(col_dim), self.col_shape())
                .expect("operand dims fit u32"),
        );
        match &self.base_map {
            None => indices,
            Some(map) => crate::view::affine_dim_indices(&map.layout, &map.base_shape)
                .expect("validated when the resolver attached the base map")
                .iter()
                .map(|index| index.to_expr(&indices))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MatMulOperation {
    pub(crate) datatype: DataTypeEnum,
    pub(crate) first: NodeIndex,
    pub(crate) second: NodeIndex,
    pub(crate) a: MatrixOperand,
    pub(crate) b: MatrixOperand,
    pub(crate) out_shape: Box<[usize]>,
    pub(crate) pre_element_wise: [UnaryFunctionChain; 2],
    pub(crate) post_element_wise: UnaryFunctionChain,
    pub(crate) parameters: MatMulParams,
}

impl Tensor {
    /// Matrix multiply, expressed as its composed form: a broadcast multiply
    /// over the `[batch.., M, N, K]` index space summed along `K`. The
    /// resolver recognizes the canonical cluster and routes it to the
    /// specialized matmul kernels; anything that composes differently lowers
    /// through the generic elementwise + reduce path.
    pub fn mat_mul(&self, other: &Self) -> Self {
        use crate::nary_wise::{ElementwiseOperation, NaryExpr};

        assert_eq!(self.datatype(), other.datatype());
        let a_shape = self.shape();
        let b_shape = other.shape();
        let rank = a_shape.len();
        assert_eq!(
            rank,
            b_shape.len(),
            "mat_mul requires equal ranks: {a_shape:?} x {b_shape:?}"
        );
        assert!(rank >= 2, "mat_mul requires rank >= 2: {a_shape:?}");
        let batch = rank - 2;
        assert_eq!(
            a_shape[..batch],
            b_shape[..batch],
            "mat_mul batch dimensions must match: {a_shape:?} x {b_shape:?}"
        );
        assert_eq!(
            a_shape[rank - 1],
            b_shape[rank - 2],
            "mat_mul contraction dimensions must match: {a_shape:?} x {b_shape:?}"
        );

        let (m, k, n) = (a_shape[batch], a_shape[batch + 1], b_shape[batch + 1]);
        let mut index_space: Vec<usize> = a_shape[..batch].to_vec();
        index_space.extend([m, n, k]);
        let (m_dim, n_dim, k_dim) = (batch, batch + 1, batch + 2);

        let a_indices: Vec<NaryExpr> = (0..batch)
            .chain([m_dim, k_dim])
            .map(NaryExpr::DimIndex)
            .collect();
        let b_indices: Vec<NaryExpr> = (0..batch)
            .chain([k_dim, n_dim])
            .map(NaryExpr::DimIndex)
            .collect();

        let datatype = self.datatype();
        let product = Tensor::from_parts(self.data.nary(ElementwiseOperation {
            inputs: vec![self.key(), other.key()],
            expression: NaryExpr::mul(
                NaryExpr::indexed_input(0, a_indices),
                NaryExpr::indexed_input(1, b_indices),
                datatype,
            ),
            shape: index_space.into(),
            output_datatype: datatype,
        }));
        product.sum(k_dim)
    }
}

#[cfg(test)]
mod selection_tests {
    use super::variants::{
        CoopTile, DenseMatmulCtx, DenseMatmulVariant, dense_matmul_selector, select_coop_kind,
    };
    use crate::kernel_selection::{
        CooperativeMatrixCaps, CooperativeMatrixKind, DeterministicShapeRng, KernelDeviceCaps,
        KernelShape,
    };

    fn caps(coop: bool) -> KernelDeviceCaps {
        KernelDeviceCaps {
            subgroups_supported: coop,
            cooperative_matrix: if coop {
                CooperativeMatrixCaps::test_dense_8x8()
            } else {
                CooperativeMatrixCaps::default()
            },
            ..KernelDeviceCaps::test_caps()
        }
    }

    #[test]
    fn dense_selector_generates_each_variant() {
        let selector = dense_matmul_selector();
        let cases = [
            (
                DenseMatmulVariant::Coop,
                DenseMatmulCtx {
                    coop_kinds: &[CooperativeMatrixKind::F32F32M8N8K8],
                },
                caps(true),
            ),
            (
                DenseMatmulVariant::Vector,
                DenseMatmulCtx { coop_kinds: &[] },
                caps(false),
            ),
            (
                DenseMatmulVariant::MatMul,
                DenseMatmulCtx { coop_kinds: &[] },
                caps(false),
            ),
        ];
        let mut rng = DeterministicShapeRng::default();

        for (variant, ctx, caps) in cases {
            let shape = selector
                .generate_for(variant, &ctx, caps, &mut rng)
                .expect("variant should generate");
            assert_eq!(selector.select(shape, &ctx, caps), Some(variant));
        }
    }

    #[test]
    fn dense_selector_gates_coop_by_scalar_property() {
        let selector = dense_matmul_selector();
        let shape = KernelShape::new([128, 256, 128]);
        let f16_ctx = DenseMatmulCtx {
            coop_kinds: &[CooperativeMatrixKind::F16F16M8N8K8],
        };
        let f32_ctx = DenseMatmulCtx {
            coop_kinds: &[CooperativeMatrixKind::F32F32M8N8K8],
        };

        assert_eq!(
            selector.select(shape, &f16_ctx, caps(true)),
            Some(DenseMatmulVariant::Coop)
        );
        assert_eq!(
            selector.select(shape, &f32_ctx, caps(true)),
            Some(DenseMatmulVariant::Coop)
        );
        assert_eq!(
            select_coop_kind(caps(true), f16_ctx.coop_kinds),
            CooperativeMatrixKind::F16F16M8N8K8
        );

        let only_f32_property = KernelDeviceCaps {
            cooperative_matrix: CooperativeMatrixCaps::from_properties(
                wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
                &[wgpu::CooperativeMatrixProperties {
                    m_size: 8,
                    n_size: 8,
                    k_size: 8,
                    ab_type: wgpu::CooperativeScalarType::F32,
                    cr_type: wgpu::CooperativeScalarType::F32,
                    saturating_accumulation: false,
                }],
            ),
            ..caps(true)
        };
        assert_eq!(
            selector.select(shape, &f16_ctx, only_f32_property),
            Some(DenseMatmulVariant::MatMul)
        );
        assert_eq!(
            selector.select(shape, &f32_ctx, only_f32_property),
            Some(DenseMatmulVariant::Coop)
        );

        let only_f16_property = KernelDeviceCaps {
            cooperative_matrix: CooperativeMatrixCaps::from_properties(
                wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16,
                &[wgpu::CooperativeMatrixProperties {
                    m_size: 8,
                    n_size: 8,
                    k_size: 8,
                    ab_type: wgpu::CooperativeScalarType::F16,
                    cr_type: wgpu::CooperativeScalarType::F16,
                    saturating_accumulation: false,
                }],
            ),
            ..caps(true)
        };
        assert_eq!(
            selector.select(shape, &f16_ctx, only_f16_property),
            Some(DenseMatmulVariant::Coop)
        );
        assert_eq!(
            selector.select(shape, &f32_ctx, only_f16_property),
            Some(DenseMatmulVariant::MatMul)
        );
        assert_eq!(
            select_coop_kind(only_f16_property, f16_ctx.coop_kinds),
            CooperativeMatrixKind::F16F16M8N8K8
        );

        let only_mixed_f16_property = KernelDeviceCaps {
            cooperative_matrix: CooperativeMatrixCaps::from_properties(
                wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16,
                &[wgpu::CooperativeMatrixProperties {
                    m_size: 8,
                    n_size: 8,
                    k_size: 8,
                    ab_type: wgpu::CooperativeScalarType::F16,
                    cr_type: wgpu::CooperativeScalarType::F32,
                    saturating_accumulation: false,
                }],
            ),
            ..caps(true)
        };
        assert_eq!(
            selector.select(shape, &f16_ctx, only_mixed_f16_property),
            Some(DenseMatmulVariant::MatMul)
        );
        assert_eq!(
            selector.select(shape, &f32_ctx, only_mixed_f16_property),
            Some(DenseMatmulVariant::MatMul)
        );
    }

    fn select_with_lanes(m: u32, k: u32, n: u32, max_lanes: u32) -> Option<CoopTile> {
        let policy = crate::occupancy::DispatchPolicy::from_parts(64 << 10, 32, max_lanes, 8 << 20);
        CoopTile::select(m, k, n, &policy, 32)
    }

    /// Every selection must be legal and within the padding bound for every
    /// (shape, caps) combination — the properties the scorer guarantees by
    /// construction, checked over a deterministic sweep.
    #[test]
    fn scored_selection_properties() {
        let mut lcg = 0x5eed_1234u64;
        let mut next = |range: u32| {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((lcg >> 33) as u32) % range + 1
        };
        for _ in 0..2000 {
            let (m, k, n) = (next(9000), next(5000), next(9000));
            let max_lanes = [128u32, 256, 512, 1024][(next(4) - 1) as usize];
            let Some(tile) = select_with_lanes(m, k, n, max_lanes) else {
                continue;
            };
            let entry = fusor_tile_ir_kernels::coop_tile_entries()
                .iter()
                .find(|entry| entry.tile.bm == tile.bm && entry.tile.bn == tile.bn)
                .expect("selected tile must exist in the kernel table");
            let threads = entry.row_groups * entry.col_groups * 32;
            assert!(
                threads <= max_lanes,
                "m={m} k={k} n={n} lanes={max_lanes}: illegal tile {tile:?}"
            );
            let padded = u64::from(m.div_ceil(tile.bm))
                * u64::from(tile.bm)
                * u64::from(n.div_ceil(tile.bn))
                * u64::from(tile.bn);
            assert!(
                padded * 4 <= u64::from(m) * u64::from(n) * 5,
                "m={m} k={k} n={n}: padding bound violated by {tile:?}"
            );
        }
    }

    /// The sgemv bucket table and the sgemm regression tree are measured
    /// policies for the non-cooperative fallback families; every cell they
    /// can produce must still be structurally legal (kernel divisibility,
    /// workgroup lane bounds, shared-memory budget) for every shape.
    #[test]
    fn fallback_family_params_are_legal_everywhere() {
        let mut lcg = 0x0fa1_1bac_c5u64;
        let mut next = |range: u32| {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((lcg >> 33) as u32) % range + 1
        };
        for _ in 0..4000 {
            let (m, k, n) = (
                next(20_000) as usize,
                next(20_000) as usize,
                next(20_000) as usize,
            );

            let gemm = crate::matmul::sgemm_params::gemm_parameters(m, n, k);
            let (bm, bn, bk) = (
                gemm.block_m_size(),
                gemm.block_n_size(),
                gemm.block_k_size(),
            );
            let (tm, tn) = (gemm.thread_m_size(), gemm.thread_n_size());
            assert!(
                bm.is_multiple_of(tm) && bn.is_multiple_of(tn),
                "m={m} n={n} k={k}: thread tile must divide the block tile ({gemm:?})"
            );
            let lanes = (bm * bn) / (tm * tn);
            assert!(
                (32..=1024).contains(&lanes),
                "m={m} n={n} k={k}: workgroup lanes {lanes} out of range ({gemm:?})"
            );
            // A and B staging tiles, doubled when double-buffered, must fit
            // Apple's 32 KB workgroup-memory floor.
            let buffers = if gemm.double_buffer() { 2 } else { 1 };
            let smem_bytes = u64::from((bm + bn) * bk) * 4 * buffers;
            assert!(
                smem_bytes <= 32 * 1024,
                "m={m} n={n} k={k}: {smem_bytes}B of workgroup memory ({gemm:?})"
            );

            let gemv = crate::matmul::sgemv_params::gemv_parameters(m, n, k);
            assert!(
                gemv.chunk_size() >= 1
                    && matches!(gemv.vector_size(), 1 | 2 | 4)
                    && (1..=32).contains(&gemv.subgroups_per_workgroup()),
                "m={m} n={n} k={k}: illegal gemv params ({gemv:?})"
            );
        }
    }
}

#[cfg(test)]
mod split_k_tests {
    //! GPU gates for automatic split-K selection and aligned-span codegen.

    use crate::{Device, Tensor};

    fn check_dense_split_k(m: usize, k: usize, n: usize) {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let values = |len: usize, freq: f32| -> Vec<f32> {
                (0..len).map(|i| ((i as f32) * freq).sin()).collect()
            };
            let a_data = values(m * k, 0.13);
            let b_data = values(k * n, 0.07);
            let a = Tensor::from_slice(&device, [m, k], &a_data);
            let b = Tensor::from_slice(&device, [k, n], &b_data);
            let out = a.mat_mul(&b);
            let actual = out.as_slice::<2, f32>().await.unwrap();
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0.0f64;
                    for ki in 0..k {
                        acc += a_data[mi * k + ki] as f64 * b_data[ki * n + ni] as f64;
                    }
                    let want = acc as f32;
                    let got = actual[[mi, ni]];
                    assert!(
                        (got - want).abs() < 2e-3 + want.abs() * 1e-3,
                        "m={m} k={k} n={n} [{mi}, {ni}]: got {got}, expected {want}"
                    );
                }
            }
        });
    }

    // The 64×2048×64 weight-gradient shape: K-tiles divide the fan-out, so
    // the spans partition K exactly and the K bounds are elided (the vec4
    // staging fast path).
    #[test]
    fn dense_split_k_elided_bounds() {
        check_dense_split_k(64, 2048, 64);
    }

    // Ragged K (1000): no useful divisor alignment, the last span overruns
    // the logical K extent and the bounds stay live under the dense flag.
    #[test]
    fn dense_split_k_ragged_k() {
        check_dense_split_k(64, 1000, 64);
    }

    // Barely past the split gate (k = 520): short trailing spans idle.
    #[test]
    fn dense_split_k_short_spans() {
        check_dense_split_k(64, 520, 64);
    }
}
