use crate::{
    Device, Layout, Tensor, compute_graph::NodeIndex, kernel_selection::CooperativeMatrixKind,
    nary_wise::UnaryFunctionChain, tensor::DataTypeEnum,
};

pub mod coop_gemm;
mod kernel;
pub mod sgemm;
mod sgemm_params;
pub mod sgemv;
mod sgemv_params;
mod variants;

pub(crate) use kernel::{MatmulMergeKey, build_merged_matmul_kernel};
pub(crate) use variants::CoopTile;
use variants::select_dense_matmul_params;

pub fn get_optimal_params(m: usize, n: usize, k: usize, device: &Device) -> MatMulParams {
    select_dense_matmul_params(m, n, k, device, &[CooperativeMatrixKind::F32F32M8N8K8])
}

#[derive(Debug, Clone, Hash)]
pub enum MatMulParams {
    Vector(sgemv::SgemvParams),
    MatMul(sgemm::SgemmParams),
    CoopMatMul(coop_gemm::CoopGemmParams),
}

/// An affine relayout between an operand's dims and its node's logical
/// space: conv's sliding windows. The kernels concretize it lazily — the
/// coop path composes it with the node's runtime buffer layout, the generic
/// reduce substitutes it into the load coordinates.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, Hash)]
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

#[derive(Debug, Clone)]
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
    /// Dense-large-graph kernel tuning: wider split-K fan-out with
    /// divisor-aligned spans and elided K bounds. Set only by
    /// `optimize_large_graph`'s dense branch, so quantized-pipeline graphs
    /// (decode f32 attention matmuls included) and small (conformance-
    /// golden) graphs keep the exact committed split-K behavior. Hashed
    /// into the split-K kernel cache key.
    pub(crate) dense_codegen: bool,
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

    /// Matrix multiply with explicit kernel parameters: a tuning/benchmark
    /// API. The parameters cannot round-trip through the composed graph, so
    /// the operation builds directly against materialized inputs and
    /// executes eagerly, returning a fresh leaf tensor.
    pub fn mat_mul_with_parameters(&self, other: &Self, parameters: MatMulParams) -> Self {
        assert_eq!(self.datatype(), other.datatype());
        self.data.materialize();
        other.data.materialize();
        let operation = MatMulOperation::new(
            self.datatype(),
            self.key(),
            other.key(),
            self.shape(),
            other.shape(),
            Some(parameters),
            self.device(),
        );
        let output = self
            .device()
            .compute_graph()
            .execute_eager(&operation)
            .unwrap_or_else(|| {
                panic!(
                    "mat_mul_with_parameters could not build a kernel for the requested parameters"
                )
            });
        Tensor::from(output)
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

    #[test]
    fn direct_tile_coop_selector_prefers_largest_supported_tile() {
        let select =
            |m, k, n, max_workgroup_size_x| CoopTile::select(m, k, n, max_workgroup_size_x, 32);
        // 4096³ (square) hits Tile128x512 — it has fewer barriers than
        // Tile256x256 because it's double-buffered.
        assert_eq!(
            select(4096, 4096, 4096, 512),
            Some(CoopTile::new(128, 512, 16))
        );
        // Shapes where N is divisible by 256 but not 512 — with enough
        // tiles — fall to Tile256x256 single-buffer.
        assert_eq!(
            select(8192, 1024, 4352, 512),
            Some(CoopTile::new(256, 256, 16))
        );
        // N=512 doesn't divide 256 on the M side... actually wait, 4096 % 256 == 0.
        // For shapes where N is divisible by 512 but M isn't by 256, fall to
        // Tile128x512.
        assert_eq!(
            select(384, 1024, 1024, 512),
            Some(CoopTile::new(128, 64, 16))
        );
        // 1024³ doesn't have enough tiles for Tile128x512 OR Tile128x256;
        // falls back to Tile128x64 for better parallelism.
        assert_eq!(
            select(1024, 1024, 1024, 512),
            Some(CoopTile::new(128, 64, 16))
        );
        // 8192x256 has tiles_for(128, 256) = 64*1 = 64 — below the threshold,
        // so it falls to Tile128x64.
        assert_eq!(
            select(8192, 1024, 256, 256),
            Some(CoopTile::new(128, 64, 16))
        );
        // M=4096, N=1024 gives tiles_for(128, 256) = 32*4 = 128. Below 256.
        // Falls to Tile128x64.
        assert_eq!(
            select(4096, 1024, 1024, 256),
            Some(CoopTile::new(128, 64, 16))
        );
        // M=8192, N=512 gives tiles_for(128, 256) = 64*2 = 128 (still <256),
        // so falls to Tile128x64. To hit Tile128x256 we need a wider shape:
        // 8192x1024 → 64*4 = 256 ✓.
        assert_eq!(
            select(8192, 1024, 1024, 256),
            Some(CoopTile::new(128, 256, 16))
        );
        // N=128 doesn't divide 256 so Tile128x256/Tile128x512 are out; falls
        // back to Tile128x64.
        assert_eq!(
            select(1024, 1024, 128, 256),
            Some(CoopTile::new(128, 64, 16))
        );
        assert_eq!(
            select(1024, 1024, 1024, 128),
            Some(CoopTile::new(64, 64, 16))
        );
        // M=1000 divides nothing: the masked-edge fallback picks the tile
        // with the least padded work (all candidates pad M to 1024; the
        // preference order breaks the tie toward the biggest tile).
        assert_eq!(
            select(1000, 1024, 1024, 512),
            Some(CoopTile::new(128, 64, 16))
        );
        // N=16 shapes route to the small-side (64, 16) tile (M pads
        // 1000 → 1024, well inside the bound).
        assert_eq!(select(1000, 1024, 16, 512), Some(CoopTile::new(64, 16, 16)));
        // True gemv shapes (one axis tiny beyond what padding allows) stay
        // on the generic path.
        assert_eq!(select(1000, 1024, 1, 512), None);
        assert_eq!(select(1, 1024, 1000, 512), None);
    }

    #[test]
    fn small_side_tiles_select_only_after_primary_declines() {
        let select =
            |m, k, n, max_workgroup_size_x| CoopTile::select(m, k, n, max_workgroup_size_x, 32);
        // Aligned small-side shapes: the attention head_dim family.
        assert_eq!(select(64, 64, 16, 512), Some(CoopTile::new(64, 16, 16)));
        assert_eq!(select(16, 64, 64, 512), Some(CoopTile::new(16, 64, 16)));
        assert_eq!(select(128, 64, 16, 512), Some(CoopTile::new(64, 16, 16)));
        // Masked-edge small-side shapes: the vocab-65 lm_head family.
        assert_eq!(select(2048, 64, 65, 512), Some(CoopTile::new(64, 16, 16)));
        assert_eq!(select(64, 2048, 65, 512), Some(CoopTile::new(64, 16, 16)));
        assert_eq!(select(65, 2048, 64, 512), Some(CoopTile::new(16, 64, 16)));
        // Padding above a quarter of the output still declines: both-tiny
        // contractions stay generic.
        assert_eq!(select(16, 64, 16, 512), None);
        assert_eq!(select(17, 64, 17, 512), None);
        // Shapes the primary ladder already served keep their exact tile.
        assert_eq!(select(64, 64, 64, 512), Some(CoopTile::new(64, 64, 16)));
        assert_eq!(select(64, 512, 112, 512), Some(CoopTile::new(64, 128, 16)));
        assert_eq!(select(128, 1024, 64, 512), Some(CoopTile::new(128, 64, 16)));
    }
}

#[cfg(test)]
mod dense_split_k_tests {
    //! GPU gates for the dense-large-graph split-K tuning
    //! (`MatMulOperation::dense_codegen`): the wider divisor-aligned
    //! fan-out and the elided-K-bounds kernel are unreachable from the
    //! public tensor API (the flag is set only by `optimize_large_graph`'s
    //! dense branch), so these tests set the flag directly and execute the
    //! operation eagerly. `tests/small_tile_matmul.rs` covers the committed
    //! (flag-unset) split-K behavior through the public API.

    use super::MatMulOperation;
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
            a.data.materialize();
            b.data.materialize();
            let mut operation = MatMulOperation::new(
                crate::DataTypeEnum::F32,
                a.key(),
                b.key(),
                &[m, k],
                &[k, n],
                None,
                &device,
            );
            operation.dense_codegen = true;
            let Some(output) = device.compute_graph().execute_eager(&operation) else {
                // Devices without cooperative matrices run the generic path;
                // nothing dense-specific to gate there.
                return;
            };
            let out: Tensor = Tensor::from(output);
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
