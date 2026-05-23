use crate::{
    Device, Tensor,
    compute_graph::NodeIndex,
    kernel_selection::CooperativeMatrixKind,
    nary_wise::UnaryFunctionChain,
    tensor::{DataType, DataTypeEnum},
};

pub mod coop_gemm;
mod direct;
mod kernel;
pub mod sgemm;
mod sgemm_params;
pub mod sgemv;
mod sgemv_params;
mod variants;

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

#[derive(Debug, Clone)]
pub(crate) struct MatMulOperation {
    pub(crate) datatype: DataTypeEnum,
    pub(crate) first: NodeIndex,
    pub(crate) second: NodeIndex,
    pub(crate) first_shape: Box<[usize]>,
    pub(crate) second_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
    pub(crate) pre_element_wise: [UnaryFunctionChain; 2],
    pub(crate) post_element_wise: UnaryFunctionChain,
    pub(crate) parameters: MatMulParams,
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn mat_mul(&self, other: &Self) -> Self {
        self.add_mat_mul(other, None)
    }

    pub fn mat_mul_with_parameters(&self, other: &Self, parameters: MatMulParams) -> Self {
        self.add_mat_mul(other, Some(parameters))
    }
}

#[cfg(test)]
mod selection_tests {
    use super::variants::{
        DenseMatmulCtx, DenseMatmulVariant, DirectTileCoopMatmulVariant, DirectTileMatmulVariant,
        dense_matmul_selector, direct_tile_matmul_selector, select_coop_kind,
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
    fn direct_tile_selector_generates_each_variant() {
        let selector = direct_tile_matmul_selector();
        let caps = KernelDeviceCaps {
            subgroups_supported: false,
            cooperative_matrix: CooperativeMatrixCaps::default(),
            min_subgroup_size: 0,
            max_subgroup_size: 0,
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroup_storage_size: 0,
            max_compute_workgroup_size_x: 0,
            backend: wgpu::Backend::Noop,
        };
        let mut rng = DeterministicShapeRng::default();

        for variant in [
            DirectTileMatmulVariant::Gemv,
            DirectTileMatmulVariant::MatMul,
        ] {
            let shape = selector
                .generate_for(variant, &(), caps, &mut rng)
                .expect("variant should generate");
            assert_eq!(selector.select(shape, &(), caps), Some(variant));
        }
    }

    #[test]
    fn direct_tile_coop_selector_prefers_largest_supported_tile() {
        // 4096³ (square) hits Tile128x512 — it has fewer barriers than
        // Tile256x256 because it's double-buffered.
        assert_eq!(
            DirectTileCoopMatmulVariant::select(4096, 4096, 4096, 512),
            DirectTileCoopMatmulVariant::Tile128x512
        );
        // Shapes where N is divisible by 256 but not 512 — with enough
        // tiles — fall to Tile256x256 single-buffer.
        assert_eq!(
            DirectTileCoopMatmulVariant::select(8192, 1024, 4352, 512),
            DirectTileCoopMatmulVariant::Tile256x256
        );
        // N=512 doesn't divide 256 on the M side... actually wait, 4096 % 256 == 0.
        // For shapes where N is divisible by 512 but M isn't by 256, fall to
        // Tile128x512.
        assert_eq!(
            DirectTileCoopMatmulVariant::select(384, 1024, 1024, 512),
            DirectTileCoopMatmulVariant::Tile128x64
        );
        // 1024³ doesn't have enough tiles for Tile128x512 OR Tile128x256;
        // falls back to Tile128x64 for better parallelism.
        assert_eq!(
            DirectTileCoopMatmulVariant::select(1024, 1024, 1024, 512),
            DirectTileCoopMatmulVariant::Tile128x64
        );
        // 8192x256 has tiles_for(128, 256) = 64*1 = 64 — below the threshold,
        // so it falls to Tile128x64.
        assert_eq!(
            DirectTileCoopMatmulVariant::select(8192, 1024, 256, 256),
            DirectTileCoopMatmulVariant::Tile128x64
        );
        // M=4096, N=1024 gives tiles_for(128, 256) = 32*4 = 128. Below 256.
        // Falls to Tile128x64.
        assert_eq!(
            DirectTileCoopMatmulVariant::select(4096, 1024, 1024, 256),
            DirectTileCoopMatmulVariant::Tile128x64
        );
        // M=8192, N=512 gives tiles_for(128, 256) = 64*2 = 128 (still <256),
        // so falls to Tile128x64. To hit Tile128x256 we need a wider shape:
        // 8192x1024 → 64*4 = 256 ✓.
        assert_eq!(
            DirectTileCoopMatmulVariant::select(8192, 1024, 1024, 256),
            DirectTileCoopMatmulVariant::Tile128x256
        );
        // N=128 doesn't divide 256 so Tile128x256/Tile128x512 are out; falls
        // back to Tile128x64.
        assert_eq!(
            DirectTileCoopMatmulVariant::select(1024, 1024, 128, 256),
            DirectTileCoopMatmulVariant::Tile128x64
        );
        assert_eq!(
            DirectTileCoopMatmulVariant::select(1024, 1024, 1024, 128),
            DirectTileCoopMatmulVariant::Tile64x64
        );
        assert_eq!(
            DirectTileCoopMatmulVariant::select(1000, 1024, 1024, 512),
            DirectTileCoopMatmulVariant::None
        );
    }
}
