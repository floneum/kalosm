use std::fmt;

use naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn,
    CooperativeData, CooperativeRole, CooperativeSize, EntryPoint, Expression, Function,
    FunctionArgument, GlobalVariable, Handle, Literal, LocalVariable, MathFunction,
    MemoryDecorations, Module, Range, ResourceBinding, Scalar, ShaderStage, Span, Statement,
    StorageAccess, Type, TypeInner, VectorSize,
};

use crate::{
    BarrierScope, BufferAccess, BufferId, DynamicOffset, ElementType, GemmOp, GemvOp, KernelIr,
    Layout, MemoryLevel, MmaOp, Op, StorageView, TileId, TileOrigin, TileRef, ViewMapping,
};

const LOCAL_INVOCATION_INDEX_ARG: u32 = 0;
const WORKGROUP_ID_ARG: u32 = 1;
const SUBGROUP_ID_ARG: u32 = 2;
const DEFAULT_WORKGROUP_INVOCATIONS: u32 = 256;
const DEFAULT_WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];
const GEMV_WORKGROUP_INVOCATIONS: u32 = 128;
const GEMV_WORKGROUP_SIZE: [u32; 3] = [128, 1, 1];
const COOP_MATRIX_WORKGROUP_INVOCATIONS: u32 = 32;
const COOP_MATRIX_WORKGROUP_SIZE: [u32; 3] = [32, 1, 1];
const COOP_MATRIX_DIM: u32 = 8;
const COOP_MATRIX_SIZE: CooperativeSize = CooperativeSize::Eight;
const PREFER_COOP_MATRIX_GEMM: bool = true;
// The staged cooperative path now matches the MLX-style 64x64/4-subgroup shape
// closely enough to beat direct cooperative loads on the current F32 Metal path.
const PREFER_SHARED_COOP_GEMM: bool = true;
// Correct but slower for scalar F32 on current wgpu/Metal; keep it opt-in until
// the cost model can pick it only for hardware/backends where it wins.
const PREFER_SHARED_GEMM: bool = false;
const COOP_MATRIX_OUTER_UNROLL: u32 = 1;
const PREFER_LINEAR_BASE_HOIST: bool = false;
const COOPERATIVE_LOAD_WIDTH: u32 = 8;

pub(crate) fn lower_to_naga(ir: &KernelIr) -> Result<NagaKernel, LowerError> {
    Lowerer::new(ir).lower()
}

/// A validated Naga lowering result.
pub struct NagaKernel {
    module: Module,
    info: naga::valid::ModuleInfo,
}

impl NagaKernel {
    /// The generated Naga module.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Naga validation metadata for the generated module.
    pub fn info(&self) -> &naga::valid::ModuleInfo {
        &self.info
    }
}

/// Errors produced by the Naga lowering pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// An event referenced a tile that was never allocated.
    UnknownTile(TileId),
    /// An operation referenced a storage buffer that was never declared.
    UnknownBuffer(BufferId),
    /// The Naga lowerer cannot emit this memory level.
    UnsupportedMemoryLevel(MemoryLevel),
    /// The typed IR operation is outside the supported lowering subset.
    UnsupportedOperation(&'static str),
    /// An operation used a tile as a different element type than its declaration.
    TileElementMismatch {
        tile: TileId,
        declared: ElementType,
        used: ElementType,
    },
    /// Naga rejected the generated module.
    Validation(String),
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTile(tile) => write!(f, "unknown tile {:?}", tile),
            Self::UnknownBuffer(buffer) => write!(f, "unknown buffer {:?}", buffer),
            Self::UnsupportedMemoryLevel(memory) => {
                write!(f, "unsupported memory level {:?}", memory)
            }
            Self::UnsupportedOperation(op) => write!(f, "unsupported operation {op}"),
            Self::TileElementMismatch {
                tile,
                declared,
                used,
            } => write!(
                f,
                "tile {:?} declared as {:?} but used as {:?}",
                tile, declared, used
            ),
            Self::Validation(error) => write!(f, "naga validation failed: {error}"),
        }
    }
}

impl std::error::Error for LowerError {}

struct Lowerer<'a> {
    ir: &'a KernelIr,
    module: Module,
    f32_ty: Handle<Type>,
    f32_vec4_ty: Handle<Type>,
    u32_ty: Handle<Type>,
    u32_vec3_ty: Handle<Type>,
    coop_f32_a_ty: Option<Handle<Type>>,
    coop_f32_b_ty: Option<Handle<Type>>,
    coop_f32_c_ty: Option<Handle<Type>>,
    buffer_globals: Vec<Option<Handle<GlobalVariable>>>,
    tile_globals: Vec<Option<Handle<GlobalVariable>>>,
    tile_locals: Vec<Option<Handle<LocalVariable>>>,
    live_tiles: Vec<bool>,
    loop_index_local: Option<Handle<LocalVariable>>,
    workgroup_invocations: u32,
    workgroup_size: [u32; 3],
    max_gemv_rows: u32,
    uses_coop_gemm: bool,
    coop_subgroups: u32,
    uses_subgroup_id: bool,
}

#[derive(Copy, Clone)]
struct ScratchLocals {
    tile_index: Handle<LocalVariable>,
    linear_index: Handle<LocalVariable>,
    store_index: Handle<LocalVariable>,
    loop_index: Handle<LocalVariable>,
    mma_i: Handle<LocalVariable>,
    mma_j: Handle<LocalVariable>,
    mma_k: Handle<LocalVariable>,
    mma_sum: Handle<LocalVariable>,
    mma_sum_1: Handle<LocalVariable>,
    mma_sum_2: Option<Handle<LocalVariable>>,
    mma_sum_3: Option<Handle<LocalVariable>>,
    mma_sum_4: Option<Handle<LocalVariable>>,
    mma_sum_5: Option<Handle<LocalVariable>>,
    mma_sum_6: Option<Handle<LocalVariable>>,
    mma_sum_7: Option<Handle<LocalVariable>>,
    coop_accs: [Option<Handle<LocalVariable>>; 16],
}

#[derive(Copy, Clone)]
enum CoopPartition {
    Single,
    Columns,
    Rows,
    InterleavedGrid { row_groups: u32, col_groups: u32 },
}

impl<'a> Lowerer<'a> {
    fn new(ir: &'a KernelIr) -> Self {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("TileElement".into()),
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: Some("SubgroupIndex".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let f32_vec4_ty = module.types.insert(
            Type {
                name: Some("Dot4".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::F32,
                },
            },
            Span::default(),
        );
        let u32_vec3_ty = module.types.insert(
            Type {
                name: Some("WorkgroupId".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            },
            Span::default(),
        );

        let coop_subgroups = if PREFER_COOP_MATRIX_GEMM {
            Self::max_coop_gemm_subgroups(ir)
        } else {
            0
        };
        let uses_coop_gemm = coop_subgroups > 0;
        let uses_subgroup_id = coop_subgroups > 1;
        let (coop_f32_a_ty, coop_f32_b_ty, coop_f32_c_ty) = if uses_coop_gemm {
            let coop_f32_a_ty = module.types.insert(
                Type {
                    name: Some("CoopA8x8F32".into()),
                    inner: TypeInner::CooperativeMatrix {
                        columns: COOP_MATRIX_SIZE,
                        rows: COOP_MATRIX_SIZE,
                        scalar: Scalar::F32,
                        role: CooperativeRole::A,
                    },
                },
                Span::default(),
            );
            let coop_f32_b_ty = module.types.insert(
                Type {
                    name: Some("CoopB8x8F32".into()),
                    inner: TypeInner::CooperativeMatrix {
                        columns: COOP_MATRIX_SIZE,
                        rows: COOP_MATRIX_SIZE,
                        scalar: Scalar::F32,
                        role: CooperativeRole::B,
                    },
                },
                Span::default(),
            );
            let coop_f32_c_ty = module.types.insert(
                Type {
                    name: Some("CoopC8x8F32".into()),
                    inner: TypeInner::CooperativeMatrix {
                        columns: COOP_MATRIX_SIZE,
                        rows: COOP_MATRIX_SIZE,
                        scalar: Scalar::F32,
                        role: CooperativeRole::C,
                    },
                },
                Span::default(),
            );
            (
                Some(coop_f32_a_ty),
                Some(coop_f32_b_ty),
                Some(coop_f32_c_ty),
            )
        } else {
            (None, None, None)
        };

        let max_gemv_rows = Self::max_gemv_rows(ir.body());
        let max_scratch_sums = max_gemv_rows.max(Self::max_gemm_sums(ir.body()));
        let (workgroup_invocations, workgroup_size) = if max_gemv_rows > 0 {
            (GEMV_WORKGROUP_INVOCATIONS, GEMV_WORKGROUP_SIZE)
        } else if uses_coop_gemm {
            (
                COOP_MATRIX_WORKGROUP_INVOCATIONS * coop_subgroups,
                [COOP_MATRIX_WORKGROUP_SIZE[0] * coop_subgroups, 1, 1],
            )
        } else {
            (DEFAULT_WORKGROUP_INVOCATIONS, DEFAULT_WORKGROUP_SIZE)
        };
        let live_tiles = Self::live_tiles(ir, workgroup_invocations);

        Self {
            ir,
            module,
            f32_ty,
            f32_vec4_ty,
            u32_ty,
            u32_vec3_ty,
            coop_f32_a_ty,
            coop_f32_b_ty,
            coop_f32_c_ty,
            buffer_globals: Vec::new(),
            tile_globals: Vec::new(),
            tile_locals: Vec::new(),
            live_tiles,
            loop_index_local: None,
            workgroup_invocations,
            workgroup_size,
            max_gemv_rows: max_scratch_sums,
            uses_coop_gemm,
            coop_subgroups,
            uses_subgroup_id,
        }
    }

    fn max_gemv_rows(block: &crate::Block) -> u32 {
        block
            .ops()
            .iter()
            .map(|op| match op {
                Op::Gemv(op) => op.rows_per_workgroup,
                Op::Block(op) => Self::max_gemv_rows(&op.body),
                Op::Loop(op) => Self::max_gemv_rows(&op.body),
                Op::Partition(op) => Self::max_gemv_rows(&op.body),
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }

    fn is_single_direct_coop_gemm(ir: &KernelIr) -> bool {
        let ops = ir.body().ops();
        if ops.len() != 3 {
            return false;
        }
        let Some((_, loop_op, store, gemm, a_load, b_load)) = Self::fused_gemm_parts(ops, 0) else {
            return false;
        };
        let crate::LoopKind::RangeStep { iterations, .. } = loop_op.kind;
        let Some(acc_layout) = Self::tile_layout_in_ir(ir, gemm.acc) else {
            return false;
        };
        Self::storage_gemm_coop8_subgroups(
            &a_load.src.layout,
            &b_load.src.layout,
            acc_layout,
            &store.dst.layout,
            iterations,
        )
        .is_some()
    }

    fn live_tiles(ir: &KernelIr, workgroup_invocations: u32) -> Vec<bool> {
        let mut live = vec![false; ir.tiles().len()];
        Self::collect_live_tiles(ir, ir.body(), &mut live, workgroup_invocations);
        live
    }

    fn collect_live_tiles(
        ir: &KernelIr,
        block: &crate::Block,
        live: &mut [bool],
        workgroup_invocations: u32,
    ) {
        let ops = block.ops();
        let mut index = 0;
        while index < ops.len() {
            if Self::mark_shared_fused_gemm_tiles(ir, ops, index, live, workgroup_invocations) {
                index += 3;
                continue;
            }
            if Self::is_direct_fused_gemm_pattern(ops, index) {
                index += 3;
                continue;
            }

            match &ops[index] {
                Op::Block(op) => {
                    Self::collect_live_tiles(ir, &op.body, live, workgroup_invocations)
                }
                Op::FillTile(op) => Self::mark_tile_live(ir, op.dst, live),
                Op::CooperativeLoad(op) => Self::mark_tile_live(ir, op.dst, live),
                Op::Partition(op) => {
                    for binding in &op.bindings {
                        Self::mark_tile_live(ir, binding.source, live);
                        Self::mark_tile_live(ir, binding.view, live);
                    }
                    Self::collect_live_tiles(ir, &op.body, live, workgroup_invocations);
                }
                Op::Barrier(_) => {}
                Op::Gemm(op) => {
                    Self::mark_tile_live(ir, op.a, live);
                    Self::mark_tile_live(ir, op.b, live);
                    Self::mark_tile_live(ir, op.acc, live);
                }
                Op::Gemv(op) => Self::mark_tile_live(ir, op.partials, live),
                Op::Mma(op) => {
                    Self::mark_tile_live(ir, op.a, live);
                    Self::mark_tile_live(ir, op.b, live);
                    Self::mark_tile_live(ir, op.acc, live);
                }
                Op::StoreTile(op) => Self::mark_tile_live(ir, op.src, live),
                Op::Loop(op) => Self::collect_live_tiles(ir, &op.body, live, workgroup_invocations),
            }
            index += 1;
        }
    }

    fn is_direct_fused_gemm_pattern(ops: &[Op], index: usize) -> bool {
        Self::fused_gemm_parts(ops, index).is_some()
    }

    fn mark_shared_fused_gemm_tiles(
        ir: &KernelIr,
        ops: &[Op],
        index: usize,
        live: &mut [bool],
        workgroup_invocations: u32,
    ) -> bool {
        let Some((_, loop_op, store, gemm, a_load, b_load)) = Self::fused_gemm_parts(ops, index)
        else {
            return false;
        };
        let crate::LoopKind::RangeStep { iterations, .. } = loop_op.kind;

        let can_lower_coop = if PREFER_COOP_MATRIX_GEMM && PREFER_SHARED_COOP_GEMM {
            match (
                Self::tile_layout_in_ir(ir, gemm.a),
                Self::tile_layout_in_ir(ir, gemm.b),
                Self::tile_layout_in_ir(ir, gemm.acc),
            ) {
                (Some(a_layout), Some(b_layout), Some(acc_layout)) => {
                    Self::can_lower_shared_gemm_coop8(
                        a_layout,
                        b_layout,
                        acc_layout,
                        &store.dst.layout,
                        iterations,
                    )
                }
                _ => false,
            }
        } else {
            false
        };

        let can_lower_scalar = PREFER_SHARED_GEMM
            && Self::can_lower_shared_gemm_4col(ir, gemm, iterations, workgroup_invocations);
        if !can_lower_coop && !can_lower_scalar {
            return false;
        }

        Self::mark_tile_live(ir, a_load.dst, live);
        Self::mark_tile_live(ir, b_load.dst, live);
        true
    }

    fn fused_gemm_parts<'ops>(
        ops: &'ops [Op],
        index: usize,
    ) -> Option<(
        &'ops crate::FillTileOp,
        &'ops crate::LoopOp,
        &'ops crate::StoreTileOp,
        &'ops GemmOp,
        &'ops crate::CooperativeLoadOp,
        &'ops crate::CooperativeLoadOp,
    )> {
        let Some(Op::FillTile(fill)) = ops.get(index) else {
            return None;
        };
        let Some(Op::Loop(loop_op)) = ops.get(index + 1) else {
            return None;
        };
        let Some(Op::StoreTile(store)) = ops.get(index + 2) else {
            return None;
        };
        if fill.value != crate::FillValue::Zero || store.src != fill.dst {
            return None;
        }
        let mut gemm = None;
        let mut loads = Vec::new();
        for op in loop_op.body.ops() {
            match op {
                Op::CooperativeLoad(op) => loads.push(op),
                Op::Barrier(_) => {}
                Op::Gemm(op) if op.acc == fill.dst && gemm.is_none() => gemm = Some(op),
                _ => return None,
            }
        }
        let gemm = gemm?;
        let a_load = loads.iter().copied().find(|load| load.dst == gemm.a)?;
        let b_load = loads.iter().copied().find(|load| load.dst == gemm.b)?;
        Some((fill, loop_op, store, gemm, a_load, b_load))
    }

    fn can_lower_shared_gemm_4col(
        ir: &KernelIr,
        gemm: &GemmOp,
        outer_iterations: u32,
        workgroup_invocations: u32,
    ) -> bool {
        if outer_iterations == 0 {
            return false;
        }
        let Some(a_layout) = Self::tile_layout_in_ir(ir, gemm.a) else {
            return false;
        };
        let Some(b_layout) = Self::tile_layout_in_ir(ir, gemm.b) else {
            return false;
        };
        let Some(acc_layout) = Self::tile_layout_in_ir(ir, gemm.acc) else {
            return false;
        };
        if a_layout.memory_level() != MemoryLevel::Workgroup
            || b_layout.memory_level() != MemoryLevel::Workgroup
            || acc_layout.memory_level() != MemoryLevel::Private
        {
            return false;
        }
        let Ok([m, k_a]) = Self::matrix_shape(a_layout) else {
            return false;
        };
        let Ok([k_b, n]) = Self::matrix_shape(b_layout) else {
            return false;
        };
        let Ok([m_acc, n_acc]) = Self::matrix_shape(acc_layout) else {
            return false;
        };
        if k_a != k_b || m != m_acc || n != n_acc || n % 4 != 0 || k_a % 4 != 0 {
            return false;
        }
        m.checked_mul(n / 4) == Some(workgroup_invocations)
    }

    fn max_coop_gemm_subgroups(ir: &KernelIr) -> u32 {
        Self::block_max_coop_gemm_subgroups(ir, ir.body())
    }

    fn block_max_coop_gemm_subgroups(ir: &KernelIr, block: &crate::Block) -> u32 {
        let ops = block.ops();
        let mut index = 0;
        let mut max_subgroups = 0;
        while index < ops.len() {
            if let Some((_, loop_op, store, gemm, a_load, b_load)) =
                Self::fused_gemm_parts(ops, index)
            {
                let crate::LoopKind::RangeStep { iterations, .. } = loop_op.kind;
                if PREFER_SHARED_COOP_GEMM {
                    if let (Some(a_layout), Some(b_layout), Some(acc_layout)) = (
                        Self::tile_layout_in_ir(ir, gemm.a),
                        Self::tile_layout_in_ir(ir, gemm.b),
                        Self::tile_layout_in_ir(ir, gemm.acc),
                    ) {
                        if let Some(subgroups) = Self::shared_gemm_coop8_subgroups(
                            a_layout,
                            b_layout,
                            acc_layout,
                            &store.dst.layout,
                            iterations,
                        ) {
                            max_subgroups = max_subgroups.max(subgroups);
                        }
                    }
                }
                if let Some(acc_layout) = Self::tile_layout_in_ir(ir, gemm.acc) {
                    if let Some(subgroups) = Self::storage_gemm_coop8_subgroups(
                        &a_load.src.layout,
                        &b_load.src.layout,
                        acc_layout,
                        &store.dst.layout,
                        iterations,
                    ) {
                        max_subgroups = max_subgroups.max(subgroups);
                    }
                }
                index += 3;
                continue;
            }

            let nested = match &ops[index] {
                Op::Block(op) => Self::block_max_coop_gemm_subgroups(ir, &op.body),
                Op::Loop(op) => Self::block_max_coop_gemm_subgroups(ir, &op.body),
                Op::Partition(op) => Self::block_max_coop_gemm_subgroups(ir, &op.body),
                _ => 0,
            };
            max_subgroups = max_subgroups.max(nested);
            index += 1;
        }
        max_subgroups
    }

    fn can_lower_storage_gemm_coop8(
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
        dst_layout: &Layout,
        outer_iterations: u32,
    ) -> bool {
        Self::storage_gemm_coop8_subgroups(
            a_layout,
            b_layout,
            acc_layout,
            dst_layout,
            outer_iterations,
        )
        .is_some()
    }

    fn can_lower_shared_gemm_coop8(
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
        dst_layout: &Layout,
        outer_iterations: u32,
    ) -> bool {
        Self::shared_gemm_coop8_subgroups(
            a_layout,
            b_layout,
            acc_layout,
            dst_layout,
            outer_iterations,
        )
        .is_some()
    }

    fn shared_gemm_coop8_subgroups(
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
        dst_layout: &Layout,
        outer_iterations: u32,
    ) -> Option<u32> {
        if outer_iterations == 0
            || a_layout.memory_level() != MemoryLevel::Workgroup
            || b_layout.memory_level() != MemoryLevel::Workgroup
            || acc_layout.memory_level() != MemoryLevel::Private
            || dst_layout.memory_level() != MemoryLevel::Storage
            || !Self::is_row_major_storage_matrix(a_layout)
            || !Self::is_row_major_storage_matrix(b_layout)
            || !Self::is_row_major_storage_matrix(dst_layout)
        {
            return None;
        }

        Self::gemm_coop8_subgroups_for_shapes(a_layout, b_layout, acc_layout, dst_layout)
    }

    fn storage_gemm_coop8_subgroups(
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
        dst_layout: &Layout,
        outer_iterations: u32,
    ) -> Option<u32> {
        if outer_iterations == 0
            || acc_layout.memory_level() != MemoryLevel::Private
            || !Self::is_row_major_storage_matrix(a_layout)
            || !Self::is_row_major_storage_matrix(b_layout)
            || !Self::is_row_major_storage_matrix(dst_layout)
        {
            return None;
        }

        Self::gemm_coop8_subgroups_for_shapes(a_layout, b_layout, acc_layout, dst_layout)
    }

    fn gemm_coop8_subgroups_for_shapes(
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
        dst_layout: &Layout,
    ) -> Option<u32> {
        let Ok([m, k_a]) = Self::matrix_shape(a_layout) else {
            return None;
        };
        let Ok([k_b, n]) = Self::matrix_shape(b_layout) else {
            return None;
        };
        let Ok([m_acc, n_acc]) = Self::matrix_shape(acc_layout) else {
            return None;
        };
        let Ok([m_dst, n_dst]) = Self::matrix_shape(dst_layout) else {
            return None;
        };
        if k_a != k_b
            || k_a % COOP_MATRIX_DIM != 0
            || m_acc != m
            || n_acc != n
            || m_dst != m
            || n_dst != n
            || m % COOP_MATRIX_DIM != 0
            || n % COOP_MATRIX_DIM != 0
        {
            return None;
        }

        let tile_rows = m / COOP_MATRIX_DIM;
        let tile_cols = n / COOP_MATRIX_DIM;
        if tile_rows == 0 || tile_cols == 0 {
            return None;
        }
        if tile_rows * tile_cols <= 16 {
            return Some(1);
        }
        if m == 32 && n % 32 == 0 {
            let subgroups = n / 32;
            if (2..=8).contains(&subgroups) {
                return Some(subgroups);
            }
        }
        if n >= m && m <= 64 && n % 16 == 0 {
            let subgroups = n / 16;
            if (2..=8).contains(&subgroups) {
                return Some(subgroups);
            }
        }
        if n <= 64 && m % 16 == 0 {
            let subgroups = m / 16;
            if (2..=8).contains(&subgroups) {
                return Some(subgroups);
            }
        }
        None
    }

    fn is_row_major_storage_matrix(layout: &Layout) -> bool {
        layout.shape().rank() == 2
            && layout.strides().rank() == 2
            && layout.strides().values()[1] == 1
    }

    fn row_major_matrix_leading_stride(layout: &Layout) -> Result<u32, LowerError> {
        if !Self::is_row_major_storage_matrix(layout) {
            return Err(LowerError::UnsupportedOperation(
                "cooperative matrix lowering currently requires row-major matrix views",
            ));
        }
        Ok(layout.strides().values()[0])
    }

    fn tile_layout_in_ir(ir: &KernelIr, tile: TileRef) -> Option<&Layout> {
        let decl = ir.tiles().get(tile.id.index())?;
        (decl.element == tile.element).then_some(&decl.layout)
    }

    fn mark_tile_live(ir: &KernelIr, tile: TileRef, live: &mut [bool]) {
        let Some(decl) = ir.tiles().get(tile.id.index()) else {
            return;
        };
        live[tile.id.index()] = true;
        if let TileOrigin::View { source, .. } = decl.origin {
            Self::mark_tile_live(ir, source, live);
        }
    }

    fn max_gemm_sums(block: &crate::Block) -> u32 {
        block
            .ops()
            .iter()
            .map(|op| match op {
                Op::Gemm(_) => 8,
                Op::Block(op) => Self::max_gemm_sums(&op.body),
                Op::Loop(op) => Self::max_gemm_sums(&op.body),
                Op::Partition(op) => Self::max_gemm_sums(&op.body),
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }

    fn lower(mut self) -> Result<NagaKernel, LowerError> {
        self.create_storage_globals()?;
        self.create_workgroup_globals()?;

        let mut arguments = vec![
            FunctionArgument {
                name: Some("local_invocation_index".into()),
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
            },
            FunctionArgument {
                name: Some("workgroup_id".into()),
                ty: self.u32_vec3_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
            },
        ];
        if self.uses_subgroup_id {
            arguments.push(FunctionArgument {
                name: Some("subgroup_id".into()),
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::SubgroupId)),
            });
        }

        let mut function = Function {
            name: Some("main".into()),
            arguments,
            ..Function::default()
        };
        let scratch = self.create_scratch_locals(&mut function);
        self.loop_index_local = Some(scratch.loop_index);
        self.create_private_locals(&mut function)?;

        function.body = self.lower_block(self.ir.body(), &mut function.expressions, scratch)?;
        function
            .body
            .push(Statement::Return { value: None }, Span::default());

        self.module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: self.workgroup_size,
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        let mut capabilities = naga::valid::Capabilities::empty();
        if self.uses_coop_gemm {
            capabilities |= naga::valid::Capabilities::COOPERATIVE_MATRIX;
        }
        if self.uses_subgroup_id {
            capabilities |= naga::valid::Capabilities::SUBGROUP;
        }
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&self.module)
            .map_err(|error| LowerError::Validation(format!("{error:#?}")))?;

        Ok(NagaKernel {
            module: self.module,
            info,
        })
    }

    fn create_storage_globals(&mut self) -> Result<(), LowerError> {
        self.buffer_globals = vec![None; self.ir.buffers().len()];
        for buffer in self.ir.buffers() {
            let ty = self.storage_type(buffer.id.index(), buffer.element, &buffer.layout);
            let access = match buffer.access {
                BufferAccess::Read => StorageAccess::LOAD,
                BufferAccess::ReadWrite => StorageAccess::LOAD | StorageAccess::STORE,
            };
            let global = self.module.global_variables.append(
                GlobalVariable {
                    name: Some(format!("buffer_{}", buffer.id.index())),
                    space: AddressSpace::Storage { access },
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: buffer.id.index() as u32,
                    }),
                    ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::default(),
            );
            self.buffer_globals[buffer.id.index()] = Some(global);
        }
        Ok(())
    }

    fn create_workgroup_globals(&mut self) -> Result<(), LowerError> {
        self.tile_globals = vec![None; self.ir.tiles().len()];
        for tile in self.ir.tiles() {
            if !self
                .live_tiles
                .get(tile.id.index())
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            if tile.layout.memory_level() != MemoryLevel::Workgroup
                || tile.origin != TileOrigin::Allocation
            {
                continue;
            }
            let ty = self.tile_type(tile.id.index(), tile.element, &tile.layout);
            let global = self.module.global_variables.append(
                GlobalVariable {
                    name: Some(format!("tile_{}", tile.id.index())),
                    space: AddressSpace::WorkGroup,
                    binding: None,
                    ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::default(),
            );
            self.tile_globals[tile.id.index()] = Some(global);
        }
        Ok(())
    }

    fn create_private_locals(&mut self, function: &mut Function) -> Result<(), LowerError> {
        self.tile_locals = vec![None; self.ir.tiles().len()];
        for tile in self.ir.tiles() {
            if !self
                .live_tiles
                .get(tile.id.index())
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            if tile.layout.memory_level() != MemoryLevel::Private
                || tile.origin != TileOrigin::Allocation
            {
                continue;
            }
            let ty = self.tile_type(tile.id.index(), tile.element, &tile.layout);
            let local = function.local_variables.append(
                LocalVariable {
                    name: Some(format!("tile_{}", tile.id.index())),
                    ty,
                    init: None,
                },
                Span::default(),
            );
            self.tile_locals[tile.id.index()] = Some(local);
        }
        Ok(())
    }

    fn create_scratch_locals(&self, function: &mut Function) -> ScratchLocals {
        let mut coop_accs = [None; 16];
        if let Some(ty) = self.coop_f32_c_ty {
            for (index, local) in coop_accs.iter_mut().enumerate() {
                *local = Some(self.create_local(function, &format!("coop_acc_{index}"), ty));
            }
        }

        if self.uses_coop_gemm
            && !PREFER_SHARED_COOP_GEMM
            && Self::is_single_direct_coop_gemm(self.ir)
        {
            let loop_index = self.create_u32_local(function, "loop_index");
            let mma_k = self.create_u32_local(function, "mma_k");
            let mma_sum = self.create_f32_local(function, "mma_sum");
            let mma_sum_1 = self.create_f32_local(function, "mma_sum_1");
            return ScratchLocals {
                tile_index: loop_index,
                linear_index: loop_index,
                store_index: loop_index,
                loop_index,
                mma_i: mma_k,
                mma_j: mma_k,
                mma_k,
                mma_sum,
                mma_sum_1,
                mma_sum_2: None,
                mma_sum_3: None,
                mma_sum_4: None,
                mma_sum_5: None,
                mma_sum_6: None,
                mma_sum_7: None,
                coop_accs,
            };
        }

        ScratchLocals {
            tile_index: self.create_u32_local(function, "tile_index"),
            linear_index: self.create_u32_local(function, "linear_index"),
            store_index: self.create_u32_local(function, "store_index"),
            loop_index: self.create_u32_local(function, "loop_index"),
            mma_i: self.create_u32_local(function, "mma_i"),
            mma_j: self.create_u32_local(function, "mma_j"),
            mma_k: self.create_u32_local(function, "mma_k"),
            mma_sum: self.create_f32_local(function, "mma_sum"),
            mma_sum_1: self.create_f32_local(function, "mma_sum_1"),
            mma_sum_2: (self.max_gemv_rows > 2)
                .then(|| self.create_f32_local(function, "mma_sum_2")),
            mma_sum_3: (self.max_gemv_rows > 3)
                .then(|| self.create_f32_local(function, "mma_sum_3")),
            mma_sum_4: (self.max_gemv_rows > 4)
                .then(|| self.create_f32_local(function, "mma_sum_4")),
            mma_sum_5: (self.max_gemv_rows > 5)
                .then(|| self.create_f32_local(function, "mma_sum_5")),
            mma_sum_6: (self.max_gemv_rows > 6)
                .then(|| self.create_f32_local(function, "mma_sum_6")),
            mma_sum_7: (self.max_gemv_rows > 7)
                .then(|| self.create_f32_local(function, "mma_sum_7")),
            coop_accs,
        }
    }

    fn create_u32_local(&self, function: &mut Function, name: &str) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty: self.u32_ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn create_f32_local(&self, function: &mut Function, name: &str) -> Handle<LocalVariable> {
        self.create_local(function, name, self.f32_ty)
    }

    fn create_local(
        &self,
        function: &mut Function,
        name: &str,
        ty: Handle<Type>,
    ) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn tile_type(&mut self, tile: usize, element: ElementType, layout: &Layout) -> Handle<Type> {
        self.array_type(format!("Tile{tile}"), element, layout)
    }

    fn storage_type(
        &mut self,
        buffer: usize,
        element: ElementType,
        layout: &Layout,
    ) -> Handle<Type> {
        self.array_type(format!("Buffer{buffer}"), element, layout)
    }

    fn array_type(&mut self, name: String, element: ElementType, layout: &Layout) -> Handle<Type> {
        let base = match element {
            ElementType::F32 => self.f32_ty,
        };

        self.module.types.insert(
            Type {
                name: Some(name),
                inner: TypeInner::Array {
                    base,
                    size: ArraySize::Constant(layout.allocation_element_count()),
                    stride: 4,
                },
            },
            Span::default(),
        )
    }

    fn lower_block(
        &self,
        ir_block: &crate::Block,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();

        let ops = ir_block.ops();
        let mut op_index = 0;
        while op_index < ops.len() {
            if let Some((statement, consumed)) =
                self.try_lower_fused_gemm_store(ops, op_index, expressions, scratch)?
            {
                body.push(statement, Span::default());
                op_index += consumed;
                continue;
            }

            let op = &ops[op_index];
            match op {
                Op::Block(op) => {
                    body.push(
                        Statement::Block(self.lower_block(&op.body, expressions, scratch)?),
                        Span::default(),
                    );
                }
                Op::FillTile(op) => match self.tile_layout(op.dst)?.memory_level() {
                    MemoryLevel::Workgroup => {
                        body.push(
                            self.store_zero_to_tile(expressions, scratch.tile_index, op.dst)?,
                            Span::default(),
                        );
                    }
                    MemoryLevel::Private => {
                        body.push(
                            self.fill_private_tile(expressions, scratch.linear_index, op.dst)?,
                            Span::default(),
                        );
                    }
                    memory => return Err(LowerError::UnsupportedMemoryLevel(memory)),
                },
                Op::CooperativeLoad(op) => {
                    body.push(
                        self.lower_cooperative_load(
                            expressions,
                            scratch.tile_index,
                            op.dst,
                            &op.src,
                        )?,
                        Span::default(),
                    );
                }
                Op::Barrier(op) => {
                    let barrier = match op.scope {
                        BarrierScope::Workgroup => Barrier::WORK_GROUP,
                    };
                    body.push(Statement::ControlBarrier(barrier), Span::default());
                }
                Op::Partition(op) => {
                    for binding in &op.bindings {
                        self.tile_layout(binding.source)?;
                        self.tile_layout(binding.view)?;
                    }
                    body.push(
                        Statement::Block(self.lower_block(&op.body, expressions, scratch)?),
                        Span::default(),
                    );
                }
                Op::Gemm(op) => {
                    body.push(self.lower_gemm(expressions, scratch, op)?, Span::default());
                }
                Op::Gemv(op) => {
                    body.push(self.lower_gemv(expressions, scratch, op)?, Span::default());
                }
                Op::Mma(op) => {
                    body.push(self.lower_mma(expressions, scratch, op)?, Span::default());
                }
                Op::StoreTile(op) => {
                    body.push(
                        self.lower_store_tile(expressions, scratch.store_index, op.src, &op.dst)?,
                        Span::default(),
                    );
                }
                Op::Loop(op) => {
                    let crate::LoopKind::RangeStep { iterations, .. } = op.kind;
                    let loop_body = self.lower_block(&op.body, expressions, scratch)?;
                    body.push(
                        self.counted_loop(expressions, scratch.loop_index, iterations, loop_body),
                        Span::default(),
                    );
                }
            }
            op_index += 1;
        }

        Ok(body)
    }

    fn try_lower_fused_gemm_store(
        &self,
        ops: &[Op],
        op_index: usize,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
    ) -> Result<Option<(Statement, usize)>, LowerError> {
        let Some(Op::FillTile(fill)) = ops.get(op_index) else {
            return Ok(None);
        };
        let Some(Op::Loop(loop_op)) = ops.get(op_index + 1) else {
            return Ok(None);
        };
        let Some(Op::StoreTile(store)) = ops.get(op_index + 2) else {
            return Ok(None);
        };
        if fill.value != crate::FillValue::Zero || store.src != fill.dst {
            return Ok(None);
        }
        let crate::LoopKind::RangeStep { iterations, .. } = loop_op.kind;

        if let Some(statement) = self.try_lower_shared_gemm_store(
            &loop_op.body,
            fill.dst,
            &store.dst,
            iterations,
            expressions,
            scratch,
        )? {
            return Ok(Some((statement, 3)));
        }

        if let Some(statement) = self.try_lower_direct_gemm_store(
            &loop_op.body,
            fill.dst,
            &store.dst,
            iterations,
            expressions,
            scratch,
        )? {
            return Ok(Some((statement, 3)));
        }

        if iterations != 1 {
            return Ok(None);
        }

        let mut body = Block::new();
        let mut fused_gemm = false;
        for op in loop_op.body.ops() {
            match op {
                Op::CooperativeLoad(op) => {
                    body.push(
                        self.lower_cooperative_load(
                            expressions,
                            scratch.tile_index,
                            op.dst,
                            &op.src,
                        )?,
                        Span::default(),
                    );
                }
                Op::Barrier(op) => {
                    if fused_gemm {
                        continue;
                    }
                    let barrier = match op.scope {
                        BarrierScope::Workgroup => Barrier::WORK_GROUP,
                    };
                    body.push(Statement::ControlBarrier(barrier), Span::default());
                }
                Op::Gemm(op) if op.acc == fill.dst && !fused_gemm => {
                    body.push(
                        self.lower_gemm_to_storage(expressions, scratch, op, &store.dst)?,
                        Span::default(),
                    );
                    fused_gemm = true;
                }
                _ => return Ok(None),
            }
        }

        if fused_gemm {
            Ok(Some((Statement::Block(body), 3)))
        } else {
            Ok(None)
        }
    }

    fn try_lower_shared_gemm_store(
        &self,
        loop_body: &crate::Block,
        acc: TileRef,
        dst: &StorageView,
        iterations: u32,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
    ) -> Result<Option<Statement>, LowerError> {
        if !PREFER_SHARED_GEMM {
            return Ok(None);
        }
        let mut loads = Vec::new();
        let mut gemm = None;
        for op in loop_body.ops() {
            match op {
                Op::CooperativeLoad(op) => loads.push(op),
                Op::Barrier(_) => {}
                Op::Gemm(op) if op.acc == acc && gemm.is_none() => gemm = Some(op),
                _ => return Ok(None),
            }
        }

        let Some(gemm) = gemm else {
            return Ok(None);
        };
        let Some(a_load) = loads.iter().find(|load| load.dst == gemm.a) else {
            return Ok(None);
        };
        let Some(b_load) = loads.iter().find(|load| load.dst == gemm.b) else {
            return Ok(None);
        };

        self.lower_shared_gemm_loop_to_storage_4col(
            expressions,
            scratch,
            a_load,
            b_load,
            gemm,
            dst,
            iterations,
        )
    }

    fn try_lower_direct_gemm_store(
        &self,
        loop_body: &crate::Block,
        acc: TileRef,
        dst: &StorageView,
        iterations: u32,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
    ) -> Result<Option<Statement>, LowerError> {
        let mut loads = Vec::new();
        let mut gemm = None;
        for op in loop_body.ops() {
            match op {
                Op::CooperativeLoad(op) => loads.push(op),
                Op::Barrier(_) => {}
                Op::Gemm(op) if op.acc == acc && gemm.is_none() => gemm = Some(op),
                _ => return Ok(None),
            }
        }

        let Some(gemm) = gemm else {
            return Ok(None);
        };
        let Some(a_load) = loads.iter().find(|load| load.dst == gemm.a) else {
            return Ok(None);
        };
        let Some(b_load) = loads.iter().find(|load| load.dst == gemm.b) else {
            return Ok(None);
        };

        if PREFER_COOP_MATRIX_GEMM && PREFER_SHARED_COOP_GEMM {
            let a_layout = self.tile_layout(gemm.a)?;
            let b_layout = self.tile_layout(gemm.b)?;
            let acc_layout = self.tile_layout(gemm.acc)?;
            let dst_layout = self.storage_layout(dst)?;
            if Self::can_lower_shared_gemm_coop8(
                a_layout, b_layout, acc_layout, dst_layout, iterations,
            ) {
                return Ok(Some(self.lower_shared_gemm_loop_to_storage_coop8(
                    expressions,
                    scratch,
                    a_load,
                    b_load,
                    gemm,
                    dst,
                    iterations,
                )?));
            }
        }

        Ok(Some(self.lower_storage_gemm_loop_to_storage(
            expressions,
            scratch,
            &a_load.src,
            &b_load.src,
            gemm,
            dst,
            iterations,
        )?))
    }

    fn fill_private_tile(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        tile: TileRef,
    ) -> Result<Statement, LowerError> {
        let layout = self.tile_layout(tile)?;
        let mut body = Block::new();
        let (index, index_emit) = self.load_u32_local(expressions, index_local);
        let (pointer, pointer_emits) = self.tile_dynamic_pointer(expressions, tile, index)?;
        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        body.push(Statement::Emit(index_emit), Span::default());
        for emit in pointer_emits {
            body.push(Statement::Emit(emit), Span::default());
        }
        body.push(
            Statement::Store {
                pointer,
                value: zero,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(expressions, index_local, layout.element_count(), body))
    }

    fn lower_gemm(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &GemmOp,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.tile_layout(op.a)?;
        let b_layout = self.tile_layout(op.b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;

        if k_a != k_b || m != m_acc || n != n_acc {
            return Err(LowerError::UnsupportedOperation("gemm shape mismatch"));
        }
        if acc_layout.memory_level() != MemoryLevel::Private {
            return Err(LowerError::UnsupportedMemoryLevel(
                acc_layout.memory_level(),
            ));
        }
        let mut body = Block::new();
        let (flat, flat_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(flat_emit), Span::default());

        let cols = expressions.append(Expression::Literal(Literal::U32(n)), Span::default());
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: flat,
                right: cols,
            },
            Span::default(),
        );
        let col = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: flat,
                right: cols,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::range_from(expressions, row, col)),
            Span::default(),
        );

        let (acc_index, acc_index_emits) =
            self.layout_index_expr(expressions, acc_layout, &[row, col])?;
        Self::push_emits(&mut body, acc_index_emits);
        let (acc_pointer, acc_pointer_emits) =
            self.tile_dynamic_pointer(expressions, op.acc, acc_index)?;
        Self::push_emits(&mut body, acc_pointer_emits);
        let acc_value = expressions.append(
            Expression::Load {
                pointer: acc_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, acc_value)),
            Span::default(),
        );
        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        body.push(
            Statement::Store {
                pointer: sum_pointer,
                value: acc_value,
            },
            Span::default(),
        );

        let (k_body, k_iterations) = if k_a % 4 == 0 {
            let mut k_body = Block::new();
            let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
            k_body.push(Statement::Emit(k_emit), Span::default());
            let mut emits = Vec::new();
            let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
            Self::push_emits(&mut k_body, emits);

            let mut a_values = Vec::with_capacity(4);
            let mut b_values = Vec::with_capacity(4);
            for lane in 0..4 {
                let mut lane_emits = Vec::new();
                let k = self.add_literal_u32_emitted(expressions, base_k, lane, &mut lane_emits);
                let (a_index, a_index_emits) =
                    self.layout_index_expr(expressions, a_layout, &[row, k])?;
                let (b_index, b_index_emits) =
                    self.layout_index_expr(expressions, b_layout, &[k, col])?;
                lane_emits.extend(a_index_emits);
                lane_emits.extend(b_index_emits);
                let (a_pointer, a_pointer_emits) =
                    self.tile_dynamic_pointer(expressions, op.a, a_index)?;
                let (b_pointer, b_pointer_emits) =
                    self.tile_dynamic_pointer(expressions, op.b, b_index)?;
                lane_emits.extend(a_pointer_emits);
                lane_emits.extend(b_pointer_emits);
                Self::push_emits(&mut k_body, lane_emits);

                let a_value =
                    expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
                let b_value =
                    expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
                k_body.push(
                    Statement::Emit(Self::range_from(expressions, a_value, b_value)),
                    Span::default(),
                );
                a_values.push(a_value);
                b_values.push(b_value);
            }

            let a_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components: a_values,
                },
                Span::default(),
            );
            let b_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components: b_values,
                },
                Span::default(),
            );
            let dot = expressions.append(
                Expression::Math {
                    fun: MathFunction::Dot,
                    arg: a_vec,
                    arg1: Some(b_vec),
                    arg2: None,
                    arg3: None,
                },
                Span::default(),
            );
            k_body.push(
                Statement::Emit(Self::range_from(expressions, a_vec, dot)),
                Span::default(),
            );

            let sum_pointer =
                expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            let value = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: sum_value,
                    right: dot,
                },
                Span::default(),
            );
            k_body.push(
                Statement::Emit(Self::range_from(expressions, sum_value, value)),
                Span::default(),
            );
            k_body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value,
                },
                Span::default(),
            );
            (k_body, k_a / 4)
        } else {
            let mut k_body = Block::new();
            let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
            k_body.push(Statement::Emit(k_emit), Span::default());

            let (a_index, a_index_emits) =
                self.layout_index_expr(expressions, a_layout, &[row, k])?;
            let (b_index, b_index_emits) =
                self.layout_index_expr(expressions, b_layout, &[k, col])?;
            Self::push_emits(&mut k_body, a_index_emits);
            Self::push_emits(&mut k_body, b_index_emits);

            let (a_pointer, a_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.a, a_index)?;
            let (b_pointer, b_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.b, b_index)?;
            Self::push_emits(&mut k_body, a_pointer_emits);
            Self::push_emits(&mut k_body, b_pointer_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            let b_value =
                expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
            let sum_pointer =
                expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            k_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            k_body.push(
                Statement::Emit(Self::single_expression_range(expressions, b_value)),
                Span::default(),
            );
            let value = expressions.append(
                Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a_value,
                    arg1: Some(b_value),
                    arg2: Some(sum_value),
                    arg3: None,
                },
                Span::default(),
            );
            k_body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            k_body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value,
                },
                Span::default(),
            );
            (k_body, k_a)
        };

        body.push(
            self.counted_loop(expressions, scratch.mma_k, k_iterations, k_body),
            Span::default(),
        );
        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum_value = expressions.append(
            Expression::Load {
                pointer: sum_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, sum_value)),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: acc_pointer,
                value: sum_value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(
            expressions,
            scratch.linear_index,
            acc_layout.element_count(),
            body,
        ))
    }

    #[allow(dead_code)]
    fn lower_gemm_2col_microtile(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &GemmOp,
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
    ) -> Result<Statement, LowerError> {
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [_, n] = Self::matrix_shape(b_layout)?;
        let lanes = std::num::NonZeroU32::new(m * (n / 2))
            .ok_or(LowerError::UnsupportedOperation("empty microtile"))?;

        let mut body = Block::new();
        let (lane, lane_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(lane_emit), Span::default());

        let col_pairs =
            expressions.append(Expression::Literal(Literal::U32(n / 2)), Span::default());
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: lane,
                right: col_pairs,
            },
            Span::default(),
        );
        let col_pair = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: lane,
                right: col_pairs,
            },
            Span::default(),
        );
        let col0 = self.mul_literal_u32(expressions, col_pair, 2);
        let col1 = self.add_literal_u32(expressions, col0, 1);
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, row)),
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, col_pair)),
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, col0)),
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, col1)),
            Span::default(),
        );

        let (acc0_index, acc0_index_emits) =
            self.layout_index_expr(expressions, acc_layout, &[row, col0])?;
        let (acc1_index, acc1_index_emits) =
            self.layout_index_expr(expressions, acc_layout, &[row, col1])?;
        Self::push_emits(&mut body, acc0_index_emits);
        Self::push_emits(&mut body, acc1_index_emits);
        let (acc0_pointer, acc0_pointer_emits) =
            self.tile_dynamic_pointer(expressions, op.acc, acc0_index)?;
        let (acc1_pointer, acc1_pointer_emits) =
            self.tile_dynamic_pointer(expressions, op.acc, acc1_index)?;
        Self::push_emits(&mut body, acc0_pointer_emits);
        Self::push_emits(&mut body, acc1_pointer_emits);
        let acc0_value = expressions.append(
            Expression::Load {
                pointer: acc0_pointer,
            },
            Span::default(),
        );
        let acc1_value = expressions.append(
            Expression::Load {
                pointer: acc1_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::range_from(expressions, acc0_value, acc1_value)),
            Span::default(),
        );
        let sum0_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum1_pointer = expressions.append(
            Expression::LocalVariable(scratch.mma_sum_1),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: sum0_pointer,
                value: acc0_value,
            },
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: sum1_pointer,
                value: acc1_value,
            },
            Span::default(),
        );

        let mut k_body = Block::new();
        let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        k_body.push(Statement::Emit(k_emit), Span::default());
        let mut emits = Vec::new();
        let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
        Self::push_emits(&mut k_body, emits);

        let mut a_values = Vec::with_capacity(4);
        let mut b0_values = Vec::with_capacity(4);
        let mut b1_values = Vec::with_capacity(4);
        for lane in 0..4 {
            let mut lane_emits = Vec::new();
            let k = self.add_literal_u32_emitted(expressions, base_k, lane, &mut lane_emits);
            let (a_index, a_index_emits) =
                self.layout_index_expr(expressions, a_layout, &[row, k])?;
            let (b0_index, b0_index_emits) =
                self.layout_index_expr(expressions, b_layout, &[k, col0])?;
            let (b1_index, b1_index_emits) =
                self.layout_index_expr(expressions, b_layout, &[k, col1])?;
            lane_emits.extend(a_index_emits);
            lane_emits.extend(b0_index_emits);
            lane_emits.extend(b1_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.a, a_index)?;
            let (b0_pointer, b0_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.b, b0_index)?;
            let (b1_pointer, b1_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.b, b1_index)?;
            lane_emits.extend(a_pointer_emits);
            lane_emits.extend(b0_pointer_emits);
            lane_emits.extend(b1_pointer_emits);
            Self::push_emits(&mut k_body, lane_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            let b0_value = expressions.append(
                Expression::Load {
                    pointer: b0_pointer,
                },
                Span::default(),
            );
            let b1_value = expressions.append(
                Expression::Load {
                    pointer: b1_pointer,
                },
                Span::default(),
            );
            k_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            k_body.push(
                Statement::Emit(Self::single_expression_range(expressions, b0_value)),
                Span::default(),
            );
            k_body.push(
                Statement::Emit(Self::single_expression_range(expressions, b1_value)),
                Span::default(),
            );
            a_values.push(a_value);
            b0_values.push(b0_value);
            b1_values.push(b1_value);
        }

        let a_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: a_values,
            },
            Span::default(),
        );
        let b0_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: b0_values,
            },
            Span::default(),
        );
        let b1_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: b1_values,
            },
            Span::default(),
        );
        let dot0 = expressions.append(
            Expression::Math {
                fun: MathFunction::Dot,
                arg: a_vec,
                arg1: Some(b0_vec),
                arg2: None,
                arg3: None,
            },
            Span::default(),
        );
        let dot1 = expressions.append(
            Expression::Math {
                fun: MathFunction::Dot,
                arg: a_vec,
                arg1: Some(b1_vec),
                arg2: None,
                arg3: None,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_vec)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b0_vec)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b1_vec)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, dot0)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, dot1)),
            Span::default(),
        );

        let sum0_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum1_pointer = expressions.append(
            Expression::LocalVariable(scratch.mma_sum_1),
            Span::default(),
        );
        let sum0_value = expressions.append(
            Expression::Load {
                pointer: sum0_pointer,
            },
            Span::default(),
        );
        let sum1_value = expressions.append(
            Expression::Load {
                pointer: sum1_pointer,
            },
            Span::default(),
        );
        let value0 = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: sum0_value,
                right: dot0,
            },
            Span::default(),
        );
        let value1 = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: sum1_value,
                right: dot1,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::range_from(expressions, sum0_value, value1)),
            Span::default(),
        );
        k_body.push(
            Statement::Store {
                pointer: sum0_pointer,
                value: value0,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Store {
                pointer: sum1_pointer,
                value: value1,
            },
            Span::default(),
        );

        body.push(
            self.counted_loop(expressions, scratch.mma_k, k_a / 4, k_body),
            Span::default(),
        );

        let sum0_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum1_pointer = expressions.append(
            Expression::LocalVariable(scratch.mma_sum_1),
            Span::default(),
        );
        let sum0_value = expressions.append(
            Expression::Load {
                pointer: sum0_pointer,
            },
            Span::default(),
        );
        let sum1_value = expressions.append(
            Expression::Load {
                pointer: sum1_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::range_from(expressions, sum0_value, sum1_value)),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: acc0_pointer,
                value: sum0_value,
            },
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: acc1_pointer,
                value: sum1_value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(expressions, scratch.linear_index, lanes, body))
    }

    fn lower_gemm_to_storage(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &GemmOp,
        dst: &StorageView,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.tile_layout(op.a)?;
        let b_layout = self.tile_layout(op.b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;

        if k_a != k_b || m != m_acc || n != n_acc {
            return Err(LowerError::UnsupportedOperation("gemm shape mismatch"));
        }
        if acc_layout.shape() != dst_layout.shape() {
            return Err(LowerError::UnsupportedOperation("store shape mismatch"));
        }
        if acc_layout.memory_level() != MemoryLevel::Private {
            return Err(LowerError::UnsupportedMemoryLevel(
                acc_layout.memory_level(),
            ));
        }

        let mut body = Block::new();
        let (flat, flat_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(flat_emit), Span::default());

        let cols = expressions.append(Expression::Literal(Literal::U32(n)), Span::default());
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: flat,
                right: cols,
            },
            Span::default(),
        );
        let col = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: flat,
                right: cols,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::range_from(expressions, row, col)),
            Span::default(),
        );

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        body.push(
            Statement::Store {
                pointer: sum_pointer,
                value: zero,
            },
            Span::default(),
        );

        let mut k_body = Block::new();
        let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        k_body.push(Statement::Emit(k_emit), Span::default());

        let (a_index, a_index_emits) = self.layout_index_expr(expressions, a_layout, &[row, k])?;
        let (b_index, b_index_emits) = self.layout_index_expr(expressions, b_layout, &[k, col])?;
        Self::push_emits(&mut k_body, a_index_emits);
        Self::push_emits(&mut k_body, b_index_emits);

        let (a_pointer, a_pointer_emits) = self.tile_dynamic_pointer(expressions, op.a, a_index)?;
        let (b_pointer, b_pointer_emits) = self.tile_dynamic_pointer(expressions, op.b, b_index)?;
        Self::push_emits(&mut k_body, a_pointer_emits);
        Self::push_emits(&mut k_body, b_pointer_emits);

        let a_value = expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
        let b_value = expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum_value = expressions.append(
            Expression::Load {
                pointer: sum_pointer,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_value)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b_value)),
            Span::default(),
        );
        let value = expressions.append(
            Expression::Math {
                fun: MathFunction::Fma,
                arg: a_value,
                arg1: Some(b_value),
                arg2: Some(sum_value),
                arg3: None,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        k_body.push(
            Statement::Store {
                pointer: sum_pointer,
                value,
            },
            Span::default(),
        );

        body.push(
            self.counted_loop(expressions, scratch.mma_k, k_a, k_body),
            Span::default(),
        );

        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum_value = expressions.append(
            Expression::Load {
                pointer: sum_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, sum_value)),
            Span::default(),
        );

        let (dst_index, dst_index_emits) =
            self.storage_index_from_coords(expressions, dst, &[row, col])?;
        Self::push_emits(&mut body, dst_index_emits);
        let (dst_pointer, dst_pointer_emits) =
            self.storage_dynamic_pointer(expressions, dst, dst_index)?;
        Self::push_emits(&mut body, dst_pointer_emits);
        body.push(
            Statement::Store {
                pointer: dst_pointer,
                value: sum_value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(
            expressions,
            scratch.linear_index,
            acc_layout.element_count(),
            body,
        ))
    }

    #[allow(dead_code)]
    fn lower_storage_gemm_to_storage(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        op: &GemmOp,
        dst: &StorageView,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(a)?;
        let b_layout = self.storage_layout(b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;

        if k_a != k_b || m != m_acc || n != n_acc {
            return Err(LowerError::UnsupportedOperation("gemm shape mismatch"));
        }
        if acc_layout.shape() != dst_layout.shape() {
            return Err(LowerError::UnsupportedOperation("store shape mismatch"));
        }
        if acc_layout.memory_level() != MemoryLevel::Private {
            return Err(LowerError::UnsupportedMemoryLevel(
                acc_layout.memory_level(),
            ));
        }

        let mut body = Block::new();
        let (flat, flat_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(flat_emit), Span::default());

        let cols = expressions.append(Expression::Literal(Literal::U32(n)), Span::default());
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: flat,
                right: cols,
            },
            Span::default(),
        );
        let col = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: flat,
                right: cols,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::range_from(expressions, row, col)),
            Span::default(),
        );

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        body.push(
            Statement::Store {
                pointer: sum_pointer,
                value: zero,
            },
            Span::default(),
        );

        let mut k_body = Block::new();
        let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        k_body.push(Statement::Emit(k_emit), Span::default());

        let (a_index, a_index_emits) = self.storage_index_from_coords(expressions, a, &[row, k])?;
        let (b_index, b_index_emits) = self.storage_index_from_coords(expressions, b, &[k, col])?;
        Self::push_emits(&mut k_body, a_index_emits);
        Self::push_emits(&mut k_body, b_index_emits);

        let (a_pointer, a_pointer_emits) = self.storage_dynamic_pointer(expressions, a, a_index)?;
        let (b_pointer, b_pointer_emits) = self.storage_dynamic_pointer(expressions, b, b_index)?;
        Self::push_emits(&mut k_body, a_pointer_emits);
        Self::push_emits(&mut k_body, b_pointer_emits);

        let a_value = expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
        let b_value = expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum_value = expressions.append(
            Expression::Load {
                pointer: sum_pointer,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_value)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b_value)),
            Span::default(),
        );
        let value = expressions.append(
            Expression::Math {
                fun: MathFunction::Fma,
                arg: a_value,
                arg1: Some(b_value),
                arg2: Some(sum_value),
                arg3: None,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        k_body.push(
            Statement::Store {
                pointer: sum_pointer,
                value,
            },
            Span::default(),
        );

        body.push(
            self.counted_loop(expressions, scratch.mma_k, k_a, k_body),
            Span::default(),
        );

        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum_value = expressions.append(
            Expression::Load {
                pointer: sum_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, sum_value)),
            Span::default(),
        );

        let (dst_index, dst_index_emits) =
            self.storage_index_from_coords(expressions, dst, &[row, col])?;
        Self::push_emits(&mut body, dst_index_emits);
        let (dst_pointer, dst_pointer_emits) =
            self.storage_dynamic_pointer(expressions, dst, dst_index)?;
        Self::push_emits(&mut body, dst_pointer_emits);
        body.push(
            Statement::Store {
                pointer: dst_pointer,
                value: sum_value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(
            expressions,
            scratch.linear_index,
            acc_layout.element_count(),
            body,
        ))
    }

    fn lower_shared_gemm_loop_to_storage_4col(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a_load: &crate::CooperativeLoadOp,
        b_load: &crate::CooperativeLoadOp,
        op: &GemmOp,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Option<Statement>, LowerError> {
        let a_layout = self.tile_layout(op.a)?;
        let b_layout = self.tile_layout(op.b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;

        if k_a != k_b || m != m_acc || n != n_acc {
            return Err(LowerError::UnsupportedOperation("gemm shape mismatch"));
        }
        if acc_layout.shape() != dst_layout.shape() {
            return Err(LowerError::UnsupportedOperation("store shape mismatch"));
        }
        if a_layout.memory_level() != MemoryLevel::Workgroup
            || b_layout.memory_level() != MemoryLevel::Workgroup
            || acc_layout.memory_level() != MemoryLevel::Private
        {
            return Ok(None);
        }
        if a_load.dst != op.a || b_load.dst != op.b {
            return Ok(None);
        }
        if outer_iterations == 0 {
            return Err(LowerError::UnsupportedOperation(
                "gemm loop iteration count must be non-zero",
            ));
        }
        if n % 4 != 0 || k_a % 4 != 0 {
            return Ok(None);
        }

        let lanes = std::num::NonZeroU32::new(m * (n / 4)).ok_or(
            LowerError::UnsupportedOperation("empty shared 4-column gemm"),
        )?;
        if lanes.get() > self.workgroup_invocations {
            return Ok(None);
        }

        let sum2 = scratch.mma_sum_2.ok_or(LowerError::UnsupportedOperation(
            "missing gemm scratch local",
        ))?;
        let sum3 = scratch.mma_sum_3.ok_or(LowerError::UnsupportedOperation(
            "missing gemm scratch local",
        ))?;

        let mut body = Block::new();
        let lane = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let lane_limit = expressions.append(
            Expression::Literal(Literal::U32(lanes.get())),
            Span::default(),
        );
        let active = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Less,
                left: lane,
                right: lane_limit,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, active)),
            Span::default(),
        );

        let col_quads =
            expressions.append(Expression::Literal(Literal::U32(n / 4)), Span::default());
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: lane,
                right: col_quads,
            },
            Span::default(),
        );
        let col_quad = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: lane,
                right: col_quads,
            },
            Span::default(),
        );
        let col0 = self.mul_literal_u32(expressions, col_quad, 4);
        let col1 = self.add_literal_u32(expressions, col0, 1);
        let col2 = self.add_literal_u32(expressions, col0, 2);
        let col3 = self.add_literal_u32(expressions, col0, 3);
        for value in [row, col_quad, col0, col1, col2, col3] {
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
        }

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        let mut init_body = Block::new();
        for sum in [scratch.mma_sum, scratch.mma_sum_1, sum2, sum3] {
            let pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
            init_body.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::default(),
            );
        }
        body.push(
            Statement::If {
                condition: active,
                accept: init_body,
                reject: Block::new(),
            },
            Span::default(),
        );

        let mut inner_body = Block::new();
        let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        inner_body.push(Statement::Emit(k_emit), Span::default());
        let mut emits = Vec::new();
        let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
        Self::push_emits(&mut inner_body, emits);

        let mut a_values = Vec::with_capacity(4);
        let mut b0_values = Vec::with_capacity(4);
        let mut b1_values = Vec::with_capacity(4);
        let mut b2_values = Vec::with_capacity(4);
        let mut b3_values = Vec::with_capacity(4);
        for lane_index in 0..4 {
            let mut lane_emits = Vec::new();
            let k = self.add_literal_u32_emitted(expressions, base_k, lane_index, &mut lane_emits);
            let (a_index, a_index_emits) =
                self.layout_index_expr(expressions, a_layout, &[row, k])?;
            lane_emits.extend(a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.a, a_index)?;
            lane_emits.extend(a_pointer_emits);

            let mut b_pointers = Vec::with_capacity(4);
            for col in [col0, col1, col2, col3] {
                let (b_index, b_index_emits) =
                    self.layout_index_expr(expressions, b_layout, &[k, col])?;
                lane_emits.extend(b_index_emits);
                let (b_pointer, b_pointer_emits) =
                    self.tile_dynamic_pointer(expressions, op.b, b_index)?;
                lane_emits.extend(b_pointer_emits);
                b_pointers.push(b_pointer);
            }
            Self::push_emits(&mut inner_body, lane_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            let b0_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[0],
                },
                Span::default(),
            );
            let b1_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[1],
                },
                Span::default(),
            );
            let b2_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[2],
                },
                Span::default(),
            );
            let b3_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[3],
                },
                Span::default(),
            );
            for value in [a_value, b0_value, b1_value, b2_value, b3_value] {
                inner_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, value)),
                    Span::default(),
                );
            }
            a_values.push(a_value);
            b0_values.push(b0_value);
            b1_values.push(b1_value);
            b2_values.push(b2_value);
            b3_values.push(b3_value);
        }

        let a_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: a_values,
            },
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_vec)),
            Span::default(),
        );

        let mut dots = Vec::with_capacity(4);
        for components in [b0_values, b1_values, b2_values, b3_values] {
            let b_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components,
                },
                Span::default(),
            );
            let dot = expressions.append(
                Expression::Math {
                    fun: MathFunction::Dot,
                    arg: a_vec,
                    arg1: Some(b_vec),
                    arg2: None,
                    arg3: None,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, b_vec, dot)),
                Span::default(),
            );
            dots.push(dot);
        }

        for (sum, dot) in [scratch.mma_sum, scratch.mma_sum_1, sum2, sum3]
            .into_iter()
            .zip(dots)
        {
            let pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
            let current = expressions.append(Expression::Load { pointer }, Span::default());
            let next = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: current,
                    right: dot,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, current, next)),
                Span::default(),
            );
            inner_body.push(
                Statement::Store {
                    pointer,
                    value: next,
                },
                Span::default(),
            );
        }

        let mut outer_body = Block::new();
        outer_body.push(
            self.lower_cooperative_load(expressions, scratch.tile_index, a_load.dst, &a_load.src)?,
            Span::default(),
        );
        outer_body.push(
            self.lower_cooperative_load(expressions, scratch.tile_index, b_load.dst, &b_load.src)?,
            Span::default(),
        );
        outer_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        outer_body.push(
            Statement::If {
                condition: active,
                accept: Block::from_vec(vec![self.counted_loop(
                    expressions,
                    scratch.mma_k,
                    k_a / 4,
                    inner_body,
                )]),
                reject: Block::new(),
            },
            Span::default(),
        );
        outer_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations,
                outer_body,
            ),
            Span::default(),
        );

        let mut store_body = Block::new();
        for (sum, col) in [scratch.mma_sum, scratch.mma_sum_1, sum2, sum3]
            .into_iter()
            .zip([col0, col1, col2, col3])
        {
            let sum_pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            store_body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum_value)),
                Span::default(),
            );
            let (dst_index, dst_index_emits) =
                self.storage_index_from_coords(expressions, dst, &[row, col])?;
            Self::push_emits(&mut store_body, dst_index_emits);
            let (dst_pointer, dst_pointer_emits) =
                self.storage_dynamic_pointer(expressions, dst, dst_index)?;
            Self::push_emits(&mut store_body, dst_pointer_emits);
            store_body.push(
                Statement::Store {
                    pointer: dst_pointer,
                    value: sum_value,
                },
                Span::default(),
            );
        }
        body.push(
            Statement::If {
                condition: active,
                accept: store_body,
                reject: Block::new(),
            },
            Span::default(),
        );

        Ok(Some(Statement::Block(body)))
    }

    fn lower_storage_gemm_loop_to_storage(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        op: &GemmOp,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(a)?;
        let b_layout = self.storage_layout(b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;

        if k_a != k_b || m != m_acc || n != n_acc {
            return Err(LowerError::UnsupportedOperation("gemm shape mismatch"));
        }
        if acc_layout.shape() != dst_layout.shape() {
            return Err(LowerError::UnsupportedOperation("store shape mismatch"));
        }
        if acc_layout.memory_level() != MemoryLevel::Private {
            return Err(LowerError::UnsupportedMemoryLevel(
                acc_layout.memory_level(),
            ));
        }
        if outer_iterations == 0 {
            return Err(LowerError::UnsupportedOperation(
                "gemm loop iteration count must be non-zero",
            ));
        }
        if PREFER_COOP_MATRIX_GEMM
            && Self::can_lower_storage_gemm_coop8(
                a_layout,
                b_layout,
                acc_layout,
                dst_layout,
                outer_iterations,
            )
        {
            return self.lower_storage_gemm_loop_to_storage_coop8(
                expressions,
                scratch,
                a,
                b,
                dst,
                outer_iterations,
            );
        }
        if n % 8 == 0 && k_a % 4 == 0 {
            return self.lower_storage_gemm_loop_to_storage_widecol(
                expressions,
                scratch,
                a,
                b,
                dst,
                outer_iterations,
                8,
            );
        }
        if n % 4 == 0 && k_a % 4 == 0 {
            return self.lower_storage_gemm_loop_to_storage_4col(
                expressions,
                scratch,
                a,
                b,
                dst,
                outer_iterations,
            );
        }
        if n % 2 == 0 && k_a % 4 == 0 {
            return self.lower_storage_gemm_loop_to_storage_2col(
                expressions,
                scratch,
                a,
                b,
                dst,
                outer_iterations,
            );
        }

        let mut body = Block::new();
        let (flat, flat_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(flat_emit), Span::default());

        let cols = expressions.append(Expression::Literal(Literal::U32(n)), Span::default());
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: flat,
                right: cols,
            },
            Span::default(),
        );
        let col = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: flat,
                right: cols,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::range_from(expressions, row, col)),
            Span::default(),
        );

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        body.push(
            Statement::Store {
                pointer: sum_pointer,
                value: zero,
            },
            Span::default(),
        );

        let mut outer_body = Block::new();
        let (inner_body, inner_iterations) = if k_a % 4 == 0 {
            let mut inner_body = Block::new();
            let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
            inner_body.push(Statement::Emit(k_emit), Span::default());
            let mut emits = Vec::new();
            let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
            Self::push_emits(&mut inner_body, emits);

            let mut a_values = Vec::with_capacity(4);
            let mut b_values = Vec::with_capacity(4);
            for lane in 0..4 {
                let mut lane_emits = Vec::new();
                let k = self.add_literal_u32_emitted(expressions, base_k, lane, &mut lane_emits);
                let (a_index, a_index_emits) =
                    self.storage_index_from_coords(expressions, a, &[row, k])?;
                let (b_index, b_index_emits) =
                    self.storage_index_from_coords(expressions, b, &[k, col])?;
                lane_emits.extend(a_index_emits);
                lane_emits.extend(b_index_emits);
                let (a_pointer, a_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, a, a_index)?;
                let (b_pointer, b_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, b, b_index)?;
                lane_emits.extend(a_pointer_emits);
                lane_emits.extend(b_pointer_emits);
                Self::push_emits(&mut inner_body, lane_emits);

                let a_value =
                    expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
                let b_value =
                    expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
                inner_body.push(
                    Statement::Emit(Self::range_from(expressions, a_value, b_value)),
                    Span::default(),
                );
                a_values.push(a_value);
                b_values.push(b_value);
            }

            let a_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components: a_values,
                },
                Span::default(),
            );
            let b_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components: b_values,
                },
                Span::default(),
            );
            let dot = expressions.append(
                Expression::Math {
                    fun: MathFunction::Dot,
                    arg: a_vec,
                    arg1: Some(b_vec),
                    arg2: None,
                    arg3: None,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, a_vec, dot)),
                Span::default(),
            );

            let sum_pointer =
                expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            let value = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: sum_value,
                    right: dot,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, sum_value, value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value,
                },
                Span::default(),
            );
            (inner_body, k_a / 4)
        } else {
            let mut inner_body = Block::new();
            let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
            inner_body.push(Statement::Emit(k_emit), Span::default());

            let (a_index, a_index_emits) =
                self.storage_index_from_coords(expressions, a, &[row, k])?;
            let (b_index, b_index_emits) =
                self.storage_index_from_coords(expressions, b, &[k, col])?;
            Self::push_emits(&mut inner_body, a_index_emits);
            Self::push_emits(&mut inner_body, b_index_emits);

            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, a, a_index)?;
            let (b_pointer, b_pointer_emits) =
                self.storage_dynamic_pointer(expressions, b, b_index)?;
            Self::push_emits(&mut inner_body, a_pointer_emits);
            Self::push_emits(&mut inner_body, b_pointer_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            let b_value =
                expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
            let sum_pointer =
                expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            let value = expressions.append(
                Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a_value,
                    arg1: Some(b_value),
                    arg2: Some(sum_value),
                    arg3: None,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, b_value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum_value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value,
                },
                Span::default(),
            );
            (inner_body, k_a)
        };

        outer_body.push(
            self.counted_loop(expressions, scratch.mma_k, inner_iterations, inner_body),
            Span::default(),
        );
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations,
                outer_body,
            ),
            Span::default(),
        );

        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum_value = expressions.append(
            Expression::Load {
                pointer: sum_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, sum_value)),
            Span::default(),
        );

        let (dst_index, dst_index_emits) =
            self.storage_index_from_coords(expressions, dst, &[row, col])?;
        Self::push_emits(&mut body, dst_index_emits);
        let (dst_pointer, dst_pointer_emits) =
            self.storage_dynamic_pointer(expressions, dst, dst_index)?;
        Self::push_emits(&mut body, dst_pointer_emits);
        body.push(
            Statement::Store {
                pointer: dst_pointer,
                value: sum_value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(
            expressions,
            scratch.linear_index,
            acc_layout.element_count(),
            body,
        ))
    }

    fn lower_storage_gemm_loop_to_storage_2col(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(a)?;
        let b_layout = self.storage_layout(b)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_dst, n_dst] = Self::matrix_shape(dst_layout)?;
        if k_a != k_b || m != m_dst || n != n_dst || n % 2 != 0 || k_a % 4 != 0 {
            return Err(LowerError::UnsupportedOperation(
                "2-column gemm shape mismatch",
            ));
        }
        let lanes = std::num::NonZeroU32::new(m * (n / 2))
            .ok_or(LowerError::UnsupportedOperation("empty 2-column gemm"))?;

        let mut body = Block::new();
        let (lane, lane_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(lane_emit), Span::default());

        let col_pairs =
            expressions.append(Expression::Literal(Literal::U32(n / 2)), Span::default());
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: lane,
                right: col_pairs,
            },
            Span::default(),
        );
        let col_pair = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: lane,
                right: col_pairs,
            },
            Span::default(),
        );
        let col0 = self.mul_literal_u32(expressions, col_pair, 2);
        let col1 = self.add_literal_u32(expressions, col0, 1);
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, row)),
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, col_pair)),
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, col0)),
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, col1)),
            Span::default(),
        );

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        let sum0_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum1_pointer = expressions.append(
            Expression::LocalVariable(scratch.mma_sum_1),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: sum0_pointer,
                value: zero,
            },
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: sum1_pointer,
                value: zero,
            },
            Span::default(),
        );

        let mut inner_body = Block::new();
        let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        inner_body.push(Statement::Emit(k_emit), Span::default());
        let mut emits = Vec::new();
        let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
        Self::push_emits(&mut inner_body, emits);

        let mut a_values = Vec::with_capacity(4);
        let mut b0_values = Vec::with_capacity(4);
        let mut b1_values = Vec::with_capacity(4);
        for lane_index in 0..4 {
            let mut lane_emits = Vec::new();
            let k = self.add_literal_u32_emitted(expressions, base_k, lane_index, &mut lane_emits);
            let (a_index, a_index_emits) =
                self.storage_index_from_coords(expressions, a, &[row, k])?;
            let (b0_index, b0_index_emits) =
                self.storage_index_from_coords(expressions, b, &[k, col0])?;
            let (b1_index, b1_index_emits) =
                self.storage_index_from_coords(expressions, b, &[k, col1])?;
            lane_emits.extend(a_index_emits);
            lane_emits.extend(b0_index_emits);
            lane_emits.extend(b1_index_emits);

            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, a, a_index)?;
            let (b0_pointer, b0_pointer_emits) =
                self.storage_dynamic_pointer(expressions, b, b0_index)?;
            let (b1_pointer, b1_pointer_emits) =
                self.storage_dynamic_pointer(expressions, b, b1_index)?;
            lane_emits.extend(a_pointer_emits);
            lane_emits.extend(b0_pointer_emits);
            lane_emits.extend(b1_pointer_emits);
            Self::push_emits(&mut inner_body, lane_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            let b0_value = expressions.append(
                Expression::Load {
                    pointer: b0_pointer,
                },
                Span::default(),
            );
            let b1_value = expressions.append(
                Expression::Load {
                    pointer: b1_pointer,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, b0_value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, b1_value)),
                Span::default(),
            );
            a_values.push(a_value);
            b0_values.push(b0_value);
            b1_values.push(b1_value);
        }

        let a_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: a_values,
            },
            Span::default(),
        );
        let b0_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: b0_values,
            },
            Span::default(),
        );
        let b1_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: b1_values,
            },
            Span::default(),
        );
        let dot0 = expressions.append(
            Expression::Math {
                fun: MathFunction::Dot,
                arg: a_vec,
                arg1: Some(b0_vec),
                arg2: None,
                arg3: None,
            },
            Span::default(),
        );
        let dot1 = expressions.append(
            Expression::Math {
                fun: MathFunction::Dot,
                arg: a_vec,
                arg1: Some(b1_vec),
                arg2: None,
                arg3: None,
            },
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_vec)),
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b0_vec)),
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b1_vec)),
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, dot0)),
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, dot1)),
            Span::default(),
        );

        let sum0_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum1_pointer = expressions.append(
            Expression::LocalVariable(scratch.mma_sum_1),
            Span::default(),
        );
        let sum0_value = expressions.append(
            Expression::Load {
                pointer: sum0_pointer,
            },
            Span::default(),
        );
        let sum1_value = expressions.append(
            Expression::Load {
                pointer: sum1_pointer,
            },
            Span::default(),
        );
        let value0 = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: sum0_value,
                right: dot0,
            },
            Span::default(),
        );
        let value1 = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: sum1_value,
                right: dot1,
            },
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::range_from(expressions, sum0_value, value1)),
            Span::default(),
        );
        inner_body.push(
            Statement::Store {
                pointer: sum0_pointer,
                value: value0,
            },
            Span::default(),
        );
        inner_body.push(
            Statement::Store {
                pointer: sum1_pointer,
                value: value1,
            },
            Span::default(),
        );

        let mut outer_body = Block::new();
        outer_body.push(
            self.counted_loop(expressions, scratch.mma_k, k_a / 4, inner_body),
            Span::default(),
        );
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations,
                outer_body,
            ),
            Span::default(),
        );

        let sum0_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum1_pointer = expressions.append(
            Expression::LocalVariable(scratch.mma_sum_1),
            Span::default(),
        );
        let sum0_value = expressions.append(
            Expression::Load {
                pointer: sum0_pointer,
            },
            Span::default(),
        );
        let sum1_value = expressions.append(
            Expression::Load {
                pointer: sum1_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::range_from(expressions, sum0_value, sum1_value)),
            Span::default(),
        );

        let (dst0_index, dst0_index_emits) =
            self.storage_index_from_coords(expressions, dst, &[row, col0])?;
        let (dst1_index, dst1_index_emits) =
            self.storage_index_from_coords(expressions, dst, &[row, col1])?;
        Self::push_emits(&mut body, dst0_index_emits);
        Self::push_emits(&mut body, dst1_index_emits);
        let (dst0_pointer, dst0_pointer_emits) =
            self.storage_dynamic_pointer(expressions, dst, dst0_index)?;
        let (dst1_pointer, dst1_pointer_emits) =
            self.storage_dynamic_pointer(expressions, dst, dst1_index)?;
        Self::push_emits(&mut body, dst0_pointer_emits);
        Self::push_emits(&mut body, dst1_pointer_emits);
        body.push(
            Statement::Store {
                pointer: dst0_pointer,
                value: sum0_value,
            },
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: dst1_pointer,
                value: sum1_value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(expressions, scratch.linear_index, lanes, body))
    }

    fn lower_storage_gemm_loop_to_storage_4col(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(a)?;
        let b_layout = self.storage_layout(b)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_dst, n_dst] = Self::matrix_shape(dst_layout)?;
        if k_a != k_b || m != m_dst || n != n_dst || n % 4 != 0 || k_a % 4 != 0 {
            return Err(LowerError::UnsupportedOperation(
                "4-column gemm shape mismatch",
            ));
        }
        let sum2 = scratch.mma_sum_2.ok_or(LowerError::UnsupportedOperation(
            "missing gemm scratch local",
        ))?;
        let sum3 = scratch.mma_sum_3.ok_or(LowerError::UnsupportedOperation(
            "missing gemm scratch local",
        ))?;
        let lanes = std::num::NonZeroU32::new(m * (n / 4))
            .ok_or(LowerError::UnsupportedOperation("empty 4-column gemm"))?;

        let mut body = Block::new();
        let (lane, lane_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(lane_emit), Span::default());

        let col_quads =
            expressions.append(Expression::Literal(Literal::U32(n / 4)), Span::default());
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: lane,
                right: col_quads,
            },
            Span::default(),
        );
        let col_quad = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: lane,
                right: col_quads,
            },
            Span::default(),
        );
        let col0 = self.mul_literal_u32(expressions, col_quad, 4);
        let col1 = self.add_literal_u32(expressions, col0, 1);
        let col2 = self.add_literal_u32(expressions, col0, 2);
        let col3 = self.add_literal_u32(expressions, col0, 3);
        for value in [row, col_quad, col0, col1, col2, col3] {
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
        }

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        for sum in [scratch.mma_sum, scratch.mma_sum_1, sum2, sum3] {
            let pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
            body.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::default(),
            );
        }

        let mut inner_body = Block::new();
        let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        inner_body.push(Statement::Emit(k_emit), Span::default());
        let mut emits = Vec::new();
        let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
        Self::push_emits(&mut inner_body, emits);

        let mut a_values = Vec::with_capacity(4);
        let mut b0_values = Vec::with_capacity(4);
        let mut b1_values = Vec::with_capacity(4);
        let mut b2_values = Vec::with_capacity(4);
        let mut b3_values = Vec::with_capacity(4);
        for lane_index in 0..4 {
            let mut lane_emits = Vec::new();
            let k = self.add_literal_u32_emitted(expressions, base_k, lane_index, &mut lane_emits);
            let (a_index, a_index_emits) =
                self.storage_index_from_coords(expressions, a, &[row, k])?;
            lane_emits.extend(a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, a, a_index)?;
            lane_emits.extend(a_pointer_emits);

            let mut b_pointers = Vec::with_capacity(4);
            for col in [col0, col1, col2, col3] {
                let (b_index, b_index_emits) =
                    self.storage_index_from_coords(expressions, b, &[k, col])?;
                lane_emits.extend(b_index_emits);
                let (b_pointer, b_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, b, b_index)?;
                lane_emits.extend(b_pointer_emits);
                b_pointers.push(b_pointer);
            }
            Self::push_emits(&mut inner_body, lane_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            let b0_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[0],
                },
                Span::default(),
            );
            let b1_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[1],
                },
                Span::default(),
            );
            let b2_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[2],
                },
                Span::default(),
            );
            let b3_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[3],
                },
                Span::default(),
            );
            for value in [a_value, b0_value, b1_value, b2_value, b3_value] {
                inner_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, value)),
                    Span::default(),
                );
            }
            a_values.push(a_value);
            b0_values.push(b0_value);
            b1_values.push(b1_value);
            b2_values.push(b2_value);
            b3_values.push(b3_value);
        }

        let a_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: a_values,
            },
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_vec)),
            Span::default(),
        );

        let mut dots = Vec::with_capacity(4);
        for components in [b0_values, b1_values, b2_values, b3_values] {
            let b_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components,
                },
                Span::default(),
            );
            let dot = expressions.append(
                Expression::Math {
                    fun: MathFunction::Dot,
                    arg: a_vec,
                    arg1: Some(b_vec),
                    arg2: None,
                    arg3: None,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, b_vec, dot)),
                Span::default(),
            );
            dots.push(dot);
        }

        for (sum, dot) in [scratch.mma_sum, scratch.mma_sum_1, sum2, sum3]
            .into_iter()
            .zip(dots)
        {
            let pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
            let current = expressions.append(Expression::Load { pointer }, Span::default());
            let next = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: current,
                    right: dot,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, current, next)),
                Span::default(),
            );
            inner_body.push(
                Statement::Store {
                    pointer,
                    value: next,
                },
                Span::default(),
            );
        }

        let mut outer_body = Block::new();
        outer_body.push(
            self.counted_loop(expressions, scratch.mma_k, k_a / 4, inner_body),
            Span::default(),
        );
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations,
                outer_body,
            ),
            Span::default(),
        );

        for (sum, col) in [scratch.mma_sum, scratch.mma_sum_1, sum2, sum3]
            .into_iter()
            .zip([col0, col1, col2, col3])
        {
            let sum_pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum_value)),
                Span::default(),
            );
            let (dst_index, dst_index_emits) =
                self.storage_index_from_coords(expressions, dst, &[row, col])?;
            Self::push_emits(&mut body, dst_index_emits);
            let (dst_pointer, dst_pointer_emits) =
                self.storage_dynamic_pointer(expressions, dst, dst_index)?;
            Self::push_emits(&mut body, dst_pointer_emits);
            body.push(
                Statement::Store {
                    pointer: dst_pointer,
                    value: sum_value,
                },
                Span::default(),
            );
        }

        Ok(self.distributed_index_loop(expressions, scratch.linear_index, lanes, body))
    }

    fn lower_shared_gemm_loop_to_storage_coop8(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a_load: &crate::CooperativeLoadOp,
        b_load: &crate::CooperativeLoadOp,
        op: &GemmOp,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Statement, LowerError> {
        let _coop_c_ty = self.coop_f32_c_ty.ok_or(LowerError::UnsupportedOperation(
            "cooperative matrix C type was not allocated",
        ))?;

        if a_load.dst != op.a || b_load.dst != op.b {
            return Err(LowerError::UnsupportedOperation(
                "shared cooperative gemm load mismatch",
            ));
        }

        let a_layout = self.tile_layout(op.a)?;
        let b_layout = self.tile_layout(op.b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;
        let [m_dst, n_dst] = Self::matrix_shape(dst_layout)?;
        let (subgroup_rows, subgroup_cols, partition) = if self.coop_subgroups > 1 {
            if m == 64 && n == 64 && self.coop_subgroups == 4 {
                (
                    32,
                    32,
                    CoopPartition::InterleavedGrid {
                        row_groups: 2,
                        col_groups: 2,
                    },
                )
            } else if n >= m && n % self.coop_subgroups == 0 {
                (m, n / self.coop_subgroups, CoopPartition::Columns)
            } else if m % self.coop_subgroups == 0 {
                (m / self.coop_subgroups, n, CoopPartition::Rows)
            } else if n % self.coop_subgroups == 0 {
                (m, n / self.coop_subgroups, CoopPartition::Columns)
            } else {
                return Err(LowerError::UnsupportedOperation(
                    "cooperative matrix multi-subgroup tile width mismatch",
                ));
            }
        } else {
            (m, n, CoopPartition::Single)
        };
        if m == 0
            || n == 0
            || m % 8 != 0
            || n % 8 != 0
            || subgroup_rows % 8 != 0
            || subgroup_cols % 8 != 0
            || (subgroup_rows / 8) * (subgroup_cols / 8) > scratch.coop_accs.len() as u32
            || m_acc != m
            || n_acc != n
            || m_dst != m
            || n_dst != n
            || k_a != k_b
            || k_a % 8 != 0
            || a_layout.memory_level() != MemoryLevel::Workgroup
            || b_layout.memory_level() != MemoryLevel::Workgroup
            || acc_layout.memory_level() != MemoryLevel::Private
        {
            return Err(LowerError::UnsupportedOperation(
                "shared cooperative matrix lowering requires workgroup A/B tiles, a private accumulator, 8x8 fragments, and K divisible by 8",
            ));
        }
        if outer_iterations == 0 {
            return Err(LowerError::UnsupportedOperation(
                "cooperative matrix gemm loop iteration count must be non-zero",
            ));
        }

        let tile_rows = subgroup_rows / 8;
        let tile_cols = subgroup_cols / 8;
        let fragment_count = (tile_rows * tile_cols) as usize;
        let acc_locals = scratch.coop_accs[..fragment_count]
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .ok_or(LowerError::UnsupportedOperation(
                "cooperative matrix accumulator locals were not allocated",
            ))?;

        let a_stride = Self::row_major_matrix_leading_stride(a_layout)?;
        let b_stride = Self::row_major_matrix_leading_stride(b_layout)?;
        let dst_stride = Self::row_major_matrix_leading_stride(dst_layout)?;
        let a_stride =
            expressions.append(Expression::Literal(Literal::U32(a_stride)), Span::default());
        let b_stride =
            expressions.append(Expression::Literal(Literal::U32(b_stride)), Span::default());
        let dst_stride = expressions.append(
            Expression::Literal(Literal::U32(dst_stride)),
            Span::default(),
        );

        let mut body = Block::new();
        let mut acc_pointers = Vec::with_capacity(acc_locals.len());
        for local in acc_locals {
            let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
            acc_pointers.push(pointer);
        }
        let active_subgroup = if self.workgroup_invocations > self.coop_subgroups * 32 {
            let subgroup_id = expressions.append(
                Expression::FunctionArgument(SUBGROUP_ID_ARG),
                Span::default(),
            );
            let subgroup_limit = expressions.append(
                Expression::Literal(Literal::U32(self.coop_subgroups)),
                Span::default(),
            );
            let active = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Less,
                    left: subgroup_id,
                    right: subgroup_limit,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::range_from(expressions, subgroup_id, active)),
                Span::default(),
            );
            Some(active)
        } else {
            None
        };

        let mut inner_body = Block::new();
        let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        inner_body.push(Statement::Emit(k_emit), Span::default());
        let mut base_k_emits = Vec::new();
        let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 8, &mut base_k_emits);
        Self::push_emits(&mut inner_body, base_k_emits);
        self.append_shared_coop_k_chunk(
            expressions,
            &mut inner_body,
            op.a,
            op.b,
            &acc_pointers,
            tile_rows,
            tile_cols,
            base_k,
            a_stride,
            b_stride,
            subgroup_cols,
            subgroup_rows,
            partition,
        )?;

        let mut outer_body = Block::new();
        outer_body.push(
            self.lower_cooperative_load(expressions, scratch.tile_index, a_load.dst, &a_load.src)?,
            Span::default(),
        );
        outer_body.push(
            self.lower_cooperative_load(expressions, scratch.tile_index, b_load.dst, &b_load.src)?,
            Span::default(),
        );
        outer_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        let mma_loop = self.counted_loop(expressions, scratch.mma_k, k_a / 8, inner_body);
        if let Some(active_subgroup) = active_subgroup {
            outer_body.push(
                Statement::If {
                    condition: active_subgroup,
                    accept: Block::from_vec(vec![mma_loop]),
                    reject: Block::new(),
                },
                Span::default(),
            );
        } else {
            outer_body.push(mma_loop, Span::default());
        }
        outer_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations,
                outer_body,
            ),
            Span::default(),
        );

        let (subgroup_row_base, subgroup_col_base) = self.subgroup_partition_bases(
            expressions,
            &mut body,
            partition,
            subgroup_rows,
            subgroup_cols,
        );
        let mut store_body = Block::new();
        for row_tile in 0..tile_rows {
            for col_tile in 0..tile_cols {
                let acc_index = (row_tile * tile_cols + col_tile) as usize;
                let acc_value = expressions.append(
                    Expression::Load {
                        pointer: acc_pointers[acc_index],
                    },
                    Span::default(),
                );
                store_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, acc_value)),
                    Span::default(),
                );
                let row = if let Some(subgroup_row_base) = subgroup_row_base {
                    let mut row_emits = Vec::new();
                    let row = self.add_literal_u32_emitted(
                        expressions,
                        subgroup_row_base,
                        Self::coop_tile_offset(partition, true, row_tile),
                        &mut row_emits,
                    );
                    Self::push_emits(&mut store_body, row_emits);
                    row
                } else {
                    expressions.append(
                        Expression::Literal(Literal::U32(Self::coop_tile_offset(
                            partition, true, row_tile,
                        ))),
                        Span::default(),
                    )
                };
                let col = if let Some(subgroup_col_base) = subgroup_col_base {
                    let mut col_emits = Vec::new();
                    let col = self.add_literal_u32_emitted(
                        expressions,
                        subgroup_col_base,
                        Self::coop_tile_offset(partition, false, col_tile),
                        &mut col_emits,
                    );
                    Self::push_emits(&mut store_body, col_emits);
                    col
                } else {
                    expressions.append(
                        Expression::Literal(Literal::U32(Self::coop_tile_offset(
                            partition, false, col_tile,
                        ))),
                        Span::default(),
                    )
                };
                let (dst_index, dst_index_emits) =
                    self.storage_index_from_coords(expressions, dst, &[row, col])?;
                Self::push_emits(&mut store_body, dst_index_emits);
                let (dst_pointer, dst_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, dst, dst_index)?;
                Self::push_emits(&mut store_body, dst_pointer_emits);
                store_body.push(
                    Statement::CooperativeStore {
                        target: acc_value,
                        data: CooperativeData {
                            pointer: dst_pointer,
                            stride: dst_stride,
                            row_major: false,
                        },
                    },
                    Span::default(),
                );
            }
        }
        if let Some(active_subgroup) = active_subgroup {
            body.push(
                Statement::If {
                    condition: active_subgroup,
                    accept: store_body,
                    reject: Block::new(),
                },
                Span::default(),
            );
        } else {
            body.push(Statement::Block(store_body), Span::default());
        }

        Ok(Statement::Block(body))
    }

    fn lower_storage_gemm_loop_to_storage_coop8(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Statement, LowerError> {
        let _coop_a_ty = self.coop_f32_a_ty.ok_or(LowerError::UnsupportedOperation(
            "cooperative matrix A type was not allocated",
        ))?;
        let _coop_b_ty = self.coop_f32_b_ty.ok_or(LowerError::UnsupportedOperation(
            "cooperative matrix B type was not allocated",
        ))?;
        let _coop_c_ty = self.coop_f32_c_ty.ok_or(LowerError::UnsupportedOperation(
            "cooperative matrix C type was not allocated",
        ))?;

        let a_layout = self.storage_layout(a)?;
        let b_layout = self.storage_layout(b)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_dst, n_dst] = Self::matrix_shape(dst_layout)?;
        let (subgroup_rows, subgroup_cols, partition) = if self.coop_subgroups > 1 {
            if m == 64 && n == 64 && self.coop_subgroups == 4 {
                (
                    32,
                    32,
                    CoopPartition::InterleavedGrid {
                        row_groups: 2,
                        col_groups: 2,
                    },
                )
            } else if n >= m && n % self.coop_subgroups == 0 {
                (m, n / self.coop_subgroups, CoopPartition::Columns)
            } else if m % self.coop_subgroups == 0 {
                (m / self.coop_subgroups, n, CoopPartition::Rows)
            } else if n % self.coop_subgroups == 0 {
                (m, n / self.coop_subgroups, CoopPartition::Columns)
            } else {
                return Err(LowerError::UnsupportedOperation(
                    "cooperative matrix multi-subgroup tile shape mismatch",
                ));
            }
        } else {
            (m, n, CoopPartition::Single)
        };
        if m == 0
            || n == 0
            || m % COOP_MATRIX_DIM != 0
            || n % COOP_MATRIX_DIM != 0
            || subgroup_rows % COOP_MATRIX_DIM != 0
            || subgroup_cols % COOP_MATRIX_DIM != 0
            || (subgroup_rows / COOP_MATRIX_DIM) * (subgroup_cols / COOP_MATRIX_DIM)
                > scratch.coop_accs.len() as u32
            || m_dst != m
            || n_dst != n
            || k_a != k_b
            || k_a % COOP_MATRIX_DIM != 0
        {
            return Err(LowerError::UnsupportedOperation(
                "cooperative matrix lowering requires output tiles made of at most sixteen cooperative fragments and compatible K",
            ));
        }
        if outer_iterations == 0 {
            return Err(LowerError::UnsupportedOperation(
                "cooperative matrix gemm loop iteration count must be non-zero",
            ));
        }

        let tile_rows = subgroup_rows / COOP_MATRIX_DIM;
        let tile_cols = subgroup_cols / COOP_MATRIX_DIM;
        let fragment_count = (tile_rows * tile_cols) as usize;
        let acc_locals = scratch.coop_accs[..fragment_count]
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .ok_or(LowerError::UnsupportedOperation(
                "cooperative matrix accumulator locals were not allocated",
            ))?;

        let a_stride = Self::row_major_matrix_leading_stride(a_layout)?;
        let b_stride = Self::row_major_matrix_leading_stride(b_layout)?;
        let dst_stride = Self::row_major_matrix_leading_stride(dst_layout)?;
        let a_stride =
            expressions.append(Expression::Literal(Literal::U32(a_stride)), Span::default());
        let b_stride =
            expressions.append(Expression::Literal(Literal::U32(b_stride)), Span::default());
        let dst_stride = expressions.append(
            Expression::Literal(Literal::U32(dst_stride)),
            Span::default(),
        );

        let mut body = Block::new();
        let mut acc_pointers = Vec::with_capacity(acc_locals.len());
        for local in acc_locals {
            let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
            acc_pointers.push(pointer);
        }

        let k_chunks = k_a / COOP_MATRIX_DIM;
        let outer_unroll = COOP_MATRIX_OUTER_UNROLL.min(outer_iterations).max(1);
        let outer_unroll = (1..=outer_unroll)
            .rev()
            .find(|unroll| outer_iterations % unroll == 0)
            .unwrap_or(1);
        let (a_linear_base, a_linear_base_emits) = if PREFER_LINEAR_BASE_HOIST {
            self.storage_linear_base_without_loop_offsets(expressions, a)?
        } else {
            (None, Vec::new())
        };
        Self::push_emits(&mut body, a_linear_base_emits);
        let (b_linear_base, b_linear_base_emits) = if PREFER_LINEAR_BASE_HOIST {
            self.storage_linear_base_without_loop_offsets(expressions, b)?
        } else {
            (None, Vec::new())
        };
        Self::push_emits(&mut body, b_linear_base_emits);
        let (dst_linear_base, dst_linear_base_emits) = if PREFER_LINEAR_BASE_HOIST {
            self.storage_linear_base_without_loop_offsets(expressions, dst)?
        } else {
            (None, Vec::new())
        };
        Self::push_emits(&mut body, dst_linear_base_emits);
        let mut outer_body = Block::new();
        let (loop_index, loop_emit) = self.load_u32_local(expressions, scratch.loop_index);
        outer_body.push(Statement::Emit(loop_emit), Span::default());
        let mut loop_base_emits = Vec::new();
        let loop_k_base = self.mul_literal_u32_emitted(
            expressions,
            loop_index,
            k_a * outer_unroll,
            &mut loop_base_emits,
        );
        Self::push_emits(&mut outer_body, loop_base_emits);
        if k_chunks <= 4 {
            for outer_chunk in 0..outer_unroll {
                for k_chunk in 0..k_chunks {
                    let mut base_k_emits = Vec::new();
                    let base_k = self.add_literal_u32_emitted(
                        expressions,
                        loop_k_base,
                        outer_chunk * k_a + k_chunk * COOP_MATRIX_DIM,
                        &mut base_k_emits,
                    );
                    Self::push_emits(&mut outer_body, base_k_emits);
                    self.append_coop_k_chunk(
                        expressions,
                        &mut outer_body,
                        a,
                        b,
                        &acc_pointers,
                        a_linear_base,
                        b_linear_base,
                        tile_rows,
                        tile_cols,
                        base_k,
                        a_stride,
                        b_stride,
                        subgroup_cols,
                        subgroup_rows,
                        partition,
                        true,
                    )?;
                }
            }
        } else {
            let mut inner_body = Block::new();
            let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
            inner_body.push(Statement::Emit(k_emit), Span::default());
            let mut base_k_emits = Vec::new();
            let inner_k = self.mul_literal_u32_emitted(
                expressions,
                k_chunk,
                COOP_MATRIX_DIM,
                &mut base_k_emits,
            );
            let base_k = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: loop_k_base,
                    right: inner_k,
                },
                Span::default(),
            );
            base_k_emits.push(Self::single_expression_range(expressions, base_k));
            Self::push_emits(&mut inner_body, base_k_emits);
            self.append_coop_k_chunk(
                expressions,
                &mut inner_body,
                a,
                b,
                &acc_pointers,
                a_linear_base,
                b_linear_base,
                tile_rows,
                tile_cols,
                base_k,
                a_stride,
                b_stride,
                subgroup_cols,
                subgroup_rows,
                partition,
                true,
            )?;
            outer_body.push(
                self.counted_loop(expressions, scratch.mma_k, k_chunks, inner_body),
                Span::default(),
            );
        }
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations / outer_unroll,
                outer_body,
            ),
            Span::default(),
        );

        let (subgroup_row_base, subgroup_col_base) = self.subgroup_partition_bases(
            expressions,
            &mut body,
            partition,
            subgroup_rows,
            subgroup_cols,
        );
        for row_tile in 0..tile_rows {
            for col_tile in 0..tile_cols {
                let acc_index = (row_tile * tile_cols + col_tile) as usize;
                let acc_value = expressions.append(
                    Expression::Load {
                        pointer: acc_pointers[acc_index],
                    },
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, acc_value)),
                    Span::default(),
                );
                let row = if let Some(subgroup_row_base) = subgroup_row_base {
                    let mut row_emits = Vec::new();
                    let row = self.add_literal_u32_emitted(
                        expressions,
                        subgroup_row_base,
                        Self::coop_tile_offset(partition, true, row_tile),
                        &mut row_emits,
                    );
                    Self::push_emits(&mut body, row_emits);
                    row
                } else {
                    expressions.append(
                        Expression::Literal(Literal::U32(Self::coop_tile_offset(
                            partition, true, row_tile,
                        ))),
                        Span::default(),
                    )
                };
                let col = if let Some(subgroup_col_base) = subgroup_col_base {
                    let mut col_emits = Vec::new();
                    let col = self.add_literal_u32_emitted(
                        expressions,
                        subgroup_col_base,
                        Self::coop_tile_offset(partition, false, col_tile),
                        &mut col_emits,
                    );
                    Self::push_emits(&mut body, col_emits);
                    col
                } else {
                    expressions.append(
                        Expression::Literal(Literal::U32(Self::coop_tile_offset(
                            partition, false, col_tile,
                        ))),
                        Span::default(),
                    )
                };
                let (dst_index, dst_index_emits) = if PREFER_LINEAR_BASE_HOIST {
                    let (dst_index, mut dst_index_emits) =
                        self.layout_index_expr(expressions, dst_layout, &[row, col])?;
                    let dst_index = self.add_optional_base_u32_emitted(
                        expressions,
                        dst_index,
                        dst_linear_base,
                        &mut dst_index_emits,
                    );
                    (dst_index, dst_index_emits)
                } else {
                    self.storage_index_from_coords(expressions, dst, &[row, col])?
                };
                Self::push_emits(&mut body, dst_index_emits);
                let (dst_pointer, dst_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, dst, dst_index)?;
                Self::push_emits(&mut body, dst_pointer_emits);
                body.push(
                    Statement::CooperativeStore {
                        target: acc_value,
                        data: CooperativeData {
                            pointer: dst_pointer,
                            stride: dst_stride,
                            row_major: false,
                        },
                    },
                    Span::default(),
                );
            }
        }

        Ok(Statement::Block(body))
    }

    fn append_shared_coop_k_chunk(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        a: TileRef,
        b: TileRef,
        acc_pointers: &[Handle<Expression>],
        tile_rows: u32,
        tile_cols: u32,
        base_k: Handle<Expression>,
        a_stride: Handle<Expression>,
        b_stride: Handle<Expression>,
        subgroup_cols: u32,
        subgroup_rows: u32,
        partition: CoopPartition,
    ) -> Result<(), LowerError> {
        let a_layout = self.tile_layout(a)?;
        let b_layout = self.tile_layout(b)?;
        let (subgroup_row_base, subgroup_col_base) = self.subgroup_partition_bases(
            expressions,
            body,
            partition,
            subgroup_rows,
            subgroup_cols,
        );

        let mut a_fragments = Vec::with_capacity(tile_rows as usize);
        for row_tile in 0..tile_rows {
            let row = if let Some(subgroup_row_base) = subgroup_row_base {
                let mut row_emits = Vec::new();
                let row = self.add_literal_u32_emitted(
                    expressions,
                    subgroup_row_base,
                    Self::coop_tile_offset(partition, true, row_tile),
                    &mut row_emits,
                );
                Self::push_emits(body, row_emits);
                row
            } else {
                expressions.append(
                    Expression::Literal(Literal::U32(Self::coop_tile_offset(
                        partition, true, row_tile,
                    ))),
                    Span::default(),
                )
            };
            let (a_index, a_index_emits) =
                self.layout_index_expr(expressions, a_layout, &[row, base_k])?;
            Self::push_emits(body, a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.tile_dynamic_pointer(expressions, a, a_index)?;
            Self::push_emits(body, a_pointer_emits);
            let a_value = expressions.append(
                Expression::CooperativeLoad {
                    columns: CooperativeSize::Eight,
                    rows: CooperativeSize::Eight,
                    role: CooperativeRole::A,
                    data: CooperativeData {
                        pointer: a_pointer,
                        stride: a_stride,
                        row_major: false,
                    },
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            a_fragments.push(a_value);
        }

        let mut b_fragments = Vec::with_capacity(tile_cols as usize);
        for col_tile in 0..tile_cols {
            let col = if let Some(subgroup_col_base) = subgroup_col_base {
                let mut col_emits = Vec::new();
                let col = self.add_literal_u32_emitted(
                    expressions,
                    subgroup_col_base,
                    Self::coop_tile_offset(partition, false, col_tile),
                    &mut col_emits,
                );
                Self::push_emits(body, col_emits);
                col
            } else {
                expressions.append(
                    Expression::Literal(Literal::U32(Self::coop_tile_offset(
                        partition, false, col_tile,
                    ))),
                    Span::default(),
                )
            };
            let (b_index, b_index_emits) =
                self.layout_index_expr(expressions, b_layout, &[base_k, col])?;
            Self::push_emits(body, b_index_emits);
            let (b_pointer, b_pointer_emits) =
                self.tile_dynamic_pointer(expressions, b, b_index)?;
            Self::push_emits(body, b_pointer_emits);
            let b_value = expressions.append(
                Expression::CooperativeLoad {
                    columns: CooperativeSize::Eight,
                    rows: CooperativeSize::Eight,
                    role: CooperativeRole::B,
                    data: CooperativeData {
                        pointer: b_pointer,
                        stride: b_stride,
                        row_major: false,
                    },
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, b_value)),
                Span::default(),
            );
            b_fragments.push(b_value);
        }

        for row_tile in 0..tile_rows {
            for col_tile in 0..tile_cols {
                let acc_index = (row_tile * tile_cols + col_tile) as usize;
                let acc_pointer = acc_pointers[acc_index];
                let acc_value = expressions.append(
                    Expression::Load {
                        pointer: acc_pointer,
                    },
                    Span::default(),
                );
                let next_acc = expressions.append(
                    Expression::CooperativeMultiplyAdd {
                        a: a_fragments[row_tile as usize],
                        b: b_fragments[col_tile as usize],
                        c: acc_value,
                    },
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, acc_value)),
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, next_acc)),
                    Span::default(),
                );
                body.push(
                    Statement::Store {
                        pointer: acc_pointer,
                        value: next_acc,
                    },
                    Span::default(),
                );
            }
        }

        Ok(())
    }

    fn append_coop_k_chunk(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        a: &StorageView,
        b: &StorageView,
        acc_pointers: &[Handle<Expression>],
        a_linear_base: Option<Handle<Expression>>,
        b_linear_base: Option<Handle<Expression>>,
        tile_rows: u32,
        tile_cols: u32,
        base_k: Handle<Expression>,
        a_stride: Handle<Expression>,
        b_stride: Handle<Expression>,
        subgroup_cols: u32,
        subgroup_rows: u32,
        partition: CoopPartition,
        _skip_loop_offsets: bool,
    ) -> Result<(), LowerError> {
        let (subgroup_row_base, subgroup_col_base) = self.subgroup_partition_bases(
            expressions,
            body,
            partition,
            subgroup_rows,
            subgroup_cols,
        );
        let mut a_fragments = Vec::with_capacity(tile_rows as usize);
        for row_tile in 0..tile_rows {
            let row = if let Some(subgroup_row_base) = subgroup_row_base {
                let mut row_emits = Vec::new();
                let row = self.add_literal_u32_emitted(
                    expressions,
                    subgroup_row_base,
                    Self::coop_tile_offset(partition, true, row_tile),
                    &mut row_emits,
                );
                Self::push_emits(body, row_emits);
                row
            } else {
                expressions.append(
                    Expression::Literal(Literal::U32(Self::coop_tile_offset(
                        partition, true, row_tile,
                    ))),
                    Span::default(),
                )
            };
            let (a_index, a_index_emits) = if PREFER_LINEAR_BASE_HOIST {
                let (a_index, mut a_index_emits) =
                    self.layout_index_expr(expressions, self.storage_layout(a)?, &[row, base_k])?;
                let a_index = self.add_optional_base_u32_emitted(
                    expressions,
                    a_index,
                    a_linear_base,
                    &mut a_index_emits,
                );
                (a_index, a_index_emits)
            } else {
                self.storage_index_from_coords_without_loop_offsets(expressions, a, &[row, base_k])?
            };
            Self::push_emits(body, a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, a, a_index)?;
            Self::push_emits(body, a_pointer_emits);
            let a_value = expressions.append(
                Expression::CooperativeLoad {
                    columns: COOP_MATRIX_SIZE,
                    rows: COOP_MATRIX_SIZE,
                    role: CooperativeRole::A,
                    data: CooperativeData {
                        pointer: a_pointer,
                        stride: a_stride,
                        row_major: false,
                    },
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            a_fragments.push(a_value);
        }

        let mut b_fragments = Vec::with_capacity(tile_cols as usize);
        for col_tile in 0..tile_cols {
            let col = if let Some(subgroup_col_base) = subgroup_col_base {
                let mut col_emits = Vec::new();
                let col = self.add_literal_u32_emitted(
                    expressions,
                    subgroup_col_base,
                    Self::coop_tile_offset(partition, false, col_tile),
                    &mut col_emits,
                );
                Self::push_emits(body, col_emits);
                col
            } else {
                expressions.append(
                    Expression::Literal(Literal::U32(Self::coop_tile_offset(
                        partition, false, col_tile,
                    ))),
                    Span::default(),
                )
            };
            let (b_index, b_index_emits) = if PREFER_LINEAR_BASE_HOIST {
                let (b_index, mut b_index_emits) =
                    self.layout_index_expr(expressions, self.storage_layout(b)?, &[base_k, col])?;
                let b_index = self.add_optional_base_u32_emitted(
                    expressions,
                    b_index,
                    b_linear_base,
                    &mut b_index_emits,
                );
                (b_index, b_index_emits)
            } else {
                self.storage_index_from_coords_without_loop_offsets(expressions, b, &[base_k, col])?
            };
            Self::push_emits(body, b_index_emits);
            let (b_pointer, b_pointer_emits) =
                self.storage_dynamic_pointer(expressions, b, b_index)?;
            Self::push_emits(body, b_pointer_emits);
            let b_value = expressions.append(
                Expression::CooperativeLoad {
                    columns: COOP_MATRIX_SIZE,
                    rows: COOP_MATRIX_SIZE,
                    role: CooperativeRole::B,
                    data: CooperativeData {
                        pointer: b_pointer,
                        stride: b_stride,
                        row_major: false,
                    },
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, b_value)),
                Span::default(),
            );
            b_fragments.push(b_value);
        }

        for row_tile in 0..tile_rows {
            for col_tile in 0..tile_cols {
                let acc_index = (row_tile * tile_cols + col_tile) as usize;
                let acc_pointer = acc_pointers[acc_index];
                let acc_value = expressions.append(
                    Expression::Load {
                        pointer: acc_pointer,
                    },
                    Span::default(),
                );
                let next_acc = expressions.append(
                    Expression::CooperativeMultiplyAdd {
                        a: a_fragments[row_tile as usize],
                        b: b_fragments[col_tile as usize],
                        c: acc_value,
                    },
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, acc_value)),
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, next_acc)),
                    Span::default(),
                );
                body.push(
                    Statement::Store {
                        pointer: acc_pointer,
                        value: next_acc,
                    },
                    Span::default(),
                );
            }
        }

        Ok(())
    }

    fn lower_storage_gemm_loop_to_storage_widecol(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        dst: &StorageView,
        outer_iterations: u32,
        columns: u32,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(a)?;
        let b_layout = self.storage_layout(b)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_dst, n_dst] = Self::matrix_shape(dst_layout)?;
        if columns == 0
            || k_a != k_b
            || m != m_dst
            || n != n_dst
            || n % columns != 0
            || k_a % 4 != 0
        {
            return Err(LowerError::UnsupportedOperation(
                "wide-column gemm shape mismatch",
            ));
        }

        let lanes = std::num::NonZeroU32::new(m * (n / columns))
            .ok_or(LowerError::UnsupportedOperation("empty wide-column gemm"))?;
        let sum_locals = (0..columns)
            .map(|index| self.gemm_sum_local(scratch, index))
            .collect::<Result<Vec<_>, _>>()?;

        let mut body = Block::new();
        let (lane, lane_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(lane_emit), Span::default());

        let col_groups = expressions.append(
            Expression::Literal(Literal::U32(n / columns)),
            Span::default(),
        );
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: lane,
                right: col_groups,
            },
            Span::default(),
        );
        let col_group = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: lane,
                right: col_groups,
            },
            Span::default(),
        );
        let col0 = self.mul_literal_u32(expressions, col_group, columns);
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, row)),
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, col_group)),
            Span::default(),
        );

        let mut cols = Vec::with_capacity(columns as usize);
        for offset in 0..columns {
            let col = self.add_literal_u32(expressions, col0, offset);
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, col)),
                Span::default(),
            );
            cols.push(col);
        }

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        for sum in &sum_locals {
            let pointer = expressions.append(Expression::LocalVariable(*sum), Span::default());
            body.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::default(),
            );
        }

        let mut inner_body = Block::new();
        let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        inner_body.push(Statement::Emit(k_emit), Span::default());
        let mut emits = Vec::new();
        let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
        Self::push_emits(&mut inner_body, emits);

        let mut a_values = Vec::with_capacity(4);
        let mut b_values_by_col: Vec<Vec<Handle<Expression>>> =
            (0..columns).map(|_| Vec::with_capacity(4)).collect();
        for lane_index in 0..4 {
            let mut lane_emits = Vec::new();
            let k = self.add_literal_u32_emitted(expressions, base_k, lane_index, &mut lane_emits);
            let (a_index, a_index_emits) =
                self.storage_index_from_coords(expressions, a, &[row, k])?;
            lane_emits.extend(a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, a, a_index)?;
            lane_emits.extend(a_pointer_emits);

            let mut b_pointers = Vec::with_capacity(columns as usize);
            for col in &cols {
                let (b_index, b_index_emits) =
                    self.storage_index_from_coords(expressions, b, &[k, *col])?;
                lane_emits.extend(b_index_emits);
                let (b_pointer, b_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, b, b_index)?;
                lane_emits.extend(b_pointer_emits);
                b_pointers.push(b_pointer);
            }
            Self::push_emits(&mut inner_body, lane_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            a_values.push(a_value);

            for (col_index, b_pointer) in b_pointers.into_iter().enumerate() {
                let b_value =
                    expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
                inner_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, b_value)),
                    Span::default(),
                );
                b_values_by_col[col_index].push(b_value);
            }
        }

        let a_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: a_values,
            },
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_vec)),
            Span::default(),
        );

        for (sum, components) in sum_locals.iter().copied().zip(b_values_by_col) {
            let b_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components,
                },
                Span::default(),
            );
            let dot = expressions.append(
                Expression::Math {
                    fun: MathFunction::Dot,
                    arg: a_vec,
                    arg1: Some(b_vec),
                    arg2: None,
                    arg3: None,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, b_vec, dot)),
                Span::default(),
            );
            let pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
            let current = expressions.append(Expression::Load { pointer }, Span::default());
            let next = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: current,
                    right: dot,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, current, next)),
                Span::default(),
            );
            inner_body.push(
                Statement::Store {
                    pointer,
                    value: next,
                },
                Span::default(),
            );
        }

        let mut outer_body = Block::new();
        outer_body.push(
            self.counted_loop(expressions, scratch.mma_k, k_a / 4, inner_body),
            Span::default(),
        );
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations,
                outer_body,
            ),
            Span::default(),
        );

        for (sum, col) in sum_locals.into_iter().zip(cols) {
            let sum_pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum_value)),
                Span::default(),
            );
            let (dst_index, dst_index_emits) =
                self.storage_index_from_coords(expressions, dst, &[row, col])?;
            Self::push_emits(&mut body, dst_index_emits);
            let (dst_pointer, dst_pointer_emits) =
                self.storage_dynamic_pointer(expressions, dst, dst_index)?;
            Self::push_emits(&mut body, dst_pointer_emits);
            body.push(
                Statement::Store {
                    pointer: dst_pointer,
                    value: sum_value,
                },
                Span::default(),
            );
        }

        Ok(self.distributed_index_loop(expressions, scratch.linear_index, lanes, body))
    }

    fn lower_gemv(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &GemvOp,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(&op.a)?;
        let x_layout = self.storage_layout(&op.x)?;
        let y_layout = self.storage_layout(&op.y)?;
        let partial_layout = self.tile_layout(op.partials)?;
        let [m, k] = Self::matrix_shape(a_layout)?;
        let [x_k, x_cols] = Self::matrix_shape(x_layout)?;
        let [y_m, y_cols] = Self::matrix_shape(y_layout)?;

        if k != x_k || x_cols != 1 || m != y_m || y_cols != 1 {
            return Err(LowerError::UnsupportedOperation("gemv shape mismatch"));
        }
        if op.rows_per_workgroup == 0 || op.rows_per_workgroup > 4 {
            return Err(LowerError::UnsupportedOperation(
                "gemv rows per workgroup must be between 1 and 4",
            ));
        }
        if m % op.rows_per_workgroup != 0 {
            return Err(LowerError::UnsupportedOperation(
                "gemv rows per workgroup must divide M",
            ));
        }
        if partial_layout.memory_level() != MemoryLevel::Workgroup {
            return Err(LowerError::UnsupportedMemoryLevel(
                partial_layout.memory_level(),
            ));
        }
        if partial_layout.shape().rank() != 1
            || partial_layout.element_count().get()
                != self.workgroup_invocations * op.rows_per_workgroup
        {
            return Err(LowerError::UnsupportedOperation(
                "gemv partials must match the selected workgroup size",
            ));
        }
        if op.vector_width == 0 {
            return Err(LowerError::UnsupportedOperation(
                "gemv vector width must be non-zero",
            ));
        }

        let mut body = Block::new();
        let workgroup_id = expressions.append(
            Expression::FunctionArgument(WORKGROUP_ID_ARG),
            Span::default(),
        );
        let row = expressions.append(
            Expression::AccessIndex {
                base: workgroup_id,
                index: 0,
            },
            Span::default(),
        );
        let mut row_emits = Vec::new();
        let row_base =
            self.mul_literal_u32_emitted(expressions, row, op.rows_per_workgroup, &mut row_emits);
        if row_emits.is_empty() {
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, row_base)),
                Span::default(),
            );
        } else {
            Self::push_emits(&mut body, row_emits);
        }

        let row_limit = expressions.append(Expression::Literal(Literal::U32(m)), Span::default());
        let row_done = expressions.append(
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: row_base,
                right: row_limit,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, row_done)),
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: row_done,
                accept: Block::from_vec(vec![Statement::Return { value: None }]),
                reject: Block::new(),
            },
            Span::default(),
        );

        for row_offset in 0..op.rows_per_workgroup {
            let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
            let sum_pointer = expressions.append(
                Expression::LocalVariable(self.gemv_sum_local(scratch, row_offset)?),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value: zero,
                },
                Span::default(),
            );
        }

        let local_invocation = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let mut start_emits = Vec::new();
        let k_start = self.mul_literal_u32_emitted(
            expressions,
            local_invocation,
            op.vector_width,
            &mut start_emits,
        );
        Self::push_emits(&mut body, start_emits);
        let k_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_k), Span::default());
        body.push(
            Statement::Store {
                pointer: k_pointer,
                value: k_start,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let (k_index, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        loop_body.push(Statement::Emit(k_emit), Span::default());
        let k_limit = expressions.append(Expression::Literal(Literal::U32(k)), Span::default());
        let k_done = expressions.append(
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: k_index,
                right: k_limit,
            },
            Span::default(),
        );
        loop_body.push(
            Statement::Emit(Self::single_expression_range(expressions, k_done)),
            Span::default(),
        );
        loop_body.push(
            Statement::If {
                condition: k_done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let needs_tail_checks = k % op.vector_width != 0;
        for lane in 0..op.vector_width {
            let mut lane_emits = Vec::new();
            let k_lane = self.add_literal_u32_emitted(expressions, k_index, lane, &mut lane_emits);
            Self::push_emits(&mut loop_body, lane_emits);
            let lane_body = self.lower_gemv_fma_lane(expressions, scratch, op, row_base, k_lane)?;
            if needs_tail_checks && lane != 0 {
                let k_limit =
                    expressions.append(Expression::Literal(Literal::U32(k)), Span::default());
                let in_bounds = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Less,
                        left: k_lane,
                        right: k_limit,
                    },
                    Span::default(),
                );
                loop_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, in_bounds)),
                    Span::default(),
                );
                loop_body.push(
                    Statement::If {
                        condition: in_bounds,
                        accept: lane_body,
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            } else {
                loop_body.push(Statement::Block(lane_body), Span::default());
            }
        }

        let stride = self
            .workgroup_invocations
            .checked_mul(op.vector_width)
            .ok_or(LowerError::UnsupportedOperation("gemv stride overflow"))?;
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::from_vec(vec![self.increment_u32_local(
                    expressions,
                    scratch.mma_k,
                    stride,
                )]),
                break_if: None,
            },
            Span::default(),
        );

        let local_invocation = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        for row_offset in 0..op.rows_per_workgroup {
            let mut partial_index_emits = Vec::new();
            let partial_index = self.add_literal_u32_emitted(
                expressions,
                local_invocation,
                row_offset * self.workgroup_invocations,
                &mut partial_index_emits,
            );
            Self::push_emits(&mut body, partial_index_emits);
            let (partial_pointer, partial_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.partials, partial_index)?;
            Self::push_emits(&mut body, partial_pointer_emits);
            let sum_pointer = expressions.append(
                Expression::LocalVariable(self.gemv_sum_local(scratch, row_offset)?),
                Span::default(),
            );
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum_value)),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: partial_pointer,
                    value: sum_value,
                },
                Span::default(),
            );
        }
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let mut reduce_stride = self.workgroup_invocations / 2;
        while reduce_stride > 1 {
            let local_invocation = expressions.append(
                Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
                Span::default(),
            );
            let limit = expressions.append(
                Expression::Literal(Literal::U32(reduce_stride)),
                Span::default(),
            );
            let participates = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Less,
                    left: local_invocation,
                    right: limit,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, participates)),
                Span::default(),
            );
            body.push(
                Statement::If {
                    condition: participates,
                    accept: self.lower_gemv_partial_add(
                        expressions,
                        op.partials,
                        local_invocation,
                        reduce_stride,
                        op.rows_per_workgroup,
                    )?,
                    reject: Block::new(),
                },
                Span::default(),
            );
            body.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            reduce_stride /= 2;
        }

        let local_invocation = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        let is_lane_zero = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Equal,
                left: local_invocation,
                right: zero,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, is_lane_zero)),
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: is_lane_zero,
                accept: self.lower_gemv_final_store(expressions, op, row_base)?,
                reject: Block::new(),
            },
            Span::default(),
        );

        Ok(Statement::Block(body))
    }

    fn lower_gemv_fma_lane(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &GemvOp,
        row_base: Handle<Expression>,
        k: Handle<Expression>,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();
        let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        let (x_index, x_index_emits) =
            self.storage_index_from_coords(expressions, &op.x, &[k, zero])?;
        Self::push_emits(&mut body, x_index_emits);
        let (x_pointer, x_pointer_emits) =
            self.storage_dynamic_pointer(expressions, &op.x, x_index)?;
        Self::push_emits(&mut body, x_pointer_emits);
        let x_value = expressions.append(Expression::Load { pointer: x_pointer }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, x_value)),
            Span::default(),
        );

        for row_offset in 0..op.rows_per_workgroup {
            let mut row_emits = Vec::new();
            let row =
                self.add_literal_u32_emitted(expressions, row_base, row_offset, &mut row_emits);
            Self::push_emits(&mut body, row_emits);
            let (a_index, a_index_emits) =
                self.storage_index_from_coords(expressions, &op.a, &[row, k])?;
            Self::push_emits(&mut body, a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, &op.a, a_index)?;
            Self::push_emits(&mut body, a_pointer_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            let sum_pointer = expressions.append(
                Expression::LocalVariable(self.gemv_sum_local(scratch, row_offset)?),
                Span::default(),
            );
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            let value = expressions.append(
                Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a_value,
                    arg1: Some(x_value),
                    arg2: Some(sum_value),
                    arg3: None,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum_value)),
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value,
                },
                Span::default(),
            );
        }
        Ok(body)
    }

    fn lower_gemv_partial_add(
        &self,
        expressions: &mut Arena<Expression>,
        partials: TileRef,
        local_invocation: Handle<Expression>,
        stride: u32,
        rows_per_workgroup: u32,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();
        for row_offset in 0..rows_per_workgroup {
            let base = row_offset * self.workgroup_invocations;
            let mut lhs_emits = Vec::new();
            let lhs_index =
                self.add_literal_u32_emitted(expressions, local_invocation, base, &mut lhs_emits);
            Self::push_emits(&mut body, lhs_emits);
            let mut rhs_emits = Vec::new();
            let rhs_index = self.add_literal_u32_emitted(
                expressions,
                local_invocation,
                base + stride,
                &mut rhs_emits,
            );
            Self::push_emits(&mut body, rhs_emits);
            let (lhs_pointer, lhs_pointer_emits) =
                self.tile_dynamic_pointer(expressions, partials, lhs_index)?;
            let (rhs_pointer, rhs_pointer_emits) =
                self.tile_dynamic_pointer(expressions, partials, rhs_index)?;
            Self::push_emits(&mut body, lhs_pointer_emits);
            Self::push_emits(&mut body, rhs_pointer_emits);

            let lhs_value = expressions.append(
                Expression::Load {
                    pointer: lhs_pointer,
                },
                Span::default(),
            );
            let rhs_value = expressions.append(
                Expression::Load {
                    pointer: rhs_pointer,
                },
                Span::default(),
            );
            let value = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: lhs_value,
                    right: rhs_value,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::range_from(expressions, lhs_value, value)),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: lhs_pointer,
                    value,
                },
                Span::default(),
            );
        }
        Ok(body)
    }

    fn lower_gemv_final_store(
        &self,
        expressions: &mut Arena<Expression>,
        op: &GemvOp,
        row_base: Handle<Expression>,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();
        for row_offset in 0..op.rows_per_workgroup {
            let base = row_offset * self.workgroup_invocations;
            let partial_0_index =
                expressions.append(Expression::Literal(Literal::U32(base)), Span::default());
            let partial_1_index =
                expressions.append(Expression::Literal(Literal::U32(base + 1)), Span::default());
            let (partial_0, partial_0_emits) =
                self.tile_dynamic_pointer(expressions, op.partials, partial_0_index)?;
            let (partial_1, partial_1_emits) =
                self.tile_dynamic_pointer(expressions, op.partials, partial_1_index)?;
            Self::push_emits(&mut body, partial_0_emits);
            Self::push_emits(&mut body, partial_1_emits);
            let value_0 =
                expressions.append(Expression::Load { pointer: partial_0 }, Span::default());
            let value_1 =
                expressions.append(Expression::Load { pointer: partial_1 }, Span::default());
            let value = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: value_0,
                    right: value_1,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::range_from(expressions, value_0, value)),
                Span::default(),
            );

            let mut row_emits = Vec::new();
            let row =
                self.add_literal_u32_emitted(expressions, row_base, row_offset, &mut row_emits);
            Self::push_emits(&mut body, row_emits);
            let zero_col =
                expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
            let (y_index, y_index_emits) =
                self.storage_index_from_coords(expressions, &op.y, &[row, zero_col])?;
            Self::push_emits(&mut body, y_index_emits);
            let (y_pointer, y_pointer_emits) =
                self.storage_dynamic_pointer(expressions, &op.y, y_index)?;
            Self::push_emits(&mut body, y_pointer_emits);
            body.push(
                Statement::Store {
                    pointer: y_pointer,
                    value,
                },
                Span::default(),
            );
        }
        Ok(body)
    }

    fn gemv_sum_local(
        &self,
        scratch: ScratchLocals,
        row_offset: u32,
    ) -> Result<Handle<LocalVariable>, LowerError> {
        match row_offset {
            0 => Ok(scratch.mma_sum),
            1 => Ok(scratch.mma_sum_1),
            2 => scratch.mma_sum_2.ok_or(LowerError::UnsupportedOperation(
                "missing gemv scratch local",
            )),
            3 => scratch.mma_sum_3.ok_or(LowerError::UnsupportedOperation(
                "missing gemv scratch local",
            )),
            _ => Err(LowerError::UnsupportedOperation(
                "gemv rows per workgroup must be between 1 and 4",
            )),
        }
    }

    fn gemm_sum_local(
        &self,
        scratch: ScratchLocals,
        index: u32,
    ) -> Result<Handle<LocalVariable>, LowerError> {
        match index {
            0 => Ok(scratch.mma_sum),
            1 => Ok(scratch.mma_sum_1),
            2 => scratch.mma_sum_2.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            3 => scratch.mma_sum_3.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            4 => scratch.mma_sum_4.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            5 => scratch.mma_sum_5.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            6 => scratch.mma_sum_6.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            7 => scratch.mma_sum_7.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            _ => Err(LowerError::UnsupportedOperation(
                "gemm microtile width is too large",
            )),
        }
    }

    fn lower_mma(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &MmaOp,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.tile_layout(op.a)?;
        let b_layout = self.tile_layout(op.b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;

        if k_a != k_b || m != m_acc || n != n_acc {
            return Err(LowerError::UnsupportedOperation("mma shape mismatch"));
        }
        if acc_layout.memory_level() != MemoryLevel::Private {
            return Err(LowerError::UnsupportedMemoryLevel(
                acc_layout.memory_level(),
            ));
        }

        let mut j_body = Block::new();
        let (i, i_emit) = self.load_u32_local(expressions, scratch.mma_i);
        let (j, j_emit) = self.load_u32_local(expressions, scratch.mma_j);
        j_body.push(Statement::Emit(i_emit), Span::default());
        j_body.push(Statement::Emit(j_emit), Span::default());

        let (acc_index, acc_index_emits) =
            self.layout_index_expr(expressions, acc_layout, &[i, j])?;
        Self::push_emits(&mut j_body, acc_index_emits);
        let (_, acc_offset) = self.storage_tile_and_offset(op.acc)?;
        let mut acc_owner_emits = Vec::new();
        let acc_owner_index =
            self.add_literal_u32_emitted(expressions, acc_index, acc_offset, &mut acc_owner_emits);
        Self::push_emits(&mut j_body, acc_owner_emits);
        let (acc_pointer, acc_pointer_emits) =
            self.tile_dynamic_pointer(expressions, op.acc, acc_index)?;
        Self::push_emits(&mut j_body, acc_pointer_emits);
        let local_invocation = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let owns_acc_element = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Equal,
                left: local_invocation,
                right: acc_owner_index,
            },
            Span::default(),
        );
        j_body.push(
            Statement::Emit(Self::single_expression_range(expressions, owns_acc_element)),
            Span::default(),
        );

        let mut k_body = Block::new();
        let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        k_body.push(Statement::Emit(k_emit), Span::default());

        let (a_index, a_index_emits) = self.layout_index_expr(expressions, a_layout, &[i, k])?;
        let (b_index, b_index_emits) = self.layout_index_expr(expressions, b_layout, &[k, j])?;
        Self::push_emits(&mut k_body, a_index_emits);
        Self::push_emits(&mut k_body, b_index_emits);

        let (a_pointer, a_pointer_emits) = self.tile_dynamic_pointer(expressions, op.a, a_index)?;
        let (b_pointer, b_pointer_emits) = self.tile_dynamic_pointer(expressions, op.b, b_index)?;
        Self::push_emits(&mut k_body, a_pointer_emits);
        Self::push_emits(&mut k_body, b_pointer_emits);

        let acc_value = expressions.append(
            Expression::Load {
                pointer: acc_pointer,
            },
            Span::default(),
        );
        let a_value = expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
        let b_value = expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, acc_value)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_value)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b_value)),
            Span::default(),
        );
        let product = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Multiply,
                left: a_value,
                right: b_value,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, product)),
            Span::default(),
        );
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: acc_value,
                right: product,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        k_body.push(
            Statement::Store {
                pointer: acc_pointer,
                value,
            },
            Span::default(),
        );

        let k_loop = self.counted_loop(expressions, scratch.mma_k, k_a, k_body);
        j_body.push(
            Statement::If {
                condition: owns_acc_element,
                accept: Block::from_vec(vec![k_loop]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let j_loop = self.counted_loop(expressions, scratch.mma_j, n, j_body);
        let i_loop =
            self.counted_loop(expressions, scratch.mma_i, m, Block::from_vec(vec![j_loop]));

        Ok(i_loop)
    }

    fn store_zero_to_tile(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        tile: TileRef,
    ) -> Result<Statement, LowerError> {
        self.lower_workgroup_tile_op(expressions, tile_index, tile, |this, expressions, index| {
            let (pointer, pointer_emits) = this.tile_index_pointer(expressions, index, tile)?;
            let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
            let mut body = Block::new();
            Self::push_emits(&mut body, pointer_emits);
            body.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::default(),
            );
            Ok(body)
        })
    }

    fn lower_cooperative_load(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        dst: TileRef,
        src: &StorageView,
    ) -> Result<Statement, LowerError> {
        if let Some(statement) =
            self.try_lower_cooperative_load_vec4(expressions, tile_index, dst, src)?
        {
            return Ok(statement);
        }

        self.lower_workgroup_tile_op(expressions, tile_index, dst, |this, expressions, index| {
            let src_base = this.storage_base_expression(expressions, src)?;
            let dst_layout = this.tile_layout(dst)?;
            let (dst_pointer, dst_emits) = this.tile_index_pointer(expressions, index, dst)?;
            let (src_pointer, src_emits) = this.storage_index_pointer_from_tile_index_with_base(
                expressions,
                index,
                dst_layout,
                src,
                src_base,
            )?;
            let value = expressions.append(
                Expression::Load {
                    pointer: src_pointer,
                },
                Span::default(),
            );

            let mut body = Block::new();
            Self::push_emits(&mut body, dst_emits);
            for emit in src_emits {
                body.push(Statement::Emit(emit), Span::default());
            }
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: dst_pointer,
                    value,
                },
                Span::default(),
            );
            Ok(body)
        })
    }

    fn try_lower_cooperative_load_vec4(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        dst: TileRef,
        src: &StorageView,
    ) -> Result<Option<Statement>, LowerError> {
        let dst_layout = self.tile_layout(dst)?;
        let src_layout = self.storage_layout(src)?;
        if dst_layout.shape() != src_layout.shape()
            || dst_layout.shape().rank() != 2
            || dst_layout.strides().values()[1] != 1
            || src_layout.strides().values()[1] != 1
        {
            return Ok(None);
        }
        let rows = dst_layout.shape().dims()[0].get();
        let cols = dst_layout.shape().dims()[1].get();
        if rows == 0 || cols == 0 || cols % COOPERATIVE_LOAD_WIDTH != 0 {
            return Ok(None);
        }
        let groups_per_row = cols / COOPERATIVE_LOAD_WIDTH;
        let Some(groups) = std::num::NonZeroU32::new(rows * groups_per_row) else {
            return Ok(None);
        };

        let src_base = self.storage_base_expression(expressions, src)?;
        let (src_dynamic_base, base_emits) = self.storage_dynamic_base_index(expressions, src)?;
        let mut prelude = Block::new();
        Self::push_emits(&mut prelude, base_emits);

        let mut body = Block::new();
        let (group, group_emit) = self.load_u32_local(expressions, tile_index);
        body.push(Statement::Emit(group_emit), Span::default());
        let mut emits = Vec::new();
        let row = self.div_literal_u32_emitted(expressions, group, groups_per_row, &mut emits);
        let col_group =
            self.mod_literal_u32_emitted(expressions, group, groups_per_row, &mut emits);
        let col0 = self.mul_literal_u32_emitted(
            expressions,
            col_group,
            COOPERATIVE_LOAD_WIDTH,
            &mut emits,
        );
        Self::push_emits(&mut body, emits);

        for lane in 0..COOPERATIVE_LOAD_WIDTH {
            let mut lane_emits = Vec::new();
            let col = self.add_literal_u32_emitted(expressions, col0, lane, &mut lane_emits);
            let (src_index, src_index_emits) =
                self.layout_index_expr(expressions, src_layout, &[row, col])?;
            lane_emits.extend(src_index_emits);
            let src_index = self.add_optional_base_u32_emitted(
                expressions,
                src_index,
                src_dynamic_base,
                &mut lane_emits,
            );
            let src_pointer = expressions.append(
                Expression::Access {
                    base: src_base,
                    index: src_index,
                },
                Span::default(),
            );
            lane_emits.push(Self::single_expression_range(expressions, src_pointer));
            let (dst_index, dst_index_emits) =
                self.layout_index_expr(expressions, dst_layout, &[row, col])?;
            lane_emits.extend(dst_index_emits);
            let (dst_pointer, dst_pointer_emits) =
                self.tile_dynamic_pointer(expressions, dst, dst_index)?;
            lane_emits.extend(dst_pointer_emits);
            Self::push_emits(&mut body, lane_emits);

            let value = expressions.append(
                Expression::Load {
                    pointer: src_pointer,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: dst_pointer,
                    value,
                },
                Span::default(),
            );
        }

        prelude.push(
            self.distributed_index_loop(expressions, tile_index, groups, body),
            Span::default(),
        );
        Ok(Some(Statement::Block(prelude)))
    }

    fn lower_store_tile(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        src: TileRef,
        dst: &StorageView,
    ) -> Result<Statement, LowerError> {
        let src_layout = self.tile_layout(src)?;
        let dst_layout = self.storage_layout(dst)?;
        if src_layout.shape() != dst_layout.shape() {
            return Err(LowerError::UnsupportedOperation("store shape mismatch"));
        }

        let mut body = Block::new();
        let (flat, flat_emit) = self.load_u32_local(expressions, index_local);
        body.push(Statement::Emit(flat_emit), Span::default());

        let (src_index, src_index_emits) =
            self.index_from_flat(expressions, flat, src_layout, src_layout, &[])?;
        let (dst_index, dst_index_emits) = self.index_from_flat(
            expressions,
            flat,
            src_layout,
            dst_layout,
            &dst.dynamic_offsets,
        )?;
        Self::push_emits(&mut body, src_index_emits);
        Self::push_emits(&mut body, dst_index_emits);

        let (src_pointer, src_pointer_emits) =
            self.tile_dynamic_pointer(expressions, src, src_index)?;
        let (dst_pointer, dst_pointer_emits) =
            self.storage_dynamic_pointer(expressions, dst, dst_index)?;
        Self::push_emits(&mut body, src_pointer_emits);
        Self::push_emits(&mut body, dst_pointer_emits);

        let value = expressions.append(
            Expression::Load {
                pointer: src_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: dst_pointer,
                value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(expressions, index_local, src_layout.element_count(), body))
    }

    fn counted_loop(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        end: u32,
        body: Block,
    ) -> Statement {
        let init = self.store_u32_literal(expressions, index_local, 0);
        let (done, done_emit) = Self::u32_done_condition(expressions, index_local, end);
        let mut loop_body = Block::new();
        loop_body.push(Statement::Emit(done_emit), Span::default());
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        loop_body.push(Statement::Block(body), Span::default());
        loop_body.push(
            self.increment_u32_local(expressions, index_local, 1),
            Span::default(),
        );

        Statement::Block(Block::from_vec(vec![
            init,
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
        ]))
    }

    fn store_u32_literal(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
        value: u32,
    ) -> Statement {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let value = expressions.append(Expression::Literal(Literal::U32(value)), Span::default());
        Statement::Store { pointer, value }
    }

    fn increment_u32_local(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
        amount: u32,
    ) -> Statement {
        let amount = expressions.append(Expression::Literal(Literal::U32(amount)), Span::default());
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let next = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: current,
                right: amount,
            },
            Span::default(),
        );
        Statement::Block(Block::from_vec(vec![
            Statement::Emit(Self::range_from(expressions, current, next)),
            Statement::Store {
                pointer,
                value: next,
            },
        ]))
    }

    fn load_u32_local(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
    ) -> (Handle<Expression>, Range<Expression>) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let value = expressions.append(Expression::Load { pointer }, Span::default());
        (value, Self::single_expression_range(expressions, value))
    }

    fn subgroup_column_base(
        &self,
        expressions: &mut Arena<Expression>,
        subgroup_cols: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        self.subgroup_base(expressions, subgroup_cols, emits)
    }

    fn subgroup_partition_bases(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        partition: CoopPartition,
        subgroup_rows: u32,
        subgroup_cols: u32,
    ) -> (Option<Handle<Expression>>, Option<Handle<Expression>>) {
        match partition {
            CoopPartition::Single => (None, None),
            CoopPartition::Rows => {
                let mut emits = Vec::new();
                let row_base = self.subgroup_base(expressions, subgroup_rows, &mut emits);
                Self::push_emits(body, emits);
                (Some(row_base), None)
            }
            CoopPartition::Columns => {
                let mut emits = Vec::new();
                let col_base = self.subgroup_column_base(expressions, subgroup_cols, &mut emits);
                Self::push_emits(body, emits);
                (None, Some(col_base))
            }
            CoopPartition::InterleavedGrid {
                row_groups: _,
                col_groups,
            } => {
                let mut row_emits = Vec::new();
                let row_base = self.subgroup_grid_base(
                    expressions,
                    COOP_MATRIX_DIM,
                    col_groups,
                    true,
                    &mut row_emits,
                );
                Self::push_emits(body, row_emits);
                let mut col_emits = Vec::new();
                let col_base = self.subgroup_grid_base(
                    expressions,
                    COOP_MATRIX_DIM,
                    col_groups,
                    false,
                    &mut col_emits,
                );
                Self::push_emits(body, col_emits);
                (Some(row_base), Some(col_base))
            }
        }
    }

    fn coop_tile_offset(partition: CoopPartition, row_axis: bool, tile: u32) -> u32 {
        let stride_groups = match partition {
            CoopPartition::InterleavedGrid {
                row_groups,
                col_groups,
            } => {
                if row_axis {
                    row_groups
                } else {
                    col_groups
                }
            }
            _ => 1,
        };
        tile * COOP_MATRIX_DIM * stride_groups
    }

    fn subgroup_grid_base(
        &self,
        expressions: &mut Arena<Expression>,
        extent_per_subgroup: u32,
        col_groups: u32,
        row_axis: bool,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if self.coop_subgroups <= 1 {
            return expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        }
        let subgroup_id = expressions.append(
            Expression::FunctionArgument(SUBGROUP_ID_ARG),
            Span::default(),
        );
        let group = if row_axis {
            self.div_literal_u32_emitted(expressions, subgroup_id, col_groups, emits)
        } else {
            self.mod_literal_u32_emitted(expressions, subgroup_id, col_groups, emits)
        };
        self.mul_literal_u32_emitted(expressions, group, extent_per_subgroup, emits)
    }

    fn subgroup_base(
        &self,
        expressions: &mut Arena<Expression>,
        extent_per_subgroup: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if self.coop_subgroups <= 1 {
            return expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        }
        let subgroup_id = expressions.append(
            Expression::FunctionArgument(SUBGROUP_ID_ARG),
            Span::default(),
        );
        self.mul_literal_u32_emitted(expressions, subgroup_id, extent_per_subgroup, emits)
    }

    fn current_loop_index(&self) -> Handle<LocalVariable> {
        self.loop_index_local
            .expect("scratch locals must be created before lowering storage offsets")
    }

    fn u32_done_condition(
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        end: u32,
    ) -> (Handle<Expression>, Range<Expression>) {
        let end = expressions.append(Expression::Literal(Literal::U32(end)), Span::default());
        let pointer = expressions.append(Expression::LocalVariable(index_local), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let condition = expressions.append(
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: current,
                right: end,
            },
            Span::default(),
        );

        (condition, Self::range_from(expressions, current, condition))
    }

    fn push_emits(body: &mut Block, emits: Vec<Range<Expression>>) {
        for emit in emits {
            body.push(Statement::Emit(emit), Span::default());
        }
    }

    fn lower_workgroup_tile_op(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        tile: TileRef,
        tile_body: impl FnOnce(
            &Self,
            &mut Arena<Expression>,
            Handle<LocalVariable>,
        ) -> Result<Block, LowerError>,
    ) -> Result<Statement, LowerError> {
        let layout = self.tile_layout(tile)?;
        let body = tile_body(self, expressions, tile_index)?;
        Ok(self.distributed_index_loop(expressions, tile_index, layout.element_count(), body))
    }

    fn distributed_index_loop(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        end: std::num::NonZeroU32,
        body: Block,
    ) -> Statement {
        let init_index = self.init_tile_index(expressions, index_local);
        if end.get() == self.workgroup_invocations {
            return Statement::Block(Block::from_vec(vec![init_index, Statement::Block(body)]));
        }
        let mut loop_body = Block::new();
        let (done, done_emit) = Self::tile_done_condition(expressions, index_local, end);
        loop_body.push(Statement::Emit(done_emit), Span::default());
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        loop_body.push(Statement::Block(body), Span::default());
        loop_body.push(
            self.advance_tile_index(expressions, index_local),
            Span::default(),
        );

        Statement::Block(Block::from_vec(vec![
            init_index,
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
        ]))
    }

    fn init_tile_index(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
    ) -> Statement {
        self.init_tile_index_with_offset(expressions, tile_index, 0)
    }

    fn init_tile_index_with_offset(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        chunk: u32,
    ) -> Statement {
        let pointer = expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let lane = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let value = if chunk == 0 {
            lane
        } else {
            self.add_literal_u32(expressions, lane, chunk * self.workgroup_invocations)
        };
        Statement::Store { pointer, value }
    }

    fn advance_tile_index(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
    ) -> Statement {
        let workgroup_size = expressions.append(
            Expression::Literal(Literal::U32(self.workgroup_invocations)),
            Span::default(),
        );
        let pointer = expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let next = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: current,
                right: workgroup_size,
            },
            Span::default(),
        );

        Statement::Block(Block::from_vec(vec![
            Statement::Emit(Self::range_from(expressions, current, next)),
            Statement::Store {
                pointer,
                value: next,
            },
        ]))
    }

    fn tile_done_condition(
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        element_count: std::num::NonZeroU32,
    ) -> (Handle<Expression>, Range<Expression>) {
        let element_count = expressions.append(
            Expression::Literal(Literal::U32(element_count.get())),
            Span::default(),
        );
        let pointer = expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let condition = expressions.append(
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: current,
                right: element_count,
            },
            Span::default(),
        );

        (condition, Self::range_from(expressions, current, condition))
    }

    fn tile_layout(&self, tile: TileRef) -> Result<&Layout, LowerError> {
        let decl = self
            .ir
            .tiles()
            .get(tile.id.index())
            .ok_or(LowerError::UnknownTile(tile.id))?;
        if decl.element != tile.element {
            return Err(LowerError::TileElementMismatch {
                tile: tile.id,
                declared: decl.element,
                used: tile.element,
            });
        }

        Ok(&decl.layout)
    }

    fn tile_index_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        tile: TileRef,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        self.tile_layout(tile)?;

        let (storage_tile, offset) = self.storage_tile_and_offset(tile)?;
        let global = self
            .tile_globals
            .get(storage_tile.id.index())
            .copied()
            .flatten()
            .ok_or_else(|| {
                self.tile_layout(storage_tile)
                    .map(|layout| LowerError::UnsupportedMemoryLevel(layout.memory_level()))
                    .unwrap_or(LowerError::UnknownTile(storage_tile.id))
            })?;
        let base = expressions.append(Expression::GlobalVariable(global), Span::default());
        let index_pointer =
            expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let flat = expressions.append(
            Expression::Load {
                pointer: index_pointer,
            },
            Span::default(),
        );
        let mut emits = Vec::new();
        emits.push(Self::single_expression_range(expressions, flat));
        let layout = self.tile_layout(tile)?;
        let index = if layout.is_row_major() {
            flat
        } else {
            self.storage_index_from_flat(expressions, flat, layout, layout, &[], &mut emits)?
        };
        let index = self.add_literal_u32_emitted(expressions, index, offset, &mut emits);
        let pointer = expressions.append(Expression::Access { base, index }, Span::default());
        emits.push(Self::single_expression_range(expressions, pointer));
        Ok((pointer, emits))
    }

    fn tile_dynamic_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        tile: TileRef,
        index: Handle<Expression>,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        self.tile_layout(tile)?;

        let base = self.tile_base_expression(expressions, tile)?;
        let (_, offset) = self.storage_tile_and_offset(tile)?;
        let mut emits = Vec::new();
        let index = self.add_literal_u32_emitted(expressions, index, offset, &mut emits);
        let pointer = expressions.append(Expression::Access { base, index }, Span::default());
        emits.push(Self::single_expression_range(expressions, pointer));
        Ok((pointer, emits))
    }

    fn tile_base_expression(
        &self,
        expressions: &mut Arena<Expression>,
        tile: TileRef,
    ) -> Result<Handle<Expression>, LowerError> {
        let (storage_tile, _) = self.storage_tile_and_offset(tile)?;
        let layout = self.tile_layout(storage_tile)?;

        match layout.memory_level() {
            MemoryLevel::Workgroup => {
                let global = self
                    .tile_globals
                    .get(storage_tile.id.index())
                    .copied()
                    .flatten()
                    .ok_or(LowerError::UnknownTile(storage_tile.id))?;
                Ok(expressions.append(Expression::GlobalVariable(global), Span::default()))
            }
            MemoryLevel::Private => {
                let local = self
                    .tile_locals
                    .get(storage_tile.id.index())
                    .copied()
                    .flatten()
                    .ok_or(LowerError::UnknownTile(storage_tile.id))?;
                Ok(expressions.append(Expression::LocalVariable(local), Span::default()))
            }
            memory => Err(LowerError::UnsupportedMemoryLevel(memory)),
        }
    }

    fn storage_index_pointer_from_tile_index_with_base(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        dst_layout: &Layout,
        view: &StorageView,
        base: Handle<Expression>,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let src_layout = self.storage_layout(view)?;
        if dst_layout.shape() != src_layout.shape() {
            return Err(LowerError::UnsupportedOperation("load shape mismatch"));
        }
        let mut emits = Vec::new();
        let index_pointer =
            expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let flat = expressions.append(
            Expression::Load {
                pointer: index_pointer,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, flat));
        let logical_index = self.storage_index_from_flat(
            expressions,
            flat,
            dst_layout,
            src_layout,
            &view.dynamic_offsets,
            &mut emits,
        )?;
        let index =
            self.add_literal_u32_emitted(expressions, logical_index, view.offset, &mut emits);
        let pointer = expressions.append(Expression::Access { base, index }, Span::default());
        emits.push(Self::single_expression_range(expressions, pointer));
        Ok((pointer, emits))
    }

    fn storage_dynamic_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
        index: Handle<Expression>,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let base = self.storage_base_expression(expressions, view)?;
        let mut emits = Vec::new();
        let index = self.add_literal_u32_emitted(expressions, index, view.offset, &mut emits);
        let pointer = expressions.append(Expression::Access { base, index }, Span::default());
        emits.push(Self::single_expression_range(expressions, pointer));
        Ok((pointer, emits))
    }

    fn storage_base_expression(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
    ) -> Result<Handle<Expression>, LowerError> {
        self.storage_layout(view)?;
        let global = self
            .buffer_globals
            .get(view.buffer.id.index())
            .copied()
            .flatten()
            .ok_or(LowerError::UnknownBuffer(view.buffer.id))?;
        Ok(expressions.append(Expression::GlobalVariable(global), Span::default()))
    }

    fn storage_layout<'view>(&self, view: &'view StorageView) -> Result<&'view Layout, LowerError> {
        let decl = self
            .ir
            .buffers()
            .get(view.buffer.id.index())
            .ok_or(LowerError::UnknownBuffer(view.buffer.id))?;
        if decl.element != view.buffer.element {
            return Err(LowerError::UnsupportedOperation("buffer element mismatch"));
        }
        Ok(&view.layout)
    }

    fn index_from_flat(
        &self,
        expressions: &mut Arena<Expression>,
        flat: Handle<Expression>,
        logical_layout: &Layout,
        target_layout: &Layout,
        dynamic_offsets: &[Option<DynamicOffset>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let mut emits = Vec::new();
        let index = self.storage_index_from_flat(
            expressions,
            flat,
            logical_layout,
            target_layout,
            dynamic_offsets,
            &mut emits,
        )?;
        Ok((index, emits))
    }

    fn layout_index_expr(
        &self,
        expressions: &mut Arena<Expression>,
        layout: &Layout,
        coords: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if layout.strides().rank() != coords.len() {
            return Err(LowerError::UnsupportedOperation("layout rank mismatch"));
        }
        let mut emits = Vec::new();
        let mut terms = Vec::with_capacity(coords.len());
        for (coord, stride) in coords.iter().zip(layout.strides().values()) {
            if Self::is_u32_literal(expressions, *coord, 0) || *stride == 0 {
                continue;
            }
            terms.push(self.mul_literal_u32_emitted(expressions, *coord, *stride, &mut emits));
        }
        let mut terms = terms.into_iter();
        let Some(mut index) = terms.next() else {
            let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
            return Ok((zero, emits));
        };
        for term in terms {
            index = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: index,
                    right: term,
                },
                Span::default(),
            );
            emits.push(Self::single_expression_range(expressions, index));
        }
        Ok((index, emits))
    }

    fn is_u32_literal(
        expressions: &Arena<Expression>,
        value: Handle<Expression>,
        expected: u32,
    ) -> bool {
        Self::u32_literal(expressions, value) == Some(expected)
    }

    fn u32_literal(expressions: &Arena<Expression>, value: Handle<Expression>) -> Option<u32> {
        match expressions[value] {
            Expression::Literal(Literal::U32(value)) => Some(value),
            _ => None,
        }
    }

    fn storage_index_from_coords(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
        coords: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        self.storage_index_from_coords_filtered(expressions, view, coords, false)
    }

    fn storage_index_from_coords_without_loop_offsets(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
        coords: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        self.storage_index_from_coords_filtered(expressions, view, coords, true)
    }

    fn storage_linear_base_without_loop_offsets(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
    ) -> Result<(Option<Handle<Expression>>, Vec<Range<Expression>>), LowerError> {
        let layout = self.storage_layout(view)?;
        let mut emits = Vec::new();
        let mut terms = Vec::new();
        for (axis_index, stride) in layout.strides().values().iter().copied().enumerate() {
            if let Some(DynamicOffset::Workgroup(offset)) =
                view.dynamic_offsets.get(axis_index).copied().flatten()
            {
                let workgroup_id = expressions.append(
                    Expression::FunctionArgument(WORKGROUP_ID_ARG),
                    Span::default(),
                );
                let axis = expressions.append(
                    Expression::AccessIndex {
                        base: workgroup_id,
                        index: offset.axis.index(),
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, axis));
                let scaled =
                    self.mul_literal_u32_emitted(expressions, axis, offset.scale, &mut emits);
                terms.push(self.mul_literal_u32_emitted(expressions, scaled, stride, &mut emits));
            }
        }

        let mut terms = terms.into_iter();
        let Some(mut base) = terms.next() else {
            return Ok((None, emits));
        };
        for term in terms {
            base = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: base,
                    right: term,
                },
                Span::default(),
            );
            emits.push(Self::single_expression_range(expressions, base));
        }
        Ok((Some(base), emits))
    }

    fn storage_dynamic_base_index(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
    ) -> Result<(Option<Handle<Expression>>, Vec<Range<Expression>>), LowerError> {
        let layout = self.storage_layout(view)?;
        let mut emits = Vec::new();
        let mut terms = Vec::new();
        if view.offset != 0 {
            terms.push(expressions.append(
                Expression::Literal(Literal::U32(view.offset)),
                Span::default(),
            ));
        }
        for (axis_index, stride) in layout.strides().values().iter().copied().enumerate() {
            let Some(offset) = view.dynamic_offsets.get(axis_index).copied().flatten() else {
                continue;
            };
            let scaled = self.dynamic_offset_scaled(expressions, offset, &mut emits);
            terms.push(self.mul_literal_u32_emitted(expressions, scaled, stride, &mut emits));
        }

        let mut terms = terms.into_iter();
        let Some(mut base) = terms.next() else {
            return Ok((None, emits));
        };
        for term in terms {
            base = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: base,
                    right: term,
                },
                Span::default(),
            );
            emits.push(Self::single_expression_range(expressions, base));
        }
        Ok((Some(base), emits))
    }

    fn add_optional_base_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        base: Option<Handle<Expression>>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        let Some(base) = base else {
            return value;
        };
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: value,
                right: base,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

    fn storage_index_from_coords_filtered(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
        coords: &[Handle<Expression>],
        skip_loop_offsets: bool,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let layout = self.storage_layout(view)?;
        if layout.strides().rank() != coords.len() {
            return Err(LowerError::UnsupportedOperation("layout rank mismatch"));
        }

        let mut emits = Vec::new();
        let mut terms = Vec::with_capacity(coords.len());
        for (axis, (coord, stride)) in coords.iter().zip(layout.strides().values()).enumerate() {
            let coord = self.apply_dynamic_offset_filtered(
                expressions,
                *coord,
                &view.dynamic_offsets,
                axis,
                skip_loop_offsets,
                &mut emits,
            );
            terms.push(self.mul_literal_u32_emitted(expressions, coord, *stride, &mut emits));
        }
        let mut terms = terms.into_iter();
        let Some(mut index) = terms.next() else {
            return Err(LowerError::UnsupportedOperation("zero-rank layout"));
        };
        for term in terms {
            index = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: index,
                    right: term,
                },
                Span::default(),
            );
            emits.push(Self::single_expression_range(expressions, index));
        }

        Ok((index, emits))
    }

    fn storage_index_from_flat(
        &self,
        expressions: &mut Arena<Expression>,
        flat: Handle<Expression>,
        dst_layout: &Layout,
        src_layout: &Layout,
        dynamic_offsets: &[Option<DynamicOffset>],
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        match dst_layout.shape().rank() {
            1 => {
                let coord = self.apply_dynamic_offset(expressions, flat, dynamic_offsets, 0, emits);
                Ok(self.mul_literal_u32_emitted(
                    expressions,
                    coord,
                    src_layout.strides().values()[0],
                    emits,
                ))
            }
            2 => {
                let cols = expressions.append(
                    Expression::Literal(Literal::U32(dst_layout.shape().dims()[1].get())),
                    Span::default(),
                );
                let row = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Divide,
                        left: flat,
                        right: cols,
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, row));
                let col = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Modulo,
                        left: flat,
                        right: cols,
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, col));
                let row = self.apply_dynamic_offset(expressions, row, dynamic_offsets, 0, emits);
                let col = self.apply_dynamic_offset(expressions, col, dynamic_offsets, 1, emits);
                let row = self.mul_literal_u32_emitted(
                    expressions,
                    row,
                    src_layout.strides().values()[0],
                    emits,
                );
                let col = self.mul_literal_u32_emitted(
                    expressions,
                    col,
                    src_layout.strides().values()[1],
                    emits,
                );
                let index = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Add,
                        left: row,
                        right: col,
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, index));
                Ok(index)
            }
            _ => Err(LowerError::UnsupportedOperation("rank > 2 storage view")),
        }
    }

    fn apply_dynamic_offset(
        &self,
        expressions: &mut Arena<Expression>,
        coord: Handle<Expression>,
        dynamic_offsets: &[Option<DynamicOffset>],
        axis_index: usize,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        self.apply_dynamic_offset_filtered(
            expressions,
            coord,
            dynamic_offsets,
            axis_index,
            false,
            emits,
        )
    }

    fn apply_dynamic_offset_filtered(
        &self,
        expressions: &mut Arena<Expression>,
        coord: Handle<Expression>,
        dynamic_offsets: &[Option<DynamicOffset>],
        axis_index: usize,
        skip_loop_offsets: bool,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        let Some(Some(offset)) = dynamic_offsets.get(axis_index) else {
            return coord;
        };
        if skip_loop_offsets && matches!(offset, DynamicOffset::Loop(_)) {
            return coord;
        }
        let scaled = self.dynamic_offset_scaled(expressions, *offset, emits);
        let coord = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: coord,
                right: scaled,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, coord));
        coord
    }

    fn dynamic_offset_scaled(
        &self,
        expressions: &mut Arena<Expression>,
        offset: DynamicOffset,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        match offset {
            DynamicOffset::Workgroup(offset) => {
                let workgroup_id = expressions.append(
                    Expression::FunctionArgument(WORKGROUP_ID_ARG),
                    Span::default(),
                );
                let axis = expressions.append(
                    Expression::AccessIndex {
                        base: workgroup_id,
                        index: offset.axis.index(),
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, axis));
                self.mul_literal_u32_emitted(expressions, axis, offset.scale, emits)
            }
            DynamicOffset::Loop(offset) => {
                let (loop_index, loop_emit) =
                    self.load_u32_local(expressions, self.current_loop_index());
                emits.push(loop_emit);
                self.mul_literal_u32_emitted(expressions, loop_index, offset.scale, emits)
            }
        }
    }

    fn storage_tile_and_offset(&self, tile: TileRef) -> Result<(TileRef, u32), LowerError> {
        let decl = self
            .ir
            .tiles()
            .get(tile.id.index())
            .ok_or(LowerError::UnknownTile(tile.id))?;
        if decl.element != tile.element {
            return Err(LowerError::TileElementMismatch {
                tile: tile.id,
                declared: decl.element,
                used: tile.element,
            });
        }

        match decl.origin {
            TileOrigin::Allocation => Ok((tile, 0)),
            TileOrigin::View { source, mapping } => {
                let (root, base_offset) = self.storage_tile_and_offset(source)?;
                let source_layout = self.tile_layout(source)?;
                let local_offset = match mapping {
                    ViewMapping::Partition { origin, .. } => {
                        Self::linear_index_prefix(source_layout, &origin)?
                    }
                };
                Ok((
                    root,
                    base_offset.checked_add(local_offset).ok_or(
                        LowerError::UnsupportedOperation("tile view offset overflow"),
                    )?,
                ))
            }
        }
    }

    fn matrix_shape(layout: &Layout) -> Result<[u32; 2], LowerError> {
        if layout.shape().rank() != 2 {
            return Err(LowerError::UnsupportedOperation("non-matrix mma"));
        }
        Ok([
            layout.shape().dims()[0].get(),
            layout.shape().dims()[1].get(),
        ])
    }

    fn linear_index_prefix(layout: &Layout, coords: &[u32]) -> Result<u32, LowerError> {
        let rank = layout.strides().rank();
        if coords.len() > rank && coords[rank..].iter().any(|coord| *coord != 0) {
            return Err(LowerError::UnsupportedOperation("layout rank mismatch"));
        }
        Ok(coords
            .iter()
            .take(rank)
            .zip(layout.strides().values())
            .map(|(coord, stride)| coord * stride)
            .sum())
    }

    fn add_literal_u32(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 0 {
            return value;
        }
        if let Some(value) = Self::u32_literal(expressions, value) {
            return expressions.append(
                Expression::Literal(Literal::U32(value + literal)),
                Span::default(),
            );
        }
        let literal =
            expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
        expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: value,
                right: literal,
            },
            Span::default(),
        )
    }

    fn mul_literal_u32(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 1 {
            return value;
        }
        if let Some(value) = Self::u32_literal(expressions, value) {
            return expressions.append(
                Expression::Literal(Literal::U32(value * literal)),
                Span::default(),
            );
        }
        let literal =
            expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
        expressions.append(
            Expression::Binary {
                op: BinaryOperator::Multiply,
                left: value,
                right: literal,
            },
            Span::default(),
        )
    }

    fn add_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if literal == 0 {
            return value;
        }
        if let Some(value) = Self::u32_literal(expressions, value) {
            return expressions.append(
                Expression::Literal(Literal::U32(value + literal)),
                Span::default(),
            );
        }
        let literal =
            expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: value,
                right: literal,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

    fn mul_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if literal == 1 {
            return value;
        }
        if let Some(value) = Self::u32_literal(expressions, value) {
            return expressions.append(
                Expression::Literal(Literal::U32(value * literal)),
                Span::default(),
            );
        }
        let literal =
            expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Multiply,
                left: value,
                right: literal,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

    fn div_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if literal == 1 {
            return value;
        }
        if let Some(value) = Self::u32_literal(expressions, value) {
            return expressions.append(
                Expression::Literal(Literal::U32(value / literal)),
                Span::default(),
            );
        }
        let value = if literal.is_power_of_two() {
            let shift = expressions.append(
                Expression::Literal(Literal::U32(literal.trailing_zeros())),
                Span::default(),
            );
            expressions.append(
                Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: value,
                    right: shift,
                },
                Span::default(),
            )
        } else {
            let literal =
                expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
            expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Divide,
                    left: value,
                    right: literal,
                },
                Span::default(),
            )
        };
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

    fn mod_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if literal == 1 {
            return expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        }
        if let Some(value) = Self::u32_literal(expressions, value) {
            return expressions.append(
                Expression::Literal(Literal::U32(value % literal)),
                Span::default(),
            );
        }
        let value = if literal.is_power_of_two() {
            let mask = expressions.append(
                Expression::Literal(Literal::U32(literal - 1)),
                Span::default(),
            );
            expressions.append(
                Expression::Binary {
                    op: BinaryOperator::And,
                    left: value,
                    right: mask,
                },
                Span::default(),
            )
        } else {
            let literal =
                expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
            expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Modulo,
                    left: value,
                    right: literal,
                },
                Span::default(),
            )
        };
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

    fn single_expression_range(
        expressions: &Arena<Expression>,
        handle: Handle<Expression>,
    ) -> Range<Expression> {
        Self::range_from(expressions, handle, handle)
    }

    fn range_from(
        expressions: &Arena<Expression>,
        first: Handle<Expression>,
        last: Handle<Expression>,
    ) -> Range<Expression> {
        Range::from_index_range(first.index() as u32..last.index() as u32 + 1, expressions)
    }
}
