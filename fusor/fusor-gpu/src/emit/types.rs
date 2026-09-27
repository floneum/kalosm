//! Kernel [`ElementType`] -> naga types, and the workgroup/storage
//! declarations. Types intern in a fixed order so emission is deterministic.

use fusor_ir::ir::kernel::{
    ArenaMode, BufferAccess, BufferDecl, CoopMatrixRole, ElementType, ScalarElement, TileDecl,
};
use fusor_ir::target::EmitError;
use naga::{
    AddressSpace, ArraySize, GlobalVariable, Handle, ResourceBinding, Scalar, Span, StorageAccess,
    Type, TypeInner, VectorSize,
};
use rustc_hash::FxHashMap;

use super::{Analysis, Emitter, key};

/// How one workgroup tile is backed: `global[base_index + tile_index]`, bitcast
/// between `canonical` and the tile's element in a heterogeneous region.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TileBacking {
    pub global: Handle<GlobalVariable>,
    pub canonical: ElementType,
    pub base_index: u32,
}

/// The two prelude handles the entry point itself needs. Everything else is
/// looked up through naga's `UniqueArena`, which interns structurally.
pub(crate) struct Prelude {
    pub u32_ty: Handle<Type>,
    pub u32_vec3_ty: Handle<Type>,
}

/// The naga scalar for one Kernel scalar element. `BF16` is storage-only and
/// never reaches Kernel as a value type.
pub(crate) fn scalar_of(scalar: ScalarElement) -> Result<Scalar, EmitError> {
    Ok(match scalar {
        ScalarElement::F32 => Scalar::F32,
        ScalarElement::F16 => Scalar::F16,
        ScalarElement::U32 => Scalar::U32,
        ScalarElement::I32 => Scalar::I32,
        ScalarElement::Bool => Scalar::BOOL,
        ScalarElement::BF16 => {
            return Err(EmitError::MissingCapability("shader-bf16"));
        }
    })
}

fn vector_size(lanes: u32) -> Result<VectorSize, EmitError> {
    Ok(match lanes {
        2 => VectorSize::Bi,
        3 => VectorSize::Tri,
        4 => VectorSize::Quad,
        _ => {
            return Err(EmitError::Unsupported(format!(
                "vectors must have 2, 3 or 4 lanes, got {lanes}"
            )));
        }
    })
}

pub(crate) fn cooperative_size(size: u32) -> Result<naga::CooperativeSize, EmitError> {
    Ok(match size {
        8 => naga::CooperativeSize::Eight,
        16 => naga::CooperativeSize::Sixteen,
        _ => {
            return Err(EmitError::Unsupported(format!(
                "cooperative-matrix size must be 8 or 16, got {size}"
            )));
        }
    })
}

pub(crate) fn naga_role(role: CoopMatrixRole) -> naga::CooperativeRole {
    match role {
        CoopMatrixRole::A => naga::CooperativeRole::A,
        CoopMatrixRole::B => naga::CooperativeRole::B,
        CoopMatrixRole::C => naga::CooperativeRole::C,
    }
}

fn type_inner(element: ElementType) -> Result<TypeInner, EmitError> {
    Ok(match element {
        ElementType::Scalar(s) => TypeInner::Scalar(scalar_of(s)?),
        ElementType::Vector { scalar, lanes } => TypeInner::Vector {
            size: vector_size(lanes)?,
            scalar: scalar_of(scalar)?,
        },
        ElementType::CoopMatrix {
            scalar,
            role,
            rows,
            cols,
        } => TypeInner::CooperativeMatrix {
            columns: cooperative_size(cols)?,
            rows: cooperative_size(rows)?,
            scalar: scalar_of(scalar)?,
            role: naga_role(role),
        },
    })
}

fn insert(module: &mut naga::Module, inner: TypeInner) -> Handle<Type> {
    module
        .types
        .insert(Type { name: None, inner }, Span::default())
}

/// Register (or reuse) the naga type for one element type.
pub(crate) fn element_type(
    module: &mut naga::Module,
    element: ElementType,
) -> Result<Handle<Type>, EmitError> {
    Ok(insert(module, type_inner(element)?))
}

/// Intern the prelude in a fixed order; the f16 quad and cooperative-matrix
/// elements only when used.
pub(crate) fn intern_prelude(
    module: &mut naga::Module,
    analysis: &Analysis,
) -> Result<Prelude, EmitError> {
    // A scalar and its vec2/3/4, in that order.
    let with_vectors = |module: &mut naga::Module, scalar| {
        [
            None,
            Some(VectorSize::Bi),
            Some(VectorSize::Tri),
            Some(VectorSize::Quad),
        ]
        .map(|size| {
            let inner = match size {
                None => TypeInner::Scalar(scalar),
                Some(size) => TypeInner::Vector { size, scalar },
            };
            insert(module, inner)
        })
    };
    with_vectors(module, Scalar::F32);
    insert(module, TypeInner::Scalar(Scalar::I32));
    let size = VectorSize::Quad;
    insert(
        module,
        TypeInner::Vector {
            size,
            scalar: Scalar::I32,
        },
    );
    let u32s = with_vectors(module, Scalar::U32);
    with_vectors(module, Scalar::BOOL);
    if analysis.uses_f16 {
        with_vectors(module, Scalar::F16);
    }
    // Cooperative-matrix types up front, in locals-list order; re-interning
    // an existing type is a no-op.
    for local in &analysis.locals {
        if matches!(local.element, ElementType::CoopMatrix { .. }) {
            element_type(module, local.element)?;
        }
    }
    Ok(Prelude {
        u32_ty: u32s[0],
        u32_vec3_ty: u32s[2],
    })
}

/// Array stride for a workgroup/storage array of `element`, from
/// [`ElementType::workgroup_array_stride`] as arena packing reads it.
fn array_stride(element: ElementType) -> Result<u32, EmitError> {
    element
        .workgroup_array_stride()
        .ok_or_else(|| EmitError::Unsupported(format!("{element:?} cannot back an array")))
}

pub(crate) fn array_type(
    module: &mut naga::Module,
    element: ElementType,
    size: ArraySize,
) -> Result<Handle<Type>, EmitError> {
    let stride = array_stride(element)?;
    let base = element_type(module, element)?;
    Ok(insert(module, TypeInner::Array { base, size, stride }))
}

fn atomic_array_type(
    module: &mut naga::Module,
    element: ElementType,
) -> Result<Handle<Type>, EmitError> {
    // f32 `AtomicAdd` is a bitcast compare-exchange loop, so the buffer is
    // `array<atomic<u32>>`.
    let scalar = match element {
        ElementType::Scalar(ScalarElement::I32) => Scalar::I32,
        ElementType::Scalar(ScalarElement::U32 | ScalarElement::F32) => Scalar::U32,
        other => {
            return Err(EmitError::Unsupported(format!(
                "atomic add is only defined for u32/i32/f32 buffers, got {other:?}"
            )));
        }
    };
    let base = insert(module, TypeInner::Atomic(scalar));
    Ok(insert(
        module,
        TypeInner::Array {
            base,
            size: ArraySize::Dynamic,
            stride: 4,
        },
    ))
}

/// Declare a storage buffer global; read-only-ness from [`BufferDecl::access`],
/// atomic typing when the analysis found an `AtomicAdd` on the binding.
pub(crate) fn storage_global_with(
    module: &mut naga::Module,
    decl: &BufferDecl,
    atomic: bool,
) -> Result<Handle<GlobalVariable>, EmitError> {
    let ty = if atomic {
        atomic_array_type(module, decl.element)?
    } else {
        array_type(module, decl.element, ArraySize::Dynamic)?
    };
    let access = match decl.access {
        BufferAccess::Read => StorageAccess::LOAD,
        BufferAccess::ReadWrite => StorageAccess::LOAD | StorageAccess::STORE,
    };
    Ok(module.global_variables.append(
        GlobalVariable {
            name: None,
            space: AddressSpace::Storage { access },
            binding: Some(ResourceBinding {
                group: 0,
                binding: decl.binding,
            }),
            ty,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::default(),
    ))
}

/// A workgroup global of type `ty`.
pub(crate) fn workgroup_var(module: &mut naga::Module, ty: Handle<Type>) -> Handle<GlobalVariable> {
    let var = GlobalVariable {
        name: None,
        space: AddressSpace::WorkGroup,
        binding: None,
        ty,
        init: None,
        memory_decorations: naga::MemoryDecorations::empty(),
    };
    module.global_variables.append(var, Span::default())
}

/// A workgroup tile global sized for its own extent.
fn workgroup_global(
    module: &mut naga::Module,
    decl: &TileDecl,
) -> Result<Handle<GlobalVariable>, EmitError> {
    let count = std::num::NonZeroU32::new(decl.layout.element_count() as u32)
        .ok_or_else(|| EmitError::Unsupported("empty workgroup tile".into()))?;
    let ty = array_type(module, decl.element, ArraySize::Constant(count))?;
    Ok(workgroup_var(module, ty))
}

/// Buffers in binding order, so the global-variable arena is independent
/// of which statement touches which buffer first.
pub(crate) fn create_storage_globals(em: &mut Emitter<'_>) -> Result<(), EmitError> {
    let mut buffers = em.analysis.buffers.clone();
    buffers.sort_by_key(|b| b.binding);
    for views in buffers.chunk_by(|a, b| a.binding == b.binding) {
        let first = &views[0];
        let mixed = views.iter().any(|b| b.element != first.element);
        let packed_half = mixed
            && views
                .iter()
                .any(|b| b.element == ElementType::Scalar(ScalarElement::F16));
        let atomic = em.analysis.atomic_buffers.contains(&first.binding) || packed_half;
        let mut decl = (**first).clone();
        if mixed {
            if views.iter().any(|b| {
                !matches!(
                    b.element,
                    ElementType::Scalar(
                        ScalarElement::F32
                            | ScalarElement::I32
                            | ScalarElement::U32
                            | ScalarElement::F16
                    )
                )
            }) {
                return Err(EmitError::Unsupported(
                    "mixed storage views require scalar numeric elements".into(),
                ));
            }
            decl.element = ElementType::Scalar(ScalarElement::U32);
        }
        if views.iter().any(|b| b.access == BufferAccess::ReadWrite) {
            decl.access = BufferAccess::ReadWrite;
        }
        // Neighboring f16 elements share a word. A compare-exchange store
        // preserves the other half even when different lanes write it.
        if atomic {
            em.analysis.atomic_buffers.insert(first.binding);
        }
        let global = storage_global_with(&mut em.module, &decl, atomic)?;
        let physical = if atomic && decl.element == ElementType::Scalar(ScalarElement::F32) {
            ElementType::Scalar(ScalarElement::U32)
        } else {
            decl.element
        };
        for buffer in views {
            em.buffer_globals.insert(key(buffer), global);
            em.buffer_elements.insert(key(buffer), physical);
        }
    }
    Ok(())
}

/// Workgroup tiles, laid out from the plan.
///
/// `ArenaMode::Regions` emits one global per shared byte offset, bitcasting
/// values (32-bit scalars only) in a heterogeneous group. `ArenaMode::ByteArena`
/// emits one `array<u32>` indexed by packed byte offset. An unplaced tile gets
/// its own allocation.
pub(crate) fn create_workgroup_globals(em: &mut Emitter<'_>) -> Result<(), EmitError> {
    let tiles = em.analysis.tiles.clone();
    // `FUSOR_NO_TILE_ALIAS` gives every tile its own allocation: the
    // bisection aid for a suspected aliasing miscompile.
    let placements: FxHashMap<usize, (u32, u32)> = if crate::flags().no_tile_alias {
        FxHashMap::default()
    } else {
        em.plan
            .placements
            .iter()
            .map(|p| (key(&p.tile), (p.byte_offset, p.byte_len)))
            .collect()
    };

    match em.plan.mode {
        ArenaMode::ByteArena if !placements.is_empty() => {
            let words = std::num::NonZeroU32::new(em.plan.total_bytes.div_ceil(4).max(1))
                .expect("max(1) is non-zero");
            let arena_ty = array_type(
                &mut em.module,
                ElementType::Scalar(ScalarElement::U32),
                ArraySize::Constant(words),
            )?;
            let arena = workgroup_var(&mut em.module, arena_ty);
            for tile in &tiles {
                match placements.get(&key(tile)) {
                    Some(&(byte_offset, _)) => {
                        if array_stride(tile.element)? != 4 {
                            return Err(EmitError::MissingCapability("workgroup-alias"));
                        }
                        em.tile_backing.insert(
                            key(tile),
                            TileBacking {
                                global: arena,
                                canonical: ElementType::Scalar(ScalarElement::U32),
                                base_index: byte_offset / 4,
                            },
                        );
                    }
                    None => standalone(em, tile)?,
                }
            }
        }
        _ => {
            // Regions: one global per distinct byte offset, in offset order.
            let mut groups: Vec<(u32, Vec<&fusor_ir::ir::kernel::Tile>)> = Vec::new();
            let mut ungrouped: Vec<&fusor_ir::ir::kernel::Tile> = Vec::new();
            for tile in &tiles {
                match placements.get(&key(tile)) {
                    Some(&(offset, _)) => match groups.iter_mut().find(|(o, _)| *o == offset) {
                        Some((_, members)) => members.push(tile),
                        None => groups.push((offset, vec![tile])),
                    },
                    None => ungrouped.push(tile),
                }
            }
            groups.sort_by_key(|(offset, _)| *offset);
            for (_, members) in &groups {
                let canonical = canonical_element(members)?;
                let stride = array_stride(canonical)?;
                let mut elements = 1u32;
                for tile in members {
                    let own = array_stride(tile.element)?;
                    if own != stride {
                        return Err(EmitError::Unsupported(
                            "a shared workgroup region needs one stride class".into(),
                        ));
                    }
                    elements = elements.max(tile.layout.element_count() as u32);
                }
                let count = std::num::NonZeroU32::new(elements)
                    .ok_or_else(|| EmitError::Unsupported("empty workgroup region".into()))?;
                let ty = array_type(&mut em.module, canonical, ArraySize::Constant(count))?;
                let global = workgroup_var(&mut em.module, ty);
                for tile in members {
                    em.tile_backing.insert(
                        key(tile),
                        TileBacking {
                            global,
                            canonical,
                            base_index: 0,
                        },
                    );
                }
            }
            for tile in ungrouped {
                standalone(em, tile)?;
            }
        }
    }
    Ok(())
}

fn standalone(em: &mut Emitter<'_>, tile: &fusor_ir::ir::kernel::Tile) -> Result<(), EmitError> {
    let global = workgroup_global(&mut em.module, tile)?;
    em.tile_backing.insert(
        key(tile),
        TileBacking {
            global,
            canonical: tile.element,
            base_index: 0,
        },
    );
    Ok(())
}

/// The element a shared region is typed with: its own when homogeneous, else
/// u32 with per-access bitcasts.
fn canonical_element(members: &[&fusor_ir::ir::kernel::Tile]) -> Result<ElementType, EmitError> {
    let first = members[0].element;
    if members.iter().all(|t| t.element == first) {
        return Ok(first);
    }
    for tile in members {
        if array_stride(tile.element)? != 4 {
            return Err(EmitError::Unsupported(
                "heterogeneous workgroup regions are limited to 32-bit scalars".into(),
            ));
        }
    }
    Ok(ElementType::Scalar(ScalarElement::U32))
}

/// Program locals, in first-use order.
pub(crate) fn create_private_locals(em: &mut Emitter<'_>) -> Result<(), EmitError> {
    let locals = em.analysis.locals.clone();
    for local in &locals {
        let ty = element_type(&mut em.module, local.element)?;
        let handle = em.fn_locals.append(
            naga::LocalVariable {
                name: None,
                ty,
                init: None,
            },
            Span::default(),
        );
        em.local_handles.insert(key(local), handle);
    }
    Ok(())
}

impl Emitter<'_> {
    /// Look up an already-interned element type; interning is idempotent in
    /// naga's `UniqueArena`, so this both reuses and registers.
    pub(crate) fn element_type(&mut self, element: ElementType) -> Result<Handle<Type>, EmitError> {
        if element.uses_f16() && !self.analysis.uses_f16 {
            // Unreachable: the analysis raises `uses_f16` for every f16.
            return Err(EmitError::MissingCapability("shader-f16"));
        }
        element_type(&mut self.module, element)
    }

    pub(crate) fn vector_type(
        &mut self,
        scalar: ScalarElement,
        lanes: u32,
    ) -> Result<Handle<Type>, EmitError> {
        self.element_type(ElementType::Vector { scalar, lanes })
    }
}
