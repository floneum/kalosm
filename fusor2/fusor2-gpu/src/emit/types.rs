//! L2 [`ElementType`] -> naga types, and the workgroup/storage address-space
//! declarations.
//!
//! Types are interned in a **fixed order** so the module's type arena is
//! deterministic: emitting the same IR twice must produce byte-identical
//! debug output, which is what makes the plan hash a usable cache key.
//!
//! Owned by W8.

use fusor2_ir::ir::level2::{
    ArenaMode, BufferAccess, BufferDecl, ElementType, ScalarElement, TileDecl,
};
use fusor2_ir::target::EmitError;
use naga::{
    AddressSpace, ArraySize, GlobalVariable, Handle, ResourceBinding, Scalar, Span, StorageAccess,
    Type, TypeInner, VectorSize,
};
use rustc_hash::FxHashMap;

use super::{Analysis, Emitter, key};

/// How one workgroup tile is backed. Access is always
/// `global[base_index + tile_index]`, with the value bitcast between
/// `canonical` and the tile's own element when a region is heterogeneous.
#[derive(Copy, Clone, Debug)]
pub struct TileBacking {
    pub global: Handle<GlobalVariable>,
    pub canonical: ElementType,
    pub base_index: u32,
}

/// The two prelude handles the entry point itself needs. Everything else is
/// looked up through naga's `UniqueArena`, which interns structurally.
pub struct Prelude {
    pub u32_ty: Handle<Type>,
    pub u32_vec3_ty: Handle<Type>,
}

/// The naga scalar for one L2 scalar element.
///
/// `BF16` has no naga 29 representation: it is a storage-only dtype whose
/// compute form the `widen-compute` L1 rule produces, so it must never reach
/// L2 as a value type.
pub fn scalar_of(scalar: ScalarElement) -> Result<Scalar, EmitError> {
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

fn cooperative_size(size: u32) -> Result<naga::CooperativeSize, EmitError> {
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
            role: match role {
                fusor2_ir::ir::level2::CoopMatrixRole::A => naga::CooperativeRole::A,
                fusor2_ir::ir::level2::CoopMatrixRole::B => naga::CooperativeRole::B,
                fusor2_ir::ir::level2::CoopMatrixRole::C => naga::CooperativeRole::C,
            },
        },
    })
}

fn insert(module: &mut naga::Module, inner: TypeInner) -> Handle<Type> {
    module
        .types
        .insert(Type { name: None, inner }, Span::default())
}

/// Register (or reuse) the naga type for one element type.
pub fn element_type(
    module: &mut naga::Module,
    element: ElementType,
) -> Result<Handle<Type>, EmitError> {
    Ok(insert(module, type_inner(element)?))
}

/// Intern the prelude in a fixed order: f32, f32x2/3/4, i32, i32x4, u32,
/// u32x2/3/4, bool, boolx2/3/4, then the f16 quad only when the analysis says
/// it is used, then every cooperative-matrix element the locals list mentions.
pub fn intern_prelude(
    module: &mut naga::Module,
    analysis: &Analysis,
) -> Result<Prelude, EmitError> {
    insert(module, TypeInner::Scalar(Scalar::F32));
    for lanes in [2u32, 3, 4] {
        insert(
            module,
            TypeInner::Vector {
                size: vector_size(lanes)?,
                scalar: Scalar::F32,
            },
        );
    }
    insert(module, TypeInner::Scalar(Scalar::I32));
    insert(
        module,
        TypeInner::Vector {
            size: VectorSize::Quad,
            scalar: Scalar::I32,
        },
    );
    let u32_ty = insert(module, TypeInner::Scalar(Scalar::U32));
    let mut u32_vec = [None; 3];
    for (slot, lanes) in [2u32, 3, 4].into_iter().enumerate() {
        u32_vec[slot] = Some(insert(
            module,
            TypeInner::Vector {
                size: vector_size(lanes)?,
                scalar: Scalar::U32,
            },
        ));
    }
    insert(module, TypeInner::Scalar(Scalar::BOOL));
    for lanes in [2u32, 3, 4] {
        insert(
            module,
            TypeInner::Vector {
                size: vector_size(lanes)?,
                scalar: Scalar::BOOL,
            },
        );
    }
    if analysis.uses_f16 {
        insert(module, TypeInner::Scalar(Scalar::F16));
        for lanes in [2u32, 3, 4] {
            insert(
                module,
                TypeInner::Vector {
                    size: vector_size(lanes)?,
                    scalar: Scalar::F16,
                },
            );
        }
    }
    // Cooperative-matrix types up front, in locals-list order.
    let mut seen: FxHashMap<ElementType, Handle<Type>> = FxHashMap::default();
    for local in &analysis.locals {
        if matches!(local.element, ElementType::CoopMatrix { .. })
            && !seen.contains_key(&local.element)
        {
            let handle = element_type(module, local.element)?;
            seen.insert(local.element, handle);
        }
    }
    Ok(Prelude {
        u32_ty,
        u32_vec3_ty: u32_vec[1].expect("u32x3 interned above"),
    })
}

/// Array stride for a workgroup/storage array of `element`. The single source
/// of stride truth: arena packing and module emission both read
/// [`ElementType::workgroup_array_stride`], so they cannot disagree.
fn array_stride(element: ElementType) -> Result<u32, EmitError> {
    element
        .workgroup_array_stride()
        .ok_or_else(|| EmitError::Unsupported(format!("{element:?} cannot back an array")))
}

fn array_type(
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
    // `AtomicAdd` on f32 runs a bitcast compare-exchange loop over a u32
    // atomic, so an f32 buffer is typed `array<atomic<u32>>` and the value is
    // bitcast at each step.
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

/// Declare a storage buffer global. Read-only-ness comes from
/// [`BufferDecl::access`] and is what [`crate::bindings`] reads back out.
///
/// The array is typed `array<atomic<..>>` when the analysis found a
/// [`fusor2_ir::ir::level2::Stmt::AtomicAdd`] on this binding.
pub fn storage_global_with(
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

/// Declare a workgroup tile global sized for its own extent.
///
/// `byte_offset` is accepted for signature compatibility with the packed-arena
/// caller; a standalone tile always starts at zero.
pub fn workgroup_global(
    module: &mut naga::Module,
    decl: &TileDecl,
    byte_offset: u32,
) -> Result<Handle<GlobalVariable>, EmitError> {
    debug_assert_eq!(byte_offset, 0, "standalone tiles are not aliased");
    let count = std::num::NonZeroU32::new(decl.layout.element_count() as u32)
        .ok_or_else(|| EmitError::Unsupported("empty workgroup tile".into()))?;
    let ty = array_type(module, decl.element, ArraySize::Constant(count))?;
    Ok(module.global_variables.append(
        GlobalVariable {
            name: None,
            space: AddressSpace::WorkGroup,
            binding: None,
            ty,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::default(),
    ))
}

// ---------------------------------------------------------------------------
// Emitter-driven declaration
// ---------------------------------------------------------------------------

/// Buffers in **binding order**, so the global-variable arena is independent
/// of which statement touches which buffer first.
pub fn create_storage_globals(em: &mut Emitter<'_>) -> Result<(), EmitError> {
    let mut buffers = em.analysis.buffers.clone();
    buffers.sort_by_key(|b| b.binding);
    for buffer in &buffers {
        let atomic = em.analysis.atomic_buffers.contains(&buffer.binding);
        let global = storage_global_with(&mut em.module, buffer, atomic)?;
        em.buffer_globals.insert(key(buffer), global);
    }
    Ok(())
}

/// Workgroup tiles, laid out from the plan.
///
/// `ArenaMode::Regions` groups placements by byte offset — tiles that share an
/// allocation share an offset — and emits one global per group, typed with the
/// group's canonical element; a heterogeneous group bitcasts the *value* at
/// each access, never the address, which is legal only between 32-bit scalars.
///
/// `ArenaMode::ByteArena` emits one `array<u32>` arena and indexes each tile
/// from its packed byte offset. Released naga has no `WorkgroupAlias`
/// decoration, so aliasing is expressed as index arithmetic instead; that
/// restricts a byte-arena tile to 4-byte scalar elements, which is a footprint
/// restriction, not a correctness one — a kernel that needs more falls back to
/// `Regions`.
///
/// A tile with no placement gets its own allocation. An empty or partial plan
/// is therefore always emittable, just larger.
pub fn create_workgroup_globals(em: &mut Emitter<'_>) -> Result<(), EmitError> {
    let tiles = em.analysis.tiles.clone();
    let placements: FxHashMap<usize, (u32, u32)> = em
        .plan
        .placements
        .iter()
        .map(|p| (key(&p.tile), (p.byte_offset, p.byte_len)))
        .collect();

    match em.plan.mode {
        ArenaMode::ByteArena if !placements.is_empty() => {
            let words = std::num::NonZeroU32::new(em.plan.total_bytes.div_ceil(4).max(1))
                .expect("max(1) is non-zero");
            let arena_ty = array_type(
                &mut em.module,
                ElementType::Scalar(ScalarElement::U32),
                ArraySize::Constant(words),
            )?;
            let arena = em.module.global_variables.append(
                GlobalVariable {
                    name: None,
                    space: AddressSpace::WorkGroup,
                    binding: None,
                    ty: arena_ty,
                    init: None,
                    memory_decorations: naga::MemoryDecorations::empty(),
                },
                Span::default(),
            );
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
            let mut groups: Vec<(u32, Vec<&fusor2_ir::ir::level2::Tile>)> = Vec::new();
            let mut ungrouped: Vec<&fusor2_ir::ir::level2::Tile> = Vec::new();
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
                let global = em.module.global_variables.append(
                    GlobalVariable {
                        name: None,
                        space: AddressSpace::WorkGroup,
                        binding: None,
                        ty,
                        init: None,
                        memory_decorations: naga::MemoryDecorations::empty(),
                    },
                    Span::default(),
                );
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

fn standalone(em: &mut Emitter<'_>, tile: &fusor2_ir::ir::level2::Tile) -> Result<(), EmitError> {
    let global = workgroup_global(&mut em.module, tile, 0)?;
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

/// The element a shared region is typed with. A homogeneous region keeps its
/// own element; a heterogeneous one goes class-neutral u32 and bitcasts values
/// at each access.
fn canonical_element(members: &[&fusor2_ir::ir::level2::Tile]) -> Result<ElementType, EmitError> {
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
pub fn create_private_locals(em: &mut Emitter<'_>) -> Result<(), EmitError> {
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
            // Unreachable: the analysis raises `uses_f16` for every f16 that
            // appears anywhere. Kept as an assertion against a future emitter
            // that synthesizes an f16 value out of thin air.
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
