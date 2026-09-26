//! Workgroup arena packing, in both modes.
//!
//! - [`ArenaMode::Regions`] (portable): tiles pack into per-stride-class typed
//!   arrays. A region holding two element types is emitted with a
//!   class-neutral u32 type and every access bitcasts the *value*, never the
//!   address — legal only for 32-bit scalars, so `f16`/`bf16` tiles never join
//!   a 4-byte region and no sub-word read-modify-write hazard exists.
//! - [`ArenaMode::ByteArena`] (needs `caps.workgroup_alias`): one byte arena,
//!   tiles at byte offsets via interval strip packing, so tiles of *different*
//!   strides (f16 staging next to f32 accumulators) reuse the same bytes.
//!
//! In **both** modes a [`Placement`]'s `[byte_offset, byte_offset + byte_len)`
//! means "these bytes": in `Regions` every tile of region `k` reports region
//! `k`'s base and full length, so [`crate::verify_arena`] needs no synthetic
//! offsets and one overlap test covers both modes.
//!
//! Sharing legality is [`LivenessInfo::can_follow_tiles`]. Both packers check
//! **every** prior occupant, not just the most recent: the loop-phase arm does
//! not compose transitively.

use fusor_ir::Result;
use fusor_ir::device::Caps;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{ArenaMode, ArenaPlan, ElementType, KernelIr, Placement, ScalarElement};
use smallvec::SmallVec;

use crate::liveness::{LivenessInfo, TileLiveness};

/// A stride-compatibility class. `lanes` is part of the key so vec3 (12 B of
/// data, 16 B of stride) never mixes with vec4 and value bitcasts stay
/// per-component.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StrideClass {
    pub stride: u32,
    pub lanes: u32,
}

/// The stride class of an element, or `None` when it cannot back an array.
pub(crate) fn stride_class(element: ElementType) -> Option<StrideClass> {
    let stride = element.workgroup_array_stride()?;
    let lanes = match element {
        ElementType::Vector { lanes, .. } => lanes,
        _ => 1,
    };
    Some(StrideClass { stride, lanes })
}

/// The scalar backing an element, or `None` for a cooperative fragment.
pub(crate) fn scalar_of(element: ElementType) -> Option<ScalarElement> {
    match element {
        ElementType::Scalar(scalar) | ElementType::Vector { scalar, .. } => Some(scalar),
        ElementType::CoopMatrix { .. } => None,
    }
}

/// Whether tiles of elements `a` and `b` may occupy one typed region: equal
/// types always; otherwise the same stride class with a value-level bitcast
/// between them. Only 4-byte scalars qualify, so a 2-byte `f16`/`bf16` tile
/// never joins a 4-byte region.
pub(crate) fn bitcast_compatible(a: ElementType, b: ElementType) -> bool {
    if a == b {
        return true;
    }
    let (Some(class_a), Some(class_b)) = (stride_class(a), stride_class(b)) else {
        return false;
    };
    class_a == class_b
        && scalar_of(a).is_some_and(|s| s.byte_size() == 4)
        && scalar_of(b).is_some_and(|s| s.byte_size() == 4)
}

/// The canonical emission type for a heterogeneous region of this class: the
/// u32-shaped type.
pub(crate) fn neutral(class: StrideClass) -> ElementType {
    if class.lanes == 1 {
        ElementType::Scalar(ScalarElement::U32)
    } else {
        ElementType::Vector {
            scalar: ScalarElement::U32,
            lanes: class.lanes,
        }
    }
}

/// Bytes one tile occupies, at its element's array stride.
fn tile_bytes(tile: &TileLiveness) -> u32 {
    tile.elements.saturating_mul(element_stride(tile.element))
}

fn element_stride(element: ElementType) -> u32 {
    element
        .workgroup_array_stride()
        .unwrap_or_else(|| element.byte_size() as u32)
}

/// Whether the kernel mixes array stride widths. Without that, the byte
/// arena's 16-byte rounding makes it a strict loss.
pub(crate) fn mixes_stride_widths(live: &LivenessInfo) -> bool {
    let mut strides: SmallVec<[u32; 4]> = SmallVec::new();
    for tile in live.iter() {
        if let Some(stride) = tile.element.workgroup_array_stride()
            && !strides.contains(&stride)
        {
            strides.push(stride);
        }
    }
    strides.len() > 1
}

/// Whether every tile can back an array at all.
pub(crate) fn all_packable(live: &LivenessInfo) -> bool {
    live.iter().all(|tile| stride_class(tile.element).is_some())
}

struct Region {
    canonical: ElementType,
    elements: u32,
}

impl Region {
    fn bytes(&self) -> u32 {
        self.elements.saturating_mul(element_stride(self.canonical))
    }
}

/// One allocation per stride class, tiles sharing a region when their live
/// ranges are barrier-separated. The universal fallback: needs no capability
/// and no aliasing proof.
pub(crate) fn regions(live: &LivenessInfo) -> ArenaPlan {
    let mut regions: Vec<Region> = Vec::new();
    // Check every occupant per region: the loop-phase arm does not compose —
    // A->B and B->C do not imply the C->A wrap is covered.
    let mut occupants: Vec<Vec<usize>> = Vec::new();
    // A coop-consumed occupant pins the region's type: widening the canonical
    // would retype the raw pointer its cooperative load/store sees.
    let mut region_coop: Vec<bool> = Vec::new();
    let mut assigned: Vec<usize> = Vec::with_capacity(live.order.len());

    for (position, &key) in live.order.iter().enumerate() {
        let tile = &live.tiles[&key];
        let reused = regions.iter().enumerate().position(|(index, region)| {
            let type_ok = region.canonical == tile.element
                || (bitcast_compatible(region.canonical, tile.element)
                    && !tile.coop
                    && !region_coop[index]);
            type_ok
                && occupants[index].iter().all(|&occupant| {
                    live.can_follow_tiles(&live.tiles[&live.order[occupant]], tile)
                })
        });
        let index = match reused {
            Some(index) => {
                let region = &mut regions[index];
                region.elements = region.elements.max(tile.elements);
                if region.canonical != tile.element {
                    region.canonical = neutral(
                        stride_class(tile.element).expect("bitcast-compatible implies a class"),
                    );
                }
                region_coop[index] |= tile.coop;
                occupants[index].push(position);
                index
            }
            None => {
                regions.push(Region {
                    canonical: tile.element,
                    elements: tile.elements,
                });
                occupants.push(vec![position]);
                region_coop.push(tile.coop);
                regions.len() - 1
            }
        };
        assigned.push(index);
    }

    // Region k's base is the sum of the byte lengths of regions 0..k, so
    // "byte intervals overlap" means "same bytes" in Regions mode too.
    let mut bases: Vec<u32> = Vec::with_capacity(regions.len());
    let mut base = 0u32;
    for region in &regions {
        bases.push(base);
        base = base.saturating_add(region.bytes());
    }
    let total_bytes = base;

    let placements = live
        .order
        .iter()
        .zip(&assigned)
        .map(|(key, &index)| Placement {
            tile: live.tiles[key].tile.clone(),
            byte_offset: bases[index],
            byte_len: regions[index].bytes(),
        })
        .collect();

    ArenaPlan {
        mode: ArenaMode::Regions,
        total_bytes,
        placements,
        barriers_inserted: SmallVec::new(),
    }
}

/// One byte arena, tiles at byte offsets by interval strip packing. Returns
/// `None` when a tile cannot back an array.
pub(crate) fn byte_arena(live: &LivenessInfo) -> Option<ArenaPlan> {
    if !all_packable(live) {
        return None;
    }
    let requests: Vec<_> = live
        .iter()
        .map(|tile| {
            (
                u64::from(tile_bytes(tile)),
                u64::from(element_stride(tile.element)),
            )
        })
        .collect();
    let (extent, offsets) =
        fusor_ir::packing::pack_interference(&requests, fusor_ir::packing::Fit::First, |a, b| {
            !live.can_follow_tiles(&live.tiles[&live.order[b]], &live.tiles[&live.order[a]])
        })
        .ok()?;
    let total_bytes = u32::try_from(extent.div_ceil(16) * 16).ok()?;
    let placements = live
        .iter()
        .zip(offsets)
        .map(|(tile, offset)| Placement {
            tile: tile.tile.clone(),
            byte_offset: offset as u32,
            byte_len: tile_bytes(tile),
        })
        .collect();

    Some(ArenaPlan {
        mode: ArenaMode::ByteArena,
        total_bytes,
        placements,
        barriers_inserted: SmallVec::new(),
    })
}

pub(crate) fn check_budget(plan: &ArenaPlan, ir: &KernelIr, caps: &Caps) -> Result<()> {
    let budget = caps.limits.max_compute_workgroup_storage_size;
    if plan.total_bytes > budget {
        return Err(Error::Legality(format!(
            "kernel {} needs {} workgroup bytes, the device allows {budget}",
            ir.name, plan.total_bytes
        )));
    }
    Ok(())
}
