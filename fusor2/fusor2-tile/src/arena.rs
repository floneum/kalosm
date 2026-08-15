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
//!
//! This is a **closed-form argmin with an independent verifier**, not an
//! e-graph alternative.

use fusor2_ir::Result;
use fusor2_ir::device::Caps;
use fusor2_ir::error::Error;
use fusor2_ir::ir::level2::{
    ArenaMode, ArenaPlan, ElementType, KernelIr, Placement, ScalarElement, Tiles,
};
use smallvec::SmallVec;

use crate::liveness::{LivenessInfo, TileLiveness};

/// A stride-compatibility class. `lanes` is part of the key so vec3 (12 B of
/// data, 16 B of stride) never mixes with vec4 and value bitcasts stay
/// per-component.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StrideClass {
    pub stride: u32,
    pub lanes: u32,
}

/// The stride class of an element, or `None` when it cannot back an array.
pub fn stride_class(element: ElementType) -> Option<StrideClass> {
    let stride = element.workgroup_array_stride()?;
    let lanes = match element {
        ElementType::Vector { lanes, .. } => lanes,
        _ => 1,
    };
    Some(StrideClass { stride, lanes })
}

/// The scalar backing an element, or `None` for a cooperative fragment.
pub fn scalar_of(element: ElementType) -> Option<ScalarElement> {
    match element {
        ElementType::Scalar(scalar) | ElementType::Vector { scalar, .. } => Some(scalar),
        ElementType::CoopMatrix { .. } => None,
    }
}

/// Whether tiles of elements `a` and `b` may occupy one typed region: equal
/// types always; otherwise the same stride class with a value-level bitcast
/// between them. Only 4-byte scalars qualify, so a 2-byte `f16`/`bf16` tile
/// never joins a 4-byte region.
pub fn bitcast_compatible(a: ElementType, b: ElementType) -> bool {
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
pub fn neutral(class: StrideClass) -> ElementType {
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
pub fn mixes_stride_widths(live: &LivenessInfo) -> bool {
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
pub fn all_packable(live: &LivenessInfo) -> bool {
    live.iter().all(|tile| stride_class(tile.element).is_some())
}

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

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
pub fn regions(live: &LivenessInfo) -> ArenaPlan {
    let mut regions: Vec<Region> = Vec::new();
    // Every occupant per region: the plain interval arm would be sound against
    // only the most recent occupant (barrier transitivity), but the loop-phase
    // arm does not compose — A->B and B->C do not imply the C->A wrap is
    // covered. Regions hold a handful of tiles, so all-occupants costs nothing.
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

// ---------------------------------------------------------------------------
// Byte arena
// ---------------------------------------------------------------------------

/// One byte arena, tiles at byte offsets by interval strip packing. Returns
/// `None` when a tile cannot back an array.
pub fn byte_arena(live: &LivenessInfo) -> Option<ArenaPlan> {
    if !all_packable(live) {
        return None;
    }
    struct Placed {
        start: u32,
        end: u32,
        position: usize,
    }
    let mut placed: Vec<Placed> = Vec::new();
    let mut placements: SmallVec<[Placement; 8]> = SmallVec::new();
    let mut arena_end = 0u32;

    for (position, &key) in live.order.iter().enumerate() {
        let tile = &live.tiles[&key];
        // Stride doubles as alignment: every supported element's array stride
        // is a power of two at least as large as its alignment (vec3 is
        // already padded to the vec4 stride).
        let align = element_stride(tile.element);
        let extent = tile_bytes(tile);
        let align_up = |value: u32| value.div_ceil(align.max(1)) * align.max(1);
        let mut candidates: Vec<u32> = std::iter::once(0)
            .chain(placed.iter().map(|entry| align_up(entry.end)))
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        let offset = candidates
            .into_iter()
            .find(|&offset| {
                // Full history, not just the most recent occupant.
                placed
                    .iter()
                    .filter(|entry| entry.start < offset + extent && entry.end > offset)
                    .all(|entry| {
                        live.can_follow_tiles(&live.tiles[&live.order[entry.position]], tile)
                    })
            })
            .expect("the offset past every placement always fits");
        let end = offset + extent;
        placed.push(Placed {
            start: offset,
            end,
            position,
        });
        arena_end = arena_end.max(end);
        placements.push(Placement {
            tile: tile.tile.clone(),
            byte_offset: offset,
            byte_len: extent,
        });
    }

    Some(ArenaPlan {
        mode: ArenaMode::ByteArena,
        total_bytes: arena_end.div_ceil(16) * 16,
        placements,
        barriers_inserted: SmallVec::new(),
    })
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Pack under one mode. Fails when the result exceeds
/// `caps.limits.max_compute_workgroup_storage_size`, or when `ByteArena` is
/// requested without `caps.workgroup_alias`.
pub fn pack(
    ir: &KernelIr,
    live: &LivenessInfo,
    mode: ArenaMode,
    caps: &Caps,
) -> Result<ArenaPlan> {
    let plan = match mode {
        ArenaMode::Regions => regions(live),
        ArenaMode::ByteArena => {
            if !caps.workgroup_alias {
                return Err(Error::Legality(
                    "byte-arena packing needs the workgroup-alias capability".into(),
                ));
            }
            byte_arena(live).ok_or_else(|| {
                Error::Legality(
                    "byte-arena packing needs every tile to back a workgroup array".into(),
                )
            })?
        }
    };
    check_budget(&plan, ir, caps)?;
    Ok(plan)
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

/// Bytes a declared tile set needs, without building a body. Delegates to the
/// shared [`crate::planner::Planner`] so this is the **same** computation the
/// emitters lay out with — there is no estimator and therefore no L1/L2
/// admission mismatch.
pub fn workgroup_bytes(tiles: &Tiles, caps: &Caps) -> Result<u32> {
    use fusor2_ir::ir::level2::ArenaPlanner;
    crate::planner::Planner::global().workgroup_bytes(tiles, caps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_never_joins_a_four_byte_region() {
        let f32e = ElementType::Scalar(ScalarElement::F32);
        let u32e = ElementType::Scalar(ScalarElement::U32);
        let f16e = ElementType::Scalar(ScalarElement::F16);
        let bf16e = ElementType::Scalar(ScalarElement::BF16);
        assert!(bitcast_compatible(f32e, u32e));
        assert!(!bitcast_compatible(f32e, f16e));
        assert!(!bitcast_compatible(f32e, bf16e));
        // bf16 and f16 share a stride class but are 2-byte scalars, so they
        // may not join either.
        assert!(!bitcast_compatible(f16e, bf16e));
        assert!(bitcast_compatible(bf16e, bf16e));
    }

    #[test]
    fn neutral_is_the_u32_shape() {
        assert_eq!(
            neutral(StrideClass {
                stride: 4,
                lanes: 1
            }),
            ElementType::Scalar(ScalarElement::U32)
        );
        assert_eq!(
            neutral(StrideClass {
                stride: 16,
                lanes: 4
            }),
            ElementType::Vector {
                scalar: ScalarElement::U32,
                lanes: 4
            }
        );
    }
}
