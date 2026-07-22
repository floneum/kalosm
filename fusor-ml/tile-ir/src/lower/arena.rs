//! Workgroup-tile allocation packing.
//!
//! Consumes [`crate::analysis::LivenessInfo`] and places each workgroup tile
//! into shared memory. Two modes:
//!
//! - [`ArenaMode::Regions`] (portable): tiles pack into per-region typed
//!   arrays, every tile at offset 0. Tiles of the same stride class share a
//!   region when their live ranges are barrier-separated; a region holding
//!   more than one element type is emitted with a class-neutral canonical
//!   type and every access bitcasts the value (never the address — within a
//!   stride class, element index `i` names the same bytes for every type).
//! - [`ArenaMode::ByteArena`] (Metal fork): one byte arena, tiles at byte
//!   offsets via interval strip-packing, so tiles of *different* strides
//!   (f16 staging next to f32 accumulators) can reuse the same bytes.
//!
//! Sharing legality is [`LivenessInfo::can_follow`]: disjoint expanded live
//! ranges plus a guaranteed uniform barrier between them. In the byte arena
//! the "previous occupant" is per byte interval, tracked as a segment list;
//! transitivity of the barrier chain applies pointwise per byte.

use rustc_hash::FxHashMap;

use crate::ElementType;
use crate::analysis::{LivenessInfo, trace_enabled};
use crate::ir::ScalarElement;

/// How this kernel's workgroup tiles are laid out.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArenaMode {
    /// One naga global per region; every tile at offset 0 in its region.
    Regions,
    /// One threadgroup byte arena; tiles at byte offsets via the workgroup
    /// alias extension. Chosen only when the kernel proves backend support
    /// (`KernelIr::byte_arena`) and mixes stride widths.
    ByteArena,
}

/// A stride-compatibility class. `lanes` is part of the key so vec3 (data
/// 12 B, stride 16 B) never mixes with vec4, and value bitcasts stay
/// per-component.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct StrideClass {
    stride: u32,
    lanes: u32,
}

fn stride_class(element: ElementType) -> Option<StrideClass> {
    let stride = element.workgroup_array_stride()?;
    let lanes = match element {
        ElementType::Vector { lanes, .. } => lanes,
        _ => 1,
    };
    Some(StrideClass { stride, lanes })
}

/// Whether tiles of elements `a` and `b` may occupy one typed region: equal
/// types always; otherwise the same stride class with a value-level bitcast
/// between them. Only 32-bit-scalar casts qualify (`f32 <-> u32`, per
/// component for vectors): f16 has no same-width partner, so 2-byte tiles
/// never join a 4-byte region and no sub-word read-modify-write hazard can
/// arise.
fn bitcast_compatible(a: ElementType, b: ElementType) -> bool {
    if a == b {
        return true;
    }
    let (Some(class_a), Some(class_b)) = (stride_class(a), stride_class(b)) else {
        return false;
    };
    class_a == class_b && scalar_of(a).is_some_and(|scalar| scalar.byte_size() == 4)
        && scalar_of(b).is_some_and(|scalar| scalar.byte_size() == 4)
}

fn scalar_of(element: ElementType) -> Option<ScalarElement> {
    match element {
        ElementType::F32 => Some(ScalarElement::F32),
        ElementType::U32 => Some(ScalarElement::U32),
        ElementType::F16 => Some(ScalarElement::F16),
        ElementType::Vector { scalar, .. } => Some(scalar),
        ElementType::Bool | ElementType::CoopMatrix { .. } => None,
    }
}

/// The canonical emission type for a heterogeneous region of this class:
/// the u32-based type of the same shape.
fn neutral(class: StrideClass) -> ElementType {
    if class.lanes == 1 {
        ElementType::U32
    } else {
        ElementType::vector(ScalarElement::U32, class.lanes)
    }
}

/// One time-shared typed allocation (Regions mode).
pub(crate) struct Region {
    /// Emission element type of the backing array: the occupant element
    /// while homogeneous (bit-identical emission to unshared lowering),
    /// widened to the class-neutral u32 form on the first cross-type join.
    pub canonical: ElementType,
    /// Array length in canonical elements (stride-equal across the class,
    /// so occupant extents compare directly).
    pub elements: u32,
}

#[derive(Clone, Copy)]
pub(crate) enum Placement {
    Region { index: usize },
    Arena { byte_offset: u32 },
}

/// The computed tile placement for one kernel.
pub(crate) struct TileArena {
    pub mode: ArenaMode,
    /// Regions in creation order (empty in ByteArena mode).
    pub regions: Vec<Region>,
    /// Packed arena extent, 16-byte aligned (0 in Regions mode).
    pub arena_bytes: u32,
    /// Tile identity (`Rc` pointer) -> placement.
    pub assignment: FxHashMap<*const (), Placement>,
}

impl TileArena {
    pub(crate) fn assign(info: &LivenessInfo, byte_arena: bool) -> Self {
        let mut strides = Vec::new();
        for key in &info.order {
            let tile = &info.tiles[key];
            match tile.element.workgroup_array_stride() {
                Some(stride) if !strides.contains(&stride) => strides.push(stride),
                _ => {}
            }
        }
        let mixed_strides = strides.len() > 1;
        let all_packable = info
            .order
            .iter()
            .all(|key| stride_class(info.tiles[key].element).is_some());
        let regions = Self::assign_regions(info);
        if byte_arena && mixed_strides && all_packable {
            // The arena only wins when cross-stride reuse actually fires:
            // without it, 16-byte rounding makes it a strict loss, so pick
            // by measured footprint.
            let packed = Self::assign_byte_arena(info);
            if packed.total_bytes() < regions.total_bytes() {
                return packed;
            }
        }
        regions
    }

    fn assign_regions(info: &LivenessInfo) -> Self {
        let mut regions: Vec<Region> = Vec::new();
        // Every occupant per region. The plain interval arm would be sound
        // checking only the most recent occupant (barrier transitivity),
        // but the loop-phase arm does not compose: A->B and B->C phase
        // separation does not imply the C->A wrap is covered. Regions hold
        // a handful of tiles, so checking all occupants costs nothing.
        let mut region_occupants: Vec<Vec<*const ()>> = Vec::new();
        // A coop-consumed occupant pins the region's type: widening the
        // canonical would retype the raw pointer its CooperativeLoad/Store
        // sees.
        let mut region_coop: Vec<bool> = Vec::new();
        let mut assignment = FxHashMap::default();
        for &key in &info.order {
            let tile = &info.tiles[&key];
            let (range, element, elements) = (tile.range, tile.element, tile.elements);
            let reused = regions.iter().enumerate().position(|(index, region)| {
                let type_ok = region.canonical == element
                    || (bitcast_compatible(region.canonical, element)
                        && !tile.coop
                        && !region_coop[index]);
                type_ok
                    && region_occupants[index]
                        .iter()
                        .all(|occupant| info.can_follow_tiles(&info.tiles[occupant], tile))
            });
            let index = match reused {
                Some(index) => {
                    if trace_enabled() {
                        eprintln!(
                            "arena-share region={index} element={:?} elems={elements} range=({},{}) occupants={}",
                            element,
                            range.first,
                            range.last,
                            region_occupants[index].len()
                        );
                    }
                    let region = &mut regions[index];
                    region.elements = region.elements.max(elements);
                    if region.canonical != element {
                        region.canonical = neutral(
                            stride_class(element).expect("bitcast-compatible implies a class"),
                        );
                    }
                    region_coop[index] |= tile.coop;
                    region_occupants[index].push(key);
                    index
                }
                None => {
                    regions.push(Region {
                        canonical: element,
                        elements,
                    });
                    region_occupants.push(vec![key]);
                    region_coop.push(tile.coop);
                    regions.len() - 1
                }
            };
            assignment.insert(key, Placement::Region { index });
        }
        Self {
            mode: ArenaMode::Regions,
            regions,
            arena_bytes: 0,
            assignment,
        }
    }

    fn assign_byte_arena(info: &LivenessInfo) -> Self {
        // Full placement history: with the loop-phase arm, legality against
        // only each byte's most recent occupant does not compose, so a
        // candidate is checked against EVERY placement its bytes overlap.
        struct Placed {
            start: u32,
            end: u32,
            key: *const (),
        }
        let mut placed: Vec<Placed> = Vec::new();
        let mut assignment = FxHashMap::default();
        let mut arena_end = 0u32;
        for &key in &info.order {
            let tile = &info.tiles[&key];
            let stride = tile
                .element
                .workgroup_array_stride()
                .expect("mode selection requires packable elements");
            let extent = tile.elements * stride;
            // Stride doubles as alignment: every supported element's Naga
            // array stride is a power of two at least as large as its
            // alignment (vec3 already padded to the vec4 stride).
            let align = stride;
            let align_up = |value: u32| value.div_ceil(align) * align;
            let mut candidates: Vec<u32> = std::iter::once(0)
                .chain(placed.iter().map(|entry| align_up(entry.end)))
                .collect();
            candidates.sort_unstable();
            candidates.dedup();
            let offset = candidates
                .into_iter()
                .find(|&offset| {
                    placed
                        .iter()
                        .filter(|entry| entry.start < offset + extent && entry.end > offset)
                        .all(|entry| info.can_follow_tiles(&info.tiles[&entry.key], tile))
                })
                .expect("the offset past every placement always fits");
            if trace_enabled() {
                eprintln!(
                    "arena-pack offset={offset} bytes={extent} element={:?} range=({},{})",
                    tile.element, tile.range.first, tile.range.last
                );
            }
            let end = offset + extent;
            placed.push(Placed {
                start: offset,
                end,
                key,
            });
            arena_end = arena_end.max(end);
            assignment.insert(key, Placement::Arena {
                byte_offset: offset,
            });
        }
        Self {
            mode: ArenaMode::ByteArena,
            regions: Vec::new(),
            arena_bytes: arena_end.div_ceil(16) * 16,
            assignment,
        }
    }

    /// Total post-arena workgroup footprint in bytes.
    pub(crate) fn total_bytes(&self) -> u64 {
        match self.mode {
            ArenaMode::Regions => self
                .regions
                .iter()
                .map(|region| {
                    let stride = region
                        .canonical
                        .workgroup_array_stride()
                        .map(u64::from)
                        // Elements that cannot back an array (rejected at
                        // emission) still count their data size.
                        .unwrap_or_else(|| region.canonical.byte_size());
                    u64::from(region.elements) * stride
                })
                .sum(),
            ArenaMode::ByteArena => u64::from(self.arena_bytes),
        }
    }
}
