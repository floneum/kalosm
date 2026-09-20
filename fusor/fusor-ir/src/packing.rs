//! Byte placement with caller-defined interference and allocation order.
use crate::{Error, Result};

#[derive(Clone, Copy)]
pub enum Fit {
    First,
    Best,
}

/// Pack `(bytes, alignment)` requests in their supplied order. `conflicts(i, j)`
/// requires disjoint bytes for requests `i` and previously placed request `j`.
/// Every prior occupant participates; sharing need not be transitive.
pub fn pack_interference(
    requests: &[(u64, u64)],
    fit: Fit,
    conflicts: impl Fn(usize, usize) -> bool,
) -> Result<(u64, Vec<u64>)> {
    let overflow = || Error::Plan("arena byte offsets overflow u64".into());
    let mut offsets = Vec::with_capacity(requests.len());
    let mut extent = 0;
    for (i, &(bytes, alignment)) in requests.iter().enumerate() {
        let alignment = alignment.max(1);
        let aligned = |value: u64| {
            value
                .checked_add(alignment - 1)
                .map(|v| v / alignment * alignment)
                .ok_or_else(overflow)
        };
        let mut occupied: Vec<_> = offsets
            .iter()
            .enumerate()
            .filter(|(j, _)| conflicts(i, *j))
            .map(|(j, &offset)| (offset, offset + requests[j].0))
            .collect();
        occupied.sort_unstable();
        let tail = occupied.iter().map(|(_, end)| *end).max().unwrap_or(0);
        occupied.push((extent, extent));
        let mut cursor = 0;
        let mut best = None;
        for (start, end) in occupied {
            let offset = aligned(cursor)?;
            let hole = start.saturating_sub(offset);
            if hole >= bytes && best.is_none_or(|(_, size)| hole < size) {
                best = Some((offset, hole));
                if matches!(fit, Fit::First) {
                    break;
                }
            }
            cursor = cursor.max(end);
        }
        let offset = match best {
            Some((offset, _)) => offset,
            None => aligned(match fit {
                Fit::First => tail,
                Fit::Best => extent,
            })?,
        };
        extent = extent.max(offset.checked_add(bytes).ok_or_else(overflow)?);
        offsets.push(offset);
    }
    Ok((extent, offsets))
}
