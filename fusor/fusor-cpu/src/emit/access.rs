//! Lane-uniformity of an expression: a barrier may only sit under a uniform branch.

use fusor_ir::ir::kernel::{Builtin, TileExpr, TileExprKind};

/// Whether an expression can vary across lanes; a `Load` counts as divergent.
pub(crate) fn is_lane_uniform(e: &TileExpr) -> bool {
    use TileExprKind as K;
    match e.kind() {
        K::Builtin(Builtin::Lane)
        | K::Builtin(Builtin::SubgroupLane)
        | K::Builtin(Builtin::SubgroupId) => false,
        K::Literal(_) | K::Builtin(_) | K::LoadLocal(_) => true,
        K::Load { .. } | K::LoadTile { .. } => false,
        K::Unary { value, .. } => is_lane_uniform(value),
        K::Binary { left, right, .. } | K::Compare { left, right, .. } => {
            is_lane_uniform(left) && is_lane_uniform(right)
        }
        K::Round { value, .. } | K::Cast { value, .. } | K::Bitcast { value, .. } => {
            is_lane_uniform(value)
        }
        K::Select {
            condition,
            accept,
            reject,
        } => is_lane_uniform(condition) && is_lane_uniform(accept) && is_lane_uniform(reject),
        K::Vec { parts, .. } => parts.iter().all(is_lane_uniform),
        K::VecComponent { vector, .. } => is_lane_uniform(vector),
        K::Dot { left, right } => is_lane_uniform(left) && is_lane_uniform(right),
        // A cross-lane reduce is uniform across its group by construction.
        K::Reduce { .. } => true,
        K::CoopLoad { .. } | K::CoopMma { .. } | K::CoopZero { .. } => false,
    }
}
