use super::*;

/// A masked value to be filled with `fill` outside the mask. `spill_depth`
/// drives demand-allocated spill local selection.
#[derive(Clone, Copy)]
pub(in crate::lower) struct MaskedF32Value<'a> {
    pub(in crate::lower) mask: &'a Expr,
    pub(in crate::lower) fill: &'a Expr,
    pub(in crate::lower) spill_depth: usize,
}

pub(in crate::lower) struct MaskedLocalValue<'a> {
    pub(in crate::lower) mask: &'a Expr,
    pub(in crate::lower) element: ElementType,
    pub(in crate::lower) fill: Handle<Expression>,
    pub(in crate::lower) spill_depth: usize,
}

pub(in crate::lower) struct StorageLoadLowering<'a> {
    pub(in crate::lower) src: &'a StorageView,
    pub(in crate::lower) mask: &'a Expr,
    pub(in crate::lower) fill: &'a Expr,
    pub(in crate::lower) spill_depth: usize,
}

mod expr;
mod load;
mod quantized;
mod scalar;
mod stmt;
mod types;
