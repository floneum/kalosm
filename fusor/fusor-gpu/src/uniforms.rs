//! Packing binding 0: `[u32 symbolic dims..., f32 uniform scalars...]`, a
//! storage buffer so the derived bind-group mechanism sees it.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::extract::Plan;
use fusor_ir::shape::{Dim, SymId};
use fusor_ir::target::Uniforms;
use rustc_hash::FxHashMap;

/// The word layout of binding 0 for one plan: a function of the plan alone,
/// so a sequence-length change recompiles nothing.
#[derive(Clone, Debug, Default)]
pub(crate) struct UniformPack {
    /// Symbols carried as `u32` extents, in `Plan::symbols` order.
    dim_syms: Vec<SymId>,
    /// Symbols carried as `f32` runtime scalars, in `Plan::symbols` order.
    scalar_syms: Vec<SymId>,
    sym_index: FxHashMap<SymId, u32>,
    scalar_index: FxHashMap<SymId, u32>,
}

impl UniformPack {
    /// The layout's identity: its dim and scalar symbol order, which kernel bodies
    /// bake as slot indices.
    pub(crate) fn digest(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = rustc_hash::FxHasher::default();
        self.dim_syms.hash(&mut h);
        self.scalar_syms.hash(&mut h);
        h.finish()
    }

    /// Derive binding 0's word layout from a plan. A symbol is a scalar when the
    /// plan names it a `Leaf::Uniform`, else a dim.
    pub(crate) fn new(plan: &Plan) -> Self {
        let mut dim_syms = Vec::new();
        let mut scalar_syms = Vec::new();
        for &sym in &plan.symbols {
            if sym == fusor_ir::shape::OPAQUE_SYM {
                continue;
            }
            if plan.scalar_symbols.contains(&sym) {
                if !scalar_syms.contains(&sym) {
                    scalar_syms.push(sym);
                }
            } else if !dim_syms.contains(&sym) {
                dim_syms.push(sym);
            }
        }

        let sym_index = dim_syms
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i as u32))
            .collect();
        let base = dim_syms.len() as u32;
        let scalar_index = scalar_syms
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, base + i as u32))
            .collect();

        Self {
            dim_syms,
            scalar_syms,
            sym_index,
            scalar_index,
        }
    }

    /// Fill a pre-derived layout. This is the per-dispatch path: it allocates
    /// two `Vec`s and does no hashing of the plan.
    pub(crate) fn fill(
        &self,
        binding: &FxHashMap<SymId, u64>,
        scalars: &FxHashMap<SymId, f32>,
    ) -> Result<Uniforms> {
        let mut dims = Vec::with_capacity(self.dim_syms.len());
        for sym in &self.dim_syms {
            let v = Dim::Sym(*sym)
                .evaluate(&mut |s| binding.get(&s).copied())
                .ok_or_else(|| {
                    Error::Plan(format!("symbolic dim {sym} has no dispatch binding"))
                })?;
            dims.push(u32::try_from(v).map_err(|_| {
                Error::Plan(format!(
                    "symbolic dim {sym} = {v} does not fit in a u32 word"
                ))
            })?);
        }

        let mut out_scalars = Vec::with_capacity(self.scalar_syms.len());
        for sym in &self.scalar_syms {
            let v = scalars.get(sym).copied().ok_or_else(|| {
                Error::Plan(format!("uniform scalar {sym} has no dispatch value"))
            })?;
            out_scalars.push(v);
        }

        Ok(Uniforms {
            dims,
            scalars: out_scalars,
        })
    }

    /// Word index of a symbolic extent at binding 0.
    pub(crate) fn dim_slot(&self, sym: SymId) -> Option<u32> {
        self.sym_index.get(&sym).copied()
    }

    /// Word index of a runtime scalar at binding 0.
    pub(crate) fn scalar_slot(&self, sym: SymId) -> Option<u32> {
        self.scalar_index.get(&sym).copied()
    }

    /// Total words. Binding 0 is always present even when this is zero, so a
    /// kernel's storage globals always start at binding 1.
    pub(crate) fn words(&self) -> usize {
        self.dim_syms.len() + self.scalar_syms.len()
    }

    /// Byte length of binding 0 — the size the pool allocates. Never zero:
    /// wgpu rejects a zero-sized buffer binding, and binding 0 is always bound.
    pub(crate) fn byte_len(&self) -> u64 {
        (self.words() as u64 * 4).max(4)
    }
}
