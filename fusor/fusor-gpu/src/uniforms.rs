//! Packing binding 0. It holds `[u32 symbolic dims..., u32 derived strides...,
//! f32 uniform scalars...]` and is a storage buffer, because the derived
//! bind-group mechanism walks storage globals.
//!
//! Binding 0 carries symbolic dimensions and scalars that would otherwise
//! need to be baked literals or constants.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::extract::Plan;
use fusor_ir::shape::{Dim, SymId};
use fusor_ir::target::Uniforms;
use rustc_hash::FxHashMap;

/// `Layout::row_major_strides` emits this symbol for every stride that sits
/// past a `Dim::Sym` axis and is therefore not a compile-time constant. It is
/// a *placeholder*, never a real binding: [`UniformPack`] resolves each
/// occurrence into a concrete `u32` dim word, and emitting the placeholder
/// into a kernel would be a compiler bug.
pub(crate) const DERIVED_STRIDE: SymId = SymId(u32::MAX);

/// Which slot of binding 0 a derived stride landed in.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct StrideKey {
    /// Index into `Plan::buffers`.
    pub buffer: u32,
    pub axis: u32,
}

/// The word layout of binding 0 for one plan, plus the packer that fills it.
///
/// The layout is a function of the *plan* alone — never of a binding — so a
/// sequence-length change re-fills the same words and recompiles nothing.
#[derive(Clone, Debug, Default)]
pub(crate) struct UniformPack {
    /// Symbols carried as `u32` extents, in `Plan::symbols` order.
    dim_syms: Vec<SymId>,
    /// Symbols carried as `f32` runtime scalars, in `Plan::symbols` order.
    scalar_syms: Vec<SymId>,
    /// Placeholder strides resolved into their own `u32` words, in a stable
    /// (buffer, axis) order.
    strides: Vec<StrideKey>,
    sym_index: FxHashMap<SymId, u32>,
    scalar_index: FxHashMap<SymId, u32>,
}

impl UniformPack {
    /// The word layout's identity. The three index maps are derived from the
    /// three lists below them, so those lists are the whole of it.
    ///
    /// A kernel body bakes these slot indices, which is why an artifact's
    /// cache key carries this and not `Plan::symbols`: two plans that agree
    /// on the pack emit the same words whatever else differs between them.
    pub(crate) fn digest(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = rustc_hash::FxHasher::default();
        self.dim_syms.hash(&mut h);
        self.scalar_syms.hash(&mut h);
        self.strides.hash(&mut h);
        h.finish()
    }

    /// Derive the word layout of binding 0 from a plan.
    ///
    /// A symbol appears in exactly one of the two groups: `scalar_syms` when
    /// the plan names it a runtime scalar (a `Leaf::Uniform`), `dim_syms`
    /// otherwise — an extent, a view offset, a stride — whether or not any
    /// buffer layout mentions it. The classification is a property of the
    /// plan, so it does not move when a value does.
    pub(crate) fn new(plan: &Plan) -> Self {
        let mut dim_syms = Vec::new();
        let mut scalar_syms = Vec::new();
        for &sym in &plan.symbols {
            if sym == DERIVED_STRIDE {
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

        let mut strides = Vec::new();
        for (bi, buf) in plan.buffers.iter().enumerate() {
            for (axis, stride) in buf.layout.strides().iter().enumerate() {
                if matches!(stride, Dim::Sym(s) if *s == DERIVED_STRIDE) {
                    strides.push(StrideKey {
                        buffer: bi as u32,
                        axis: axis as u32,
                    });
                }
            }
        }

        let sym_index = dim_syms
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i as u32))
            .collect();
        let dim_words = dim_syms.len() as u32;
        let base = dim_words + strides.len() as u32;
        let scalar_index = scalar_syms
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, base + i as u32))
            .collect();

        Self {
            dim_syms,
            scalar_syms,
            strides,
            sym_index,
            scalar_index,
        }
    }

    /// Fill a pre-derived layout. This is the per-dispatch path: it allocates
    /// two `Vec`s and does no hashing of the plan.
    pub(crate) fn fill(
        &self,
        plan: &Plan,
        binding: &FxHashMap<SymId, u64>,
        scalars: &FxHashMap<SymId, f32>,
    ) -> Result<Uniforms> {
        let mut dims = Vec::with_capacity(self.dim_syms.len() + self.strides.len());
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

        for key in &self.strides {
            dims.push(self.resolve_stride(plan, *key, binding)?);
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
        self.dim_syms.len() + self.strides.len() + self.scalar_syms.len()
    }

    /// Byte length of binding 0 — the size the pool allocates. Never zero:
    /// wgpu rejects a zero-sized buffer binding, and binding 0 is always bound.
    pub(crate) fn byte_len(&self) -> u64 {
        (self.words() as u64 * 4).max(4)
    }

    /// Resolve one `row_major_strides` placeholder into a concrete word.
    ///
    /// The stride of axis `a` is the product of the extents of axes after it.
    /// Every one of those extents is either a constant or a bound symbol; a
    /// second placeholder there would mean the plan carried a stride it never
    /// derived, which is `Error::Plan`, not a fallback.
    fn resolve_stride(
        &self,
        plan: &Plan,
        key: StrideKey,
        binding: &FxHashMap<SymId, u64>,
    ) -> Result<u32> {
        let buf = plan
            .buffers
            .get(key.buffer as usize)
            .ok_or_else(|| Error::Plan(format!("derived stride names buffer {}", key.buffer)))?;
        let shape = buf.layout.shape();
        let mut acc: u64 = 1;
        for d in shape.iter().skip(key.axis as usize + 1) {
            let extent = match d {
                Dim::Const(v) => *v,
                Dim::Sym(s) if *s == DERIVED_STRIDE => {
                    return Err(Error::Plan(
                        "a shape extent is the derived-stride placeholder".into(),
                    ));
                }
                d @ Dim::Sym(s) => {
                    d.evaluate(&mut |x| binding.get(&x).copied())
                        .ok_or_else(|| {
                            Error::Plan(format!("derived stride needs unbound symbol {s}"))
                        })?
                }
            };
            acc = acc
                .checked_mul(extent)
                .ok_or_else(|| Error::Plan("derived stride overflows a u64".into()))?;
        }
        u32::try_from(acc)
            .map_err(|_| Error::Plan(format!("derived stride {acc} does not fit in a u32 word")))
    }
}
