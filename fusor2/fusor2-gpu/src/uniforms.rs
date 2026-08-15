//! Packing binding 0. It holds `[u32 symbolic dims..., u32 derived strides...,
//! f32 uniform scalars...]` and is a **storage** buffer, because the derived
//! bind-group mechanism walks storage globals.
//!
//! That one buffer kills trainer constraints 1 and 2 together: `m * lr_f32`
//! produces a `Uniform`, not a baked literal, and a sequence length is a `Sym`
//! read from binding 0.
//!
//! Owned by W9.

use fusor2_ir::Result;
use fusor2_ir::error::Error;
use fusor2_ir::extract::Plan;
use fusor2_ir::shape::{Dim, SymId};
use fusor2_ir::target::Uniforms;
use rustc_hash::FxHashMap;

/// `Layout::row_major_strides` emits this symbol for every stride that sits
/// past a `Dim::Sym` axis and is therefore not a compile-time constant. It is
/// a *placeholder*, never a real binding: [`UniformPack`] resolves each
/// occurrence into a concrete `u32` dim word, and emitting the placeholder
/// into a kernel would be a compiler bug.
pub const DERIVED_STRIDE: SymId = SymId(u32::MAX);

/// Which slot of binding 0 a derived stride landed in.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StrideKey {
    /// Index into `Plan::buffers`.
    pub buffer: u32,
    pub axis: u32,
}

/// The word layout of binding 0 for one plan, plus the packer that fills it.
///
/// The layout is a function of the *plan* alone — never of a binding — so a
/// sequence-length change re-fills the same words and recompiles nothing.
#[derive(Clone, Debug, Default)]
pub struct UniformPack {
    /// Symbols carried as `u32` extents, in `Plan::symbols` order.
    dim_syms: Vec<SymId>,
    /// Symbols carried as `f32` runtime scalars, in `Plan::symbols` order.
    scalar_syms: Vec<SymId>,
    /// Placeholder strides resolved into their own `u32` words, in a stable
    /// (buffer, axis) order.
    strides: Vec<StrideKey>,
    sym_index: FxHashMap<SymId, u32>,
    scalar_index: FxHashMap<SymId, u32>,
    stride_index: FxHashMap<StrideKey, u32>,
}

impl UniformPack {
    /// The word layout's identity. The three index maps are derived from the
    /// three lists below them, so those lists are the whole of it.
    ///
    /// A kernel body bakes these slot indices, which is why an artifact's
    /// cache key carries this and not `Plan::symbols`: two plans that agree
    /// on the pack emit the same words whatever else differs between them.
    pub fn digest(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = rustc_hash::FxHasher::default();
        self.dim_syms.hash(&mut h);
        self.scalar_syms.hash(&mut h);
        self.strides.hash(&mut h);
        h.finish()
    }

    /// Derive the word layout of binding 0 from a plan.
    ///
    /// A symbol appears in exactly one of the two groups: `dim_syms` when any
    /// buffer shape or stride mentions it, `scalar_syms` otherwise. The
    /// classification is structural, so it does not move when a value does.
    pub fn new(plan: &Plan) -> Self {
        let mut mentioned_as_dim: FxHashMap<SymId, ()> = FxHashMap::default();
        for buf in &plan.buffers {
            for d in buf.layout.shape().iter().chain(buf.layout.strides()) {
                if let Dim::Sym(s) = d
                    && *s != DERIVED_STRIDE
                {
                    mentioned_as_dim.insert(*s, ());
                }
            }
            if let Dim::Sym(s) = buf.elements
                && s != DERIVED_STRIDE
            {
                mentioned_as_dim.insert(s, ());
            }
            if let Dim::Sym(s) = buf.layout.offset()
                && s != DERIVED_STRIDE
            {
                mentioned_as_dim.insert(s, ());
            }
        }

        let mut dim_syms = Vec::new();
        let mut scalar_syms = Vec::new();
        for &sym in &plan.symbols {
            if sym == DERIVED_STRIDE {
                continue;
            }
            if mentioned_as_dim.contains_key(&sym) {
                if !dim_syms.contains(&sym) {
                    dim_syms.push(sym);
                }
            } else if !scalar_syms.contains(&sym) {
                scalar_syms.push(sym);
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
        let stride_index = strides
            .iter()
            .enumerate()
            .map(|(i, k)| (*k, dim_words + i as u32))
            .collect();
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
            stride_index,
        }
    }

    /// Pack, in `plan.symbols` order, every `Dim::Sym` extent as `u32`, then
    /// every derived stride as `u32`, then every `Leaf::Uniform` scalar as
    /// `f32`.
    pub fn build(
        plan: &Plan,
        binding: &FxHashMap<SymId, u64>,
        scalars: &FxHashMap<SymId, f32>,
    ) -> Result<Uniforms> {
        Self::new(plan).fill(plan, binding, scalars)
    }

    /// Fill a pre-derived layout. This is the per-dispatch path: it allocates
    /// two `Vec`s and does no hashing of the plan.
    pub fn fill(
        &self,
        plan: &Plan,
        binding: &FxHashMap<SymId, u64>,
        scalars: &FxHashMap<SymId, f32>,
    ) -> Result<Uniforms> {
        let mut dims = Vec::with_capacity(self.dim_syms.len() + self.strides.len());
        for sym in &self.dim_syms {
            let v = binding.get(sym).copied().ok_or_else(|| {
                Error::Plan(format!("symbolic dim {sym} has no dispatch binding"))
            })?;
            dims.push(u32::try_from(v).map_err(|_| {
                Error::Plan(format!("symbolic dim {sym} = {v} does not fit in a u32 word"))
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
    pub fn dim_slot(&self, sym: SymId) -> Option<u32> {
        self.sym_index.get(&sym).copied()
    }

    /// Word index of a runtime scalar at binding 0.
    pub fn scalar_slot(&self, sym: SymId) -> Option<u32> {
        self.scalar_index.get(&sym).copied()
    }

    /// Word index of a resolved derived stride at binding 0.
    pub fn stride_slot(&self, buffer: u32, axis: u32) -> Option<u32> {
        self.stride_index.get(&StrideKey { buffer, axis }).copied()
    }

    /// Total words. Binding 0 is always present even when this is zero, so a
    /// kernel's storage globals always start at binding 1.
    pub fn words(&self) -> usize {
        self.dim_syms.len() + self.strides.len() + self.scalar_syms.len()
    }

    /// Byte length of binding 0 — the size the pool allocates. Never zero:
    /// wgpu rejects a zero-sized buffer binding, and binding 0 is always bound.
    pub fn byte_len(&self) -> u64 {
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
        let buf = plan.buffers.get(key.buffer as usize).ok_or_else(|| {
            Error::Plan(format!("derived stride names buffer {}", key.buffer))
        })?;
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
                Dim::Sym(s) => binding.get(s).copied().ok_or_else(|| {
                    Error::Plan(format!("derived stride needs unbound symbol {s}"))
                })?,
            };
            acc = acc.checked_mul(extent).ok_or_else(|| {
                Error::Plan("derived stride overflows a u64".into())
            })?;
        }
        u32::try_from(acc)
            .map_err(|_| Error::Plan(format!("derived stride {acc} does not fit in a u32 word")))
    }
}

/// Bind every symbol the plan declares to its runtime extent, in plan order.
///
/// Convenience wrapper over [`UniformPack::build`] for callers that already
/// hold slices rather than maps.
pub fn pack(plan: &Plan, bindings: &[(SymId, Dim)], scalars: &[(SymId, f32)]) -> Result<Uniforms> {
    let mut binding_map: FxHashMap<SymId, u64> = FxHashMap::default();
    for (sym, dim) in bindings {
        let v = dim.as_const().ok_or_else(|| {
            Error::Plan(format!("dispatch binding for {sym} is itself symbolic"))
        })?;
        binding_map.insert(*sym, v);
    }
    let scalar_map: FxHashMap<SymId, f32> = scalars.iter().copied().collect();
    UniformPack::build(plan, &binding_map, &scalar_map)
}

/// Byte length of binding 0 for this plan — the size the pool allocates.
pub fn byte_len(plan: &Plan) -> u64 {
    UniformPack::new(plan).byte_len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::cost::Picoseconds;
    use fusor2_ir::dtype::{Dtype, Persistence};
    use fusor2_ir::egraph::Id;
    use fusor2_ir::extract::{Extraction, PlanHash};
    use fusor2_ir::shape::Layout;

    fn plan_with(buffers: Vec<fusor2_ir::extract::BufferPlan>, symbols: Vec<SymId>) -> Plan {
        Plan {
            extraction: Extraction::default(),
            launches: Vec::new(),
            buffers,
            symbols,
            hash: PlanHash(0),
            cost: Picoseconds(0),
        }
    }

    fn buffer(shape: &[Dim]) -> fusor2_ir::extract::BufferPlan {
        fusor2_ir::extract::BufferPlan {
            value: Id(0),
            layout: Layout::contiguous(shape),
            elements: Dim::Const(1),
            dtype: Dtype::F32,
            persistence: Persistence::Step,
        }
    }

    /// Test 2: three syms and two uniform scalars produce 20 bytes, dims at
    /// words 0..3 as LE u32 and scalars at words 3..5 as LE f32, with
    /// `dim_slot`/`scalar_slot` agreeing with `plan.symbols` order.
    #[test]
    fn uniform_block_layout() {
        let (s0, s1, s2) = (SymId(0), SymId(1), SymId(2));
        let (lr, scale) = (SymId(10), SymId(11));
        // One buffer whose shape mentions all three dim symbols, so the
        // classification puts them in the dim group and nothing else there.
        let plan = plan_with(
            vec![buffer(&[Dim::Sym(s0), Dim::Sym(s1), Dim::Sym(s2)])],
            vec![s0, s1, s2, lr, scale],
        );
        let pack = UniformPack::new(&plan);

        assert_eq!(pack.dim_slot(s0), Some(0));
        assert_eq!(pack.dim_slot(s1), Some(1));
        assert_eq!(pack.dim_slot(s2), Some(2));
        // Rank 3 with symbolic axes 1 and 2 leaves axes 0 and 1 derived; axis
        // 2's stride is the constant 1.
        assert_eq!(pack.stride_slot(0, 0), Some(3));
        assert_eq!(pack.stride_slot(0, 1), Some(4));
        assert_eq!(pack.stride_slot(0, 2), None);
        assert_eq!(pack.scalar_slot(lr), Some(5));
        assert_eq!(pack.scalar_slot(scale), Some(6));

        let binding: FxHashMap<SymId, u64> = [(s0, 2), (s1, 3), (s2, 4)].into_iter().collect();
        let scalars: FxHashMap<SymId, f32> =
            [(lr, 1e-3f32), (scale, 1024.0f32)].into_iter().collect();
        let u = pack.fill(&plan, &binding, &scalars).unwrap();
        assert_eq!(u.dims, vec![2, 3, 4, 12, 4]);
        assert_eq!(u.scalars, vec![1e-3f32, 1024.0f32]);

        let bytes = u.to_bytes();
        assert_eq!(bytes.len(), 4 * 7);
        assert_eq!(&bytes[0..4], &2u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &3u32.to_le_bytes());
        assert_eq!(&bytes[8..12], &4u32.to_le_bytes());
        assert_eq!(&bytes[20..24], &1e-3f32.to_le_bytes());
        assert_eq!(&bytes[24..28], &1024.0f32.to_le_bytes());
    }

    /// The exact 20-byte shape the spec's test names: three syms and two
    /// scalars with no derived strides in play.
    #[test]
    fn uniform_block_layout_without_derived_strides() {
        let (s0, s1, s2) = (SymId(0), SymId(1), SymId(2));
        let (lr, scale) = (SymId(10), SymId(11));
        // Three rank-1 buffers, one symbolic extent each: every stride is the
        // constant 1, so no placeholder appears.
        let plan = plan_with(
            vec![
                buffer(&[Dim::Sym(s0)]),
                buffer(&[Dim::Sym(s1)]),
                buffer(&[Dim::Sym(s2)]),
            ],
            vec![s0, s1, s2, lr, scale],
        );
        let pack = UniformPack::new(&plan);
        assert_eq!(pack.words(), 5);
        assert_eq!(pack.byte_len(), 20);
        assert_eq!(pack.dim_slot(s0), Some(0));
        assert_eq!(pack.scalar_slot(lr), Some(3));
        assert_eq!(pack.scalar_slot(scale), Some(4));

        let binding: FxHashMap<SymId, u64> =
            [(s0, 256), (s1, 8), (s2, 4)].into_iter().collect();
        let scalars: FxHashMap<SymId, f32> = [(lr, 5e-4), (scale, 1.0)].into_iter().collect();
        let bytes = pack.fill(&plan, &binding, &scalars).unwrap().to_bytes();
        assert_eq!(bytes.len(), 20);
        assert_eq!(&bytes[0..4], &256u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &5e-4f32.to_le_bytes());
    }

    /// A learning-rate change re-fills the same words: the *layout* is a
    /// function of the plan, so nothing about the kernel changes.
    #[test]
    fn scalar_value_does_not_move_a_slot() {
        let lr = SymId(7);
        let plan = plan_with(vec![buffer(&[Dim::Const(4)])], vec![lr]);
        let pack = UniformPack::new(&plan);
        let binding = FxHashMap::default();
        let a = pack
            .fill(&plan, &binding, &[(lr, 1e-3f32)].into_iter().collect())
            .unwrap();
        let b = pack
            .fill(&plan, &binding, &[(lr, 5e-4f32)].into_iter().collect())
            .unwrap();
        assert_eq!(pack.scalar_slot(lr), Some(0));
        assert_eq!(a.dims, b.dims);
        assert_ne!(a.scalars, b.scalars);
    }

    #[test]
    fn unbound_symbol_is_a_plan_error() {
        let s = SymId(3);
        let plan = plan_with(vec![buffer(&[Dim::Sym(s)])], vec![s]);
        let pack = UniformPack::new(&plan);
        let err = pack
            .fill(&plan, &FxHashMap::default(), &FxHashMap::default())
            .unwrap_err();
        assert!(matches!(err, Error::Plan(_)), "{err}");
    }

    /// The placeholder never reaches a kernel: it is resolved into a concrete
    /// word here or the pack fails.
    #[test]
    fn derived_stride_is_resolved_not_emitted() {
        let s = SymId(1);
        // Only an axis to the *left* of a symbolic extent has a symbolic
        // stride; axes to its right keep constant strides.
        let plan = plan_with(
            vec![buffer(&[Dim::Const(4), Dim::Sym(s), Dim::Const(64)])],
            vec![s],
        );
        let pack = UniformPack::new(&plan);
        assert_eq!(pack.dim_slot(s), Some(0));
        assert_eq!(pack.stride_slot(0, 0), Some(1));
        assert_eq!(pack.stride_slot(0, 1), None);
        let binding: FxHashMap<SymId, u64> = [(s, 512)].into_iter().collect();
        let u = pack.fill(&plan, &binding, &FxHashMap::default()).unwrap();
        assert_eq!(u.dims, vec![512, 512 * 64]);
        assert!(pack.dim_slot(DERIVED_STRIDE).is_none());
    }
}
