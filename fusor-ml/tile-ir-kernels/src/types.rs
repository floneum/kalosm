use std::sync::Arc;

use fusor_tile_ir::tile::Tile;
use fusor_tile_ir::{Layout, TileLiteral};

/// Names a paired-FFN preset for which a tensor-level evaluation exists.
/// Used by the CPU fallback in higher layers; arbitrary user epilogues built
/// via [`PairedEpilogue::new`] return `None` from
/// [`PairedEpilogue::cpu_preset`] and run on GPU only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairedEpiloguePreset {
    SwiGLU,
    GeGLU,
    ReGLU,
}

/// Paired matmul epilogue. The matmul produces concatenated `[gate; up]`
/// columns; the kernel reduces each pair separately and applies this epilogue
/// as `build(gate, up)` before storing a single output column per pair.
///
/// Construct via [`PairedEpilogue::swiglu`] / [`geglu`] / [`reglu`] for the
/// standard FFN flavors, or [`PairedEpilogue::new`] for an arbitrary tile-IR
/// expression. The closure runs at kernel-build time and produces a Tile-IR
/// `Expr` tree; pipelines are cached by the structural hash of that tree, so
/// two epilogues with identical generated expressions share a compiled kernel.
#[derive(Clone)]
pub struct PairedEpilogue {
    label: &'static str,
    identity: u64,
    cpu_preset: Option<PairedEpiloguePreset>,
    // Closure is BLOCK-agnostic — it sees `Tile<1>` and the caller re-tags the
    // const generic on both sides of the call. `Tile<N>` carries no runtime
    // state beyond its `Expr`, so this is a host-time shape cast only.
    build: Arc<dyn Fn(Tile<1>, Tile<1>) -> Tile<1> + Send + Sync>,
}

impl PairedEpilogue {
    /// Build a paired epilogue from an arbitrary tile-IR closure. `label`
    /// appears in graph visualizations and kernel names; the structural hash
    /// of the closure's output drives pipeline caching, so two closures that
    /// produce identical Expr trees will share a pipeline regardless of label.
    pub fn new<F>(label: &'static str, build: F) -> Self
    where
        F: Fn(Tile<1>, Tile<1>) -> Tile<1> + Send + Sync + 'static,
    {
        Self::with_cpu_preset(label, None, build)
    }

    fn with_cpu_preset<F>(
        label: &'static str,
        cpu_preset: Option<PairedEpiloguePreset>,
        build: F,
    ) -> Self
    where
        F: Fn(Tile<1>, Tile<1>) -> Tile<1> + Send + Sync + 'static,
    {
        // Probe the closure with two distinguishable placeholder tiles so that
        // commutative differences (`gate * up` vs `up * gate`) yield distinct
        // structural hashes.
        let gate_probe = Tile::<1>::literal(TileLiteral::f32(f32::from_bits(0x5EED_CA7E)));
        let up_probe = Tile::<1>::literal(TileLiteral::f32(f32::from_bits(0xBADF_00D5)));
        let identity = build(gate_probe, up_probe).signature_hash();
        Self {
            label,
            identity,
            cpu_preset,
            build: Arc::new(build),
        }
    }

    /// `silu(gate) * up` — Llama / Mistral / Qwen / Gemma SwiGLU FFNs.
    pub fn swiglu() -> Self {
        Self::with_cpu_preset(
            "swiglu",
            Some(PairedEpiloguePreset::SwiGLU),
            |gate, up| gate.silu() * up,
        )
    }

    /// `gelu(gate) * up` — GeGLU-style FFNs.
    pub fn geglu() -> Self {
        Self::with_cpu_preset(
            "geglu",
            Some(PairedEpiloguePreset::GeGLU),
            |gate, up| gate.gelu() * up,
        )
    }

    /// `relu(gate) * up` — ReGLU-style FFNs.
    pub fn reglu() -> Self {
        Self::with_cpu_preset(
            "reglu",
            Some(PairedEpiloguePreset::ReGLU),
            |gate, up| gate.relu() * up,
        )
    }

    /// Returns the preset tag if this epilogue was built via one of the named
    /// constructors. Tensor-level evaluators (CPU fallback) use this to apply
    /// the activation on already-materialized gate/up tensors. Arbitrary
    /// closures from [`PairedEpilogue::new`] return `None` and are GPU-only.
    pub fn cpu_preset(&self) -> Option<PairedEpiloguePreset> {
        self.cpu_preset
    }

    /// Build the per-output tile expression for this epilogue.
    pub fn apply<const BLOCK: usize>(
        &self,
        gate: Tile<BLOCK>,
        up: Tile<BLOCK>,
    ) -> Tile<BLOCK> {
        (self.build)(gate.retag_block::<1>(), up.retag_block::<1>()).retag_block::<BLOCK>()
    }

    /// Stable structural hash of the produced Tile-IR Expr tree. Mix into
    /// pipeline cache keys so distinct epilogues do not alias.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Human-readable label for graph visualization and kernel names.
    pub fn label(&self) -> &'static str {
        self.label
    }
}

impl std::fmt::Debug for PairedEpilogue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairedEpilogue")
            .field("label", &self.label)
            .field("identity", &format_args!("{:#018x}", self.identity))
            .finish()
    }
}

impl PartialEq for PairedEpilogue {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

impl Eq for PairedEpilogue {}

impl std::hash::Hash for PairedEpilogue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
    }
}

pub fn matrix_shape(layout: &Layout) -> [u32; 2] {
    assert_eq!(layout.shape().rank(), 2, "matrix operands must be rank-2");
    [
        layout.shape().dims()[0].get(),
        layout.shape().dims()[1].get(),
    ]
}

pub fn cooperative_store_layout_supported(layout: &Layout) -> bool {
    if !layout.is_affine() || layout.shape().rank() != 2 {
        return false;
    }
    let strides = layout.affine_strides();
    strides[0] == 1 || strides[1] == 1
}
