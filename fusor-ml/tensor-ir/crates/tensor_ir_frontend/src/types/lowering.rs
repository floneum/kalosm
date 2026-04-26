//! Lowering options shared by saturation and effect-program construction.
//!
//! These toggles control structural choices (loop unrolling, inlining, etc.)
//! that need to stay consistent across lowering rules and effect IR creation.

/// Knobs that affect how lowering rules shape the generated kernel. Customize
/// to trade kernel performance for IR readability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoweringOptions {
    /// Expand fixed-iteration inner loops into straight-line code. When
    /// disabled, loops are emitted as `Theta` nodes — slightly slower at
    /// runtime, but much easier to follow when inspecting the IR.
    pub unroll: bool,
}

impl LoweringOptions {
    /// Default settings: unroll is enabled for performance.
    #[must_use]
    pub const fn default_const() -> Self {
        Self { unroll: true }
    }

    /// Settings tuned for IR readability: disables unrolling so loop
    /// structure stays explicit in the extracted kernel.
    #[must_use]
    pub const fn readable() -> Self {
        Self { unroll: false }
    }
}

impl Default for LoweringOptions {
    fn default() -> Self {
        Self::default_const()
    }
}
