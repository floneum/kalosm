use super::*;

mod layout;
mod lowering;

/// Lower `ir` and panic with a labelled message on failure. Returns the
/// lowered Naga kernel for tests that want to inspect the module.
fn lower_or_fail(ir: &KernelIr, label: &str) -> NagaKernel {
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("{label} lowering failed: {error}"))
}
