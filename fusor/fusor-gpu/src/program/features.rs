//! Backend instructions are independent of scheduling and storage allocation.
use super::ProgramOptions;
use crate::device::GpuDevice;
use fusor_ir::dtype::Dtype;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MatrixInstructions {
    #[default]
    Portable,
    Native,
    Browser,
}

/// Instructions actually selected for a compiled program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgramAcceleration {
    pub subgroups: bool,
    /// Stable indexed reductions currently specialize for 32-wide subgroups.
    pub subgroup_scatter: bool,
    pub matrices: MatrixInstructions,
}
impl ProgramAcceleration {
    pub(crate) fn select(device: &GpuDevice, options: ProgramOptions) -> Self {
        let subgroups =
            options.subgroup_acceleration && device.features().contains(wgpu::Features::SUBGROUP);
        let fixed32 = device
            .caps()
            .subgroups
            .is_some_and(|w| w.min == 32 && w.max == 32);
        let matrices = if options.matrix_acceleration && subgroups && fixed32 {
            let kinds = &device.caps().coop;
            if kinds
                .iter()
                .any(|k| k.operand == Dtype::F32 && k.acc == Dtype::F32)
            {
                if device.adapter_info().backend == wgpu::Backend::BrowserWebGpu {
                    MatrixInstructions::Browser
                } else {
                    MatrixInstructions::Native
                }
            } else {
                MatrixInstructions::Portable
            }
        } else {
            MatrixInstructions::Portable
        };
        Self {
            subgroups,
            subgroup_scatter: subgroups && fixed32,
            matrices,
        }
    }
    pub(crate) fn cooperative(self) -> bool {
        self.matrices != MatrixInstructions::Portable
    }
    #[cfg(any(not(target_arch = "wasm32"), test, feature = "compiler-tests"))]
    pub(crate) fn native_validation(self) -> Self {
        Self {
            matrices: if self.cooperative() {
                MatrixInstructions::Native
            } else {
                MatrixInstructions::Portable
            },
            ..self
        }
    }
}
