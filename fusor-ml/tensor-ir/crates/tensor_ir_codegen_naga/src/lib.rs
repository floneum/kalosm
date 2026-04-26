pub mod analysis {
    pub use tensor_ir_egraph::analysis::*;
}
pub mod language {
    pub use tensor_ir_egraph::language::*;
}
pub mod naga_codegen;
pub mod types {
    pub use tensor_ir_frontend::types::*;
}

pub use naga_codegen::{lower_effect_program, module_to_msl, module_to_wgsl};
