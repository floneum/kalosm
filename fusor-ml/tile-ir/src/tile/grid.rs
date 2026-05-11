use crate::ir::KernelIr;
use super::*;

pub fn build(f: impl FnOnce(&mut Program)) -> KernelIr {
    let mut program = Program::new();
    f(&mut program);
    program.ir
}
