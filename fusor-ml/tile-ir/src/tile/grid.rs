use super::*;
use crate::ir::KernelIr;

pub fn build(f: impl FnOnce(&mut Program)) -> KernelIr {
    let mut program = Program::new();
    f(&mut program);
    program.ir
}
