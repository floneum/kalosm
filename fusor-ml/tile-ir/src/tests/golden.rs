//! Self-contained structural golden for the one corpus entry that uses only the
//! builder DSL (no `fusor-tile-ir-kernels` high-level builders): the contiguous
//! f16 workgroup-tile copy.
//!
//! The full corpus (qgemv/qmatmul/coop/flash) lives in
//! `tile-ir-kernels/tests/golden.rs` because it needs the kernel-crate
//! builders; this module proves the rewritten *builder* reaches the same module
//! for the one kernel it can build against `tile-ir` alone.
//!
//! It compares the **types arena, global_variables and the function expression
//! arena + body statement tree** — i.e. everything the builder shapes — but
//! canonicalizes the `local_variables` arena and `LocalVariable([n])` handle
//! numbering. Demand-allocated scratch locals may be numbered differently, so
//! the gate compares structure and expression-arena order rather than local
//! handles.

use std::path::PathBuf;

use super::*;
use crate::{ScalarElement, TileLiteral};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden_modules")
}

/// Strip the `local_variables: { .. }` arena block and rewrite every
/// `LocalVariable([n])` reference to a sentinel, so the comparison is
/// insensitive to local-variable numbering.
fn canonicalize_locals(serialized: &str) -> String {
    let no_local_refs = regex_replace_local_variable_refs(serialized);
    strip_local_variables_block(&no_local_refs)
}

fn regex_replace_local_variable_refs(input: &str) -> String {
    // Replace the multi-line `LocalVariable(\n   [n],\n)` index with `[*]`.
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(pos) = rest.find("LocalVariable(") {
        out.push_str(&rest[..pos + "LocalVariable(".len()]);
        rest = &rest[pos + "LocalVariable(".len()..];
        // Skip to the closing ')', replacing the inner index with a sentinel.
        if let Some(close) = rest.find(')') {
            out.push_str("[*]");
            rest = &rest[close..];
        }
    }
    out.push_str(rest);
    out
}

fn strip_local_variables_block(input: &str) -> String {
    let Some(start) = input.find("local_variables: {") else {
        return input.to_string();
    };
    // Find the matching close brace for this block.
    let after = start + "local_variables: {".len();
    let bytes = input.as_bytes();
    let mut depth = 1usize;
    let mut idx = after;
    while idx < bytes.len() && depth > 0 {
        match bytes[idx] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => {}
        }
        idx += 1;
    }
    let mut out = String::with_capacity(input.len());
    out.push_str(&input[..start]);
    out.push_str("local_variables: { .. }");
    out.push_str(&input[idx..]);
    out
}

fn check_golden_structural(name: &str, ir: &KernelIr) {
    let lowered = lower_or_fail(ir, name);
    let serialized = format!("{:#?}", lowered.module());
    let path = golden_dir().join(format!("{name}.txt"));
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("golden snapshot missing: {}", path.display()));
    assert_eq!(
        canonicalize_locals(&expected),
        canonicalize_locals(&serialized),
        "golden module structure for `{name}` changed \
         (locals canonicalized); snapshot: {}",
        path.display()
    );
}

#[test]
fn golden_f16_workgroup_tile_copy_structure() {
    const COLS: u32 = 128;
    let ir = tile::build(|program| {
        let f16 = ScalarElement::F16.element();
        let x = program.storage_read(f16, Shape::new([1, COLS]));
        let y = program.storage_write(f16, Shape::new([1, COLS]));
        let tile_buf = program.alloc_workgroup_tile(ScalarElement::F16, 1, COLS);
        program.program_grid(COLS, [1, 1, 1], |program| {
            let lane = program.lane();
            let mask = lane.clone().lt(COLS);
            let value = program.load(
                x.at((0u32, lane.clone())),
                mask.clone(),
                TileLiteral::F16(0),
            );
            program.store_workgroup(&tile_buf, lane.clone(), value);
            program.workgroup_barrier();
            let staged = program.load_workgroup(&tile_buf, lane.clone());
            program.store(y.at((0u32, lane)), staged, mask);
        });
    });
    check_golden_structural("f16_workgroup_tile_copy", &ir);
}
