//! Two golden families: FNV-1a hashes of exact output bytes, and `PlanHash`
//! goldens over the calibration shape set. There are no golden shader bytes.
//!
//! The recorded tables live in this file as `&'static str` blocks. Both
//! families print the measured value on mismatch, so a deliberate numeric
//! change is re-recorded by copying one line.

use fusor2_ir::extract::PlanHash;

use crate::harness::CaseError;

/// FNV-1a over exact bytes. Not a cryptographic hash.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// The hash of a tensor's exact little-endian f32 bytes. The caller hands
/// over flattened values, so the hash is layout-independent.
pub fn tensor_hash(values: &[f32]) -> u64 {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fnv1a(&bytes)
}

/// One recorded output hash.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OutputGolden {
    pub name: &'static str,
    pub hash: u64,
}

/// The traces pinned by output hash, by name. An unrecorded name fails with
/// the measured value rather than passing vacuously.
pub const OUTPUT_GOLDEN_NAMES: [&str; 6] = [
    "attention_gqa_causal_fwd_bwd",
    "bilstm_trace",
    "qgemv_decode_ggml_q4k_4096x8192",
    "qgemv_decode_ggml_q4k_4096x8193",
    "qgemv_decode_ggml_q4k_4096x5120",
    "qgemv_decode_ggml_q6k_4096x8192",
];

/// `name hash` per line, `#` comments ignored. Empty until the traces run on
/// the baseline machine; see [`record_output`].
const OUTPUT_GOLDENS_TXT: &str = "\
# name                                     fnv1a(le bytes)
# Recorded on the baseline machine by running the trace and pasting the line
# `record_output` prints. A hash here that stops matching is a numeric change:
# review it, do not re-record it reflexively.
";

/// The recorded output-hash table.
pub fn output_goldens() -> Vec<OutputGolden> {
    parse_output_table(OUTPUT_GOLDENS_TXT)
}

fn parse_output_table(text: &'static str) -> Vec<OutputGolden> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let (name, hash) = l.split_once(char::is_whitespace)?;
            let hash = hash.trim();
            let parsed = hash.strip_prefix("0x").map_or_else(
                || hash.parse::<u64>().ok(),
                |h| u64::from_str_radix(h, 16).ok(),
            )?;
            Some(OutputGolden {
                name: name.trim(),
                hash: parsed,
            })
        })
        .collect()
}

/// The line to paste into [`OUTPUT_GOLDENS_TXT`] to record `name`.
pub fn record_output(name: &str, hash: u64) -> String {
    format!("{name} 0x{hash:016x}")
}

/// Check one output hash against the table. An unrecorded name is a failure,
/// not a pass.
pub fn check_output(name: &str, values: &[f32]) -> Result<(), CaseError> {
    let hash = tensor_hash(values);
    match output_goldens().iter().find(|g| g.name == name) {
        Some(g) if g.hash == hash => Ok(()),
        Some(g) => Err(format!(
            "{name} output-hash golden mismatch: recorded 0x{:016x}, measured 0x{hash:016x}.\n\
             This is a numeric change. Review it; re-record only deliberately:\n  {}",
            g.hash,
            record_output(name, hash)
        )
        .into()),
        None => Err(format!(
            "{name} has no recorded output hash. Measured on this machine:\n  {}",
            record_output(name, hash)
        )
        .into()),
    }
}

/// One golden: a named shape and the plan hash it must extract to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Golden {
    pub name: &'static str,
    pub hash: PlanHash,
}

/// The calibration shape set the `PlanHash` goldens cover: one entry per
/// kernel family the extractor can select, at one pinned shape.
pub const CALIBRATION_SHAPES: [(&str, [u64; 3]); 8] = [
    ("dense_4096_cube", [4096, 4096, 4096]),
    ("gemv_1x4096x4096", [1, 4096, 4096]),
    ("skinny_32x4096x4096", [32, 4096, 4096]),
    ("wide_4096x4096x64", [4096, 4096, 64]),
    ("split_k_64x64x16384", [64, 64, 16384]),
    ("conv_epilogue_128x768x24", [128, 768, 24]),
    ("qgemv_1x8192x4096", [1, 8192, 4096]),
    ("embedding_scatter_384x1024x24", [384, 1024, 24]),
];

/// `name hash` per line, hash as `0x…` u128. Empty until extraction runs.
const PLAN_HASH_GOLDENS_TXT: &str = "\
# name                                     PlanHash (0x, u128)
# `PlanHash` folds the realized DAG term, M, theta and the DeviceFacts
# fingerprint. `Dim::Sym` and `Leaf::Uniform` hash as symbols, so one plan
# serves a whole shape family — which is precisely what the structural asserts
# below check without needing a recorded constant.
";

/// The recorded goldens.
pub fn plan_hash_goldens() -> Vec<Golden> {
    PLAN_HASH_GOLDENS_TXT
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let (name, hash) = l.split_once(char::is_whitespace)?;
            let hash = hash.trim().strip_prefix("0x")?;
            Some(Golden {
                name: name.trim(),
                hash: PlanHash(u128::from_str_radix(hash, 16).ok()?),
            })
        })
        .collect()
}

/// The line to paste into [`PLAN_HASH_GOLDENS_TXT`] to record `name`.
pub fn record_plan(name: &str, hash: PlanHash) -> String {
    format!("{name} 0x{:032x}", hash.0)
}

/// Compare a fresh extraction against the table. A mismatch is a real change
/// in what the compiler decided and must be reviewed, not re-recorded.
pub fn check(name: &str, hash: PlanHash) -> Result<(), String> {
    match plan_hash_goldens().iter().find(|g| g.name == name) {
        Some(g) if g.hash == hash => Ok(()),
        Some(g) => Err(format!(
            "{name} PlanHash golden mismatch: recorded 0x{:032x}, measured 0x{:032x}.\n\
             Extraction chose a different plan. Review it; re-record only deliberately:\n  {}",
            g.hash.0,
            hash.0,
            record_plan(name, hash)
        )),
        None => Err(format!(
            "{name} has no recorded PlanHash. Measured on this machine:\n  {}",
            record_plan(name, hash)
        )),
    }
}

/// One plan must serve a whole `Dim::Sym` family: the same L0 term extracted
/// at three different bindings hashes identically. Needs no recorded constant.
pub fn assert_one_plan_per_symbolic_family(
    name: &str,
    hashes: &[(u64, PlanHash)],
) -> Result<(), CaseError> {
    let Some(((first_binding, first), rest)) = hashes.split_first() else {
        return Err(format!("{name}: no bindings were extracted").into());
    };
    if rest.len() < 2 {
        return Err(format!(
            "{name}: a symbolic family needs at least three bindings, got {}",
            hashes.len()
        )
        .into());
    }
    for (binding, hash) in rest {
        if hash != first {
            return Err(format!(
                "{name}: binding {binding} extracted plan 0x{:032x} but binding \
                 {first_binding} extracted 0x{:032x}. A Dim::Sym must hash as a \
                 symbol, not as a value — otherwise every length bucket recompiles.",
                hash.0, first.0
            )
            .into());
        }
    }
    Ok(())
}

/// `specialize_dim` must change the hash only after the binding has recurred;
/// on first sighting the generic symbolic variant wins outright.
pub fn assert_specialization_waits_for_reuse(
    name: &str,
    generic: PlanHash,
    first_sighting: PlanHash,
    expected_reuse: u32,
    after_reuse: PlanHash,
) -> Result<(), CaseError> {
    if first_sighting != generic {
        return Err(format!(
            "{name}: the first sighting of a binding already specialized \
             (0x{:032x} vs generic 0x{:032x}). ShapeStats::expected_reuse is 1 \
             there, so specialization must not have paid for itself yet.",
            first_sighting.0, generic.0
        )
        .into());
    }
    if expected_reuse <= 1 {
        return Err(format!(
            "{name}: expected_reuse is {expected_reuse}; the second half of this \
             assert is vacuous unless the binding actually recurred"
        )
        .into());
    }
    if after_reuse == generic {
        return Err(format!(
            "{name}: expected_reuse reached {expected_reuse} but the plan did not \
             specialize — `specialize_dim` never won, so compile amortization is \
             not reaching the cost model."
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole `resolve` pipeline — saturate, extract, derive — is a pure
    /// function of `(graph, device)`. No budget is a wall clock, so the plan
    /// may not depend on machine load. Runs the pipeline twenty times and
    /// asserts the plan is identical each time.
    #[test]
    fn the_same_graph_on_the_same_device_yields_the_same_plan() {
        use fusor2::{Device, Dim, Dtype, Graph, Session, Tensor};
        use fusor2_ir::egraph::{Id, Saturate, SaturationBudget};
        use fusor2_ir::extract::{ExtractBudget, Extractor};
        use fusor2_ir::saturate::Driver;

        let Ok(cpu) = Device::cpu() else { return };
        let session = Session::new(cpu).unwrap();
        let caps = session.caps();
        let cost = fusor2_cost::Roofline::new(session.device().target().facts().clone());
        let extractor = fusor2_cost::LocalSearch::new(fusor2_tile::Planner::shared(), caps.clone())
            .with_registry(session.registry().clone());

        let plan_of = || {
            let graph = Graph::new(&session);
            let g = graph.handle();
            let shape: Vec<Dim> = [3u64, 5].iter().copied().map(Dim::Const).collect();
            let data: Vec<f32> = (0..15).map(|i| i as f32 * 0.1 - 0.5).collect();
            let x = Tensor::from_elements(g, &shape, &data).unwrap();
            let y = x.rms_norm_no_weight(1e-3).unwrap();
            let _ = Dtype::F32;
            g.with_egraph(|eg| {
                eg.add_root(y.id());
                let report = Driver::new().saturate(
                    eg,
                    &caps,
                    session.rules(),
                    SaturationBudget::default(),
                )?;
                let roots: Vec<Id> = eg.roots().to_vec();
                let plan = extractor.extract(eg, &roots, &cost, ExtractBudget::default())?;
                Ok((report, plan))
            })
            .unwrap()
        };

        let (first_report, first) = plan_of();
        assert!(
            first_report.saturated,
            "the graph this pins must saturate: {first_report:?}"
        );
        for _ in 0..20 {
            let (report, plan) = plan_of();
            assert_eq!(report.final_nodes, first_report.final_nodes);
            assert_eq!(report.applications, first_report.applications);
            assert_eq!(report.truncated, first_report.truncated);
            assert_eq!(plan.hash, first.hash, "the plan moved between two runs");
            assert_eq!(plan.cost, first.cost);
            assert_eq!(plan.launches.len(), first.launches.len());
            assert_eq!(plan.buffers, first.buffers);
        }
    }

    #[test]
    fn fnv1a_matches_the_published_vectors() {
        assert_eq!(fnv1a(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn tensor_hash_is_over_little_endian_bytes() {
        let values = [1.0f32, -2.5];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.5f32).to_le_bytes());
        assert_eq!(tensor_hash(&values), fnv1a(&bytes));
    }

    #[test]
    fn tensor_hash_separates_a_sign_flip() {
        assert_ne!(tensor_hash(&[0.0]), tensor_hash(&[-0.0]));
        assert_ne!(tensor_hash(&[1.0, 2.0]), tensor_hash(&[2.0, 1.0]));
    }

    #[test]
    fn an_unrecorded_output_golden_fails_with_the_measured_line() {
        let err = check_output("attention_gqa_causal_fwd_bwd", &[1.0, 2.0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("no recorded output hash"), "{err}");
        assert!(err.contains("0x"), "{err}");
    }

    #[test]
    fn an_unrecorded_plan_hash_fails_with_the_measured_line() {
        let err = check("dense_4096_cube", PlanHash(0xdead_beef)).unwrap_err();
        assert!(err.contains("no recorded PlanHash"), "{err}");
        assert!(err.contains("000000000000000000000000deadbeef"), "{err}");
    }

    #[test]
    fn the_output_table_parser_reads_both_radixes() {
        let table =
            parse_output_table("# comment\n\nalpha 0x0000000000000001\nbeta 2\n  gamma   0xff  \n");
        assert_eq!(table.len(), 3);
        assert_eq!(table[0].hash, 1);
        assert_eq!(table[1].hash, 2);
        assert_eq!(table[2].hash, 0xff);
        assert_eq!(table[2].name, "gamma");
    }

    #[test]
    fn the_shipped_tables_parse_even_while_empty() {
        assert!(output_goldens().is_empty());
        assert!(plan_hash_goldens().is_empty());
    }

    #[test]
    fn record_lines_round_trip_through_the_parsers() {
        let line: &'static str = Box::leak(record_output("demo", 0x1234).into_boxed_str());
        let parsed = parse_output_table(line);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "demo");
        assert_eq!(parsed[0].hash, 0x1234);
    }

    #[test]
    fn one_plan_serves_a_symbolic_family() {
        let same = PlanHash(7);
        assert!(
            assert_one_plan_per_symbolic_family("seq", &[(128, same), (768, same), (2048, same)])
                .is_ok()
        );
    }

    #[test]
    fn a_binding_that_moved_the_plan_hash_is_a_failure() {
        let err = assert_one_plan_per_symbolic_family(
            "seq",
            &[(128, PlanHash(7)), (768, PlanHash(7)), (2048, PlanHash(9))],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("binding 2048"), "{err}");
        assert!(err.contains("hash as a symbol"), "{err}");
    }

    #[test]
    fn a_two_binding_family_is_rejected_as_too_thin() {
        let same = PlanHash(7);
        assert!(assert_one_plan_per_symbolic_family("seq", &[(128, same), (768, same)]).is_err());
        assert!(assert_one_plan_per_symbolic_family("seq", &[]).is_err());
    }

    #[test]
    fn specialization_must_wait_for_reuse_and_then_happen() {
        let generic = PlanHash(1);
        let special = PlanHash(2);
        assert!(assert_specialization_waits_for_reuse("seq", generic, generic, 4, special).is_ok());
        // Specialized on first sighting: a length bucket compiled speculatively.
        assert!(
            assert_specialization_waits_for_reuse("seq", generic, special, 4, special).is_err()
        );
        // Never specialized despite reuse: compile amortization is not wired up.
        assert!(
            assert_specialization_waits_for_reuse("seq", generic, generic, 4, generic).is_err()
        );
        // Reuse never happened, so the assert would be vacuous.
        assert!(
            assert_specialization_waits_for_reuse("seq", generic, generic, 1, special).is_err()
        );
    }

    #[test]
    fn the_calibration_shape_set_is_distinct() {
        let mut names: Vec<&str> = CALIBRATION_SHAPES.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len());
        assert_eq!(OUTPUT_GOLDEN_NAMES.len(), 6);
    }
}
