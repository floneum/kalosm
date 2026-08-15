//! Launch-count asserts. `resolves_in::<N>` is the tripwire that turns a
//! non-firing fusion rule into a hard test failure rather than a quiet 5-10x
//! throughput regression.
//!
//! The eight named backward shapes are pinned here because those are exactly
//! the hand-fused kernels this design gives up in exchange for deriving them
//! by rewrite. `attention_grads`, `rms_norm_fused`'s backward and the analytic
//! softmax Jacobian are not written by hand anywhere in fusor2; they exist
//! only if a rule mints them.
//!
//! **[`NAMED_BACKWARD_PINS`] is not wired to the suite yet.** Nothing calls
//! [`check_pin`] on those eight names, so each `launches: 1` states where its
//! law must land, not what was measured. Reading the table as a live tripwire
//! is the one mistake it invites: `attention_grads` resolves in **17**
//! dispatches today, not 1. [`BACKWARD_CEILINGS`] is the measured half, and
//! [`check_ceiling`] is what a case can call now.
//!
//! [`FORWARD_CEILINGS`] and [`FORWARD_PINS`] *are* wired, so the shapes whose
//! counts a landing law is about to move are the ones actually guarded.

use fusor2::{Session, };
use fusor2::tensor::Dyn as Tensor;

use crate::harness::CaseError;

/// Resolve `values` and report how many dispatches it took.
///
/// `Session::launch_count` counts dispatches, not encoder submissions, which
/// is what makes the pins below meaningful.
pub fn launches_to_resolve(session: &Session, values: &[Tensor]) -> Result<u64, CaseError> {
    let before = session.launch_count();
    session.resolve(values)?;
    let after = session.launch_count();
    Ok(after.saturating_sub(before))
}

/// Assert that resolving `values` costs exactly `N` dispatches.
pub fn resolves_in<const N: u64>(session: &Session, values: &[Tensor]) -> Result<(), String> {
    resolves_in_n(session, values, N).map_err(|e| e.to_string())
}

/// [`resolves_in`] with the count as a value rather than a const parameter,
/// so a table can drive it.
pub fn resolves_in_n(session: &Session, values: &[Tensor], n: u64) -> Result<(), CaseError> {
    let actual = launches_to_resolve(session, values)?;
    if actual == n {
        Ok(())
    } else {
        Err(format!("expected {n} dispatches to resolve, took {actual}").into())
    }
}

/// The failure message every pinned count shares: which rule must have fired.
pub fn assert_launches(name: &str, expected: u64, actual: u64) -> Result<(), CaseError> {
    if expected == actual {
        return Ok(());
    }
    let rule = rule_for(name);
    Err(format!(
        "{name}: expected {expected} launch(es), measured {actual}. \
         The rule that must have fired is `{rule}`; a non-firing rule is a \
         hard failure here rather than a quiet throughput regression."
    )
    .into())
}

/// One pinned shape: its name, the dispatch count it must resolve in, and the
/// rewrite rule whose absence would explain a miss.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Pin {
    pub name: &'static str,
    pub launches: u64,
    /// The `Rule::name` (SCREAMING_CASE) that mints the
    /// fused form.
    pub rule: &'static str,
}

/// The eight named backward shapes, with the count each must resolve in.
///
/// `conv_epilogue_backward_sunk` is two, not one: the weight gradient and the
/// input gradient are different contractions over the same activation, and
/// only the epilogue sinks.
pub const NAMED_BACKWARD_PINS: [Pin; 8] = [
    Pin {
        name: "attention_grads_kv_single_launch",
        launches: 1,
        rule: "ABSORB+RETARGET+TUPLE",
    },
    Pin {
        name: "attention_lse_recompute",
        launches: 1,
        rule: "ABSORB+RETARGET+TUPLE",
    },
    Pin {
        name: "rms_norm_backward_fused",
        launches: 1,
        rule: "MAP_INTO_FOLD",
    },
    Pin {
        name: "layer_norm_backward_fused",
        launches: 1,
        rule: "MAP_INTO_FOLD",
    },
    Pin {
        name: "softmax_last_dim_backward_analytic",
        launches: 1,
        rule: "SOFTMAX_JACOBIAN",
    },
    Pin {
        name: "softplus_bce_adjoint_single_sigmoid",
        launches: 1,
        rule: "SOFTPLUS_BCE_ADJOINT",
    },
    Pin {
        name: "embedding_scatter_add_backward",
        launches: 1,
        rule: "SCATTER_WG_PRIVATE_MERGE",
    },
    Pin {
        name: "conv_epilogue_backward_sunk",
        launches: 2,
        rule: "SINK_EPILOGUE",
    },
];

/// The eight names, for the acceptance checklist.
pub const NAMED_BACKWARD_SHAPES: [&str; 8] = [
    "attention_grads_kv_single_launch",
    "attention_lse_recompute",
    "rms_norm_backward_fused",
    "layer_norm_backward_fused",
    "softmax_last_dim_backward_analytic",
    "softplus_bce_adjoint_single_sigmoid",
    "embedding_scatter_add_backward",
    "conv_epilogue_backward_sunk",
];

/// Forward counts. These are the shapes whose whole point is that fusion
/// collapses them to one dispatch.
pub const FORWARD_PINS: [Pin; 3] = [
    Pin {
        name: "dense_matmul_with_epilogue",
        launches: 1,
        rule: "SINK_EPILOGUE",
    },
    Pin {
        name: "single_pass_softmax",
        launches: 1,
        rule: "MAP_INTO_FOLD",
    },
    Pin {
        name: "rms_norm_with_bias",
        launches: 1,
        rule: "MAP_INTO_FOLD",
    },
];

/// A launch-count **ceiling**, as opposed to a [`Pin`]'s exact count.
///
/// The pins above state where a shape must land once the rule that fuses it
/// exists. A ceiling states where it lands *today* and forbids getting worse,
/// which is the only honest shape for a count a landing law is about to
/// improve: `resolves_in::<1>` on a shape that takes four today is a failing
/// assert wearing an aspiration's clothes, and a suite full of those cannot
/// tell a regression from an unlanded feature.
///
/// A ceiling that is met with room to spare is not silently fine — it is
/// reported, so tightening it is a one-line edit rather than an
/// archaeological dig.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Ceiling {
    pub name: &'static str,
    /// The measured count. Exceeding it is a regression.
    pub launches: u64,
    /// Where this count must end up once the law named in `rule` lands.
    pub target: u64,
    pub rule: &'static str,
}

/// Forward shapes whose count the fold laws are expected to collapse.
///
/// `attention_with_lse` measures **6** and must become **one**: the output and
/// the log-sum-exp are two slots of one carrier.
///
/// Each number here is the measured value on the hardware it must work on.
/// One constant serves both sessions and the slack on the faster backend is
/// reported by [`check_ceiling`] rather than hidden.
pub const FORWARD_CEILINGS: [Ceiling; 3] = [
    Ceiling {
        name: "attention_forward",
        launches: 5,
        target: 1,
        rule: "ABSORB+RETARGET+TUPLE",
    },
    Ceiling {
        name: "attention_with_lse",
        launches: 6,
        target: 1,
        rule: "ABSORB+RETARGET+TUPLE",
    },
    Ceiling {
        name: "attention_causal_forward",
        launches: 5,
        target: 1,
        rule: "ABSORB+RETARGET+TUPLE+STRIP",
    },
];

/// Backward shapes whose count the fold laws are expected to collapse.
///
/// [`NAMED_BACKWARD_PINS`] states where `attention_grads` **must** land — one
/// dispatch — and nothing in the suite calls [`check_pin`] on it, so that 1 is
/// a target, not a measurement. This table is the measurement.
///
/// The count is **17**, taken on `grads_case`'s shape with `dq`, `dk` and `dv`
/// resolved together.
pub const BACKWARD_CEILINGS: [Ceiling; 1] = [Ceiling {
    name: "attention_grads_all_three",
    launches: 17,
    target: 1,
    rule: "ABSORB+RETARGET+TUPLE",
}];

/// The ceiling with this name, from either table.
pub fn ceiling(name: &str) -> Option<Ceiling> {
    FORWARD_CEILINGS
        .iter()
        .chain(BACKWARD_CEILINGS.iter())
        .find(|c| c.name == name)
        .copied()
}

/// Resolve `values` and check the count against the named ceiling.
///
/// **At or below** the ceiling passes: a law that collapses the shape is an
/// improvement, not a failure, and four laws are landing against this table at
/// once. Over the ceiling is a regression and the message names both the
/// shape and the law that must have stopped firing.
pub fn check_ceiling(session: &Session, name: &str, values: &[Tensor]) -> Result<(), CaseError> {
    let Some(c) = ceiling(name) else {
        return Err(format!("{name} has no launch ceiling").into());
    };
    let actual = launches_to_resolve(session, values)?;
    if actual > c.launches {
        return Err(format!(
            "{name}: {actual} dispatches, ceiling {}. The target is {} once `{}` lands; \
             this is a regression away from it.",
            c.launches, c.target, c.rule
        )
        .into());
    }
    // Slack is reported, per this table's own contract: a ceiling met with
    // room to spare is a one-line edit away from being tightened, and a
    // silent one is an archaeological dig.
    if actual < c.launches {
        eprintln!(
            "note: {name} resolves in {actual}, ceiling {} (target {} via `{}`) — \
             the ceiling can be tightened to {actual}.",
            c.launches, c.target, c.rule
        );
    }
    Ok(())
}

fn rule_for(name: &str) -> &'static str {
    NAMED_BACKWARD_PINS
        .iter()
        .chain(FORWARD_PINS.iter())
        .find(|p| p.name == name)
        .map(|p| p.rule)
        .unwrap_or("<unpinned>")
}

/// The pin with this name, from either table.
pub fn pin(name: &str) -> Option<Pin> {
    NAMED_BACKWARD_PINS
        .iter()
        .chain(FORWARD_PINS.iter())
        .find(|p| p.name == name)
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_eight_names_and_the_eight_pins_agree() {
        assert_eq!(NAMED_BACKWARD_PINS.len(), NAMED_BACKWARD_SHAPES.len());
        for (pin, name) in NAMED_BACKWARD_PINS.iter().zip(NAMED_BACKWARD_SHAPES) {
            assert_eq!(pin.name, name);
        }
    }

    #[test]
    fn every_pin_is_unique_and_positive() {
        let mut seen = Vec::new();
        for p in NAMED_BACKWARD_PINS.iter().chain(FORWARD_PINS.iter()) {
            assert!(p.launches >= 1, "{} pins zero launches", p.name);
            assert!(!seen.contains(&p.name), "{} is pinned twice", p.name);
            seen.push(p.name);
        }
    }

    #[test]
    fn only_the_conv_epilogue_pin_costs_two() {
        for p in NAMED_BACKWARD_PINS {
            let expected = if p.name == "conv_epilogue_backward_sunk" {
                2
            } else {
                1
            };
            assert_eq!(p.launches, expected, "{}", p.name);
        }
    }

    #[test]
    fn a_missed_pin_names_the_rule_that_should_have_fired() {
        let err = assert_launches("rms_norm_backward_fused", 1, 4)
            .unwrap_err()
            .to_string();
        assert!(err.contains("measured 4"), "{err}");
        assert!(err.contains("MAP_INTO_FOLD"), "{err}");
    }

    #[test]
    fn a_hit_pin_is_silent() {
        assert!(assert_launches("attention_lse_recompute", 1, 1).is_ok());
    }

    /// Every ceiling states where its shape is today and where the law must
    /// put it. A ceiling already at its target is not a ceiling — it is a pin,
    /// and it belongs in [`FORWARD_PINS`].
    #[test]
    fn every_ceiling_names_a_target_it_has_not_reached() {
        let mut seen = Vec::new();
        for c in FORWARD_CEILINGS
            .iter()
            .chain(BACKWARD_CEILINGS.iter())
            .copied()
        {
            assert!(c.target >= 1, "{} targets zero launches", c.name);
            assert!(
                c.launches > c.target,
                "{} is already at its target of {}; pin it instead",
                c.name,
                c.target
            );
            assert!(!seen.contains(&c.name), "{} has two ceilings", c.name);
            seen.push(c.name);
        }
        assert!(ceiling("attention_with_lse").is_some());
        assert!(ceiling("no_such_shape").is_none());
    }

    /// `attention_with_lse` must not be cheaper than plain attention forward
    /// once the laws land: `o` and `lse` are two slots of **one** carrier, so
    /// the two shapes converge on the same single dispatch — which is
    /// strictly better than `KFlash`, where `FlashOut::Output` and
    /// `FlashOut::LogSumExp` were two dispatches of two kernels.
    #[test]
    fn the_lse_shape_targets_one_dispatch_like_the_plain_one() {
        let plain = ceiling("attention_forward").expect("attention_forward");
        let lse = ceiling("attention_with_lse").expect("attention_with_lse");
        assert_eq!(plain.target, 1);
        assert_eq!(lse.target, 1);
        assert!(
            lse.launches > plain.launches,
            "the lse shape costs {} today and plain attention {}; if they are already \
             equal the second output is free and the target is met",
            lse.launches,
            plain.launches
        );
    }

    #[test]
    fn pin_lookup_covers_both_tables_and_rejects_strangers() {
        assert!(pin("single_pass_softmax").is_some());
        assert!(pin("embedding_scatter_add_backward").is_some());
        assert!(pin("no_such_shape").is_none());
        assert_eq!(rule_for("no_such_shape"), "<unpinned>");
    }
}
