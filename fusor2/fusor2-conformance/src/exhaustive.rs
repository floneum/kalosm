//! `--exhaustive`: sweep the full schedule domain instead of the shipped move
//! budget, so a domain point the local search never reaches still gets
//! correctness coverage.
//!
//! Two separate claims:
//!
//! * Every point is correct. For each `SchedPoint` a lowering rule mints, lower
//!   to `KernelIr`, run `verify_l2` + `verify_arena` + `verify_uniformity`,
//!   emit, launch, and diff against the CPU reference.
//! * The chosen point is nearly the best. Across a shape list, the point
//!   extraction selected must be within [`WITHIN_FRACTION`] of the measured
//!   best. This is the only place in the suite that measures rather than
//!   models.

use std::sync::Arc;
use std::time::{Duration, Instant};

use fusor2_ir::device::Caps;
use fusor2_ir::ir::level1::{SchedPoint, ScheduleDomain};
use fusor2_ir::ir::level2::{ArenaPlanner, KernelIr};

use crate::harness::CaseError;

/// The chosen schedule point must be within this fraction of the measured
/// best. 15% is the spread between neighbouring coop tiles.
pub const WITHIN_FRACTION: f64 = 0.15;

/// Shapes the `--exhaustive` proximity claim is asserted on, spanning every
/// family a lowering rule mints a domain for.
pub const PROXIMITY_SHAPES: [(&str, [u64; 3]); 8] = [
    ("coop_4096_cube", [4096, 4096, 4096]),
    ("coop_skinny", [64, 4096, 4096]),
    ("sgemm_768_cube", [768, 768, 768]),
    ("sgemm_ragged", [500, 700, 900]),
    ("sgemv_decode", [1, 4096, 4096]),
    ("sgemv_wide", [1, 8192, 1024]),
    ("fold_long_axis", [128, 1, 16384]),
    ("map_conv_epilogue", [128, 768, 64]),
];

/// Sweep every point of `domain`, running `body` at each index.
///
/// Errors short-circuit: the first miscompiled point is the interesting one,
/// and a `CoopDomain` has thousands of siblings that would fail the same way.
pub fn sweep(
    domain: &ScheduleDomain,
    body: &mut dyn FnMut(usize) -> Result<(), String>,
) -> Result<(), String> {
    if domain.is_empty() {
        return Err("the schedule domain is empty: the node is unselectable".to_string());
    }
    for index in 0..domain.len() {
        body(index).map_err(|e| {
            let point = domain
                .point(index)
                .map_or_else(|| "<out of range>".to_string(), |p| format!("{p:?}"));
            format!("schedule point {index} of {} ({point}): {e}", domain.len())
        })?;
    }
    Ok(())
}

/// [`sweep`] handing the resolved [`SchedPoint`] to `body` rather than its
/// index, which is what a lowering call wants.
pub fn sweep_points(
    domain: &ScheduleDomain,
    body: &mut dyn FnMut(SchedPoint) -> Result<(), String>,
) -> Result<(), String> {
    let points: Vec<SchedPoint> = domain.iter().collect();
    sweep(domain, &mut |i| body(points[i]))
}

/// The three level verifiers every swept point must pass before it is allowed
/// anywhere near an emitter.
///
/// `verify_arena`'s all-pairs recheck re-derives, from liveness facts alone,
/// that every byte-overlapping tile pair is separated by a guaranteed-uniform
/// barrier. A failure here is a lowering error, never a runtime race.
pub fn verify_point(
    ir: &KernelIr,
    planner: &dyn ArenaPlanner,
    caps: &Caps,
) -> Result<(), CaseError> {
    // `verify_l2` takes `Caps`: the f16/bf16 gate is a capability question.
    fusor2_tile::verify_l2(ir, caps)
        .map_err(|e| -> CaseError { format!("verify_l2: {e}").into() })?;
    let plan = planner
        .arena_plan(ir, caps)
        .map_err(|e| -> CaseError { format!("arena_plan: {e}").into() })?;
    planner
        .verify_arena(ir, &plan)
        .map_err(|e| -> CaseError { format!("verify_arena: {e}").into() })?;
    planner
        .verify_uniformity(ir)
        .map_err(|e| -> CaseError { format!("verify_uniformity: {e}").into() })?;
    Ok(())
}

/// One measured point.
#[derive(Copy, Clone, Debug)]
pub struct Measured {
    pub index: usize,
    pub point: SchedPoint,
    pub elapsed: Duration,
}

/// Time `run` at every point of `domain`, keeping the wall clock of each.
///
/// Points that fail to lower are skipped rather than reported: an illegal
/// geometry is the domain generator's business, and `verify_point` already
/// covers the ones that do lower.
pub fn measure(
    domain: &ScheduleDomain,
    repeats: u32,
    mut run: impl FnMut(SchedPoint) -> Result<(), CaseError>,
) -> Result<Vec<Measured>, CaseError> {
    let repeats = repeats.max(1);
    let mut out = Vec::with_capacity(domain.len());
    for (index, point) in domain.iter().enumerate() {
        // One untimed warm-up so compilation does not land in the sample.
        if run(point).is_err() {
            continue;
        }
        let start = Instant::now();
        let mut ok = true;
        for _ in 0..repeats {
            if run(point).is_err() {
                ok = false;
                break;
            }
        }
        if ok {
            out.push(Measured {
                index,
                point,
                elapsed: start.elapsed() / repeats,
            });
        }
    }
    if out.is_empty() {
        return Err("no schedule point in the domain ran successfully".into());
    }
    Ok(out)
}

/// The fastest measured point.
pub fn best(measured: &[Measured]) -> Option<Measured> {
    measured.iter().copied().min_by_key(|m| m.elapsed)
}

/// Assert the extractor's choice is within [`WITHIN_FRACTION`] of the
/// measured best.
///
/// Reported as a fraction rather than a rank, since a rank says nothing about
/// how much slower the choice is.
pub fn assert_choice_is_near_best(
    shape: &str,
    chosen: SchedPoint,
    measured: &[Measured],
) -> Result<(), CaseError> {
    let Some(best) = best(measured) else {
        return Err(format!("{shape}: nothing was measured").into());
    };
    let Some(chosen_row) = measured.iter().find(|m| m.point == chosen) else {
        return Err(format!(
            "{shape}: extraction chose {chosen:?}, which is not a point the sweep \
             could run — the domain the extractor saw and the one the sweep \
             enumerated disagree"
        )
        .into());
    };
    let best_ns = best.elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    let excess = chosen_row.elapsed.as_secs_f64() / best_ns - 1.0;
    if excess <= WITHIN_FRACTION {
        return Ok(());
    }
    Err(format!(
        "{shape}: extraction chose {chosen:?} at {:.3} ms, but {:?} measured \
         {:.3} ms — {:.1}% slower, over the {:.0}% bar",
        chosen_row.elapsed.as_secs_f64() * 1e3,
        best.point,
        best.elapsed.as_secs_f64() * 1e3,
        excess * 100.0,
        WITHIN_FRACTION * 100.0
    )
    .into())
}

/// Whether `--exhaustive` was asked for. The binary parses the flag; a case
/// running under `cargo test` reads the environment.
pub fn requested(args: &[String]) -> bool {
    args.iter().any(|a| a == "--exhaustive")
        || std::env::var("FUSOR2_CONFORMANCE_EXHAUSTIVE").is_ok_and(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
        })
}

/// The planner the sweep verifies against: the same memoized `arena_plan`
/// `verify_l1` and the L2 emitter use. An estimator here would let extraction
/// commit a plan that fails L2 verification and silently falls back.
pub fn planner() -> Arc<dyn ArenaPlanner> {
    fusor2_tile::Planner::shared()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_domain_sweeps_exactly_once() {
        let mut seen = Vec::new();
        sweep(&ScheduleDomain::Point, &mut |i| {
            seen.push(i);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec![0]);
    }

    #[test]
    fn sweep_points_resolves_every_index() {
        let mut seen = Vec::new();
        sweep_points(&ScheduleDomain::Point, &mut |p| {
            seen.push(p);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec![SchedPoint::Point]);
    }

    #[test]
    fn a_failing_point_names_its_index_and_the_point() {
        let err = sweep(&ScheduleDomain::Point, &mut |_| Err("bad tile".into())).unwrap_err();
        assert!(err.contains("schedule point 0 of 1"), "{err}");
        assert!(err.contains("Point"), "{err}");
        assert!(err.contains("bad tile"), "{err}");
    }

    #[test]
    fn an_empty_domain_is_a_failure_not_a_silent_pass() {
        let empty = ScheduleDomain::Sgemm(Default::default());
        assert!(empty.is_empty());
        let err = sweep(&empty, &mut |_| Ok(())).unwrap_err();
        assert!(err.contains("unselectable"), "{err}");
    }

    #[test]
    fn proximity_accepts_within_the_bar_and_rejects_past_it() {
        let rows = [
            Measured {
                index: 0,
                point: SchedPoint::Point,
                elapsed: Duration::from_micros(100),
            },
            Measured {
                index: 1,
                point: SchedPoint::Fold(fusor2_ir::ir::level1::FoldStrat::Subgroup),
                elapsed: Duration::from_micros(110),
            },
        ];
        // 10% slower than the best: inside the 15% bar.
        assert!(assert_choice_is_near_best("demo", rows[1].point, &rows).is_ok());
        // The best itself is trivially inside.
        assert!(assert_choice_is_near_best("demo", rows[0].point, &rows).is_ok());

        let far = [
            rows[0],
            Measured {
                elapsed: Duration::from_micros(200),
                ..rows[1]
            },
        ];
        let err = assert_choice_is_near_best("demo", far[1].point, &far)
            .unwrap_err()
            .to_string();
        assert!(err.contains("100.0% slower"), "{err}");
        assert!(err.contains("15% bar"), "{err}");
    }

    #[test]
    fn a_chosen_point_the_sweep_never_saw_is_a_disagreement_not_a_pass() {
        let rows = [Measured {
            index: 0,
            point: SchedPoint::Point,
            elapsed: Duration::from_micros(100),
        }];
        let stranger = SchedPoint::Fold(fusor2_ir::ir::level1::FoldStrat::Subgroup);
        let err = assert_choice_is_near_best("demo", stranger, &rows)
            .unwrap_err()
            .to_string();
        assert!(err.contains("disagree"), "{err}");
    }

    #[test]
    fn measure_reports_one_row_per_runnable_point() {
        let rows = measure(&ScheduleDomain::Point, 2, |_| Ok(())).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].point, SchedPoint::Point);
        assert!(best(&rows).is_some());
    }

    #[test]
    fn measure_fails_when_nothing_ran() {
        assert!(measure(&ScheduleDomain::Point, 1, |_| Err("no".into())).is_err());
    }

    #[test]
    fn the_flag_is_read_from_argv_or_the_environment() {
        assert!(requested(&["--exhaustive".to_string()]));
        assert!(!requested(&["--other".to_string()]));
    }

    #[test]
    fn the_proximity_shape_list_is_eight_distinct_shapes() {
        assert_eq!(PROXIMITY_SHAPES.len(), 8);
        let mut names: Vec<&str> = PROXIMITY_SHAPES.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len());
    }
}
