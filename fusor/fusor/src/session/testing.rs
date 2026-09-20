//! Differential compiler checks, compiled only for the conformance harness.
//! Never writes timing observations or selects a production candidate.
use super::*;

/// Class members the differential harness found computing wrong values, process
/// wide. Every entry is a live miscompile: a member of some e-class whose
/// value disagrees with its siblings'. The conformance harness executes every
/// class member (`FUSOR_VERIFY_MEMBERS`) and fails the run when this is
/// nonzero.
static WRONG_MEMBERS: AtomicU64 = AtomicU64::new(0);

/// Number of live member-verification failures observed by this process.
pub fn wrong_member_count() -> u64 {
    WRONG_MEMBERS.load(Ordering::Relaxed)
}

/// Whether the test harness checks class members during resolves.
///
/// Starts from `FUSOR_VERIFY_MEMBERS` and is settable from there on, because
/// the sweep is a per-kernel correctness pass and a suite that reruns a case
/// at several shapes does not need to pay for it at every one. It is by far
/// the most expensive thing a resolve can do: one small sampling case races
/// about 470 candidates under it.
static VERIFY_MEMBERS: std::sync::OnceLock<std::sync::atomic::AtomicBool> =
    std::sync::OnceLock::new();

fn verify_members_flag() -> &'static std::sync::atomic::AtomicBool {
    VERIFY_MEMBERS.get_or_init(|| {
        std::sync::atomic::AtomicBool::new(std::env::var_os("FUSOR_VERIFY_MEMBERS").is_some())
    })
}

/// Whether the member sweep is currently on. See [`set_verify_members`].
pub fn verify_members() -> bool {
    verify_members_flag().load(Ordering::Relaxed)
}

/// Turn the member sweep on or off for the resolves that follow.
pub fn set_verify_members(on: bool) {
    verify_members_flag().store(on, Ordering::Relaxed);
}

impl Session {
    pub(super) fn check_members(
        &self,
        guard: &ResolveGuard<'_>,
        graph: &GraphRef,
        roots: &[Id],
        base: Arc<Plan>,
        values: &[Tensor],
    ) -> Result<Arc<Plan>> {
        {
            let g = graph.state().egraph.lock();
            if base.launches.iter().any(|l| {
                l.members
                    .iter()
                    .any(|m| g.semantics().effect(&g.node(*m).op) != Effect::Pure)
            }) {
                return Ok(base);
            }
        }
        let read = || -> Result<Vec<(Dtype, Vec<u8>)>> {
            values
                .iter()
                .map(|v| {
                    Ok((
                        graph.facts(v.id).dtype,
                        self.read_bytes_locked(guard, graph, v.id)?,
                    ))
                })
                .collect()
        };
        self.run(graph, &base, values)?;
        let expected = read()?;
        let mut checked = vec![Arc::clone(&base)];
        let mut wrong = 0;
        for ix in 0..base.launches.len() {
            let variants = {
                let g = graph.state().egraph.lock();
                self.inner.extractor.test_launch_variants(
                    &g,
                    roots,
                    &base,
                    ix,
                    self.inner.cost.as_ref(),
                )
            };
            for (label, plan) in variants {
                let plan = Arc::new(plan);
                checked.push(Arc::clone(&plan));
                self.run(graph, &plan, values).map_err(|e| {
                    Error::Plan(format!(
                        "candidate `{label}` of launch {ix} failed to execute: {e}"
                    ))
                })?;
                let actual = read()?;
                if expected
                    .iter()
                    .zip(&actual)
                    .any(|((dt, a), (got_dt, b))| dt != got_dt || !agrees(*dt, a, b))
                {
                    WRONG_MEMBERS.fetch_add(1, Ordering::Relaxed);
                    wrong += 1;
                    let detail = expected.iter().zip(&actual).enumerate().find_map(
                        |(o, ((dt, a), (_, b)))| first_mismatch(*dt, a, b).map(|m| (o, m)),
                    );
                    eprintln!(
                        "[compiler-test] MISCOMPILE: candidate `{label}` of launch {ix}: {detail:?}"
                    );
                }
            }
        }
        let arena = graph.state().egraph.lock().arena_id();
        self.inner.device.release_candidates(arena, &checked, &base);
        if wrong != 0 {
            return Err(Error::Plan(format!(
                "compiler test found {wrong} non-equivalent class members"
            )));
        }
        Ok(base)
    }
}

/// The first disagreeing float element and the worst one, for the MISCOMPILE
/// report: `(first_index, expected, got, worst_abs_diff)`.
pub(super) fn first_mismatch(dtype: Dtype, a: &[u8], b: &[u8]) -> Option<(usize, f32, f32, f32)> {
    if !dtype.is_float() {
        return None;
    }
    let f = |s: &[u8]| {
        s.chunks_exact(dtype.byte_size() as usize)
            .map(|c| match dtype {
                Dtype::F16 => half::f16::from_le_bytes(c.try_into().unwrap()).to_f32(),
                Dtype::BF16 => half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32(),
                _ => f32::from_le_bytes(c.try_into().unwrap()),
            })
            .collect::<Vec<f32>>()
    };
    let (x, y) = (f(a), f(b));
    let scale = x
        .iter()
        .filter(|v| v.is_finite())
        .fold(1.0f32, |m, v| m.max(v.abs()));
    let tolerance = if dtype == Dtype::BF16 { 1e-2 } else { 1e-3 } * scale;
    let mut first = None;
    let mut worst = 0.0f32;
    for (i, (p, q)) in x.iter().zip(&y).enumerate() {
        if p == q {
            continue;
        }
        let d = (p - q).abs();
        if (!p.is_finite() || !q.is_finite() || d > tolerance) && first.is_none() {
            first = Some((i, *p, *q));
        }
        worst = worst.max(d);
    }
    first.map(|(i, p, q)| (i, p, q, worst))
}

pub(super) fn agrees(dtype: Dtype, a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    if a == b {
        return true;
    }
    dtype.is_float()
        && a.len().is_multiple_of(dtype.byte_size() as usize)
        && first_mismatch(dtype, a, b).is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn member_comparison_handles_narrow_float_rounding() {
        let a = half::f16::from_f32(1.000_976_6).to_f32();
        let b = half::f16::from_f32(1.018_554_7).to_f32();
        let separate = half::f16::from_f32(half::f16::from_f32(a * b).to_f32() + a);
        let fused = half::f16::from_f32(a.mul_add(b, a));
        assert_ne!(separate, fused);
        assert!(agrees(
            Dtype::F16,
            &separate.to_le_bytes(),
            &fused.to_le_bytes()
        ));
        for dtype in [Dtype::F16, Dtype::BF16, Dtype::F32] {
            assert!(!agrees(dtype, &[0], &[1]));
            let bytes = |v| match dtype {
                Dtype::F16 => half::f16::from_f32(v).to_le_bytes().to_vec(),
                Dtype::BF16 => half::bf16::from_f32(v).to_le_bytes().to_vec(),
                _ => v.to_le_bytes().to_vec(),
            };
            let rounded = if dtype == Dtype::BF16 {
                2.015625
            } else {
                2.001
            };
            assert!(agrees(dtype, &bytes(2.0), &bytes(rounded)));
            for wrong in [2.125, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
                assert!(!agrees(dtype, &bytes(2.0), &bytes(wrong)));
            }
            assert!(!agrees(
                dtype,
                &[bytes(f32::INFINITY), bytes(2.0)].concat(),
                &[bytes(f32::INFINITY), bytes(2.125)].concat()
            ));
        }
        assert!(!agrees(
            Dtype::U32,
            &10000u32.to_le_bytes(),
            &10001u32.to_le_bytes()
        ));
    }
}
