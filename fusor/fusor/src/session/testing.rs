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
                        |(o, ((dt, a), (_, b)))| {
                            (*dt == Dtype::F32)
                                .then(|| first_mismatch(a, b).map(|m| (o, m)))
                                .flatten()
                        },
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

/// The first disagreeing f32 element and the worst one, for the MISCOMPILE
/// report: `(first_index, expected, got, worst_abs_diff)`.
pub(super) fn first_mismatch(a: &[u8], b: &[u8]) -> Option<(usize, f32, f32, f32)> {
    let f = |s: &[u8]| {
        s.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect::<Vec<f32>>()
    };
    let (x, y) = (f(a), f(b));
    let scale = x.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let mut first = None;
    let mut worst = 0.0f32;
    for (i, (p, q)) in x.iter().zip(&y).enumerate() {
        let d = (p - q).abs();
        if d > 1e-3 * scale && first.is_none() {
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
    if dtype != Dtype::F32 {
        return false;
    }
    let f = |s: &[u8]| {
        s.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect::<Vec<f32>>()
    };
    let (x, y) = (f(a), f(b));
    let scale = x.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    x.iter().zip(&y).all(|(p, q)| (p - q).abs() <= 1e-3 * scale)
}
