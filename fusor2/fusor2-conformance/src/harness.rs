//! The test harness: build one named case, run it on every available backend,
//! report rather than panic.
//!
//! Cases return `Err` instead of asserting: the browser runner cannot recover
//! from a wasm panic. [`run_one`] wraps each case in `catch_unwind`, so a
//! panic is one failed row rather than a dead run.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, MutexGuard, OnceLock};

use fusor2::graph::GraphRef;
use fusor2::{Device, Dim, Dtype, Session, Tensor};

/// Error returned by a conformance case. Boxed so a case can return a
/// comparison mismatch or a free-form message uniformly.
pub type CaseError = Box<dyn std::error::Error>;

/// Result of a conformance case.
pub type CaseResult = Result<(), CaseError>;

/// `assert!` that returns `Err` from a `CaseResult`-returning case instead of
/// panicking.
#[macro_export]
macro_rules! ensure {
    ($cond:expr $(,)?) => {
        if $cond {
        } else {
            return ::core::result::Result::Err(
                format!("assertion failed: {}", ::core::stringify!($cond)).into(),
            );
        }
    };
    ($cond:expr, $($arg:tt)+) => {
        if $cond {
        } else {
            return ::core::result::Result::Err(format!($($arg)+).into());
        }
    };
}

/// `assert_eq!` counterpart of [`ensure!`].
#[macro_export]
macro_rules! ensure_eq {
    ($a:expr, $b:expr $(,)?) => {{
        let (a, b) = (&$a, &$b);
        if a != b {
            return ::core::result::Result::Err(
                format!(
                    "assertion failed: `{} == {}`\n  left: `{:?}`\n right: `{:?}`",
                    ::core::stringify!($a),
                    ::core::stringify!($b),
                    a,
                    b
                )
                .into(),
            );
        }
    }};
    ($a:expr, $b:expr, $($arg:tt)+) => {{
        let (a, b) = (&$a, &$b);
        if a != b {
            return ::core::result::Result::Err(format!($($arg)+).into());
        }
    }};
}

/// `assert_ne!` counterpart of [`ensure!`].
#[macro_export]
macro_rules! ensure_ne {
    ($a:expr, $b:expr $(,)?) => {{
        let (a, b) = (&$a, &$b);
        if a == b {
            return ::core::result::Result::Err(
                format!(
                    "assertion failed: `{} != {}`\n  both: `{:?}`",
                    ::core::stringify!($a),
                    ::core::stringify!($b),
                    a
                )
                .into(),
            );
        }
    }};
    ($a:expr, $b:expr, $($arg:tt)+) => {{
        let (a, b) = (&$a, &$b);
        if a == b {
            return ::core::result::Result::Err(format!($($arg)+).into());
        }
    }};
}

/// The body of a case. `Fn` rather than `FnOnce` so one case runs on every
/// session in [`sessions`] without being rebuilt per backend.
pub type CaseFn = Box<dyn Fn(&Session) -> CaseResult + Send + Sync>;

/// One conformance case, named `area::case`.
pub struct Case {
    pub name: String,
    pub area: &'static str,
    pub run: CaseFn,
}

impl Case {
    pub fn new(
        area: &'static str,
        case: impl Into<String>,
        run: impl Fn(&Session) -> CaseResult + Send + Sync + 'static,
    ) -> Self {
        let case = case.into();
        Self {
            name: format!("{area}::{case}"),
            area,
            run: Box::new(run),
        }
    }

    /// The case name without its area prefix.
    pub fn short(&self) -> &str {
        self.name
            .split_once("::")
            .map_or(&self.name[..], |(_, c)| c)
    }
}

impl std::fmt::Debug for Case {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Case({})", self.name)
    }
}

/// A list of cases. Every `suite::<area>::cases()` returns one.
#[derive(Default)]
pub struct Cases(Vec<Case>);

impl Cases {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn push(
        &mut self,
        area: &'static str,
        name: impl Into<String>,
        run: impl Fn(&Session) -> CaseResult + Send + Sync + 'static,
    ) -> &mut Self {
        self.0.push(Case::new(area, name, run));
        self
    }

    /// Push an already-built [`Case`], for tables that produce one directly.
    pub fn push_case(&mut self, case: Case) -> &mut Self {
        self.0.push(case);
        self
    }

    pub fn extend(&mut self, other: Cases) -> &mut Self {
        self.0.extend(other.0);
        self
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Case> {
        self.0.iter()
    }

    /// The names, in registration order.
    pub fn names(&self) -> Vec<&str> {
        self.0.iter().map(|c| c.name.as_str()).collect()
    }
}

impl IntoIterator for Cases {
    type Item = Case;
    type IntoIter = std::vec::IntoIter<Case>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

fn require_gpu() -> bool {
    std::env::var("FUSOR2_CONFORMANCE_REQUIRE_GPU")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn acquire_gpu() -> Option<Device> {
    match Device::gpu_blocking() {
        Ok(gpu) => Some(gpu),
        Err(err) => {
            assert!(
                !require_gpu(),
                "GPU conformance is required but no GPU device was available: {err}"
            );
            None
        }
    }
}

/// Acquire the GPU device; `None` means none is available.
///
/// On wasm the handle is memoized for the life of the page, since acquiring a
/// device per case is prohibitively slow. Natively it is not: a thread-local
/// holding a wgpu `Device` panics when dropped during thread teardown, because
/// wgpu's `Drop` touches already-destroyed thread-locals.
#[cfg(not(target_arch = "wasm32"))]
fn cached_gpu() -> Option<Device> {
    acquire_gpu()
}

#[cfg(target_arch = "wasm32")]
fn cached_gpu() -> Option<Device> {
    thread_local! {
        static GPU: std::cell::RefCell<Option<Option<Device>>> =
            const { std::cell::RefCell::new(None) };
    }
    if let Some(cached) = GPU.with(|cell| cell.borrow().clone()) {
        return cached;
    }
    let acquired = acquire_gpu();
    GPU.with(|cell| *cell.borrow_mut() = Some(acquired.clone()));
    acquired
}

/// Always CPU, plus GPU when one is available. Every case runs on every
/// session returned here; nothing in the suite mentions a concrete backend.
pub fn sessions() -> Vec<Session> {
    let mut out = Vec::new();
    if let Ok(cpu) = Device::cpu()
        && let Ok(session) = Session::new(cpu)
    {
        out.push(session);
    }
    if let Some(gpu) = cached_gpu()
        && let Ok(session) = Session::new(gpu)
    {
        out.push(session);
    }
    out
}

/// True when `session` runs on the GPU.
pub fn is_gpu(session: &Session) -> bool {
    matches!(session.device(), Device::Gpu(_))
}

/// Serializes GPU cases across the process. A launch-count assert is
/// meaningless if another case is dispatching into the same device
/// concurrently under `cargo test`'s thread pool.
pub fn gpu_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        // A previous case that panicked poisoned the mutex; the guard exists
        // for serialization, not for data integrity, so recover.
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The LCG every golden depends on; changing a constant here invalidates every
/// recorded output hash.
///
/// `state = state * 6364136223846793005 + 1442695040888963407`, value
/// `((state >> 33) as f32 / 2^31) - 0.5`, in `[-0.5, 0.5)`.
///
/// The seed enters the state unmodified, so distinct seeds give distinct
/// streams.
pub fn fill(seed: u32, len: usize) -> Vec<f32> {
    let mut state = seed as u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

/// `len` values in `[lo, hi)` from the same LCG, for ops with a restricted
/// domain (`acosh` needs `x >= 1`, `log` needs `x > 0`).
pub fn fill_range(seed: u32, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    fill(seed, len)
        .into_iter()
        .map(|v| lo + (v + 0.5) * (hi - lo))
        .collect()
}

/// Element count of a fully constant shape. Panics on a symbolic extent, which
/// the host-side references cannot size.
pub fn dense_len(shape: &[Dim]) -> usize {
    shape
        .iter()
        .map(|d| match d {
            Dim::Const(n) => *n as usize,
            Dim::Sym(s) => panic!("dense_len over a symbolic extent {s:?}"),
        })
        .product()
}

/// `Dim::Const` extents from plain integers, which is what every case wants.
pub fn dims(shape: &[u64]) -> Vec<Dim> {
    shape.iter().map(|n| Dim::Const(*n)).collect()
}

/// Upload f32 host data. `Tensor::from_slice` takes bytes, so this is the one
/// place the little-endian encoding lives.
pub fn from_f32(graph: &GraphRef, shape: &[Dim], data: &[f32]) -> fusor2::Result<Tensor> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Tensor::from_slice(graph, Dtype::F32, shape, &bytes)
}

/// Upload u32 host data (indices).
pub fn from_u32(graph: &GraphRef, shape: &[Dim], data: &[u32]) -> fusor2::Result<Tensor> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Tensor::from_slice(graph, Dtype::U32, shape, &bytes)
}

/// Outcome of one case on one backend.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Pass,
    Fail(String),
    Skipped(String),
}

impl Outcome {
    pub fn is_pass(&self) -> bool {
        matches!(self, Outcome::Pass)
    }
    pub fn is_fail(&self) -> bool {
        matches!(self, Outcome::Fail(_))
    }
    pub fn is_skipped(&self) -> bool {
        matches!(self, Outcome::Skipped(_))
    }
}

/// The marker a case body puts at the front of its error to say "this device
/// cannot run this row" rather than "this row is wrong".
///
/// [`guard`] sorts the two apart, so a run on a device without bf16 reports
/// `skip` rather than `ok` or `FAILED`.
pub const SKIP_PREFIX: &str = "skipped: ";

/// Build the `Err` that [`guard`] turns into [`Outcome::Skipped`].
pub fn skip(reason: impl std::fmt::Display) -> CaseError {
    format!("{SKIP_PREFIX}{reason}").into()
}

/// One row of a run: which case, which backend, what happened.
#[derive(Clone, Debug)]
pub struct Report {
    pub case: String,
    pub backend: &'static str,
    pub outcome: Outcome,
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panicked with a non-string payload".to_string()
    }
}

fn backend_name(session: &Session) -> &'static str {
    match session.device() {
        Device::Cpu(_) => "cpu",
        Device::Gpu(_) => "gpu",
    }
}

/// Run a case body, converting a panic into a failure message and a
/// [`SKIP_PREFIX`] error into [`Outcome::Skipped`]. A panic is never a skip,
/// however it is worded.
pub fn guard(body: impl FnOnce() -> CaseResult) -> Outcome {
    let hushed = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(body));
    std::panic::set_hook(hushed);
    match result {
        Ok(Ok(())) => Outcome::Pass,
        Ok(Err(err)) => {
            let message = err.to_string();
            match message.strip_prefix(SKIP_PREFIX) {
                Some(why) => Outcome::Skipped(why.to_string()),
                None => Outcome::Fail(message),
            }
        }
        Err(payload) => Outcome::Fail(format!("panicked: {}", panic_message(payload))),
    }
}

/// Run one case against one session.
pub fn run_one(case: &Case, session: &Session) -> Outcome {
    guard(|| (case.run)(session))
}

/// Run every case whose name contains `filter` (all of them on `None`) on
/// every available session, reporting progress as it goes.
pub fn run_filtered(filter: Option<&str>, mut progress: impl FnMut(&Report)) -> Vec<Report> {
    let registry = crate::suite::registry();
    let sessions = sessions();
    let mut out = Vec::with_capacity(registry.len() * sessions.len().max(1));
    for case in registry
        .iter()
        .filter(|c| filter.is_none_or(|f| c.name.contains(f)))
    {
        if sessions.is_empty() {
            let report = Report {
                case: case.name.clone(),
                backend: "none",
                outcome: Outcome::Skipped("no device available".to_string()),
            };
            progress(&report);
            out.push(report);
            continue;
        }
        for session in &sessions {
            let _guard = is_gpu(session).then(gpu_test_guard);
            let report = Report {
                case: case.name.clone(),
                backend: backend_name(session),
                outcome: run_one(case, session),
            };
            progress(&report);
            out.push(report);
        }
    }
    out
}

/// Print one line per report and return the failure count.
pub fn summarize(reports: &[Report]) -> usize {
    let mut failures = 0;
    for r in reports {
        match &r.outcome {
            Outcome::Pass => println!("ok      {} [{}]", r.case, r.backend),
            Outcome::Skipped(why) => println!("skip    {} [{}]: {why}", r.case, r.backend),
            Outcome::Fail(err) => {
                failures += 1;
                println!("FAILED  {} [{}]: {err}", r.case, r.backend);
            }
        }
    }
    // A skip is counted separately, never folded into the pass count.
    let skipped = reports.iter().filter(|r| r.outcome.is_skipped()).count();
    let passed = reports.iter().filter(|r| r.outcome.is_pass()).count();
    println!(
        "{} results: {passed} passed, {skipped} skipped, {failures} failed",
        reports.len()
    );
    failures
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_is_the_exact_reference_generator() {
        // Seed 1 enters as state 1, so the first value is
        // ((s0 >> 33) / 2^31) - 0.5 with s0 the first recurrence step.
        let s0 = 1u64
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let expected = ((s0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
        assert_eq!(fill(1, 1)[0], expected);
    }

    #[test]
    fn fill_is_deterministic_and_seed_sensitive() {
        assert_eq!(fill(4, 32), fill(4, 32));
        assert_ne!(fill(4, 32), fill(5, 32));
    }

    #[test]
    fn no_two_seeds_share_a_stream() {
        let mut streams: Vec<Vec<u32>> = (0..256)
            .map(|seed| fill(seed, 4).iter().map(|v| v.to_bits()).collect())
            .collect();
        let before = streams.len();
        streams.sort_unstable();
        streams.dedup();
        assert_eq!(before, streams.len(), "two seeds produced the same stream");
    }

    #[test]
    fn the_seeds_the_suite_actually_uses_are_distinct() {
        // Area files draw operands from nearby primes and adjacent integers.
        for pair in [
            (11u32, 23),
            (307, 311),
            (1009, 1013),
            (2, 3),
            (4, 5),
            (0, 1),
        ] {
            assert_ne!(fill(pair.0, 8), fill(pair.1, 8), "seeds {pair:?} collide");
        }
    }

    #[test]
    fn fill_stays_in_the_documented_half_open_range() {
        for v in fill(9, 4096) {
            assert!((-0.5..0.5).contains(&v), "{v} left [-0.5, 0.5)");
        }
    }

    #[test]
    fn fill_range_maps_onto_the_requested_interval() {
        for v in fill_range(3, 512, 1.0, 4.0) {
            assert!((1.0..4.0).contains(&v), "{v} left [1, 4)");
        }
    }

    #[test]
    fn case_names_are_area_qualified() {
        let case = Case::new("elementwise", "abs", |_| Ok(()));
        assert_eq!(case.name, "elementwise::abs");
        assert_eq!(case.short(), "abs");
    }

    #[test]
    fn guard_turns_a_panic_into_a_named_failure() {
        let outcome = guard(|| panic!("todo!(\"W13: conv\")"));
        match outcome {
            Outcome::Fail(message) => assert!(message.contains("W13: conv"), "{message}"),
            other => panic!("expected a failure, got {other:?}"),
        }
    }

    #[test]
    fn guard_passes_through_ok_and_err() {
        assert_eq!(guard(|| Ok(())), Outcome::Pass);
        assert!(guard(|| Err("nope".into())).is_fail());
    }

    #[test]
    fn a_skip_is_neither_a_pass_nor_a_failure() {
        let outcome = guard(|| Err(skip("this adapter has no bf16 support")));
        match &outcome {
            Outcome::Skipped(why) => assert_eq!(why, "this adapter has no bf16 support"),
            other => panic!("expected a skip, got {other:?}"),
        }
        assert!(!outcome.is_pass());
        assert!(!outcome.is_fail());
        assert!(outcome.is_skipped());
    }

    #[test]
    fn a_panic_is_never_a_skip_however_it_is_worded() {
        let outcome = guard(|| panic!("skipped: not yet implemented"));
        assert!(outcome.is_fail(), "{outcome:?}");
    }

    #[test]
    fn the_skip_marker_round_trips_through_the_error_type() {
        let err = skip("no adapter");
        assert!(err.to_string().starts_with(SKIP_PREFIX));
        assert_eq!(
            err.to_string().strip_prefix(SKIP_PREFIX),
            Some("no adapter")
        );
    }

    #[test]
    fn summarize_counts_skips_apart_from_passes() {
        let reports = vec![
            Report {
                case: "a::x".into(),
                backend: "cpu",
                outcome: Outcome::Pass,
            },
            Report {
                case: "a::y".into(),
                backend: "gpu",
                outcome: Outcome::Skipped("no bf16".into()),
            },
            Report {
                case: "a::z".into(),
                backend: "cpu",
                outcome: Outcome::Fail("wrong".into()),
            },
        ];
        assert_eq!(summarize(&reports), 1);
        assert_eq!(reports.iter().filter(|r| r.outcome.is_pass()).count(), 1);
        assert_eq!(reports.iter().filter(|r| r.outcome.is_skipped()).count(), 1);
    }

    #[test]
    fn gpu_guard_is_reentrant_across_calls() {
        drop(gpu_test_guard());
        drop(gpu_test_guard());
    }

    #[test]
    fn dims_builds_constant_extents() {
        assert_eq!(dense_len(&dims(&[2, 3, 4])), 24);
    }
}
