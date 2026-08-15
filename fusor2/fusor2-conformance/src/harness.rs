//! The test harness: build one named case, run it on every available backend,
//! report rather than panic.
//!
//! Cases return `Err` instead of asserting because the browser conformance
//! runner cannot recover from a wasm panic, and because a half-built backend
//! should surface as a named failure rather than aborting the whole matrix.
//! That is also why [`run_one`] wraps each case in `catch_unwind`: an
//! unfinished op in a dependency is one red row, not a dead run.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, MutexGuard, OnceLock};

use fusor2::graph::GraphRef;
use fusor2::session::Backend;
use fusor2::{Dim, Dtype, Session};
use fusor2::tensor::Dyn as Tensor;

// ---------------------------------------------------------------------------
// Case results and the `ensure!` family
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

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

    /// The names, in registration order. The acceptance bar is "every case in
    /// the named list is present", so this is asserted on directly.
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

/// Build one case per row of a table. The shape every area file uses.
pub fn cases_from_rows<T: Send + Sync + 'static>(
    area: &'static str,
    rows: impl IntoIterator<Item = (&'static str, T)>,
    body: impl Fn(&Session, &T) -> CaseResult + Copy + Send + Sync + 'static,
) -> Cases {
    let mut cases = Cases::new();
    for (name, row) in rows {
        cases.push(area, name, move |s| body(s, &row));
    }
    cases
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

fn require_gpu() -> bool {
    std::env::var("FUSOR2_CONFORMANCE_REQUIRE_GPU")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn acquire_gpu() -> Option<Backend> {
    match Backend::gpu_blocking() {
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
/// In the browser the full suite runs hundreds of cases that each ask for a
/// device, and acquiring a fresh wgpu device per case is prohibitively slow,
/// so the handle is memoized for the life of the page. Natively the cache is
/// skipped: a thread-local holding a wgpu `Device` panics when dropped during
/// thread teardown (wgpu's `Drop` touches already-destroyed thread-locals),
/// and re-acquiring per run is cheap enough off the web.
#[cfg(not(target_arch = "wasm32"))]
fn cached_gpu() -> Option<Backend> {
    acquire_gpu()
}

#[cfg(target_arch = "wasm32")]
fn cached_gpu() -> Option<Backend> {
    thread_local! {
        static GPU: std::cell::RefCell<Option<Option<Backend>>> =
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
    if let Ok(cpu) = Backend::cpu()
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
    matches!(session.device(), Backend::Gpu(_))
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

// ---------------------------------------------------------------------------
// Deterministic data
// ---------------------------------------------------------------------------

/// The LCG every golden depends on. Changing a single constant here
/// invalidates every recorded output hash, which is why it lives in exactly
/// one place and is never re-derived.
///
/// `state = state * 6364136223846793005 + 1442695040888963407`, value
/// `((state >> 33) as f32 / 2^31) - 0.5`.
///
/// The seed enters the state unmodified. Forcing the low bit (`seed | 1`)
/// would fold every even seed onto its odd successor, so `fill(4, ..)` and
/// `fill(5, ..)` would be the same stream — and a case table that draws its
/// two operands from consecutive seeds would silently be testing `f(x, x)`.
/// The increment is nonzero, so state 0 is not a fixed point and needs no
/// special casing.
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

/// `len` indices in `[0, modulus)`, from the same generator.
pub fn fill_indices(seed: u32, len: usize, modulus: u32) -> Vec<u32> {
    let modulus = modulus.max(1);
    fill(seed, len)
        .into_iter()
        .map(|v| (((v + 0.5) * modulus as f32) as u32) % modulus)
        .collect()
}

// ---------------------------------------------------------------------------
// Shape fuzzing
// ---------------------------------------------------------------------------

/// How many times a fuzzed case re-samples its shapes and data. Every run of
/// one case sees a different size, so an op that is correct only at its
/// authoring shape fails by the second run. `FUSOR2_CONFORMANCE_RUNS`
/// overrides, and `1` degenerates to fusor1's single-sample style.
pub fn runs() -> u32 {
    static RUNS: OnceLock<u32> = OnceLock::new();
    *RUNS.get_or_init(|| {
        std::env::var("FUSOR2_CONFORMANCE_RUNS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(3)
    })
}

/// The seed a case's run draws everything from: FNV-1a over the case name
/// plus the run index. Deterministic and distinct per (case, run), so a
/// failure report naming the case and run reproduces the exact shapes and
/// data with no state carried between cases.
pub fn case_seed(name: &str, run: u32) -> u32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes().chain(run.to_le_bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    // Fold to 32 bits without losing the high half.
    (h ^ (h >> 32)) as u32
}

/// A deterministic RNG over the same LCG as [`fill`], for shape sampling.
/// Kept separate from the data stream so adding a dimension to a spec does
/// not shift every operand's values.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        // One warm-up step so nearby seeds separate immediately.
        let mut rng = Self {
            state: seed as u64,
        };
        rng.next_bits();
        rng
    }

    fn next_bits(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state >> 33
    }

    /// A value in `[lo, hi]`, inclusive on both ends.
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        debug_assert!(lo <= hi, "empty range [{lo}, {hi}]");
        lo + self.next_bits() % (hi - lo + 1)
    }

    /// One element of a non-empty list.
    pub fn choose(&mut self, items: &[u64]) -> u64 {
        items[(self.next_bits() % items.len() as u64) as usize]
    }
}

/// One dimension of a fuzzed shape.
#[derive(Copy, Clone, Debug)]
pub enum FuzzDim {
    /// Always this extent (an op-mandated size, e.g. a quantized block).
    Fixed(u64),
    /// Any extent in `[lo, hi]`, inclusive.
    Range(u64, u64),
    /// A multiple of `step` in `[lo, hi]`, for alignment-constrained axes
    /// (quantized K must be a block multiple, attention lengths that must
    /// stay flash-aligned). `lo` and `hi` are themselves multiples.
    Mult(u64, u64, u64),
    /// One of an explicit extent list.
    Choices(&'static [u64]),
}

impl FuzzDim {
    pub fn sample(&self, rng: &mut Rng) -> u64 {
        match *self {
            FuzzDim::Fixed(n) => n,
            FuzzDim::Range(lo, hi) => rng.range(lo, hi),
            FuzzDim::Mult(step, lo, hi) => {
                debug_assert!(step > 0 && lo % step == 0 && hi % step == 0);
                rng.range(lo / step, hi / step) * step
            }
            FuzzDim::Choices(items) => rng.choose(items),
        }
    }
}

/// Sample every dimension of `spec`, in order.
pub fn sample_shape(rng: &mut Rng, spec: &[FuzzDim]) -> Vec<u64> {
    spec.iter().map(|d| d.sample(rng)).collect()
}

/// Build one fuzzed case: `body` runs [`runs`] times, each with a fresh shape
/// sampled from `spec` and a distinct data seed. A failing run names its run
/// index and the sampled shape, so any failure reproduces exactly from the
/// report line.
///
/// `body(session, shape, seed)` draws its operand data from `seed` (and any
/// derived seeds like `seed + 1`); the harness owns which seed a run gets.
pub fn fuzz_case(
    area: &'static str,
    name: &'static str,
    spec: &'static [FuzzDim],
    body: impl Fn(&Session, &[u64], u32) -> CaseResult + Send + Sync + 'static,
) -> Case {
    Case::new(area, name, move |session| {
        for run in 0..runs() {
            let seed = case_seed(name, run);
            let shape = sample_shape(&mut Rng::new(seed), spec);
            body(session, &shape, seed).map_err(|e| -> CaseError {
                // A skip must stay a skip: the marker is a prefix, so the run
                // context goes after it, not in front of it.
                let message = e.to_string();
                match message.strip_prefix(SKIP_PREFIX) {
                    Some(why) => skip(format!("run {run} at shape {shape:?}: {why}")),
                    None => {
                        format!("run {run} at shape {shape:?} (seed {seed}): {message}").into()
                    }
                }
            })?;
        }
        Ok(())
    })
}

/// Element count of a fully constant shape. Panics on a symbolic extent: the
/// host-side references need a concrete length, and a `Dim::Sym` here is a
/// bug in the case rather than in the compiler.
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

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

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
/// A case returns `Result<(), CaseError>` and has no third variant, so the
/// alternative would be for every capability-gated row to return `Ok(())` on a
/// device that never ran it — and a skipped f16 matrix would read as a passing
/// one. Prefixing instead keeps the case signature and lets [`guard`] sort the
/// two apart, so a run on a device without bf16 reports `skip`, not `ok` and
/// not `FAILED`.
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
        Backend::Cpu(_) => "cpu",
        Backend::Gpu(_) => "gpu",
    }
}

/// Run a case body, converting a panic into a failure message and a
/// [`SKIP_PREFIX`] error into [`Outcome::Skipped`].
///
/// A panic is never a skip, however it is worded: an unfinished op elsewhere
/// in the workspace must stay one red row.
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

/// Run one named case on every session.
pub fn run_case(name: &str) -> Vec<Report> {
    let registry = crate::suite::registry();
    let sessions = sessions();
    let mut out = Vec::new();
    for case in registry.iter().filter(|c| c.name == name) {
        for session in &sessions {
            let _guard = is_gpu(session).then(gpu_test_guard);
            out.push(Report {
                case: case.name.clone(),
                backend: backend_name(session),
                outcome: run_one(case, session),
            });
        }
    }
    out
}

/// Run the whole registry on every session, reporting progress as it goes.
pub fn run_all(mut progress: impl FnMut(&Report)) -> Vec<Report> {
    let registry = crate::suite::registry();
    let sessions = sessions();
    let mut out = Vec::with_capacity(registry.len() * sessions.len().max(1));
    if sessions.is_empty() {
        for case in registry.iter() {
            let report = Report {
                case: case.name.clone(),
                backend: "none",
                outcome: Outcome::Skipped("no device available".to_string()),
            };
            progress(&report);
            out.push(report);
        }
        return out;
    }
    for case in registry.iter() {
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

/// Runs cases against one or both backends, optionally filtered by a
/// substring of the case name.
pub struct Harness {
    pub filter: Option<String>,
}

impl Harness {
    pub fn new() -> Self {
        Self { filter: None }
    }

    pub fn with_filter(filter: impl Into<String>) -> Self {
        Self {
            filter: Some(filter.into()),
        }
    }

    fn matches(&self, name: &str) -> bool {
        self.filter.as_ref().is_none_or(|f| name.contains(f))
    }

    /// Every matching case on every available backend.
    pub fn run(&self) -> Vec<Report> {
        let registry = crate::suite::registry();
        let sessions = sessions();
        let mut out = Vec::new();
        for case in registry.iter().filter(|c| self.matches(&c.name)) {
            for session in &sessions {
                let _guard = is_gpu(session).then(gpu_test_guard);
                out.push(Report {
                    case: case.name.clone(),
                    backend: backend_name(session),
                    outcome: run_one(case, session),
                });
            }
        }
        out
    }

    /// Per-dtype absolute/relative tolerance. Delegates to the one table in
    /// [`crate::compare::DTYPE_TOL`] so nothing carries a second copy.
    pub fn tolerance(dtype: Dtype) -> (f32, f32) {
        crate::compare::tol_for(dtype)
    }
}

impl Default for Harness {
    fn default() -> Self {
        Self::new()
    }
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
    // A skip is reported separately, never folded into the pass count: a
    // suite that skipped the whole f16 matrix must not read as one that ran
    // it.
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
        // Hand-evaluated from the documented recurrence: seed 1 forces
        // state = 1, so the first value is ((s0 >> 33) / 2^31) - 0.5 with
        // s0 = 1 * 6364136223846793005 + 1442695040888963407.
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
        // The regression this guards: `seed | 1` folded every even seed onto
        // its odd successor, so `fill(4, ..)` and `fill(5, ..)` were the same
        // numbers and a case table drawing two operands from consecutive
        // seeds was silently testing `f(x, x)`.
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
        // Area files draw operands from nearby primes and from adjacent
        // small integers; both must separate.
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
    fn fill_indices_stay_in_bounds() {
        for i in fill_indices(3, 512, 7) {
            assert!(i < 7);
        }
    }

    #[test]
    fn case_names_are_area_qualified() {
        let case = Case::new("elementwise", "abs", |_| Ok(()));
        assert_eq!(case.name, "elementwise::abs");
        assert_eq!(case.short(), "abs");
    }

    #[test]
    fn cases_from_rows_names_every_row() {
        let cases = cases_from_rows("demo", [("a", 1u32), ("b", 2u32)], |_, _| Ok(()));
        assert_eq!(cases.names(), vec!["demo::a", "demo::b"]);
    }

    #[test]
    fn guard_turns_a_panic_into_a_named_failure() {
        // The whole reason `guard` exists: an unfinished op elsewhere in the
        // workspace must be one red row, not a dead run.
        let outcome = guard(|| panic!("todo!(\"conv\")"));
        match outcome {
            Outcome::Fail(message) => assert!(message.contains("conv"), "{message}"),
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
        // The whole point: a device that cannot run a row must not report it
        // as `ok`, and must not report it as broken either.
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
        // An unfinished op must stay one red row even if its message happens
        // to start with the marker.
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

    #[test]
    fn fuzz_dims_sample_inside_their_own_domains() {
        let mut rng = Rng::new(7);
        for _ in 0..256 {
            assert_eq!(FuzzDim::Fixed(5).sample(&mut rng), 5);
            let r = FuzzDim::Range(2, 9).sample(&mut rng);
            assert!((2..=9).contains(&r), "{r} left [2, 9]");
            let m = FuzzDim::Mult(32, 32, 256).sample(&mut rng);
            assert!(m % 32 == 0 && (32..=256).contains(&m), "{m}");
            let c = FuzzDim::Choices(&[1, 8, 64]).sample(&mut rng);
            assert!([1, 8, 64].contains(&c), "{c}");
        }
    }

    #[test]
    fn a_range_actually_varies() {
        let mut rng = Rng::new(3);
        let samples: std::collections::HashSet<u64> =
            (0..64).map(|_| FuzzDim::Range(1, 16).sample(&mut rng)).collect();
        assert!(samples.len() > 4, "only {} distinct extents in 64 draws", samples.len());
    }

    #[test]
    fn case_seeds_separate_by_name_and_run() {
        assert_ne!(case_seed("a::x", 0), case_seed("a::x", 1));
        assert_ne!(case_seed("a::x", 0), case_seed("a::y", 0));
        assert_eq!(case_seed("a::x", 2), case_seed("a::x", 2));
    }

    #[test]
    fn a_failing_fuzz_run_names_its_run_and_shape() {
        // The whole point of deriving everything from (name, run): the report
        // line alone reproduces the failure.
        let case = fuzz_case("demo", "boom", &[FuzzDim::Range(1, 4)], |_, shape, _| {
            Err(format!("bad at {shape:?}").into())
        });
        let sessions = sessions();
        let Some(session) = sessions.first() else {
            return;
        };
        match run_one(&case, session) {
            Outcome::Fail(message) => {
                assert!(message.contains("run 0 at shape ["), "{message}");
                assert!(message.contains("seed "), "{message}");
            }
            other => panic!("expected a failure, got {other:?}"),
        }
    }

    #[test]
    fn a_skip_survives_the_fuzz_wrapper() {
        // A device that cannot run a row must stay a skip when the row is
        // fuzzed: burying the marker under the run context turned the whole
        // bf16 matrix into failures.
        let case = fuzz_case("demo", "no_bf16", &[FuzzDim::Fixed(2)], |_, _, _| {
            Err(skip("this adapter has no bf16 support"))
        });
        let sessions = sessions();
        let Some(session) = sessions.first() else {
            return;
        };
        match run_one(&case, session) {
            Outcome::Skipped(why) => assert!(why.contains("no bf16 support"), "{why}"),
            other => panic!("expected a skip, got {other:?}"),
        }
    }

    #[test]
    fn a_passing_fuzz_case_runs_every_run() {
        let hits = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let case = {
            let hits = hits.clone();
            fuzz_case("demo", "count", &[FuzzDim::Fixed(2)], move |_, _, _| {
                hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            })
        };
        let sessions = sessions();
        let Some(session) = sessions.first() else {
            return;
        };
        assert_eq!(run_one(&case, session), Outcome::Pass);
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), runs());
    }
}
