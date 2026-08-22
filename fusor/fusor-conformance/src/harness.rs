//! The test harness: build one named case, run it on every available backend,
//! report rather than panic.
//!
//! Cases return `Err` instead of asserting because the browser conformance
//! runner cannot recover from a wasm panic. [`run_one`] wraps each case in
//! `catch_unwind` so a panicking op is one red row, not a dead run.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, MutexGuard, OnceLock};

use fusor::graph::GraphRef;
use fusor::session::Backend;
use fusor::tensor::Dyn as Tensor;
use fusor::{Dim, Dtype, Session};

/// Error returned by a conformance case.
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

/// The body of a case. `Fn` so one case can run on every session in
/// [`sessions`].
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
    std::env::var("FUSOR_CONFORMANCE_REQUIRE_GPU")
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
/// The browser memoizes the handle for the life of the page. Natively there
/// is no cache: a thread-local holding a wgpu `Device` panics when dropped
/// during thread teardown.
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
/// session returned here.
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
        // The guard exists for serialization, not data integrity, so recover
        // from poisoning.
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The LCG every golden depends on. Changing a constant here invalidates
/// every recorded output hash.
///
/// `state = state * 6364136223846793005 + 1442695040888963407`, value
/// `((state >> 33) as f32 / 2^31) - 0.5`.
///
/// The seed must enter the state unmodified: masking bits folds distinct
/// seeds onto the same stream.
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

/// How many times a fuzzed case re-samples its shapes and data. Every run of
/// one case sees a different size, so an op that is correct only at its
/// authoring shape fails by the second run. `FUSOR_CONFORMANCE_RUNS`
/// overrides.
pub fn runs() -> u32 {
    static RUNS: OnceLock<u32> = OnceLock::new();
    *RUNS.get_or_init(|| {
        std::env::var("FUSOR_CONFORMANCE_RUNS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(3)
    })
}

/// The seed a case's run draws everything from: FNV-1a over the case name
/// plus the run index. Deterministic per (case, run), so a failure report
/// naming the case and run reproduces the exact shapes and data.
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
/// Separate from the data stream so adding a dimension to a spec does not
/// shift every operand's values.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        // One warm-up step so nearby seeds separate immediately.
        let mut rng = Self { state: seed as u64 };
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
                // A skip must stay a skip: the run context goes after the
                // marker, not in front of it.
                let message = e.to_string();
                match message.strip_prefix(SKIP_PREFIX) {
                    Some(why) => skip(format!("run {run} at shape {shape:?}: {why}")),
                    None => format!("run {run} at shape {shape:?} (seed {seed}): {message}").into(),
                }
            })?;
        }
        Ok(())
    })
}

/// Element count of a fully constant shape. Panics on a symbolic extent: a
/// `Dim::Sym` here is a bug in the case.
pub fn dense_len(shape: &[Dim]) -> usize {
    shape
        .iter()
        .map(|d| match d {
            Dim::Const(n) => *n as usize,
            Dim::Sym(s) => panic!("dense_len over a symbolic extent {s:?}"),
        })
        .product()
}

/// `Dim::Const` extents from plain integers.
pub fn dims(shape: &[u64]) -> Vec<Dim> {
    shape.iter().map(|n| Dim::Const(*n)).collect()
}

/// Upload f32 host data as little-endian bytes.
pub fn from_f32(graph: &GraphRef, shape: &[Dim], data: &[f32]) -> fusor::Result<Tensor> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Tensor::from_slice(graph, Dtype::F32, shape, &bytes)
}

/// Upload u32 host data (indices).
pub fn from_u32(graph: &GraphRef, shape: &[Dim], data: &[u32]) -> fusor::Result<Tensor> {
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
/// cannot run this row" rather than "this row is wrong". [`guard`] turns a
/// prefixed error into [`Outcome::Skipped`].
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
/// A panic is never a skip, however it is worded.
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

    /// Per-dtype absolute/relative tolerance, from [`crate::compare::DTYPE_TOL`].
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
    // A skip is never folded into the pass count.
    let skipped = reports.iter().filter(|r| r.outcome.is_skipped()).count();
    let passed = reports.iter().filter(|r| r.outcome.is_pass()).count();
    println!(
        "{} results: {passed} passed, {skipped} skipped, {failures} failed",
        reports.len()
    );
    failures
}
