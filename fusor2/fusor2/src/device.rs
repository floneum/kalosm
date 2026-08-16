//! The const-rank facade's device handle.
//!
//! This `Device` is a backend plus the session and the graph that values
//! built from it live in, so `Tensor::from_slice(device, [n], xs)` needs no
//! graph argument.
//!
//! # The ambient graph
//!
//! Every fusor2 value is a node in one e-graph, and an op across two graphs is
//! an error. Constructors that take only a `&Device` (or nothing) build into
//! the graph the most recently created device installed. Creating a second
//! device replaces it, so mixing two devices in one process is not supported
//! here.
//!
//! Nodes accumulate in that graph for the process lifetime; nothing here
//! collects them.

use std::sync::{Arc, Mutex, OnceLock};

use crate::graph::{Graph, GraphRef};
use crate::session::{Backend, Session};
use crate::Result;

/// One backend, its session and its graph.
struct Inner {
    session: Session,
    graph: Graph,
}

/// A GPU device. Held by [`Device::Gpu`].
#[derive(Clone)]
pub struct Gpu(Arc<Inner>);

/// A CPU device. Held by [`Device::Cpu`].
#[derive(Clone)]
pub struct Cpu(Arc<Inner>);

/// Which backend the const-rank API is building for.
#[derive(Clone)]
pub enum Device {
    /// A CPU-backed graph and session.
    Cpu(Cpu),
    /// A GPU-backed graph and session.
    Gpu(Gpu),
}

/// The graph const-rank constructors build into. Installed by the most recent
/// successful device creation.
static AMBIENT: OnceLock<Mutex<Option<Graph>>> = OnceLock::new();
/// One CPU device per process, so two `Device::cpu()` calls agree on a graph.
static CPU: OnceLock<Device> = OnceLock::new();

fn ambient() -> &'static Mutex<Option<Graph>> {
    AMBIENT.get_or_init(|| Mutex::new(None))
}

fn install(graph: &Graph) {
    *ambient().lock().expect("ambient graph lock") = Some(graph.clone());
}

/// The graph a `&Device`-only constructor builds into.
///
/// # Panics
/// If no device has been created yet.
pub(crate) fn ambient_graph() -> Graph {
    ambient()
        .lock()
        .expect("ambient graph lock")
        .clone()
        .expect("no fusor2 Device has been created yet; make one before a Graph or a Tensor")
}

impl Inner {
    fn new(backend: Backend) -> Result<Arc<Self>> {
        let session = Session::new(backend)?;
        let graph = Graph::new(&session);
        install(&graph);
        Ok(Arc::new(Self { session, graph }))
    }
}

impl Device {
    /// The CPU backend. One per process.
    ///
    /// Re-installs the ambient graph on every call, not just the first: a
    /// `Device::cpu()` after a `Device::gpu_blocking()` must make the CPU
    /// graph ambient again, or `Graph::new()` and `Tensor::from_slice(&cpu, ..)`
    /// would name different graphs.
    ///
    /// # Panics
    /// If the CPU target cannot be built. [`Device::try_cpu`] is the checked
    /// spelling.
    pub fn cpu() -> Self {
        let device = CPU
            .get_or_init(|| Self::try_cpu().expect("cpu device"))
            .clone();
        install(device.graph());
        device
    }

    /// [`Device::cpu`], reporting the failure instead of panicking.
    pub fn try_cpu() -> Result<Self> {
        Ok(Self::Cpu(Cpu(Inner::new(Backend::cpu()?)?)))
    }

    /// The GPU backend, blocking on adapter acquisition. `Err` when there is
    /// no usable adapter — the caller is expected to fall back.
    pub fn gpu_blocking() -> Result<Self> {
        Ok(Self::Gpu(Gpu(Inner::new(Backend::gpu_blocking()?)?)))
    }

    /// The GPU backend, awaiting adapter acquisition.
    pub async fn gpu() -> Result<Self> {
        Ok(Self::Gpu(Gpu(Inner::new(Backend::gpu().await?)?)))
    }

    /// The GPU if one is usable, otherwise the CPU.
    pub fn auto() -> Self {
        Self::gpu_blocking().unwrap_or_else(|_| Self::cpu())
    }

    /// The device a value in `graph` was built from. Wraps the same session
    /// and graph handle, so it is the same device by every observable.
    pub(crate) fn of_graph(graph: &GraphRef) -> Self {
        let inner = Arc::new(Inner {
            session: graph.session().clone(),
            graph: Graph::from_handle(graph.clone()),
        });
        match inner.session.device() {
            Backend::Cpu(_) => Self::Cpu(Cpu(inner)),
            Backend::Gpu(_) => Self::Gpu(Gpu(inner)),
        }
    }

    fn inner(&self) -> &Arc<Inner> {
        match self {
            Self::Cpu(d) => &d.0,
            Self::Gpu(d) => &d.0,
        }
    }

    /// The session that resolves this device's graph.
    pub fn session(&self) -> &Session {
        &self.inner().session
    }

    /// The graph values built from this device live in.
    pub fn graph(&self) -> &Graph {
        &self.inner().graph
    }

    pub(crate) fn handle(&self) -> &GraphRef {
        self.inner().graph.handle()
    }

    /// The backend name, either `"cpu"` or `"gpu"`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            Self::Gpu(_) => "gpu",
        }
    }

    /// The backend selector this device was built from.
    pub fn backend(&self) -> &Backend {
        self.session().device()
    }

    /// Submit whatever is queued. Errors are reported, not swallowed.
    pub fn try_flush(&self) -> Result<()> {
        self.session().flush()
    }

    /// Submit whatever is queued.
    ///
    /// # Panics
    /// If submission fails, which means the device is gone.
    pub fn flush(&self) {
        self.try_flush().expect("device flush");
    }

    /// Block until every submitted plan has retired.
    ///
    /// # Panics
    /// If the wait fails, which means the device is gone.
    pub fn wait(&self) {
        self.session().wait().expect("device wait");
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Device({})", self.name())
    }
}

impl Gpu {
    /// The GPU session.
    pub fn session(&self) -> &Session {
        &self.0.session
    }

    /// Block until every submitted plan has retired.
    ///
    /// # Panics
    /// If the wait fails, which means the device is gone.
    pub fn poll_wait(&self) {
        self.0.session.wait().expect("gpu wait");
    }

    /// Drain the per-resolve kernel timings collected since the last call.
    /// Empty unless the backend was built with profiling on.
    pub fn take_kernel_profiles(&self) -> Vec<KernelProfile> {
        let Backend::Gpu(target) = self.0.session.device() else {
            return Vec::new();
        };
        target
            .take_kernel_profiles()
            .into_iter()
            .map(KernelProfile::from_backend)
            .collect()
    }
}

impl Cpu {
    /// The CPU session.
    pub fn session(&self) -> &Session {
        &self.0.session
    }

    /// Block until every submitted plan has retired.
    pub fn poll_wait(&self) {
        self.0.session.wait().expect("cpu wait");
    }
}

/// One kernel's aggregated timing across a resolve.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelProfileRow {
    /// Kernel label.
    pub name: String,
    /// Number of dispatches with this label.
    pub count: usize,
    /// Total device time in milliseconds.
    pub total_ms: f64,
    /// Average dispatch time in microseconds.
    pub average_us: f64,
    /// Slowest dispatch time in microseconds.
    pub max_us: f64,
}

/// One resolve's timing.
///
/// `span_ms` is `None` when the backend could not time the submission — a
/// profile with no wall clock is not a zero-length one.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelProfile {
    /// Total timed submission span in milliseconds, when available.
    pub span_ms: Option<f64>,
    /// Number of kernel dispatches in the resolve.
    pub kernels: usize,
    /// The slowest kernel labels by aggregate time.
    pub top_names: Vec<KernelProfileRow>,
}

impl KernelProfile {
    fn from_backend(p: fusor2_gpu::launch::KernelProfile) -> Self {
        Self {
            span_ms: (p.span_ms > 0.0).then_some(p.span_ms),
            kernels: p.kernels,
            top_names: p
                .top_names
                .into_iter()
                .map(|r| KernelProfileRow {
                    name: r.name,
                    count: r.count as usize,
                    total_ms: r.total_ms,
                    average_us: r.average_us,
                    max_us: r.max_us,
                })
                .collect(),
        }
    }
}

/// Turn a `Result` from a fallible builder into a panic naming the op that
/// produced it. The const-rank API is panic-on-error throughout.
#[track_caller]
pub(crate) fn ok<T>(what: &str, r: Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("fusor2 {what}: {e}"),
    }
}
