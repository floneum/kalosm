//! The const-rank facade's device handle.
//!
//! [`crate::session::Device`] is the backend selector, an enum over
//! `CpuTarget`/`GpuTarget` that a [`Session`] is built from. This `Device` is a
//! backend plus the session and the graph that values built from it live in, so
//! `Tensor::from_slice(device, [n], xs)` needs no graph argument.
//!
//! Every fusor2 value is a node in one e-graph and an op across two graphs is
//! an error. Constructors that take only a `&Device` build into the graph the
//! most recent device installed; creating a second device replaces it, so
//! mixing two devices in one process is unsupported here. The runtime-rank API,
//! where every constructor names its graph, supports it.
//!
//! Nodes accumulate in that graph for the process lifetime; nothing here
//! collects them.

use std::sync::{Arc, Mutex, OnceLock};

use crate::graph::{Graph, GraphRef};
use crate::session::{Device as Backend, Session};
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
    Cpu(Cpu),
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
    /// Re-installs the ambient graph on every call, not just the first:
    /// `CPU.get_or_init` builds the device once, so a `Device::cpu()` after a
    /// `Device::gpu_blocking()` would otherwise hand back the CPU device while
    /// leaving the GPU's graph ambient.
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

    /// The GPU backend, blocking on adapter acquisition. `Err` when there is no
    /// usable adapter.
    pub fn gpu_blocking() -> Result<Self> {
        Ok(Self::Gpu(Gpu(Inner::new(Backend::gpu_blocking()?)?)))
    }

    /// The GPU backend, awaiting adapter acquisition.
    pub async fn gpu() -> Result<Self> {
        Ok(Self::Gpu(Gpu(Inner::new(Backend::gpu().await?)?)))
    }

    fn inner(&self) -> &Arc<Inner> {
        match self {
            Self::Cpu(d) => &d.0,
            Self::Gpu(d) => &d.0,
        }
    }

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

    /// Submit whatever is queued, reporting failure.
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
    pub name: String,
    pub count: usize,
    pub total_ms: f64,
    pub average_us: f64,
    pub max_us: f64,
}

/// One resolve's timing.
///
/// `span_ms` is `None` when the backend could not time the submission, which is
/// distinct from a zero-length span.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelProfile {
    pub span_ms: Option<f64>,
    pub kernels: usize,
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

/// Serializes tests that assert on shared device state (the ambient graph,
/// launch counts). Correctness of concurrent resolves is
/// `GraphInner::resolve_lock`'s job. Poisoning is ignored, so a
/// `#[should_panic]` test holding this lock does not fail every test after it.
#[cfg(test)]
pub(crate) fn test_device_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cpu_device_is_one_per_process_and_owns_one_graph() {
        let _serial = crate::device::test_device_lock();
        let a = Device::cpu();
        let b = Device::cpu();
        assert!(Arc::ptr_eq(a.handle(), b.handle()));
        assert_eq!(a.name(), "cpu");
    }

    #[test]
    fn a_device_installs_the_ambient_graph() {
        let _serial = crate::device::test_device_lock();
        let d = Device::cpu();
        // Another test may have installed a GPU device since, so this asserts
        // only that a graph is installed and this device's handle is live.
        let _ = ambient_graph();
        assert!(Arc::strong_count(d.handle()) >= 1);
    }

    /// `Device::cpu()` installs its graph on every call, not only the first.
    /// Installing any other graph and calling `cpu()` again brings the CPU
    /// device's graph back.
    #[test]
    fn a_second_cpu_call_reinstalls_its_own_graph() {
        let _serial = crate::device::test_device_lock();
        let cpu = Device::cpu();
        assert!(Arc::ptr_eq(
            ambient_graph().handle(),
            cpu.graph().handle()
        ));

        // Stand in for another device having been created since.
        let other = Graph::new(cpu.session());
        install(&other);
        assert!(Arc::ptr_eq(ambient_graph().handle(), other.handle()));

        let cpu_again = Device::cpu();
        assert!(
            Arc::ptr_eq(ambient_graph().handle(), cpu_again.graph().handle()),
            "a cached Device::cpu() must still install its own graph"
        );
    }

    /// The const-rank API shares one `Session` and one e-graph process-wide, so
    /// two threads computing on it resolve one graph. Each thread uses a
    /// distinct constant, so a result belonging to another thread fails as well
    /// as an unwritten buffer.
    #[test]
    fn the_shared_device_survives_concurrent_resolves() {
        let _serial = crate::device::test_device_lock();
        const THREADS: u32 = 8;
        const ROUNDS: u32 = 40;
        let device = Device::cpu();

        let bad: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let device = device.clone();
                    s.spawn(move || {
                        let mut wrong = Vec::new();
                        for r in 0..ROUNDS {
                            let k = (t * ROUNDS + r) as f32 + 1.0;
                            let x = crate::root::typed::Tensor::<1, f32>::from_slice(
                                &device,
                                [4],
                                &[k, k + 1.0, k + 2.0, k + 3.0],
                            );
                            let got = x.mul_scalar(2.0).to_flat();
                            let want: Vec<f32> =
                                (0..4).map(|i| (k + i as f32) * 2.0).collect();
                            if got != want {
                                wrong.push(format!("t{t} r{r}: got {got:?} want {want:?}"));
                            }
                        }
                        wrong
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });

        assert!(
            bad.is_empty(),
            "{} of {} concurrent readbacks were wrong:\n{}",
            bad.len(),
            THREADS * ROUNDS,
            bad.join("\n")
        );
    }
}
