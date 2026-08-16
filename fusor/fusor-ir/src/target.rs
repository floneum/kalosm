//! The backend lowering trait and the runtime handles it deals in.

use crate::cost::DeviceFacts;
use crate::device::Caps;
use crate::egraph::{Id, Rule};
use crate::error::Result;
use crate::extract::{Dispatch, Plan};
use crate::ir::Node;
use crate::ir::kernel::KernelIr;
use crate::ir::launch::SchedPoint;
use crate::shape::SymId;
use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// A backend-owned compiled artifact (shader module + pipeline, or a
/// specialized CPU loop nest). Opaque above the backend.
#[derive(Clone)]
pub struct Artifact(Arc<dyn Any + Send + Sync>);

impl Artifact {
    pub fn new<T: Any + Send + Sync>(inner: T) -> Self {
        Self(Arc::new(inner))
    }
    pub fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }
}

impl fmt::Debug for Artifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Artifact(..)")
    }
}

/// A backend-owned buffer handle. Opaque and `Arc`-shared so [`Target`]
/// stays object-safe and the pooled allocator's `strong_count == 1` reuse
/// test still works.
#[derive(Clone)]
pub struct Buf(Arc<dyn Any + Send + Sync>);

impl Buf {
    pub fn new<T: Any + Send + Sync>(inner: T) -> Self {
        Self(Arc::new(inner))
    }
    pub fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }
    /// Pointer identity, for the aliasing-pattern check a replayed plan
    /// requires.
    pub fn addr(&self) -> usize {
        Arc::as_ptr(&self.0) as *const () as usize
    }
    /// `1` means the pool may recycle it.
    pub fn refcount(&self) -> usize {
        Arc::strong_count(&self.0)
    }
    /// A non-owning handle. Holding one does **not** hold the buffer: the
    /// pool's `strong_count == 1` reuse test still sees the buffer as free,
    /// which is what lets a cache key a derived object on [`Self::addr`]
    /// without pinning the allocation.
    pub fn downgrade(&self) -> WeakBuf {
        WeakBuf(Arc::downgrade(&self.0))
    }
}

/// A non-owning [`Buf`] handle.
///
/// Its purpose is address disambiguation: a live `WeakBuf` keeps the
/// allocation from being reused, so a `Buf` whose [`Buf::addr`] equals the one
/// this was taken from *is* that buffer. Without it an address is only valid
/// while the buffer lives, and a freed-then-reallocated `Buf` can land on it.
#[derive(Clone)]
pub struct WeakBuf(std::sync::Weak<dyn Any + Send + Sync>);

impl WeakBuf {
    pub fn alive(&self) -> bool {
        self.0.strong_count() > 0
    }
}

impl fmt::Debug for WeakBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WeakBuf(..)")
    }
}

impl fmt::Debug for Buf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Buf(0x{:x})", self.addr())
    }
}

/// The contents of binding 0. Always a storage buffer, holding
/// `[u32 symbolic dims..., f32 uniform scalars...]`. A uniform-address-space
/// block would break the derived-bind-group mechanism, which walks storage
/// globals.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Uniforms {
    pub dims: Vec<u32>,
    pub scalars: Vec<f32>,
}

impl Uniforms {
    /// Pack into the byte layout binding 0 expects.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 * (self.dims.len() + self.scalars.len()));
        for d in &self.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        for s in &self.scalars {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }
}

/// Why a backend refused a kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmitError {
    Unsupported(String),
    MissingCapability(&'static str),
    Validation(String),
    LimitExceeded(String),
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
            Self::MissingCapability(c) => write!(f, "missing capability {c}"),
            Self::Validation(m) => write!(f, "validation failed: {m}"),
            Self::LimitExceeded(m) => write!(f, "limit exceeded: {m}"),
        }
    }
}
impl std::error::Error for EmitError {}

/// What a target needs to lower one selected node into Kernel.
pub struct LowerCtx<'a> {
    pub plan: &'a Plan,
    pub launch: &'a Dispatch,
    pub graph: &'a crate::egraph::EGraph,
    pub symbols: &'a [SymId],
}

impl LowerCtx<'_> {
    /// The class member the extraction selected for `id`.
    ///
    /// An `Operand::src` names the node the *rule author* wrote. That is a
    /// member of the operand's e-class, but generally not the member the
    /// extractor selected, and the plan names buffers and bindings by the
    /// selected member. Every buffer lookup has to go through here or it
    /// looks its key up in a map the key was never inserted into.
    pub fn selected(&self, id: crate::egraph::Id) -> crate::egraph::Id {
        let class = self.graph.class_of(id);
        self.plan.extraction.selected(class).unwrap_or(id)
    }
}

/// A compute backend. Object-safe: the session holds `Arc<dyn Target>`.
pub trait Target: Send + Sync {
    /// Stable name; keys `OpDef::lower_per_target` and the calibration
    /// cache.
    fn name(&self) -> &'static str;

    /// What this device can do. Legality only.
    fn caps(&self) -> &Caps;

    /// Calibrated rates. Everything the cost model reads.
    fn facts(&self) -> &DeviceFacts;

    /// Target-exclusive lowering rules (lane/subgroup geometry). Every Logical
    /// rule is inherited.
    fn rules(&self) -> &'static [Rule];

    /// Lower one selected Launch node at one schedule point into Kernel.
    fn lower(&self, node: &Node, id: Id, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr>;

    fn emit(&self, ir: &KernelIr) -> std::result::Result<Artifact, EmitError>;

    /// Run one dispatch. `binds` is positional against the emitted module's
    /// storage globals sorted by binding, so binding order and codegen
    /// cannot drift.
    fn launch(
        &self,
        artifact: &Artifact,
        grid: [u32; 3],
        binds: &[Buf],
        uniforms: &Uniforms,
    ) -> Result<()>;

    fn alloc(&self, bytes: u64, persistence: crate::dtype::Persistence) -> Result<Buf>;

    /// Block until every submitted dispatch has retired. The only host
    /// syncs are this, explicit readback, and the allocator's cap retry.
    fn wait(&self) -> Result<()>;
}
