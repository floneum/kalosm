//! The backend lowering trait and the runtime handles it deals in.

use crate::cost::DeviceFacts;
use crate::device::Caps;
use crate::egraph::{Id, Rule};
use crate::error::Result;
use crate::extract::{Launch, Plan};
use crate::ir::Node;
use crate::ir::level1::SchedPoint;
use crate::ir::level2::KernelIr;
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
/// stays object-safe and the pooled allocator can test reuse with
/// `strong_count == 1`.
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
}

impl fmt::Debug for Buf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Buf(0x{:x})", self.addr())
    }
}

/// The contents of binding 0. **Always a storage buffer**, holding
/// `[u32 symbolic dims..., f32 uniform scalars...]`. That one buffer is what
/// keeps host scalars out of the kernel identity: `m * lr_f32` produces a
/// `Uniform`, not a baked literal, and a sequence length is a `Sym` read
/// from binding 0. A uniform-address-space block would break the
/// derived-bind-group mechanism, which walks storage globals.
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

/// What a target needs to lower one selected node into L2.
pub struct LowerCtx<'a> {
    pub plan: &'a Plan,
    pub launch: &'a Launch,
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

/// A compute backend. **Object-safe**: the session holds
/// `Arc<dyn Target>`. A third backend is ~1,500 lines (facts, caps,
/// emitter, launcher) and inherits every L0 rule, autograd, the cost model
/// and the plan cache for free.
pub trait Target: Send + Sync {
    /// Stable name; keys `OpDef::lower_per_target` and the calibration
    /// cache.
    fn name(&self) -> &'static str;

    /// What this device can do. Legality only.
    fn caps(&self) -> &Caps;

    /// Calibrated rates. Everything the cost model reads.
    fn facts(&self) -> &DeviceFacts;

    /// Target-exclusive lowering rules (lane/subgroup geometry). Every L0
    /// rule is inherited.
    fn rules(&self) -> &'static [Rule];

    /// Lower one selected L1 node at one schedule point into L2.
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

/// `L1::Ext` lowering: the one escape hatch out of the closed `L0`/`L1` enums,
/// shared by every target and keyed by the target's name.
pub mod ext {
    use super::*;
    use crate::error::Error;
    use crate::ir::{OpDefId, OpDefRegistry};
    use std::sync::RwLock;

    /// The registry `L1::Ext` lowering resolves `OpDefId` against.
    ///
    /// [`LowerCtx`] carries the plan, the launch, the graph and the symbol
    /// list — not the [`OpDefRegistry`] the graph was built with — and
    /// `Semantics`, which *does* hold one, exposes no accessor for it. So a
    /// target handed one selected `L1::Ext { def }` reaches that def's
    /// `lower_per_target` row only through this registry: the embedder installs
    /// here the same registry it installed on the e-graph's semantics.
    /// Registration order is id order and must match, which is the same
    /// contract `CoreSemantics::with_registry` imposes.
    static DEFS: RwLock<Option<OpDefRegistry>> = RwLock::new(None);

    /// Install the extension registry this process lowers against. Idempotent
    /// and last-write-wins; a second install with a differently ordered
    /// registry would silently rename every `OpDefId`, so callers pass the
    /// registry the graph was built with, unchanged.
    pub fn install(registry: OpDefRegistry) {
        *DEFS.write().expect("the OpDef registry lock is poisoned") = Some(registry);
    }

    /// The installed registry, if the embedder installed one.
    pub fn installed() -> Option<OpDefRegistry> {
        DEFS.read()
            .expect("the OpDef registry lock is poisoned")
            .clone()
    }

    /// Lower one registered extension op through its `target` row.
    pub fn lower(
        target: &'static str,
        def: OpDefId,
        node: &Node,
        theta: SchedPoint,
    ) -> Result<KernelIr> {
        let registry = installed().ok_or_else(|| {
            Error::Plan(format!(
                "{def:?} is an extension op, but no OpDefRegistry is installed on the \
                 \"{target}\" target; call fusor2_ir::target::ext::install"
            ))
        })?;
        let entry = registry
            .get(def)
            .ok_or_else(|| Error::Plan(format!("no OpDef is registered as {def:?}")))?;
        let lower = entry
            .lower_per_target
            .iter()
            .find(|(t, _)| *t == target)
            .map(|(_, f)| *f)
            .ok_or_else(|| {
                Error::Plan(format!(
                    "OpDef \"{}\" declares no \"{target}\" lowering; its \
                     lower_per_target names {:?}",
                    entry.name,
                    entry
                        .lower_per_target
                        .iter()
                        .map(|(t, _)| *t)
                        .collect::<Vec<_>>()
                ))
            })?;
        lower(node, &theta)
    }
}
