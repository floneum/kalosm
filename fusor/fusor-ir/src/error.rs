//! Crate-level error type and `Result` alias.

use crate::egraph::Id;
use crate::ir::Level;
use std::fmt;

/// The one `Result` alias every fusor crate uses.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Every way a fusor compilation can fail. Flat and `Clone`: errors cross
/// thread boundaries (parallel kernel build) and are compared in goldens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A level verifier rejected a node. A rule or the frontend built
    /// something illegal; never recoverable.
    Verify {
        level: Level,
        node: Option<Id>,
        msg: String,
    },
    Shape(String),
    Dtype(String),
    /// A rewrite would have violated a `NumericContract`.
    Numeric(String),
    /// A lowering was structurally illegal on this device.
    Legality(String),
    /// Saturation/extraction hit a budget the caller asked to be told
    /// about. The default driver degrades instead of producing this.
    Budget(String),
    /// `verify_plan` rejected an extraction. A hard conformance assert.
    Plan(String),
    Lower(crate::ir::kernel::LowerError),
    Emit(crate::target::EmitError),
    Device(String),
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verify { level, node, msg } => match node {
                Some(id) => write!(f, "verify_{level}: {id}: {msg}"),
                None => write!(f, "verify_{level}: {msg}"),
            },
            Self::Shape(m) => write!(f, "shape: {m}"),
            Self::Dtype(m) => write!(f, "dtype: {m}"),
            Self::Numeric(m) => write!(f, "numeric contract: {m}"),
            Self::Legality(m) => write!(f, "lowering illegal: {m}"),
            Self::Budget(m) => write!(f, "budget: {m}"),
            Self::Plan(m) => write!(f, "plan: {m}"),
            Self::Lower(e) => write!(f, "lower: {e}"),
            Self::Emit(e) => write!(f, "emit: {e}"),
            Self::Device(m) => write!(f, "device: {m}"),
            Self::Io(m) => write!(f, "io: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<crate::ir::kernel::LowerError> for Error {
    fn from(e: crate::ir::kernel::LowerError) -> Self {
        Self::Lower(e)
    }
}

impl From<crate::target::EmitError> for Error {
    fn from(e: crate::target::EmitError) -> Self {
        Self::Emit(e)
    }
}

impl Error {
    pub fn verify(level: Level, node: Id, msg: impl Into<String>) -> Self {
        Self::Verify {
            level,
            node: Some(node),
            msg: msg.into(),
        }
    }

    pub fn verify_global(level: Level, msg: impl Into<String>) -> Self {
        Self::Verify {
            level,
            node: None,
            msg: msg.into(),
        }
    }
}
