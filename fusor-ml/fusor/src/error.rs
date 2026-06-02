//! Error types for the fusor crate

/// Error type for fusor operations.
#[derive(Debug)]
pub enum Error {
    /// GPU device error from fusor-core.
    Gpu(crate::gpu::Error),
    /// Device mismatch error - tensors are on different devices.
    DeviceMismatch {
        /// Description of what operation failed.
        operation: &'static str,
    },
    /// VarBuilder error (key not found, IO error, etc.)
    VarBuilder(String),
    /// Generic error message.
    Other(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Gpu(e) => write!(f, "GPU error: {}", e),
            Error::DeviceMismatch { operation } => {
                write!(
                    f,
                    "Device mismatch in {}: tensors must be on the same device",
                    operation
                )
            }
            Error::VarBuilder(msg) => write!(f, "VarBuilder error: {}", msg),
            Error::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl Error {
    /// Create a generic error message.
    pub fn msg<S: Into<String>>(s: S) -> Self {
        Error::Other(s.into())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Gpu(e) => Some(e),
            Error::DeviceMismatch { .. } => None,
            Error::VarBuilder(_) => None,
            Error::Other(_) => None,
        }
    }
}

impl From<crate::gpu::Error> for Error {
    fn from(e: crate::gpu::Error) -> Self {
        Error::Gpu(e)
    }
}

impl From<crate::gpu::GgufReadError> for Error {
    fn from(e: crate::gpu::GgufReadError) -> Self {
        // Convert GGUF errors to our error type
        Error::VarBuilder(e.to_string())
    }
}
