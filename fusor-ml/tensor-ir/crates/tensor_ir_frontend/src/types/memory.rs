//! Memory tier, buffer reference, and GPU thread-index level enums.

use std::fmt;
use std::str::FromStr;

/// Stable identifier for a tensor buffer in lowered program IR.
///
/// External inputs use their existing user-facing ids through
/// [`BufferRef::External`]. Computed tensors use deterministic ids assigned
/// from the frontend expression that produces them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(pub u32);

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

impl FromStr for TensorId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let rest = s
            .strip_prefix('t')
            .ok_or_else(|| format!("bad tensor id: {s}"))?;
        let id: u32 = rest
            .parse()
            .map_err(|error| format!("bad tensor id: {error}"))?;
        Ok(Self(id))
    }
}

/// Threadgroup tile reference. The `source` identifies the device-side buffer
/// being staged and `region` disambiguates multiple tiles of the same source
/// inside one dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TgRef {
    pub source: BufferRef,
    pub region: u32,
}

/// Typed reference to a kernel-level buffer slot.
///
/// Uses positional indexing for inputs and outputs. The threadgroup-staging
/// variant is encoded by `MemTier::Threadgroup(BufferRef)` rather than by
/// name conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BufferRef {
    /// External program input at slot `index`.
    External(u32),
    /// Program tensor buffer produced by a lowered tensor expression.
    Tensor(TensorId),
    /// Kernel input at slot `index`.
    Input(u32),
    /// Kernel output at slot `index`.
    Output(u32),
}

impl BufferRef {
    /// True if this ref names an input slot.
    #[must_use]
    pub const fn is_input(self) -> bool {
        matches!(self, Self::Input(_) | Self::External(_))
    }

    /// Slot index of an input, or None for outputs.
    #[must_use]
    pub const fn input_index(self) -> Option<u32> {
        match self {
            Self::Input(i) | Self::External(i) => Some(i),
            Self::Tensor(_) | Self::Output(_) => None,
        }
    }
}

impl fmt::Display for BufferRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::External(i) => write!(f, "ext:{i}"),
            Self::Tensor(id) => write!(f, "tensor:{id}"),
            Self::Input(i) => write!(f, "in:{i}"),
            Self::Output(i) => write!(f, "out:{i}"),
        }
    }
}

impl FromStr for BufferRef {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        if let Some(rest) = s.strip_prefix("ext:") {
            let idx: u32 = rest
                .parse()
                .map_err(|e| format!("bad external slot: {e}"))?;
            Ok(Self::External(idx))
        } else if let Some(rest) = s.strip_prefix("tensor:") {
            Ok(Self::Tensor(rest.parse()?))
        } else if let Some(rest) = s.strip_prefix("in:") {
            let idx: u32 = rest.parse().map_err(|e| format!("bad input slot: {e}"))?;
            Ok(Self::Input(idx))
        } else if let Some(rest) = s.strip_prefix("out:") {
            let idx: u32 = rest.parse().map_err(|e| format!("bad output slot: {e}"))?;
            Ok(Self::Output(idx))
        } else {
            Err(format!("bad buffer ref: {s}"))
        }
    }
}

/// Memory tier for Load/Store in low-level IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemTier {
    Device(BufferRef),
    Threadgroup(BufferRef),
}

impl MemTier {
    #[must_use]
    pub const fn buffer(&self) -> BufferRef {
        match self {
            Self::Device(b) | Self::Threadgroup(b) => *b,
        }
    }

    #[must_use]
    pub const fn is_device(&self) -> bool {
        matches!(self, Self::Device(_))
    }

    #[must_use]
    pub const fn is_threadgroup(&self) -> bool {
        matches!(self, Self::Threadgroup(_))
    }

    /// Lift this tier to its threadgroup-staging variant, keeping the same
    /// underlying `BufferRef`. Idempotent.
    #[must_use]
    pub const fn to_threadgroup(&self) -> Self {
        match self {
            Self::Device(b) => Self::Threadgroup(*b),
            Self::Threadgroup(_) => *self,
        }
    }
}

impl fmt::Display for MemTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(b) => write!(f, "dev:{b}"),
            Self::Threadgroup(b) => write!(f, "tg:{b}"),
        }
    }
}

impl FromStr for MemTier {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        if let Some(rest) = s.strip_prefix("dev:") {
            Ok(Self::Device(rest.parse()?))
        } else if let Some(rest) = s.strip_prefix("tg:") {
            Ok(Self::Threadgroup(rest.parse()?))
        } else {
            Err(format!("bad mem tier: {s}"))
        }
    }
}

/// GPU thread index hierarchy level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IndexLevel {
    Lane,
    Simdgroup,
    Workgroup,
}

impl fmt::Display for IndexLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lane => write!(f, "lane"),
            Self::Simdgroup => write!(f, "simdgroup"),
            Self::Workgroup => write!(f, "workgroup"),
        }
    }
}

impl FromStr for IndexLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "lane" => Ok(Self::Lane),
            "simdgroup" => Ok(Self::Simdgroup),
            "workgroup" => Ok(Self::Workgroup),
            _ => Err(format!("unknown index level: {s}")),
        }
    }
}
