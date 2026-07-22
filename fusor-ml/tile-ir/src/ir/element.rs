//! Runtime element types.
//!
//! tile-ir carries element types as runtime data on `Node.ty`, not Rust marker
//! types or const generics. The typed frontend (`fusor`) owns compile-time
//! element/shape/rank correctness; this module only models the runtime element
//! enums and the cooperative-matrix role.

/// Scalar elements that can back scalar, vector, and cooperative-matrix IR
/// values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarElement {
    /// 32-bit floating point scalar.
    F32,
    /// 16-bit floating point scalar.
    F16,
    /// 32-bit unsigned integer scalar.
    U32,
    /// Boolean scalar.
    Bool,
}

impl ScalarElement {
    /// Byte size of one scalar as stored in memory.
    pub const fn byte_size(self) -> u64 {
        match self {
            Self::F32 | Self::U32 | Self::Bool => 4,
            Self::F16 => 2,
        }
    }

    /// Element type for this scalar by itself.
    pub const fn element(self) -> ElementType {
        match self {
            Self::F32 => ElementType::F32,
            Self::F16 => ElementType::F16,
            Self::U32 => ElementType::U32,
            Self::Bool => ElementType::Bool,
        }
    }
}

/// Cooperative-matrix role encoded in an [`ElementType::CoopMatrix`].
///
/// A data enum, not typestate. Making invalid cooperative-matrix shapes
/// unrepresentable is `fusor`'s job, not tile-ir's.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CoopMatrixRole {
    /// Left-hand MMA operand.
    A,
    /// Right-hand MMA operand.
    B,
    /// Accumulator/result fragment.
    C,
}

/// Element types represented by the runtime-typed IR.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// 32-bit floating point scalar.
    F32,
    /// 16-bit floating point scalar.
    F16,
    /// 32-bit unsigned integer scalar.
    U32,
    /// Boolean scalar.
    Bool,
    /// Packed vector value.
    Vector {
        /// Scalar component type.
        scalar: ScalarElement,
        /// Vector lane count. Naga lowering supports 2, 3, and 4 lanes.
        lanes: u32,
    },
    /// Cooperative-matrix value of the given scalar, role, and shape. Coop
    /// fragment dims are runtime `u32` — there is no `CoopSize` const-generic.
    CoopMatrix {
        /// Scalar component type.
        scalar: ScalarElement,
        /// Cooperative role.
        role: CoopMatrixRole,
        /// Matrix rows.
        rows: u32,
        /// Matrix columns.
        cols: u32,
    },
}

impl ElementType {
    /// Construct a vector element.
    pub const fn vector(scalar: ScalarElement, lanes: u32) -> Self {
        Self::Vector { scalar, lanes }
    }

    /// Construct a cooperative-matrix element.
    pub const fn coop_matrix(
        scalar: ScalarElement,
        role: CoopMatrixRole,
        rows: u32,
        cols: u32,
    ) -> Self {
        Self::CoopMatrix {
            scalar,
            role,
            rows,
            cols,
        }
    }

    /// Byte size of one element as allocated in an array of this type.
    /// Cooperative fragments live in registers, not addressable arrays, and
    /// report the size of their scalar so footprint sums stay conservative.
    pub const fn byte_size(self) -> u64 {
        match self {
            Self::F32 | Self::U32 | Self::Bool => 4,
            Self::F16 => 2,
            Self::Vector { scalar, lanes } => scalar.byte_size() * lanes as u64,
            Self::CoopMatrix { scalar, .. } => scalar.byte_size(),
        }
    }

    /// Array stride of one element in a workgroup array, or `None` for
    /// elements that cannot back one (bool, cooperative fragments). The
    /// single source of stride truth: allocation packing and Naga array
    /// emission both read this, so they can never disagree. Differs from
    /// [`Self::byte_size`] for vec3, which pads to the vec4 stride.
    pub const fn workgroup_array_stride(self) -> Option<u32> {
        match self {
            Self::F32 | Self::U32 => Some(4),
            Self::F16 => Some(2),
            Self::Vector { scalar, lanes } => {
                let size = match scalar {
                    ScalarElement::F32 | ScalarElement::U32 => 4,
                    ScalarElement::F16 => 2,
                    ScalarElement::Bool => return None,
                };
                match lanes {
                    2 => Some(2 * size),
                    3 | 4 => Some(4 * size),
                    _ => None,
                }
            }
            Self::Bool | Self::CoopMatrix { .. } => None,
        }
    }

    /// Returns true when this element stores or computes with f16 data.
    pub const fn uses_f16(self) -> bool {
        matches!(
            self,
            Self::F16
                | Self::Vector {
                    scalar: ScalarElement::F16,
                    ..
                }
                | Self::CoopMatrix {
                    scalar: ScalarElement::F16,
                    ..
                }
        )
    }
}
