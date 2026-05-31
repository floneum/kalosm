//! Runtime element types.
//!
//! tile-ir is runtime-typed (see ARBOR_DESIGN.md §2): element types are *data*
//! carried on `Node.ty`, not Rust marker types or const generics. The typed
//! frontend (`fusor`) owns compile-time element/shape/rank correctness; this
//! module only models the runtime element enums and the cooperative-matrix
//! role (also data, not typestate).

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
/// A data enum (not typestate): it collapses the old `CoopOperandRole` +
/// `CoopRole` chain. Making `{8, 16}` shapes unrepresentable is `fusor`'s job,
/// not tile-ir's (see ARBOR_DESIGN.md §2/§3).
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
