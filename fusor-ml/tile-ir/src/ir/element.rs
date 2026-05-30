use std::marker::PhantomData;

macro_rules! numeric_markers {
    ($(($(#[$meta:meta])* $name:ident, $scalar:expr, $element:expr)),+ $(,)?) => {
        $(
            $(#[$meta])*
            #[derive(Copy, Clone, Debug)]
            pub struct $name;

            impl ScalarMarker for $name {
                const SCALAR: ScalarElement = $scalar;
            }

            impl Numeric for $name {
                const ELEMENT: ElementType = $element;
            }
        )+
    };
}

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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CoopMatrixRole {
    /// Left-hand MMA operand.
    A,
    /// Right-hand MMA operand.
    B,
    /// Accumulator/result fragment.
    C,
}

/// Element types represented by the typed IR.
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
    /// Cooperative-matrix value of the given scalar, role, and shape.
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

/// Marker for scalar types that can be named by typed IR wrappers.
pub trait ScalarMarker {
    /// Scalar represented by this marker.
    const SCALAR: ScalarElement;
}

/// Numeric element markers that can appear in the typed IR.
pub trait Numeric {
    /// Element type represented by this marker.
    const ELEMENT: ElementType;
}

/// Marker for scalar types accepted by vector dot products.
pub trait FloatElement: ScalarMarker + Numeric {}

/// Marker for scalar types accepted by cooperative-matrix fragments.
pub trait CoopElement: FloatElement {}

/// Packed vector marker.
///
/// ```
/// use fusor_tile_ir::{ElementType, Numeric, Vector, F32};
///
/// type F32x2 = Vector<F32, 2>;
/// assert_eq!(
///     F32x2::ELEMENT,
///     ElementType::vector(fusor_tile_ir::ScalarElement::F32, 2)
/// );
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Vector<T, const LANES: usize>(PhantomData<T>);

impl<T: ScalarMarker, const LANES: usize> Numeric for Vector<T, LANES> {
    const ELEMENT: ElementType = ElementType::Vector {
        scalar: T::SCALAR,
        lanes: LANES as u32,
    };
}

numeric_markers!(
    (
        /// A sample numeric marker.
        F32,
        ScalarElement::F32,
        ElementType::F32
    ),
    (
        /// Half-precision floating point storage marker.
        F16,
        ScalarElement::F16,
        ElementType::F16
    ),
    (
        /// Packed u32 storage marker.
        U32,
        ScalarElement::U32,
        ElementType::U32
    ),
    (
        /// Boolean private/control value marker.
        Bool,
        ScalarElement::Bool,
        ElementType::Bool
    ),
);

impl FloatElement for F32 {}
impl FloatElement for F16 {}

impl CoopElement for F32 {}
impl CoopElement for F16 {}
