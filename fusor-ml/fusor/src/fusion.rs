//! Facade-owned CPU fusion markers.
//!
//! `fusor::Tensor` keeps a third generic parameter so CPU expression fusion can
//! remain zero-cost, but callers should depend on this module instead of the
//! backend crates.

/// Hidden bridge to the CPU backend's lazy backing trait.
pub(crate) use crate::cpu::TensorBacking as BackendFusion;

/// A facade-owned marker for tensor CPU fusion state.
///
/// This is intentionally a thin layer over the current CPU lazy backing model:
/// `fusor` owns the public bound while the CPU backend remains free to change
/// how expressions are represented internally.
pub trait Fusion<const R: usize, D>: BackendFusion<R, Elem = D> {}

impl<const R: usize, D, T> Fusion<R, D> for T where T: BackendFusion<R, Elem = D> {}

/// Materialized tensor backing.
pub type Concrete<D, const R: usize> = crate::cpu::ConcreteTensor<D, R>;

/// Layout/view fusion marker.
pub type View<T, const R: usize> = crate::cpu::MapLayout<T, R>;

/// Elementwise unary fusion markers.
pub type Abs<D, const R: usize, T> = crate::cpu::Abs<D, R, T>;
pub type Acos<D, const R: usize, T> = crate::cpu::Acos<D, R, T>;
pub type Acosh<D, const R: usize, T> = crate::cpu::Acosh<D, R, T>;
pub type Asin<D, const R: usize, T> = crate::cpu::Asin<D, R, T>;
pub type Asinh<D, const R: usize, T> = crate::cpu::Asinh<D, R, T>;
pub type Atan<D, const R: usize, T> = crate::cpu::Atan<D, R, T>;
pub type Atanh<D, const R: usize, T> = crate::cpu::Atanh<D, R, T>;
pub type Cos<D, const R: usize, T> = crate::cpu::Cos<D, R, T>;
pub type Cosh<D, const R: usize, T> = crate::cpu::Cosh<D, R, T>;
pub type Exp<D, const R: usize, T> = crate::cpu::Exp<D, R, T>;
pub type Exp2<D, const R: usize, T> = crate::cpu::Exp2<D, R, T>;
pub type Log<D, const R: usize, T> = crate::cpu::Log<D, R, T>;
pub type Log2<D, const R: usize, T> = crate::cpu::Log2<D, R, T>;
pub type Neg<D, const R: usize, T> = crate::cpu::Neg<D, R, T>;
pub type Sin<D, const R: usize, T> = crate::cpu::Sin<D, R, T>;
pub type Sinh<D, const R: usize, T> = crate::cpu::Sinh<D, R, T>;
pub type Sqrt<D, const R: usize, T> = crate::cpu::Sqrt<D, R, T>;
pub type Tan<D, const R: usize, T> = crate::cpu::Tan<D, R, T>;
pub type Tanh<D, const R: usize, T> = crate::cpu::Tanh<D, R, T>;

/// Elementwise binary fusion markers.
pub type Add<D, const R: usize, T, U> = crate::cpu::Add<D, R, T, U>;
pub type Div<D, const R: usize, T, U> = crate::cpu::Div<D, R, T, U>;
pub type Mul<D, const R: usize, T, U> = crate::cpu::Mul<D, R, T, U>;
pub type Rem<D, const R: usize, T, U> = crate::cpu::Rem<D, R, T, U>;
pub type Sub<D, const R: usize, T, U> = crate::cpu::Sub<D, R, T, U>;

/// Scalar fusion markers.
pub type AddScalar<D, const R: usize, T> = crate::cpu::AddScalar<D, R, T>;
pub type DivScalar<D, const R: usize, T> = crate::cpu::DivScalar<D, R, T>;
pub type MulScalar<D, const R: usize, T> = crate::cpu::MulScalar<D, R, T>;
pub type SubScalar<D, const R: usize, T> = crate::cpu::SubScalar<D, R, T>;
