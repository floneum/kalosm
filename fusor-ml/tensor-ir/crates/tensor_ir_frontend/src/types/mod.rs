//! Typed building blocks for the tensor IR — shapes, scalars, ops, memory
//! tiers, dependence sets, device profile, and binder identifiers. Every
//! identifier-shaped value in the IR is encoded as one of the typed enums
//! re-exported below; there are no stringly-typed `Symbol`s in the IR.

mod binder;
mod dep;
mod device;
mod lowering;
mod memory;
mod scalar;
mod shape;
mod var;

pub use binder::{BinderInfo, BinderKind, HasBinder, slots};
pub use dep::{AddressProfile, DepSet, VarDepSet};
pub use device::DeviceProfile;
pub use lowering::LoweringOptions;
pub use memory::{BufferRef, IndexLevel, MemTier};
pub use scalar::{BinaryOp, DType, ReduceOp, ScalarValue, TernaryOp, UnaryOp};
pub use shape::{Dim, Shape, Strides};
pub use var::{VarRef, index_level_from_slot};
