/// Trait for tensor types that can be created from array-like data.
///
/// This trait is generic over:
/// - `R` - tensor rank (const generic)
/// - `D` - data type (f32, i32, etc.)
/// - `T` - input data type (array, slice, iterator, etc.)
/// - `Dev` - device type
///
/// By having the tensor type implement this trait (rather than the input type),
/// we satisfy Rust's orphan rules and can implement this for generic input types.
pub trait FromArray<const R: usize, D, T, Dev> {
    fn from_array(data: T, device: &Dev) -> Self;
}
