//! Data-table helpers that collapse the per-op `fuzz_unary!` / `fuzz_same_shape_binary!`
//! macro families into a single function call per `(op, host-reference, tolerance)` row.
//!
//! Each helper builds exactly the same `assert(op).arg(gen)...compare_with(cmp).runs(n)
//! .into_case(name)` chain the macros emit today, but applies the host reference
//! function element-wise over the resolved tensor regardless of rank, so one helper
//! serves rank-1 large-tensor cases and rank-2 elementwise cases alike.

use fusor::{Tensor, WasmNotSend};

use crate::{AssertionCase, CompareFut, FuzzGenerator, ItemMismatchError, assert};

/// Iterate every logical index of `shape` in row-major order.
fn row_major_indices<const R: usize>(shape: [usize; R]) -> impl Iterator<Item = [usize; R]> {
    let total: usize = shape.iter().product();
    (0..total).map(move |flat| {
        let mut idx = [0usize; R];
        let mut rem = flat;
        for d in (0..R).rev() {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        idx
    })
}

/// Materialize a reference tensor by mapping `host` element-wise over `input`,
/// preserving its logical shape, on `input`'s device.
async fn map_unary<const R: usize>(
    input: Tensor<R, f32>,
    host: fn(f32) -> f32,
) -> Tensor<R, f32> {
    let device = input.device();
    let shape = input.shape();
    let slice = input.as_slice().await.unwrap();
    let data: Vec<f32> = row_major_indices(shape)
        .map(|idx| host(slice[idx]))
        .collect();
    Tensor::from_slice(&device, shape, &data)
}

/// Materialize a reference tensor by mapping `host` element-wise over the
/// same-shape operands `a` and `b`, on `a`'s device.
async fn map_binary<const R: usize>(
    a: Tensor<R, f32>,
    b: Tensor<R, f32>,
    host: fn(f32, f32) -> f32,
) -> Tensor<R, f32> {
    let device = a.device();
    let shape = a.shape();
    let sa = a.as_slice().await.unwrap();
    let sb = b.as_slice().await.unwrap();
    let data: Vec<f32> = row_major_indices(shape)
        .map(|idx| host(sa[idx], sb[idx]))
        .collect();
    Tensor::from_slice(&device, shape, &data)
}

/// One unary elementwise data-table row.
///
/// Builds `assert(op).arg(gen).equal_to(host-mapped reference)
/// .compare_with(compare).runs(runs).into_case(name)`, applying `host` element-wise
/// over the resolved input tensor regardless of rank.
///
/// ```ignore
/// let case = unary_fuzz_case(
///     "elementwise_ops::sin",
///     FuzzGenerator::<2, f32>::new([45, 45]),
///     |x: Tensor<2, f32>| async move { x.sin().to_concrete() },
///     f32::sin,
///     approx_compare::<2, f32>(1e-4),
///     3,
/// );
/// ```
pub fn unary_fuzz_case<F, Fut, const R: usize>(
    name: impl Into<String>,
    generator: FuzzGenerator<R, f32>,
    op: F,
    host: fn(f32) -> f32,
    compare: impl for<'a> Fn(&'a Tensor<R, f32>, &'a Tensor<R, f32>) -> CompareFut<'a, ItemMismatchError>
    + Clone
    + 'static,
    runs: usize,
) -> AssertionCase
where
    F: FnMut(Tensor<R, f32>) -> Fut + Clone + 'static,
    Fut: std::future::Future<Output = Tensor<R, f32>> + WasmNotSend + 'static,
{
    assert(op)
        .arg(generator)
        .equal_to(move |x: Tensor<R, f32>| async move { map_unary(x, host).await })
        .compare_with(compare)
        .runs(runs)
        .into_case(name)
}

/// One same-shape binary elementwise data-table row.
///
/// Builds `assert(op).arg(gen_a).arg(gen_b).equal_to(host-mapped reference)
/// .compare_with(compare).runs(runs).into_case(name)`, applying `host` element-wise
/// over both resolved operands regardless of rank.
///
/// ```ignore
/// let case = binary_fuzz_case(
///     "elementwise_ops::add",
///     FuzzGenerator::<2, f32>::new([45, 45]),
///     FuzzGenerator::<2, f32>::new([45, 45]),
///     |a: Tensor<2, f32>, b: Tensor<2, f32>| async move { (a + b).to_concrete() },
///     |l, r| l + r,
///     approx_compare::<2, f32>(1e-6),
///     3,
/// );
/// ```
pub fn binary_fuzz_case<F, Fut, const R: usize>(
    name: impl Into<String>,
    gen_a: FuzzGenerator<R, f32>,
    gen_b: FuzzGenerator<R, f32>,
    op: F,
    host: fn(f32, f32) -> f32,
    compare: impl for<'a> Fn(&'a Tensor<R, f32>, &'a Tensor<R, f32>) -> CompareFut<'a, ItemMismatchError>
    + Clone
    + 'static,
    runs: usize,
) -> AssertionCase
where
    F: FnMut(Tensor<R, f32>, Tensor<R, f32>) -> Fut + Clone + 'static,
    Fut: std::future::Future<Output = Tensor<R, f32>> + WasmNotSend + 'static,
{
    assert(op)
        .arg(gen_a)
        .arg(gen_b)
        .equal_to(move |a: Tensor<R, f32>, b: Tensor<R, f32>| async move {
            map_binary(a, b, host).await
        })
        .compare_with(compare)
        .runs(runs)
        .into_case(name)
}
