use std::{
    error::Error,
    fmt::{Debug, Display},
    ops::Sub,
    pin::Pin,
};

use fusor::{DataType, Device, SimdElement, Tensor};
use half::f16;
use thiserror::Error;

fn index_iter<const R: usize>(shape: [usize; R]) -> impl Iterator<Item = [usize; R]> {
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

/// Assert that two f32 tensors are element-wise close within `tol`.
pub async fn eq_with<const R: usize, T: DataType + SimdElement>(
    a: &Tensor<R, T>,
    b: &Tensor<R, T>,
    eq: impl Fn(T, T) -> bool,
) -> Result<(), ItemMismatchError> {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    let shape = a.shape();
    let sa = a.as_slice().await.unwrap();
    let sb = b.as_slice().await.unwrap();

    for index in index_iter(shape) {
        let va = sa[index];
        let vb = sb[index];
        if !eq(va, vb) {
            return Err(ItemMismatchError::new(
                a.device(),
                index,
                format!("{:?}", va),
                format!("{:?}", vb),
            ));
        }
    }

    Ok(())
}

/// Assert that two f32 tensors are element-wise close within `tol`.
pub async fn approx_eq<const R: usize, T: Sub + PartialOrd + DataType + SimdElement>(
    a: &Tensor<R, T>,
    b: &Tensor<R, T>,
    tol: T,
) -> Result<(), ItemMismatchError> {
    eq_with(a, b, |va, vb| {
        let diff = if va > vb { va - vb } else { vb - va };
        diff <= tol
    })
    .await
}

/// Assert that two tensors are element-wise equal.
pub async fn exact_eq<const R: usize, T: DataType + SimdElement + PartialEq>(
    a: &Tensor<R, T>,
    b: &Tensor<R, T>,
) -> Result<(), ItemMismatchError> {
    eq_with(a, b, |va, vb| va == vb).await
}

/// Assert that two f32 tensors are element-wise close within a *relative*
/// tolerance: `|a - b| <= rel_tol * max(|a|, |b|, eps)`.
///
/// Use this when reduction outputs grow with the reduced axis size and an
/// absolute tolerance becomes meaningless (e.g. a sum of 2025 values with
/// magnitude up to 5 has expected ~5e3 but absolute roundoff scales with
/// the magnitude of the result).
pub async fn relative_eq<const R: usize>(
    a: &Tensor<R, f32>,
    b: &Tensor<R, f32>,
    rel_tol: f32,
) -> Result<(), ItemMismatchError> {
    eq_with(a, b, |va, vb| {
        let diff = (va - vb).abs();
        let scale = va.abs().max(vb.abs()).max(f32::MIN_POSITIVE);
        diff <= rel_tol * scale
    })
    .await
}

/// Assert that two f32 tensors are element-wise close within either an
/// absolute tolerance or a relative tolerance.
///
/// Use this for outputs that can be near zero for some inputs but grow large
/// enough elsewhere that absolute roundoff alone becomes brittle.
pub async fn approx_or_relative_eq<const R: usize>(
    a: &Tensor<R, f32>,
    b: &Tensor<R, f32>,
    abs_tol: f32,
    rel_tol: f32,
) -> Result<(), ItemMismatchError> {
    eq_with(a, b, |va, vb| {
        let diff = (va - vb).abs();
        let scale = va.abs().max(vb.abs()).max(f32::MIN_POSITIVE);
        diff <= abs_tol || diff <= rel_tol * scale
    })
    .await
}

#[derive(Error, Debug)]
pub struct ItemMismatchError {
    device: Device,
    position: Vec<usize>,
    expected: String,
    actual: String,
}

impl ItemMismatchError {
    pub fn new(
        device: Device,
        position: impl IntoIterator<Item = usize>,
        expected: impl ToString,
        actual: impl ToString,
    ) -> Self {
        Self {
            device,
            position: position.into_iter().collect(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }
}

impl Display for ItemMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let position = if self.position.is_empty() {
            String::from("<scalar>")
        } else {
            format!("{:?}", self.position)
        };
        write!(
            f,
            "Item mismatch on device {:?} at {}: expected {}, got {}",
            self.device, position, self.expected, self.actual
        )
    }
}

/// Boxed future returned by a comparator: `&'a U, &'a U -> Result<(), E>`.
/// Aliased so the comparator type signatures stay readable.
pub type CompareFut<'a, E> = Pin<Box<dyn std::future::Future<Output = Result<(), E>> + 'a>>;

#[doc(hidden)]
pub trait IntoCompare<U> {
    type Error: Error;

    fn into_compare(self)
    -> impl for<'a> Fn(&'a U, &'a U) -> CompareFut<'a, Self::Error> + 'static;
}

impl<U, Cmp, E: Error> IntoCompare<U> for Cmp
where
    Cmp: for<'a> Fn(&'a U, &'a U) -> CompareFut<'a, E> + 'static,
{
    type Error = E;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(&'a U, &'a U) -> CompareFut<'a, Self::Error> + 'static {
        self
    }
}

impl<const R: usize> IntoCompare<Tensor<R, u32>> for () {
    type Error = ItemMismatchError;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(&'a Tensor<R, u32>, &'a Tensor<R, u32>) -> CompareFut<'a, Self::Error> + 'static
    {
        |a, b| Box::pin(exact_eq(a, b))
    }
}

impl<const R: usize> IntoCompare<Tensor<R, f32>> for () {
    type Error = ItemMismatchError;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(&'a Tensor<R, f32>, &'a Tensor<R, f32>) -> CompareFut<'a, Self::Error> + 'static
    {
        |a, b| Box::pin(approx_eq(a, b, 1e-5))
    }
}

impl<const R: usize> IntoCompare<Tensor<R, f16>> for () {
    type Error = ItemMismatchError;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(&'a Tensor<R, f16>, &'a Tensor<R, f16>) -> CompareFut<'a, Self::Error> + 'static
    {
        |a, b| Box::pin(approx_eq(a, b, f16::from_f32(1e-3)))
    }
}

pub fn exact_compare<const R: usize, T>()
-> impl for<'a> Fn(&'a Tensor<R, T>, &'a Tensor<R, T>) -> CompareFut<'a, ItemMismatchError> + Clone
where
    T: DataType + SimdElement + PartialEq,
{
    |a, b| Box::pin(exact_eq(a, b))
}

pub fn approx_compare<const R: usize, T>(
    tol: T,
) -> impl for<'a> Fn(&'a Tensor<R, T>, &'a Tensor<R, T>) -> CompareFut<'a, ItemMismatchError> + Clone
where
    T: Sub<Output = T> + PartialOrd + DataType + SimdElement + Copy,
{
    move |a, b| Box::pin(approx_eq(a, b, tol))
}

/// Compare-fn factory for [`relative_eq`]: pass `rel_tol` as a fraction
/// (e.g. `1e-3` for 0.1%).
pub fn relative_compare<const R: usize>(
    rel_tol: f32,
) -> impl for<'a> Fn(&'a Tensor<R, f32>, &'a Tensor<R, f32>) -> CompareFut<'a, ItemMismatchError> + Clone
{
    move |a, b| Box::pin(relative_eq(a, b, rel_tol))
}

pub fn approx_or_relative_compare<const R: usize>(
    abs_tol: f32,
    rel_tol: f32,
) -> impl for<'a> Fn(&'a Tensor<R, f32>, &'a Tensor<R, f32>) -> CompareFut<'a, ItemMismatchError> + Clone
{
    move |a, b| Box::pin(approx_or_relative_eq(a, b, abs_tol, rel_tol))
}
