//! Top-k over a logits row, and the on-device sampled-token handle that lets a
//! decode loop avoid a host round trip.

use crate::Result;
use crate::tensor::Tensor;
use crate::{Dtype, Error};

use super::row;

/// A token index that still lives on the device.
///
/// `sample` hands one of these back without resolving anything: `value` is a
/// one-element `U32` tensor that a decode loop can feed straight into the next
/// step's embedding lookup. Only [`GpuSampledToken::to_u32`] costs a sync.
#[derive(Clone)]
pub struct GpuSampledToken {
    /// One-element `u32` tensor containing the sampled token id.
    pub value: Tensor,
}

impl GpuSampledToken {
    /// Read the token back; costs a host sync.
    pub fn to_u32(&self) -> Result<u32> {
        let v = self.value.to_vec_u32()?;
        v.first()
            .copied()
            .ok_or_else(|| Error::Shape("the sampled token tensor came back empty".into()))
    }

    /// [`Self::to_u32`], awaited: the form a browser can use.
    pub async fn to_u32_async(&self) -> Result<u32> {
        let v = self.value.to_vec_u32_async().await?;
        v.first()
            .copied()
            .ok_or_else(|| Error::Shape("the sampled token tensor came back empty".into()))
    }
}

/// `(values, indices)` of the k largest entries along the last axis.
///
/// The order is value descending, and **on an exact tie the larger token id
/// comes first**. `values` is `F32` and `indices` is `U32`, both of shape `[k]`.
///
/// A non-finite logit is treated as `-f32::MAX`, so `NaN` and the infinities
/// sort below every real token and are reported with that sentinel as their
/// value rather than the original bit pattern.
pub fn top_k_pairs(logits: &Tensor, k: u32) -> Result<(Tensor, Tensor)> {
    let n = row::row_len(logits)?;
    if k == 0 {
        return Err(Error::Shape("top_k_pairs needs k >= 1".into()));
    }
    if u64::from(k) > n {
        return Err(Error::Shape(format!(
            "top_k_pairs({k}) on a row of {n}: k cannot exceed the vocabulary"
        )));
    }
    let (values, ids) = row::sort_desc(logits, n)?;
    let values = values.narrow(0, 0, k as usize)?;
    let ids = ids.narrow(0, 0, k as usize)?;
    let shape = row::dims(&[u64::from(k)]);
    let values = values.reshape_dims(&shape)?;
    let ids = ids.reshape_dims(&shape)?.cast(Dtype::U32)?;
    Ok((values, ids))
}
