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
    pub value: Tensor,
}

impl GpuSampledToken {
    /// Read the token back. One of exactly three host syncs.
    pub fn to_u32(&self) -> Result<u32> {
        let v = self.value.to_vec_u32()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::test_support::{conformance_row, cpu_row, host_sorted};

    #[test]
    fn top_k_matches_a_host_sort_at_every_k() {
        let values = conformance_row();
        let (_s, _g, t) = cpu_row(&values);
        let want = host_sorted(&values);
        for k in 1..=values.len() {
            let (gv, gi) = top_k_pairs(&t, k as u32).unwrap();
            assert_eq!(gi.dtype(), Dtype::U32, "indices must be U32");
            let gv = gv.to_vec_f32().unwrap();
            let gi = gi.to_vec_u32().unwrap();
            let wv: Vec<f32> = want[..k].iter().map(|p| p.0).collect();
            let wi: Vec<u32> = want[..k].iter().map(|p| p.1).collect();
            assert_eq!(gv, wv, "k={k} values");
            assert_eq!(gi, wi, "k={k} ids");
        }
    }

    /// The declared rule: on an exact tie the larger token id sorts first.
    #[test]
    fn ties_break_towards_the_larger_token_id() {
        let mut values = vec![0.0f32; 16];
        values[3] = 2.0;
        values[9] = 2.0;
        let (_s, _g, t) = cpu_row(&values);
        let (_, ids) = top_k_pairs(&t, 2).unwrap();
        assert_eq!(ids.to_vec_u32().unwrap(), vec![9, 3]);
    }

    /// A whole row of equal logits is the strongest form of the tie rule: the
    /// ids must come back in strictly descending order.
    #[test]
    fn an_all_tied_row_sorts_by_descending_id() {
        let values = vec![1.5f32; 8];
        let (_s, _g, t) = cpu_row(&values);
        let (_, ids) = top_k_pairs(&t, 8).unwrap();
        assert_eq!(ids.to_vec_u32().unwrap(), vec![7, 6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn non_finite_logits_sort_below_every_real_one() {
        let values = vec![0.25, f32::NAN, 7.0, -3.0, 2.5, f32::INFINITY, 8.5, 9.0];
        let (_s, _g, t) = cpu_row(&values);
        let (_, ids) = top_k_pairs(&t, 5).unwrap();
        // The five finite winners, descending: 9.0@7, 8.5@6, 7.0@2, 2.5@4, 0.25@0.
        assert_eq!(ids.to_vec_u32().unwrap(), vec![7, 6, 2, 4, 0]);
    }

    #[test]
    fn a_degenerate_k_is_an_error_not_a_panic() {
        let (_s, _g, t) = cpu_row(&[1.0, 2.0, 3.0]);
        assert!(top_k_pairs(&t, 0).is_err());
        assert!(top_k_pairs(&t, 4).is_err());
    }
}
