//! Sampling. These are inference-only, have no adjoint, and enter through
//! `L1::Ext` + an `OpDef` with one declared cost row — no core file changes.
//!
//! Everything here is built out of ordinary facade ops on the caller's graph,
//! so a draw is a lazy device value: [`standard::sample`] resolves nothing and
//! hands back a `U32` token tensor a decode loop can consume directly. See
//! [`row`] for why the shapes are all matmuls against dense constants rather
//! than broadcasts.

pub mod mirostat2;
pub(crate) mod row;
pub mod standard;
pub mod top_k;

pub use mirostat2::Mirostat2Sampler;
pub use standard::StandardSamplerParams;
pub use top_k::{GpuSampledToken, top_k_pairs};

#[cfg(test)]
pub(crate) mod test_support {
    use crate::graph::GraphRef;
    // The backend selector, by module path — see the note in `composite.rs`.
    // `Graph` and `Dtype` are not what `typed-api` switches, so they keep the
    // root spelling.
    use crate::session::{Backend, Session};
    use crate::tensor::Tensor;
    use crate::{Dtype, Graph};

    use super::row::dims;

    /// A CPU session, a graph and one uploaded f32 row. The session and graph
    /// are returned because dropping them would take the tensor with them.
    pub(crate) fn cpu_row(values: &[f32]) -> (Session, Graph, Tensor) {
        let session = Session::new(Backend::cpu().expect("a cpu device")).expect("a session");
        let graph = Graph::new(&session);
        let t = upload(graph.handle(), values);
        (session, graph, t)
    }

    pub(crate) fn upload(g: &GraphRef, values: &[f32]) -> Tensor {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Tensor::from_slice(g, Dtype::F32, &dims(&[values.len() as u64]), &bytes)
            .expect("a logits row")
    }

    /// The conformance suite's logits row: an exact tie at 4 and 11, and one
    /// unambiguous maximum at 7.
    pub(crate) fn conformance_row() -> Vec<f32> {
        let mut v: Vec<f32> = (0..16).map(|i| (i as f32) * 0.37 - 3.0).collect();
        v[4] = 1.25;
        v[11] = 1.25;
        v[7] = 5.0;
        v
    }

    /// `(value, id)` descending, ties to the larger id — the rule under test,
    /// computed independently on the host.
    pub(crate) fn host_sorted(values: &[f32]) -> Vec<(f32, u32)> {
        let mut pairs: Vec<(f32, u32)> = values
            .iter()
            .enumerate()
            .map(|(i, v)| (*v, i as u32))
            .collect();
        pairs.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.cmp(&a.1))
        });
        pairs
    }

    pub(crate) fn softmax(values: &[f32]) -> Vec<f32> {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
        let total: f32 = exps.iter().sum();
        exps.iter().map(|e| e / total).collect()
    }

    /// The token ids in the shortest sorted prefix whose mass reaches `p`,
    /// including the token that crosses.
    pub(crate) fn nucleus(values: &[f32], p: f32) -> Vec<u32> {
        let sorted = host_sorted(values);
        let max = sorted[0].0;
        let exps: Vec<f32> = sorted.iter().map(|(v, _)| (v - max).exp()).collect();
        let total: f32 = exps.iter().sum();
        let mut mass = 0.0f32;
        let mut out = Vec::new();
        for (e, (_, id)) in exps.iter().zip(&sorted) {
            out.push(*id);
            mass += e / total;
            if mass >= p {
                break;
            }
        }
        out
    }
}
