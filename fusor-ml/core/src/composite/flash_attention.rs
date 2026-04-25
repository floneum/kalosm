use std::sync::Arc;

use crate::{
    DataType, LastRank, Tensor,
    compute_graph::NodeIndex,
    mir::{inputs::MirValue, operation::Operation},
};

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Computes flash attention with optional masking.
    pub fn flash_attention<const R2: usize>(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        mask: Option<&Tensor<2, T>>,
    ) -> Self
    where
        Tensor<R, T>: LastRank<R2, T>,
        T: crate::FloatDataType,
    {
        let operation = FlashAttentionOperation::new(
            self.key(),
            k.key(),
            v.key(),
            mask.map(|m| m.key()),
            self.datatype(),
            self.shape(),
            k.shape(),
            scale,
        );
        let data = self.data();

        Self::from_parts(data.custom(Arc::new(operation)))
    }
}

#[derive(Debug, Clone)]
struct FlashAttentionOperation {
    pub(crate) q: NodeIndex,
    pub(crate) k: NodeIndex,
    pub(crate) v: NodeIndex,
    pub(crate) mask: Option<NodeIndex>,
}

impl FlashAttentionOperation {
    #[allow(clippy::too_many_arguments)]
    fn new(
        q: NodeIndex,
        k: NodeIndex,
        v: NodeIndex,
        mask: Option<NodeIndex>,
        _datatype: crate::DataTypeEnum,
        q_shape: &[usize],
        kv_shape: &[usize],
        _scale: f32,
    ) -> Self {
        let num_heads = q_shape[1];
        let num_kv_heads = kv_shape[1];
        assert!(
            num_heads.is_multiple_of(num_kv_heads),
            "Number of Q heads ({}) must be divisible by number of K/V heads ({})",
            num_heads,
            num_kv_heads
        );
        Self { q, k, v, mask }
    }
}

impl Operation for FlashAttentionOperation {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.q);
        f(self.k);
        f(self.v);
        if let Some(mask) = self.mask {
            f(mask);
        }
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let mut inputs = vec![
            MirValue::Tensor(nodes.get_cached_result(self.q).unwrap().clone()),
            MirValue::Tensor(nodes.get_cached_result(self.k).unwrap().clone()),
            MirValue::Tensor(nodes.get_cached_result(self.v).unwrap().clone()),
        ];
        if let Some(mask_idx) = self.mask {
            inputs.push(MirValue::Tensor(
                nodes.get_cached_result(mask_idx).unwrap().clone(),
            ));
        }
        inputs
    }

    fn name(&self) -> String {
        if self.mask.is_some() {
            "flash_attention_masked".to_string()
        } else {
            "flash_attention".to_string()
        }
    }

    fn output_layout(
        &self,
        map: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        map.get(&self.q).unwrap().clone()
    }
}
