use std::sync::Arc;

use crate::{
    CastTensor, DataType, DataTypeEnum, MaxRank, Tensor,
    compute_graph::NodeIndex,
    mir::{inputs::MirValue, operation::Operation},
};

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// RMSNorm expressed as a graph operation for tensor_ir lowering.
    pub fn rms_norm_fused<const W: usize>(
        &self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
    ) -> Self
    where
        T: CastTensor<f32>,
        f32: CastTensor<T>,
        (Tensor<R, T>, Tensor<W, T>): MaxRank<R, T>,
    {
        let operation = RmsNormOperation::new(
            self.key(),
            weight.key(),
            bias.map(|b| b.key()),
            self.datatype(),
            weight.datatype(),
            self.shape(),
            weight.shape(),
            eps,
        );
        let data = self.data();

        Self::from_parts(data.custom(Arc::new(operation)))
    }

    pub fn rms_norm_fused_no_bias<const W: usize>(&self, weight: &Tensor<W, T>, eps: f32) -> Self
    where
        T: CastTensor<f32>,
        f32: CastTensor<T>,
        (Tensor<R, T>, Tensor<W, T>): MaxRank<R, T>,
    {
        self.rms_norm_fused(weight, None, eps)
    }
}

#[derive(Debug, Clone)]
struct RmsNormOperation {
    input: NodeIndex,
    weight: NodeIndex,
    bias: Option<NodeIndex>,
    input_dtype: DataTypeEnum,
    input_shape: Box<[usize]>,
}

impl RmsNormOperation {
    #[allow(clippy::too_many_arguments)]
    fn new(
        input: NodeIndex,
        weight: NodeIndex,
        bias: Option<NodeIndex>,
        input_dtype: DataTypeEnum,
        _weight_dtype: DataTypeEnum,
        input_shape: &[usize],
        _weight_shape: &[usize],
        _eps: f32,
    ) -> Self {
        Self {
            input,
            weight,
            bias,
            input_dtype,
            input_shape: input_shape.into(),
        }
    }
}

impl Operation for RmsNormOperation {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
        f(self.weight);
        if let Some(bias) = self.bias {
            f(bias);
        }
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let mut inputs = vec![
            MirValue::Tensor(nodes.get_cached_result(self.input).unwrap().clone()),
            MirValue::Tensor(nodes.get_cached_result(self.weight).unwrap().clone()),
        ];
        if let Some(bias) = self.bias {
            inputs.push(MirValue::Tensor(
                nodes.get_cached_result(bias).unwrap().clone(),
            ));
        }
        inputs
    }

    fn name(&self) -> String {
        format!(
            "rms_norm_fused_{}_{}{}",
            self.input_shape.len(),
            self.input_dtype,
            if self.bias.is_some() { "_bias" } else { "" }
        )
    }

    fn output_layout(
        &self,
        map: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        map.get(&self.input).unwrap().clone()
    }
}
