use fusor::{
    Device, Tensor as RawTensor,
    autograd::{Gradients, Tensor},
};

#[derive(Clone, Copy, Debug)]
pub struct AdamWSettings {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    adam_eps: f32,
    weight_decay: f32,
}

pub trait AdamWModel: Sized {
    type State;

    fn adamw_state(device: &Device, model: &Self) -> Self::State;

    fn adamw_step(
        self,
        state: &mut Self::State,
        gradients: &Gradients,
        step: usize,
        settings: AdamWSettings,
    ) -> Self;
}

pub struct AdamW<M: AdamWModel> {
    step: usize,
    settings: AdamWSettings,
    state: M::State,
}

pub struct AdamMoments<const R: usize> {
    m: RawTensor<R, f32>,
    v: RawTensor<R, f32>,
}

impl AdamWSettings {
    pub const fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        adam_eps: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            adam_eps,
            weight_decay,
        }
    }
}

impl<M: AdamWModel> AdamW<M> {
    pub fn new(device: &Device, model: &M, settings: AdamWSettings) -> Self {
        Self {
            step: 0,
            settings,
            state: M::adamw_state(device, model),
        }
    }

    pub fn step(&mut self, model: M, gradients: Gradients) -> M {
        self.step += 1;
        model.adamw_step(&mut self.state, &gradients, self.step, self.settings)
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.settings.learning_rate = learning_rate;
    }

    pub fn settings(&self) -> AdamWSettings {
        self.settings
    }
}

impl<const R: usize> AdamMoments<R> {
    pub fn zeros_like(device: &Device, parameter: &Tensor<R>) -> Self {
        let shape = parameter.shape();
        Self {
            m: RawTensor::zeros(device, shape),
            v: RawTensor::zeros(device, shape),
        }
    }
}

fn detach_persistent<const R: usize>(tensor: RawTensor<R, f32>) -> RawTensor<R, f32> {
    match tensor {
        RawTensor::Cpu(tensor) => RawTensor::Cpu(tensor.to_concrete()),
        RawTensor::Gpu(tensor) => RawTensor::Gpu(tensor.detach()),
    }
}

/// Extract the gradient for a parameter without resolving it. Returns the lazy
/// gradient `RawTensor` and the parameter's raw value, both of which can be
/// resolved later after the autograd graph has been dropped.
pub fn extract_gradient<const R: usize>(
    parameter: &Tensor<R>,
    gradients: &Gradients,
) -> Option<(RawTensor<R, f32>, RawTensor<R, f32>)> {
    gradients
        .get(parameter)
        .map(|gradient| (gradient, parameter.raw().clone()))
}

/// Apply the AdamW update given a pre-extracted gradient and parameter value.
/// This variant does not hold any reference to the autograd graph, allowing the
/// old graph (and its backward-closure captured tensors) to be dropped before
/// GPU resolution occurs.
pub fn adamw_update_raw<const R: usize>(
    param_value: RawTensor<R, f32>,
    gradient: RawTensor<R, f32>,
    moments: &mut AdamMoments<R>,
    step: usize,
    settings: AdamWSettings,
) -> RawTensor<R, f32> {
    // Persistent optimizer state must be detached on GPU. `to_concrete()` is a no-op for the
    // GPU backend, so reusing it here would keep chaining prior-step compute graphs into the
    // moments and parameters.
    let next_m = detach_persistent(
        ((moments.m.clone() * settings.beta1) + (gradient.clone() * (1.0 - settings.beta1)))
            .to_concrete(),
    );
    let next_v = detach_persistent(
        ((moments.v.clone() * settings.beta2)
            + (gradient.sqr().to_concrete() * (1.0 - settings.beta2)))
            .to_concrete(),
    );

    let bias_correction1 = 1.0 - settings.beta1.powi(step as i32);
    let bias_correction2 = 1.0 - settings.beta2.powi(step as i32);
    let m_hat = next_m.clone().div_scalar(bias_correction1).to_concrete();
    let v_hat = next_v.clone().div_scalar(bias_correction2).to_concrete();
    let adam_update =
        (m_hat / (v_hat.add_scalar(settings.adam_eps).sqrt().to_concrete())).to_concrete();
    let weight_decay = (param_value.clone() * settings.weight_decay).to_concrete();
    let next_parameter = detach_persistent(
        (param_value - ((adam_update + weight_decay).to_concrete() * settings.learning_rate))
            .to_concrete(),
    );

    moments.m = next_m;
    moments.v = next_v;
    next_parameter
}

pub fn adamw_update<const R: usize>(
    parameter: &Tensor<R>,
    moments: &mut AdamMoments<R>,
    gradients: &Gradients,
    step: usize,
    settings: AdamWSettings,
) -> Tensor<R> {
    let Some(gradient) = gradients.get(parameter) else {
        return parameter.clone();
    };
    let next_parameter =
        adamw_update_raw(parameter.raw().clone(), gradient, moments, step, settings);
    Tensor::from_raw(&parameter.graph(), next_parameter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor::autograd::Graph;

    #[test]
    fn gpu_adamw_update_detaches_persistent_state() {
        let Ok(device) = Device::gpu_blocking() else {
            eprintln!("skipping GPU AdamW detach regression test: GPU unavailable");
            return;
        };

        let graph = Graph::new();
        let parameter: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, -2.0]);
        let loss = parameter.sqr().sum();
        let gradients = loss.backward().unwrap().into_detached();
        let mut moments = AdamMoments::zeros_like(&device, &parameter);
        let settings = AdamWSettings::new(0.01, 0.9, 0.999, 1e-8, 0.01);

        let next = adamw_update(&parameter, &mut moments, &gradients, 1, settings);

        assert_eq!(
            next.raw()
                .as_gpu()
                .expect("expected GPU parameter")
                .count_kernels_to_resolve(),
            0,
            "updated parameters should not retain prior-step compute graphs",
        );
        assert_eq!(
            moments
                .m
                .as_gpu()
                .expect("expected GPU first moment")
                .count_kernels_to_resolve(),
            0,
            "first-moment state should be detached after the optimizer step",
        );
        assert_eq!(
            moments
                .v
                .as_gpu()
                .expect("expected GPU second moment")
                .count_kernels_to_resolve(),
            0,
            "second-moment state should be detached after the optimizer step",
        );
    }
}
