#![cfg(not(target_arch = "wasm32"))]
use fusor::{
    Device, Tensor,
    autograd::{BackwardTarget, Graph, Tensor as Tracked},
};

#[test]
fn custom_backward_preserves_broadcast_reduction_and_loss_scaling() {
    let mut devices = Vec::new();
    #[cfg(feature = "cpu")]
    devices.push(Device::cpu());
    #[cfg(feature = "gpu")]
    devices.push(Device::gpu_blocking().unwrap());
    for device in devices {
        let graph = Graph::over(device.graph().clone());
        let weight = graph.leaf(Tensor::<1>::from_slice(&device, [3], &[-0.9, 0.3, 1.2]));
        let input = graph.constant(Tensor::<2>::from_slice(
            &device,
            [2, 3],
            &[1., 2., 3., 2., 3., 4.],
        ));
        let quantized = weight.raw().gte_scalar(0.5) - weight.raw().lte_scalar(-0.5);
        let slot = weight.slot();
        let quantized = Tracked::constant_from_raw(&graph, quantized)
            .with_backwards([weight.parent()], move |gradient| {
                Ok(vec![BackwardTarget::to(slot, gradient)])
            });
        let loss = input.mul_::<1, 2>(&quantized).sum::<1>(1).sum::<0>(0);
        assert_eq!(loss.raw().to_scalar(), 4.);
        let gradients = loss.backward_with(Tensor::splat(&device, 3., [])).unwrap();
        assert_eq!(gradients.get(&weight).unwrap().to_flat(), [9., 15., 21.]);
    }
}
