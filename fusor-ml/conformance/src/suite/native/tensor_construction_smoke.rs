//! Smoke tests for tensor construction aliases and device/variant accessors.

use fusor::{Device, Tensor, arange, arange_step};

use crate::{AssertionCase, AssertionCases, exact_compare, exact_value_compare};

pub fn construction_aliases_match_on_varied_shapes() -> AssertionCases {
    let mut assertions = AssertionCases::new();

    let vector = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    assertions.push(
        fusor_conformance::assert(
            move |device: Device| async move { Tensor::new(&device, &vector) },
        )
        .arg(|device: &Device| device.clone())
        .equal_to(move |device: Device| async move {
            Tensor::from_slice(&device, [vector.len()], &vector)
        })
        .compare_with(exact_compare::<1, f32>())
        .runs(1)
        .into_case(
            "tensor_construction_smoke::construction_aliases_match_on_varied_shapes::vector_new",
        ),
    );

    let matrix = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    assertions.push(
        fusor_conformance::assert(
            move |device: Device| async move { Tensor::new(&device, &matrix) },
        )
        .arg(|device: &Device| device.clone())
        .equal_to(move |device: Device| async move {
            Tensor::from_slice(&device, [3, 3], &matrix.concat())
        })
        .compare_with(exact_compare::<2, f32>())
        .runs(1)
        .into_case(
            "tensor_construction_smoke::construction_aliases_match_on_varied_shapes::matrix_new",
        ),
    );

    assertions.push(
        fusor_conformance::assert(|device: Device| async move {
            Tensor::<2, f32>::zeros(&device, [3, 3])
        })
        .arg(|device: &Device| device.clone())
        .equal_to(move |device: Device| async move { Tensor::new(&device, &matrix).zeros_like() })
        .compare_with(exact_compare::<2, f32>())
        .runs(1)
        .into_case(
            "tensor_construction_smoke::construction_aliases_match_on_varied_shapes::matrix_zeros",
        ),
    );

    assertions.push(
        fusor_conformance::assert(async |device: Device| {
            Tensor::<2, f32>::full(&device, [3, 3], 7.0)
        })
        .arg(|device: &Device| device.clone())
        .equal_to(async |device: Device| Tensor::<2, f32>::splat(&device, 7.0, [3, 3]))
        .compare_with(exact_compare::<2, f32>())
        .runs(1)
        .into_case(
            "tensor_construction_smoke::construction_aliases_match_on_varied_shapes::matrix_full",
        ),
    );

    let cube = [
        [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
    ];
    assertions.push(
        fusor_conformance::assert(move |device: Device| async move { Tensor::new(&device, &cube) })
            .arg(|device: &Device| device.clone())
            .equal_to(move |device: Device| async move {
                Tensor::from_slice(
                    &device,
                    [2, 3, 2],
                    &cube.concat().into_iter().flatten().collect::<Vec<_>>(),
                )
            })
            .compare_with(exact_compare::<3, f32>())
            .runs(1)
            .into_case(
                "tensor_construction_smoke::construction_aliases_match_on_varied_shapes::cube_new",
            ),
    );

    assertions.push(
        fusor_conformance::assert(|device: Device| async move {
            Tensor::<3, f32>::zeros(&device, [2, 3, 2])
        })
        .arg(|device: &Device| device.clone())
        .equal_to(move |device: Device| async move { Tensor::new(&device, &cube).zeros_like() })
        .compare_with(exact_compare::<3, f32>())
        .runs(1)
        .into_case(
            "tensor_construction_smoke::construction_aliases_match_on_varied_shapes::cube_zeros",
        ),
    );

    assertions.push(
        fusor_conformance::assert(async |device: Device| {
            Tensor::<3, f32>::full(&device, [2, 3, 2], -2.5)
        })
        .arg(|device: &Device| device.clone())
        .equal_to(async |device: Device| Tensor::<3, f32>::splat(&device, -2.5, [2, 3, 2]))
        .compare_with(exact_compare::<3, f32>())
        .runs(1)
        .into_case(
            "tensor_construction_smoke::construction_aliases_match_on_varied_shapes::cube_full",
        ),
    );

    for shape in [[2, 3], [3, 4], [4, 2]] {
        let total = shape.iter().product::<usize>() as f32;
        assertions.push(fusor_conformance::assert(move |device: Device| async move {
            arange_step(&device, 0.0f32, total * 2.0, 2.0)
                .reshape(shape)
                .to_concrete()
        })
        .arg(|device: &Device| device.clone())
        .equal_to(move |device: Device| async move {
            arange(&device, 0.0f32, total)
                .reshape(shape)
                .to_concrete()
                .mul_scalar(2.0)
                .to_concrete()
        })
        .compare_with(exact_compare::<2, f32>())
        .runs(1)
        .into_case(format!(
            "tensor_construction_smoke::construction_aliases_match_on_varied_shapes::arange_{shape:?}"
        )));
    }
    assertions
}

pub fn device_wrappers_and_variant_accessors_work() -> AssertionCase {
    fusor_conformance::assert(async |device: Device| {
        let tensor: Tensor<1, f32> = Tensor::from_slice(&device, [5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let concrete = tensor.clone().to_concrete();
        (
            tensor.is_cpu(),
            tensor.is_gpu(),
            tensor.as_cpu().is_some(),
            tensor.as_gpu().is_some(),
            tensor.clone().to_cpu().is_some(),
            tensor.clone().to_gpu().is_some(),
            tensor.shape(),
            tensor.gpu_key().is_some(),
            tensor.rank(),
            tensor.to_scalar().await.unwrap(),
            concrete.is_cpu(),
            concrete.is_gpu(),
        )
    })
    .arg(|device: &Device| device.clone())
    .equal_to(async |device: Device| {
        let is_gpu = device.is_gpu();
        (
            !is_gpu,
            is_gpu,
            !is_gpu,
            is_gpu,
            !is_gpu,
            is_gpu,
            [5],
            is_gpu,
            1,
            1.0,
            !is_gpu,
            is_gpu,
        )
    })
    .compare_with(exact_value_compare())
    .runs(1)
    .into_case("tensor_construction_smoke::device_wrappers_and_variant_accessors_work")
}
