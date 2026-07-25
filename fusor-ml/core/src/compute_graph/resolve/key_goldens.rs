//! Byte goldens for the resolver's structural cache keys.
//!
//! Every key here selects a cache entry that is trusted without re-deriving
//! what it stands for: kernel plans replayed by positional rebind, merged
//! dispatch plans shared across processes through the persistent store, and
//! flush plans replayed instead of a resolve. A recipe change that keeps
//! correctness intact is therefore still a regression — it invalidates the
//! warm stores of every machine that ever ran an older build — and no
//! correctness test can see it. These goldens pin the exact key words for a
//! spread of real graphs so a refactor of the folds has to be byte-preserving.
//!
//! Re-capture procedure after an INTENTIONAL recipe change: run the test and
//! paste the measured lines from the failure message into the tables below,
//! bumping the affected recipe version (`REPLAY_RECIPE_VERSION` for flush
//! plans, the leading literal in `kernel_cache_key_with_dispatch` for kernel
//! plans) in the same change.

use fusor_gguf::GgmlType;

use super::merge_horizontal::MergedSegments;
use super::*;
use crate::reduce::{ReduceFunction, ReduceOp, ReduceOperation};
use crate::region::{ElementwiseRegionOperation, RegionStatement};
use crate::row_program::RowProgramOperation;
use crate::{Device, QMatrix, Tensor};

/// `structural_kernel_key` over each single-operation lowering arm, keyed by
/// the graph that produced it.
const KERNEL_KEY_GOLDENS: &[(&str, &str)] = &[
    (
        "step_elementwise",
        "KernelCacheKey([16471553309229238721, 5488565396097342240])",
    ),
    (
        "step_reduce",
        "KernelCacheKey([14461885498005454950, 3478897584873558469])",
    ),
    (
        "step_view",
        "KernelCacheKey([10229830369531949427, 17693586530176713425])",
    ),
    (
        "step_assign",
        "KernelCacheKey([10527443992534951948, 17991200153112607082])",
    ),
    (
        "decode_elementwise",
        "KernelCacheKey([5744166422789977743, 13207922583434741741])",
    ),
    (
        "decode_reduce",
        "KernelCacheKey([14004285640874462945, 3021297727809675328])",
    ),
    (
        "quantized_elementwise",
        "KernelCacheKey([17079719460357749957, 6096731547292962340])",
    ),
];

/// `merged_plan_cache_key` over both fold arms: the region arm hashes kernel
/// fields and MIR values inline, every other segment kind folds its own
/// structural kernel key.
const MERGED_KEY_GOLDENS: &[(&str, &str)] = &[
    (
        "region_x1",
        "KernelCacheKey([16903751727878911052, 1182021694381039177])",
    ),
    (
        "region_x2",
        "KernelCacheKey([10258795302309731177, 1156476658944166214])",
    ),
    (
        "row_x1",
        "KernelCacheKey([13902752502679103160, 16611653512458465905])",
    ),
    (
        "row_x2",
        "KernelCacheKey([10371129316798825822, 15253781405597726022])",
    ),
];

/// `FlushPlanKey` over whole pending subgraphs: a transformer step, a decode
/// token, and a quantized decode token.
const FLUSH_KEY_GOLDENS: &[(&str, &str)] = &[
    (
        "transformer_step",
        "FlushPlanKey([13924756313179939098, 12746842292723456953])",
    ),
    (
        "decode_token",
        "FlushPlanKey([2248803744916415424, 13788545245063220203])",
    ),
    (
        "quantized_decode",
        "FlushPlanKey([9842396758258840322, 12416291958004416485])",
    ),
];

fn assert_goldens(what: &str, goldens: &[(&str, &str)], measured: &[(&str, String)]) {
    let expected: Vec<(&str, String)> = goldens
        .iter()
        .map(|(name, key)| (*name, (*key).to_string()))
        .collect();
    if expected != measured {
        let lines = measured
            .iter()
            .map(|(name, key)| format!("    (\"{name}\", \"{key}\"),"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!("{what} key bytes changed. measured values:\n{lines}");
    }
}

/// Cache everything under `node` so `Operation::inputs` can gather it,
/// exactly as the resolver's queue does before it keys a kernel plan: leaves
/// contribute their own storage, intermediates a freshly allocated output.
fn cache_dependencies(graph: &mut ComputeGraphInner, node: NodeIndex) {
    let mut deps = Vec::new();
    graph.visit_dependencies(node, &mut |dep| deps.push(dep));
    for dep in deps {
        if graph.get_cached_result(dep).is_some() {
            continue;
        }
        match &graph
            .nodes
            .nodes
            .node_weight(dep)
            .expect("live node")
            .variant
        {
            // Raw block storage is gathered straight off the node.
            ComputeGraphNodeVariant::QMatrix(_) => continue,
            ComputeGraphNodeVariant::Tensor(data) => {
                let data = data.clone();
                graph.set_cached_result(dep, data);
                continue;
            }
            _ => {}
        }
        cache_dependencies(graph, dep);
        let output = {
            let variant = &graph
                .nodes
                .nodes
                .node_weight(dep)
                .expect("live node")
                .variant;
            let operation = operation_of(variant);
            let inputs = operation.inputs(graph);
            operation.output(graph, &inputs)
        };
        let MirValue::Tensor(output) = output else {
            panic!("golden dependency output is not a tensor");
        };
        graph.set_cached_result(dep, output);
    }
}

fn operation_of(variant: &ComputeGraphNodeVariant) -> &dyn Operation {
    match variant {
        ComputeGraphNodeVariant::Elementwise(op) => op,
        ComputeGraphNodeVariant::Reduce(op) => op,
        ComputeGraphNodeVariant::View(op) => op,
        ComputeGraphNodeVariant::Assign(op) => op,
        ComputeGraphNodeVariant::Tensor(_) | ComputeGraphNodeVariant::QMatrix(_) => {
            panic!("golden target is not a lowered operation")
        }
    }
}

/// The structural kernel key of `tensor`'s own operation, gathered and solved
/// the way the queue executor does.
fn kernel_key(device: &Device, tensor: &Tensor) -> String {
    let node = tensor.data().key;
    device.compute_graph().with_mut(|graph| {
        cache_dependencies(graph, node);
        let variant = &graph
            .nodes
            .nodes
            .node_weight(node)
            .expect("live node")
            .variant;
        let operation = operation_of(variant);
        let inputs = operation.inputs(graph);
        let workgroup = operation
            .workgroup_shape_constraints(device)
            .solve(device.max_subgroup_size(), &device.limits())
            .expect("golden operation has a solvable workgroup shape");
        format!(
            "{:?}",
            structural_kernel_key(operation, &inputs, &workgroup)
        )
    })
}

fn flush_key(device: &Device, targets: &[NodeIndex]) -> String {
    device.compute_graph().with_mut(|graph| {
        let fingerprint =
            flush_replay::fingerprint_pending(graph, targets).expect("golden graph fingerprints");
        format!("{:?}", fingerprint.key)
    })
}

fn tensor_value(device: &Device, shape: &[usize]) -> MirValue {
    MirValue::Tensor(TensorData::new_for_shape(
        device,
        shape,
        crate::DataTypeEnum::F32,
    ))
}

fn region_segment(inputs: usize, outputs: usize, shape: &[usize]) -> ElementwiseRegionOperation {
    ElementwiseRegionOperation {
        inputs: (0..inputs).map(NodeIndex::new).collect(),
        statements: (0..outputs)
            .map(|statement| RegionStatement {
                expression: crate::nary_wise::NaryExpr::input(statement % inputs, inputs),
                datatype: crate::DataTypeEnum::F32,
                output: Some(NodeIndex::new(100 + statement)),
            })
            .collect(),
        shape: shape.to_vec().into_boxed_slice(),
    }
}

fn row_segment(axis_len: usize) -> RowProgramOperation {
    RowProgramOperation::from_reduce(&ReduceOperation {
        inputs: vec![NodeIndex::new(0)],
        expression: crate::nary_wise::NaryExpr::indexed_input(
            0,
            vec![
                crate::nary_wise::NaryExpr::DimIndex(0),
                crate::nary_wise::NaryExpr::DimIndex(1),
            ],
        ),
        shape: vec![1, axis_len].into_boxed_slice(),
        function: ReduceFunction {
            name: Some("sum".to_string()),
            op: ReduceOp::Sum,
            initial_value: crate::nary_wise::NaryScalar::F32(0.0),
            datatype: crate::DataTypeEnum::F32,
        },
        post_element_wise: crate::nary_wise::UnaryFunctionChain::empty(crate::DataTypeEnum::F32),
        axis: 1,
    })
}

pub(super) fn q4k_weight(device: &Device, rows: usize, cols: usize) -> QMatrix {
    let blocks = rows * cols / 256;
    let bytes: Vec<u8> = (0..blocks * 144)
        .map(|index| ((index * 31 + 7) % 251) as u8)
        .collect();
    QMatrix::from_parts(
        device,
        &bytes,
        vec![rows, cols].into_boxed_slice(),
        GgmlType::Q4K,
    )
    .expect("q4k weight")
}

/// A transformer-step-shaped graph: a projection composed as broadcast
/// multiply plus reduction, a residual add, a reshape, and a slice assign.
fn transformer_step(device: &Device) -> Tensor {
    let patch = Tensor::splat::<f32>(device, 0.0, &[1, 4, 64]);
    step_view(device).slice_assign([0..1, 0..4, 0..64], &patch)
}

fn step_view(device: &Device) -> Tensor {
    step_elementwise(device).reshape([2, 4, 64])
}

fn step_elementwise(device: &Device) -> Tensor {
    let x = Tensor::splat::<f32>(device, 0.0, &[8, 64]);
    let w = Tensor::splat::<f32>(device, 0.0, &[64, 64]);
    &x.mat_mul(&w) + &x
}

fn step_reduce(device: &Device) -> Tensor {
    let x = Tensor::splat::<f32>(device, 0.0, &[8, 64]);
    let w = Tensor::splat::<f32>(device, 0.0, &[64, 64]);
    x.mat_mul(&w)
}

/// A decode-token-shaped graph: one row against a resident weight.
fn decode_token(device: &Device) -> Tensor {
    let projected = decode_reduce(device);
    &projected * &projected
}

fn decode_reduce(device: &Device) -> Tensor {
    let x = Tensor::splat::<f32>(device, 0.0, &[1, 512]);
    let w = Tensor::splat::<f32>(device, 0.0, &[512, 512]);
    x.mat_mul(&w)
}

/// A quantized-decode-shaped graph: one row against a Q4K weight.
fn quantized_decode(device: &Device) -> Tensor {
    let x = Tensor::splat::<f32>(device, 0.0, &[1, 512]);
    let weight = q4k_weight(device, 512, 512);
    let projected = x.q_mat_mul(&weight);
    &projected + &projected
}

#[test]
fn structural_kernel_key_bytes_are_stable() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let cases: &[(&str, fn(&Device) -> Tensor)] = &[
            ("step_elementwise", step_elementwise),
            ("step_reduce", step_reduce),
            ("step_view", step_view),
            ("step_assign", transformer_step),
            ("decode_elementwise", decode_token),
            ("decode_reduce", decode_reduce),
            ("quantized_elementwise", quantized_decode),
        ];
        let measured: Vec<(&str, String)> = cases
            .iter()
            .map(|(name, build)| (*name, kernel_key(&device, &build(&device))))
            .collect();
        assert_goldens("structural_kernel_key", KERNEL_KEY_GOLDENS, &measured);
    });
}

#[test]
fn merged_plan_cache_key_bytes_are_stable() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let region_inputs = |count: usize| -> Vec<Vec<MirValue>> {
            (0..count)
                .map(|_| vec![tensor_value(&device, &[8, 64]); 2])
                .collect()
        };
        let row_inputs = |count: usize| -> Vec<Vec<MirValue>> {
            (0..count)
                .map(|_| {
                    vec![
                        tensor_value(&device, &[1, 512]),
                        tensor_value(&device, &[1, 1]),
                    ]
                })
                .collect()
        };
        let cases: Vec<(&str, MergedSegments, Vec<Vec<MirValue>>)> = vec![
            (
                "region_x1",
                MergedSegments::Region(vec![(NodeIndex::new(1), region_segment(1, 1, &[8, 64]))]),
                region_inputs(1),
            ),
            (
                "region_x2",
                MergedSegments::Region(vec![
                    (NodeIndex::new(1), region_segment(1, 1, &[8, 64])),
                    (NodeIndex::new(2), region_segment(1, 2, &[8, 64])),
                ]),
                region_inputs(2),
            ),
            (
                "row_x1",
                MergedSegments::Row(vec![(NodeIndex::new(1), row_segment(512))]),
                row_inputs(1),
            ),
            (
                "row_x2",
                MergedSegments::Row(vec![
                    (NodeIndex::new(1), row_segment(512)),
                    (NodeIndex::new(2), row_segment(256)),
                ]),
                row_inputs(2),
            ),
        ];
        let measured: Vec<(&str, String)> = cases
            .iter()
            .map(|(name, merged, inputs)| {
                (
                    *name,
                    format!(
                        "{:?}",
                        super::queue_executor::merged_plan_cache_key(merged, inputs)
                    ),
                )
            })
            .collect();
        assert_goldens("merged_plan_cache_key", MERGED_KEY_GOLDENS, &measured);
    });
}

#[test]
fn flush_plan_key_bytes_are_stable() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let cases: &[(&str, fn(&Device) -> Tensor)] = &[
            ("transformer_step", transformer_step),
            ("decode_token", decode_token),
            ("quantized_decode", quantized_decode),
        ];
        let measured: Vec<(&str, String)> = cases
            .iter()
            .map(|(name, build)| {
                let target = build(&device);
                (*name, flush_key(&device, &[target.data().key]))
            })
            .collect();
        assert_goldens("FlushPlanKey", FLUSH_KEY_GOLDENS, &measured);
    });
}
