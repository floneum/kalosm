//! Matmul optimization example using the staged tensor IR pipeline.

use egg::Language;
use tensor_ir::*;

const M: u32 = 64;
const N: u32 = 64;
const K: u32 = 64;

fn main() {
    let mut builder = TensorExprBuilder::new();
    let a = builder.input(0, Shape(vec![Dim::Const(M), Dim::Const(K)]), DType::F32);
    let b = builder.input(1, Shape(vec![Dim::Const(K), Dim::Const(N)]), DType::F32);
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let body = builder.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let matmul = builder.contraction(
        Shape(vec![Dim::Const(M), Dim::Const(N), Dim::Const(K)]),
        &[
            (
                a,
                Strides(vec![Dim::Const(K), Dim::Const(0), Dim::Const(1)]),
            ),
            (
                b,
                Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(N)]),
            ),
        ],
        body,
        &[(2, ReduceOp::Add)],
    );
    let expr = builder.build(matmul).expect("valid tensor expression");

    println!("=== TensorExpr ===");
    println!("root: {:?}", expr.root());
    println!("nodes: {}", expr.nodes().len());

    // Disable inner-loop unrolling so the extracted kernel keeps its loop
    // structure visible — easier to read when inspecting the optimization
    // output, at a small runtime cost.
    let mut config = StageConfig::default();
    config.runner.lowering = LoweringOptions::readable();
    let pipeline = StagedPipeline::new(config);
    println!("=== Tensor Summary ===");
    println!("summary: {:?}", expr.summary().expect("summary"));
    println!();

    let kernel = pipeline.lower(&expr).expect("kernel lowering");
    println!("=== Kernel ===");
    println!("cost: {:.1}", kernel.cost());
    println!("nodes: {}", kernel.extracted().as_ref().len());
    for (i, node) in kernel.extracted().as_ref().iter().enumerate() {
        let children: Vec<String> = node
            .children()
            .iter()
            .map(|child| format!("%{}", usize::from(*child)))
            .collect();
        if children.is_empty() {
            println!("  %{i} = {node:?}");
        } else {
            println!("  %{i} = {node:?}({})", children.join(", "));
        }
    }
    println!();

    let simd = pipeline.compile(kernel).expect("simd lowering");
    println!("=== Dispatch Skeleton ===");
    println!("{}", simd.dispatch_program());

    match lower_to_wgsl(simd.dispatch_program()) {
        Ok(wgsl) => {
            println!("=== WGSL ===");
            println!("{wgsl}");
        }
        Err(err) => eprintln!("WGSL generation failed: {err}"),
    }

    match lower_to_msl(simd.dispatch_program()) {
        Ok(msl) => {
            println!("=== MSL ===");
            println!("{msl}");
        }
        Err(err) => eprintln!("MSL generation failed: {err}"),
    }
}
