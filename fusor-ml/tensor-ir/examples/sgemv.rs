//! SGEMV example: y[M] = A[M,K] @ x[K].

use egg::Language;
use tensor_ir::*;

const M: u32 = 128;
const K: u32 = 256;

fn main() {
    let mut builder = TensorExprBuilder::new();
    let a = builder.input(0, Shape(vec![Dim::Const(M), Dim::Const(K)]), DType::F32);
    let x = builder.input(1, Shape(vec![Dim::Const(K), Dim::Const(1)]), DType::F32);
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let body = builder.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let y = builder.contraction(
        Shape(vec![Dim::Const(M), Dim::Const(1), Dim::Const(K)]),
        &[
            (
                a,
                Strides(vec![Dim::Const(K), Dim::Const(0), Dim::Const(1)]),
            ),
            (
                x,
                Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(1)]),
            ),
        ],
        body,
        &[(2, ReduceOp::Add)],
    );
    let expr = builder.build(y).expect("valid tensor expression");

    let pipeline = StagedPipeline::default();
    let kernel = pipeline.lower(&expr).expect("kernel lowering");
    let simd = pipeline.compile(kernel).expect("simd lowering");

    println!("=== SGEMV Tensor Summary ===");
    println!("summary: {:?}", expr.summary().expect("summary"));
    println!();

    println!("=== SGEMV Kernel ===");
    for (i, node) in simd.kernel().extracted().as_ref().iter().enumerate() {
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

    println!("=== Dispatch Skeleton ===");
    println!("{}", simd.dispatch_program());

    match lower_to_wgsl(simd.dispatch_program()) {
        Ok(wgsl) => println!("{wgsl}"),
        Err(err) => eprintln!("WGSL generation failed: {err}"),
    }
}
