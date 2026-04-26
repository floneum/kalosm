//! SGEMV example: y[M] = A[M,K] @ x[K].

use egg::Language;
use tensor_ir::*;

fn main() {
    let mut builder = TensorExprBuilder::new();
    let m = Dim::Symbol(0);
    let k = Dim::Symbol(1);
    let a = builder.input(0, Shape(vec![m.clone(), k.clone()]), DType::F32);
    let x = builder.input(1, Shape(vec![k.clone(), Dim::Const(1)]), DType::F32);
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let body = builder.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let y = builder.contraction(
        Shape(vec![m, Dim::Const(1), k.clone()]),
        &[
            (a, Strides(vec![k, Dim::Const(0), Dim::Const(1)])),
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

    println!("=== SGEMV Tensor Summary ===");
    println!("summary: {:?}", expr.summary().expect("summary"));
    println!();

    println!("=== SGEMV Kernel ===");
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

    match lower_to_wgsl(&kernel) {
        Ok(wgsl) => println!("{wgsl}"),
        Err(err) => eprintln!("WGSL generation failed: {err}"),
    }
}
