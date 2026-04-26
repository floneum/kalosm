//! Sum-reduce example using the staged tensor IR pipeline.

use egg::Language;
use tensor_ir::*;

fn main() {
    let mut builder = TensorExprBuilder::new();
    let rows = Dim::Symbol(0);
    let cols = Dim::Symbol(1);
    let a = builder.input(0, Shape(vec![rows, cols]), DType::F32);
    let reduce = builder.reduce(a, 1, ReduceOp::Add);
    let expr = builder.build(reduce).expect("valid tensor expression");

    let pipeline = StagedPipeline::default();
    let kernel = pipeline.lower(&expr).expect("kernel lowering");

    println!("=== Tensor Summary ===");
    println!("summary: {:?}", expr.summary().expect("summary"));
    println!();

    println!("=== Kernel ===");
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
