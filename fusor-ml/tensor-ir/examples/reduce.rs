//! Sum-reduce example using the staged tensor IR pipeline.

use egg::Language;
use tensor_ir::*;

const ROWS: u32 = 32;
const COLS: u32 = 64;

fn main() {
    let mut builder = TensorExprBuilder::new();
    let a = builder.input(0, Shape(vec![Dim::Lit(ROWS), Dim::Lit(COLS)]), DType::F32);
    let reduce = builder.reduce(a, 1, ReduceOp::Add);
    let expr = builder.build(reduce).expect("valid tensor expression");

    let pipeline = StagedPipeline::default();
    let kernel = pipeline.lower(&expr).expect("kernel lowering");
    let simd = pipeline.compile(kernel).expect("simd lowering");

    println!("=== Tensor Summary ===");
    println!("summary: {:?}", expr.summary().expect("summary"));
    println!();

    println!("=== Kernel ===");
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
