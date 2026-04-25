//! Dump the best-extracted RecExpr after each optimization phase for a matmul.
//!
//! Steps the pipeline one phase at a time, extracts the current best
//! candidate, and pretty-prints it so you can see the IR evolve.

use egg::Language;
use tensor_ir::*;

const M: u32 = 64;
const N: u32 = 64;
const K: u32 = 64;

fn build_matmul() -> egg::RecExpr<TensorIr> {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(M), Dim::Lit(K)]), DType::F32);
    let rhs = b.input(1, Shape(vec![Dim::Lit(K), Dim::Lit(N)]), DType::F32);
    let tile = Shape(vec![Dim::Lit(M), Dim::Lit(N), Dim::Lit(K)]);
    let a_r = b.restride(a, tile.clone(), Strides(vec![i64::from(K), 0, 1]));
    let b_r = b.restride(rhs, tile.clone(), Strides(vec![0, 1, i64::from(N)]));
    let arg0 = b.scalar_arg(0);
    let arg1 = b.scalar_arg(1);
    let mul_body = b.bin_op(BinaryOp::Mul, arg0, arg1);
    let mul = b.elementwise(tile, &[a_r, b_r], mul_body);
    let _root = b.reduce(mul, 2, ReduceOp::Add);
    b.expr
}

fn print_recexpr(label: &str, expr: &egg::RecExpr<TensorIr>) {
    println!("=== {label}  ({} nodes) ===", expr.as_ref().len());
    for (i, node) in expr.as_ref().iter().enumerate() {
        let children: Vec<String> = node
            .children()
            .iter()
            .map(|c| format!("%{}", usize::from(*c)))
            .collect();
        if children.is_empty() {
            println!("  %{i:>3} = {node:?}");
        } else {
            println!("  %{i:>3} = {node:?}({})", children.join(", "));
        }
    }
    println!();
}

fn main() {
    let expr = build_matmul();
    print_recexpr("phase0: initial", &expr);

    let mut egraph = TensorEGraph::default();
    let root = egraph.add_expr(&expr);
    egraph.rebuild();

    let mut config = RunnerConfig {
        node_limit: 10_000,
        iter_limit: 20,
        ..RunnerConfig::default()
    };
    config.lowering = LoweringOptions::readable();
    let beam = BeamConfig::default();

    for (i, phase) in Phase::all().iter().enumerate() {
        egraph = saturate_phases(egraph, &[*phase], &config);
        let extracted = beam_extract_valid_candidates(
            &egraph,
            root,
            &beam,
            &config.device,
            &config.lowering,
            1,
        )
        .into_iter()
        .next()
        .map(|(_c, e)| e)
        .unwrap_or_else(|| beam_extract(&egraph, root, &beam).1);
        let label = format!("phase{}: {phase:?}", i + 1);

        // Per-phase e-graph stats (what the phase ADDED, even if the
        // extractor prefers an older form).
        let (th, tiled_th, redsimd, shuf, tg_load, dev_load, disp) = count_node_kinds(&egraph);
        println!(
            "  egraph: {classes} classes, {nodes} nodes  \
             [Theta={th} (tiled={tiled_th}) ReduceSimd={redsimd} Shuffle={shuf} \
             Load@TG={tg_load} Load@Dev={dev_load} Dispatch={disp}]",
            classes = egraph.number_of_classes(),
            nodes = egraph.total_size(),
        );
        print_recexpr(&label, &extracted);
    }

    // Finally: run the full tensor-expression pipeline with its beam-tuned
    // config to show the kernel the backend actually receives.
    println!("=== Full pipeline (expr -> lower -> compile) ===");
    let mut builder = TensorExprBuilder::new();
    let a = builder.input(0, Shape(vec![Dim::Lit(M), Dim::Lit(K)]), DType::F32);
    let b = builder.input(1, Shape(vec![Dim::Lit(K), Dim::Lit(N)]), DType::F32);
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let body = builder.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let matmul = builder.contraction(
        Shape(vec![Dim::Lit(M), Dim::Lit(N), Dim::Lit(K)]),
        &[
            (a, Strides(vec![i64::from(K), 0, 1])),
            (b, Strides(vec![0, 1, i64::from(N)])),
        ],
        body,
        &[(2, ReduceOp::Add)],
    );
    let tx = builder.build(matmul).expect("expr");
    let mut sc = StageConfig::default();
    sc.runner.lowering = LoweringOptions::readable();
    let pipe = StagedPipeline::new(sc);
    match pipe.lower(&tx) {
        Ok(kernel) => print_recexpr("final kernel", kernel.extracted()),
        Err(e) => println!("(full pipeline failed: {e})"),
    }
}

fn count_node_kinds(egraph: &TensorEGraph) -> (usize, usize, usize, usize, usize, usize, usize) {
    let mut theta = 0;
    let mut tiled_theta = 0;
    let mut reduce_simd = 0;
    let mut shuffle = 0;
    let mut tg_load = 0;
    let mut dev_load = 0;
    let mut dispatch = 0;
    for class in egraph.classes() {
        for n in &class.nodes {
            match n {
                TensorIr::Simd(SimdNode::Theta { .. }) => {
                    theta += 1;
                    // Structural "tiled" signal: nested Thetas imply tiling.
                    // We don't read a role tag; just count Thetas.
                    let _ = &mut tiled_theta;
                }
                TensorIr::Simd(SimdNode::ReduceSimd { .. }) => reduce_simd += 1,
                TensorIr::Simd(SimdNode::Shuffle(_)) => shuffle += 1,
                TensorIr::Simd(SimdNode::Load { tier, .. }) => {
                    let s = format!("{tier:?}");
                    if s.contains("Threadgroup") {
                        tg_load += 1;
                    } else if s.contains("Device") {
                        dev_load += 1;
                    }
                }
                TensorIr::Dispatch(DispatchNode::Dispatch { .. }) => dispatch += 1,
                _ => {}
            }
        }
    }
    (
        theta,
        tiled_theta,
        reduce_simd,
        shuffle,
        tg_load,
        dev_load,
        dispatch,
    )
}
