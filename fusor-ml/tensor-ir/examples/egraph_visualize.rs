//! Visualize the e-graph as several expressions progress through each
//! optimization phase.
//!
//! For each scenario, runs the phase's rules on the running e-graph and writes
//! a GraphViz `.dot` and rendered `.svg` to `target/egraph_phases/`, then emits
//! a self-contained `index.html` report with all SVGs inlined.
//!
//! Two scenarios:
//!   1. A full matmul run through every `Phase` — rules mostly rewrite semantic
//!      nodes into fresh dispatch subgraphs, so e-classes usually have a
//!      single member.
//!   2. A U32 scalar expression `((a + 0) + (b * 1)) * 1` run through Phase 1
//!      only — `arith-identity` produces e-classes with multiple equivalent
//!      forms, showing the standard e-graph "dotted box around multiple
//!      enodes" shape.
//!
//! Requires the `dot` binary on `PATH` (egg's `to_svg` shells out).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use tensor_ir::*;

struct Scenario {
    name: &'static str,
    description: &'static str,
    build: fn() -> egg::RecExpr<TensorIr>,
    phases: Vec<Phase>,
}

struct Snapshot {
    title: String,
    description: String,
    classes: usize,
    nodes: usize,
    multi_classes: usize,
    dot_path: PathBuf,
    svg_path: PathBuf,
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new("target/egraph_phases");
    fs::create_dir_all(out_dir)?;

    let config = RunnerConfig {
        node_limit: 5_000,
        iter_limit: 15,
        ..RunnerConfig::default()
    };

    let scenarios = vec![
        Scenario {
            name: "matmul",
            description: "Matmul (M=8, N=8, K=8). Walks all 7 phases. Rules \
                          mostly rewrite semantic nodes into fresh dispatch \
                          subgraphs, so e-classes rarely accumulate multiple \
                          members.",
            build: build_matmul_expr,
            phases: Phase::all().to_vec(),
        },
        Scenario {
            name: "flash-attention",
            description: "Flash attention: softmax(Q·Kᵀ)·V with seq=4, \
                          d_head=4. Walks all phases. The generic \
                          softmax-weighted-reduce rule can emit a single \
                          online-softmax dispatch, then the standard tiling, \
                          promotion, reduction, shuffle, and extraction \
                          rewrites keep operating on its nested IR.",
            build: build_attention_expr,
            phases: Phase::all().to_vec(),
        },
        Scenario {
            name: "arith-identity",
            description: "U32 scalar: ((a + 0) + (b * 1)) * 1. Phase 1's \
                          `arith-identity` rule unions identity-reduced forms \
                          back into the same e-class — so you should see \
                          dotted cluster boxes with 2–3 equivalent enodes.",
            build: build_arith_identity_expr,
            phases: vec![Phase::Lowering],
        },
    ];

    let mut groups = Vec::new();
    for scenario in &scenarios {
        let mut egraph = TensorEGraph::default();
        let _root = egraph.add_expr(&(scenario.build)());
        egraph.rebuild();

        let mut snapshots = Vec::new();
        snapshots.push(snapshot(
            &egraph,
            out_dir,
            scenario.name,
            "phase0",
            "initial",
        )?);

        for (idx, phase) in scenario.phases.iter().enumerate() {
            egraph = saturate_phases(egraph, &[*phase], &config);
            let tag = format!("phase{}", idx + 1);
            let label = format!("{phase:?}");
            snapshots.push(snapshot(&egraph, out_dir, scenario.name, &tag, &label)?);
        }
        groups.push((scenario, snapshots));
    }

    let report = out_dir.join("index.html");
    fs::write(&report, render_html(&groups)?)?;

    println!();
    println!("Report:  file://{}", report.canonicalize()?.display());
    Ok(())
}

fn snapshot(
    egraph: &TensorEGraph,
    out_dir: &Path,
    scenario: &str,
    tag: &str,
    label: &str,
) -> std::io::Result<Snapshot> {
    let slug = format!("{scenario}_{tag}_{}", label.to_ascii_lowercase());
    let dot_path = out_dir.join(format!("{slug}.dot"));
    let svg_path = out_dir.join(format!("{slug}.svg"));

    egraph.dot().to_dot(&dot_path)?;
    egraph.dot().to_svg(&svg_path)?;

    let classes = egraph.number_of_classes();
    let nodes = egraph.total_size();
    let multi_classes = egraph.classes().filter(|c| c.nodes.len() > 1).count();

    let description = match label {
        "initial" => "Input expression before any rewrite rules fire.".to_string(),
        _ => {
            if let Ok(phase) = parse_phase(label) {
                phase_description(phase).to_string()
            } else {
                String::new()
            }
        }
    };

    println!(
        "{scenario}/{tag} {label:<16} : classes={classes:>5}  nodes={nodes:>5}  \
         multi-node classes={multi_classes:>3}  -> {}",
        svg_path.display()
    );

    Ok(Snapshot {
        title: format!("{tag}: {label}"),
        description,
        classes,
        nodes,
        multi_classes,
        dot_path,
        svg_path,
    })
}

fn render_html(groups: &[(&Scenario, Vec<Snapshot>)]) -> std::io::Result<String> {
    let mut body = String::new();
    writeln!(body, "<!doctype html>").unwrap_or(());
    body.push_str(HTML_HEAD);

    body.push_str(
        "<header><h1>E-graph optimization phases</h1>\
         <p>Each figure is the full e-graph produced by <code>egraph.dot()</code>. \
         Dotted boxes are e-classes; when a class has multiple equivalent \
         enodes they sit inside the same dotted box.</p></header>",
    );

    body.push_str("<nav><h3>Scenarios</h3><ul>");
    for (scenario, snaps) in groups {
        let _ = write!(
            body,
            "<li><a href=\"#scenario-{name}\"><strong>{name}</strong></a> \
             <span class=\"pill\">{n} snapshots</span></li>",
            name = scenario.name,
            n = snaps.len(),
        );
    }
    body.push_str("</ul></nav>");

    for (scenario, snaps) in groups {
        let _ = write!(
            body,
            "<h2 id=\"scenario-{name}\">Scenario: {name}</h2>\
             <p class=\"scenario-desc\">{desc}</p>",
            name = scenario.name,
            desc = escape_html(scenario.description),
        );

        for s in snaps {
            let svg = fs::read_to_string(&s.svg_path)?;
            let svg = strip_xml_prolog(&svg);
            let anchor_id = format!("{}-{}", scenario.name, anchor(&s.title));

            let _ = write!(
                body,
                "<section id=\"{anchor_id}\">\
                   <h3>{title}</h3>\
                   <p class=\"desc\">{desc}</p>\
                   <div class=\"toolbar\">\
                     <button onclick=\"fit(this)\">Fit width</button>\
                     <button onclick=\"actual(this)\">Actual size</button>\
                     <button onclick=\"zoom(this, 1.25)\">+</button>\
                     <button onclick=\"zoom(this, 0.8)\">−</button>\
                     <span class=\"stats\">\
                       <span>{classes} e-classes</span>\
                       <span>{nodes} e-nodes</span>\
                       <span class=\"emph\">{multi} multi-node classes</span>\
                       <span><a href=\"{dot}\">.dot</a></span>\
                       <span><a href=\"{svg_name}\">.svg</a></span>\
                     </span>\
                   </div>\
                   <div class=\"figure\" data-fit=\"true\">{inline_svg}</div>\
                 </section>",
                title = escape_html(&s.title),
                desc = escape_html(&s.description),
                classes = s.classes,
                nodes = s.nodes,
                multi = s.multi_classes,
                dot = filename(&s.dot_path),
                svg_name = filename(&s.svg_path),
                inline_svg = svg,
            );
        }
    }

    body.push_str("</body></html>");
    Ok(body)
}

const HTML_HEAD: &str = r#"<html lang="en"><head><meta charset="utf-8">
<title>tensor_ir e-graph phases</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, "Helvetica Neue", sans-serif; margin: 2rem auto; max-width: 1400px; padding: 0 1rem; }
  h1 { margin-bottom: 0.25rem; }
  h2 { margin-top: 2.5rem; padding-top: 1rem; border-top: 2px solid #bbb; }
  header p, .scenario-desc { color: #666; }
  nav h3 { margin-bottom: 0.25rem; }
  nav ul { list-style: none; padding: 0; }
  .pill { background: #eee; color: #555; border-radius: 4px; padding: 0 0.4rem; font-size: 0.8em; margin-left: 0.4rem; }
  section { border-top: 1px solid #ddd; padding-top: 1rem; margin-top: 1.5rem; }
  h3 { margin-bottom: 0.25rem; }
  .desc { color: #555; margin-top: 0; }
  .toolbar { display: flex; gap: 0.5rem; align-items: center; margin: 0.5rem 0; flex-wrap: wrap; }
  .toolbar button { font: inherit; padding: 0.2rem 0.6rem; border: 1px solid #ccc; background: #f7f7f7; border-radius: 4px; cursor: pointer; }
  .toolbar button:hover { background: #eee; }
  .stats { display: flex; gap: 1rem; flex-wrap: wrap; color: #777; font-size: 0.9em; margin-left: auto; }
  .stats a { color: inherit; }
  .stats .emph { color: #b8521a; font-weight: 600; }
  .figure { overflow: auto; height: 80vh; resize: vertical; border: 1px solid #ddd; border-radius: 4px; background: white; padding: 0.5rem; }
  .figure svg { display: block; }
  .figure[data-fit="true"] svg { width: 100% !important; height: auto !important; }
  @media (prefers-color-scheme: dark) {
    body { background: #111; color: #ddd; }
    h2 { border-color: #555; }
    .pill { background: #333; color: #ccc; }
    header p, .scenario-desc, .desc, .stats { color: #999; }
    section, .toolbar button { border-color: #333; }
    .toolbar button { background: #1a1a1a; color: #ddd; }
    .toolbar button:hover { background: #252525; }
    .figure { background: #fafafa; border-color: #333; }
    .stats .emph { color: #ff9a3d; }
  }
</style>
<script>
  function svgOf(btn) { return btn.closest('section').querySelector('.figure svg'); }
  function zoom(btn, factor) {
    const svg = svgOf(btn);
    const fig = svg.parentElement;
    fig.dataset.fit = "false";
    const vb = svg.viewBox.baseVal;
    const cur = parseFloat(svg.dataset.zoom || "1");
    const next = Math.max(0.1, Math.min(8, cur * factor));
    svg.dataset.zoom = next;
    svg.style.width = (vb.width * next) + "px";
    svg.style.height = (vb.height * next) + "px";
  }
  function fit(btn) {
    const svg = svgOf(btn);
    svg.style.width = "";
    svg.style.height = "";
    svg.dataset.zoom = "1";
    svg.parentElement.dataset.fit = "true";
  }
  function actual(btn) {
    const svg = svgOf(btn);
    const vb = svg.viewBox.baseVal;
    svg.style.width = vb.width + "px";
    svg.style.height = vb.height + "px";
    svg.dataset.zoom = "1";
    svg.parentElement.dataset.fit = "false";
  }
</script></head><body>"#;

fn anchor(title: &str) -> String {
    title
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

fn filename(path: &Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

fn strip_xml_prolog(svg: &str) -> &str {
    let trimmed = svg.trim_start();
    let rest = trimmed
        .strip_prefix("<?xml")
        .and_then(|s| s.find("?>").map(|i| &s[i + 2..]))
        .unwrap_or(trimmed);
    let rest = rest.trim_start();
    rest.strip_prefix("<!DOCTYPE")
        .and_then(|s| s.find('>').map(|i| &s[i + 1..]))
        .unwrap_or(rest)
        .trim_start()
}

fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

fn build_matmul_expr() -> egg::RecExpr<TensorIr> {
    let mut b = IrBuilder::new();
    let m = Dim::Symbol(0);
    let n = Dim::Symbol(1);
    let k = Dim::Symbol(2);
    let a = b.input(0, Shape(vec![m.clone(), k.clone()]), DType::F32);
    let rhs = b.input(1, Shape(vec![k.clone(), n.clone()]), DType::F32);

    let tile_shape = Shape(vec![m, n.clone(), k.clone()]);
    let a_r = b.restride(
        a,
        tile_shape.clone(),
        Strides(vec![k.clone(), Dim::Const(0), Dim::Const(1)]),
    );
    let b_r = b.restride(
        rhs,
        tile_shape.clone(),
        Strides(vec![Dim::Const(0), Dim::Const(1), n]),
    );

    let arg0 = b.scalar_arg(0);
    let arg1 = b.scalar_arg(1);
    let mul_body = b.bin_op(BinaryOp::Mul, arg0, arg1);
    let mul = b.elementwise(tile_shape, &[a_r, b_r], mul_body);
    let _root = b.reduce(mul, 2, ReduceOp::Add);
    b.expr
}

/// Build a single-head flash-attention expression: `softmax(Q·Kᵀ, axis=1)·V`
/// with `Q, K, V ∈ [seq, d]`. Emitted as the decomposed form (two matmuls +
/// a scalar-lowered softmax) so the generic lowering, tiling, promotion,
/// and reduction passes can be visualized on a realistic attention-shaped
/// workload.
fn build_attention_expr() -> egg::RecExpr<TensorIr> {
    let mut b = IrBuilder::new();
    let seq = Dim::Symbol(0);
    let d = Dim::Symbol(1);

    let q = b.input(0, Shape(vec![seq.clone(), d.clone()]), DType::F32);
    let k = b.input(1, Shape(vec![seq.clone(), d.clone()]), DType::F32);
    let v = b.input(2, Shape(vec![seq.clone(), d.clone()]), DType::F32);

    // S = Q · Kᵀ: reduce over d. In the [seq, seq, d] tile, Q[i, k] has
    // strides [d, 0, 1]; Kᵀ viewed from K[j, k] has strides [0, d, 1].
    let qk_tile = Shape(vec![seq.clone(), seq.clone(), d.clone()]);
    let q_r = b.restride(
        q,
        qk_tile.clone(),
        Strides(vec![d.clone(), Dim::Const(0), Dim::Const(1)]),
    );
    let k_r = b.restride(
        k,
        qk_tile.clone(),
        Strides(vec![Dim::Const(0), d.clone(), Dim::Const(1)]),
    );
    let arg0 = b.scalar_arg(0);
    let arg1 = b.scalar_arg(1);
    let mul_body = b.bin_op(BinaryOp::Mul, arg0, arg1);
    let qk_mul = b.elementwise(qk_tile, &[q_r, k_r], mul_body);
    let scores = b.reduce(qk_mul, 2, ReduceOp::Add);

    // P = softmax(scores, axis=1). Kept decomposed so the generic softmax
    // lowering and subsequent memory/reduction rewrites stay visible.
    let scores_shape = Shape(vec![seq.clone(), seq.clone()]);
    let probs = b.softmax(scores, scores_shape, 1);

    // O = P · V: reduce over the inner seq axis. In the [seq, d, seq] tile,
    // P[i, k] has strides [seq, 0, 1]; V[k, j] has strides [0, 1, d].
    let pv_tile = Shape(vec![seq.clone(), d.clone(), seq.clone()]);
    let p_r = b.restride(
        probs,
        pv_tile.clone(),
        Strides(vec![seq, Dim::Const(0), Dim::Const(1)]),
    );
    let v_r = b.restride(
        v,
        pv_tile.clone(),
        Strides(vec![Dim::Const(0), Dim::Const(1), d]),
    );
    let arg0 = b.scalar_arg(0);
    let arg1 = b.scalar_arg(1);
    let mul_body = b.bin_op(BinaryOp::Mul, arg0, arg1);
    let pv_mul = b.elementwise(pv_tile, &[p_r, v_r], mul_body);
    let _root = b.reduce(pv_mul, 2, ReduceOp::Add);
    b.expr
}

/// Build `((a + 0) + (b * 1)) * 1` where `a` and `b` are symbolic `Param`s.
/// The `0` and `1` are U32 constants so `arith-identity` fires on each pair,
/// unioning the identity-stripped form with the BinOp form in the same
/// e-class. Using `Param` instead of literal constants matters: if `a`/`b`
/// were `Const(U32)`, the analysis would constant-fold the whole tree to a
/// single scalar before any rewrite ran.
fn build_arith_identity_expr() -> egg::RecExpr<TensorIr> {
    let mut b = IrBuilder::new();
    let a = b.scalar_arg(0);
    let bb = b.scalar_arg(1);
    let zero = b.scalar_u32(0);
    let one = b.scalar_u32(1);

    let a_plus_zero = b.bin_op(BinaryOp::Add, a, zero);
    let b_times_one = b.bin_op(BinaryOp::Mul, bb, one);
    let sum = b.bin_op(BinaryOp::Add, a_plus_zero, b_times_one);
    let _root = b.bin_op(BinaryOp::Mul, sum, one);
    b.expr
}

fn parse_phase(label: &str) -> Result<Phase, ()> {
    match label {
        "Lowering" => Ok(Phase::Lowering),
        "LateDispatch" => Ok(Phase::LateDispatch),
        "StateThreading" => Ok(Phase::StateThreading),
        _ => Err(()),
    }
}

const fn phase_description(phase: Phase) -> &'static str {
    match phase {
        Phase::Lowering => {
            "Phase 1 — lower high-level tensor ops to naive per-lane Dispatches, \
             plus scalar arith identities and softmax-style algebraic normalization."
        }
        Phase::LateDispatch => {
            "Late dispatch — unified shape saturation (tiling + promotion + \
             cooperative split + shuffle tree + fusion) with cost-driven \
             variant selection at extraction."
        }
        Phase::StateThreading => {
            "State threading — materialize state-threaded tiled dispatches \
             (explicit token flow between dispatches)."
        }
    }
}
