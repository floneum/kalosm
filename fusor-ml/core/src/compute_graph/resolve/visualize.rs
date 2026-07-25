//! Graphviz dumps of the execution graph between resolver stages.
//!
//! The lazy-graph `graphvis` renders what the user built; this renders what
//! the resolver did to it. One digraph per stage boundary makes each stage's
//! job legible: recognition collapsing a broadcast-mul + sum cluster into one
//! `matmul`, extraction folding producers into epilogues, region formation
//! and horizontal merge grouping survivors into dispatches.

use std::fmt::Write as _;
use std::path::Path;

use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use super::{ExecutionGraph, ExecutionVariant};

/// Stage boundaries, in the order [`super::run`] reaches them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Pass 1 output: one node per lazy-graph node, nothing fused.
    Built,
    /// After the pre-ingest recognizers (matmul, embedding, attention, row
    /// programs, assign chains).
    Recognized,
    /// After e-graph extraction and its deltas are applied: fusion committed,
    /// killed producers gone.
    Extracted,
    /// After multi-output elementwise regions form.
    Regions,
}

impl Stage {
    fn slug(self) -> &'static str {
        match self {
            Stage::Built => "1-built",
            Stage::Recognized => "2-recognized",
            Stage::Extracted => "3-extracted",
            Stage::Regions => "4-regions",
        }
    }
}

/// Node fill per variant family: leaves grey, recognized regions saturated,
/// the three-op core in one hue so fusion progress reads at a glance.
fn style(variant: &ExecutionVariant) -> (&'static str, &'static str) {
    match variant {
        ExecutionVariant::Tensor(_) => ("box", "#e8e8e8"),
        ExecutionVariant::QMatrix(_) => ("box", "#d8d0e8"),
        ExecutionVariant::Elementwise(_) => ("ellipse", "#cfe4f7"),
        ExecutionVariant::Reduce(_) => ("ellipse", "#a8cbe8"),
        ExecutionVariant::View(_) => ("ellipse", "#eef3f7"),
        ExecutionVariant::Assign(_) => ("ellipse", "#f7e4cf"),
        ExecutionVariant::Region(_) => ("octagon", "#bfe8cf"),
        ExecutionVariant::MatMul(_) => ("doubleoctagon", "#f7cfcf"),
        ExecutionVariant::QMatMul(_) => ("doubleoctagon", "#e8bfd8"),
        ExecutionVariant::QEmbedding(_) => ("doubleoctagon", "#e8dcbf"),
        ExecutionVariant::RowProgram(_) => ("doubleoctagon", "#cfd8f7"),
        ExecutionVariant::Attention(_) => ("doubleoctagon", "#f7bfbf"),
    }
}

fn label(variant: &ExecutionVariant) -> String {
    match variant {
        ExecutionVariant::Tensor(data) => format!("tensor\\n{:?}", data.layout().shape()),
        ExecutionVariant::QMatrix(_) => "qmatrix".to_string(),
        ExecutionVariant::Elementwise(op) => {
            format!("elementwise x{}\\n{:?}", op.inputs.len(), op.shape)
        }
        ExecutionVariant::Reduce(op) => format!("reduce {}", op.function.name()),
        ExecutionVariant::View(_) => "view".to_string(),
        ExecutionVariant::Assign(_) => "slice_assign".to_string(),
        ExecutionVariant::Region(op) => format!("region\\n{} statements", op.statements.len()),
        ExecutionVariant::MatMul(_) => "matmul".to_string(),
        ExecutionVariant::QMatMul(op) => format!(
            "qmatmul\\n{:?} -> {:?}{}{}",
            op.in_shape,
            op.out_shape,
            op.pre_element_wise_expr
                .as_ref()
                .map_or("", |_| "\\n+pre epilogue"),
            op.post_element_wise_expr
                .as_ref()
                .map_or("", |_| "\\n+post epilogue"),
        ),
        ExecutionVariant::QEmbedding(_) => "qembedding".to_string(),
        ExecutionVariant::RowProgram(op) => format!("row_program\\n{} steps", op.steps.len()),
        ExecutionVariant::Attention(op) => format!("attention\\n{:?}", op.kind),
    }
}

/// One stage's execution graph as a Graphviz digraph.
pub(crate) fn execution_graph_dot(graph: &ExecutionGraph, stage: Stage) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "digraph \"{}\" {{", stage.slug());
    let _ = writeln!(out, "  rankdir=BT;");
    let _ = writeln!(
        out,
        "  label=\"{} — {} nodes\";\n  labelloc=t;",
        stage.slug(),
        graph.node_count()
    );
    let _ = writeln!(
        out,
        "  node [style=filled, fontname=\"Helvetica\", fontsize=10];"
    );
    for idx in graph.node_indices() {
        let node = &graph[idx];
        let (shape, fill) = style(&node.variant);
        let _ = writeln!(
            out,
            "  n{} [label=\"{}\\n#{}\", shape={shape}, fillcolor=\"{fill}\"];",
            idx.index(),
            label(&node.variant),
            node.inner_idx.index(),
        );
    }
    for edge in graph.edge_references() {
        let _ = writeln!(
            out,
            "  n{} -> n{};",
            edge.source().index(),
            edge.target().index()
        );
    }
    let _ = writeln!(out, "}}");
    out
}

/// Resolves are numbered so a decode trace's per-token graphs stay apart.
static RESOLVE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Write one stage dump. Failures are traced, never fatal: this is a
/// debugging aid and must not change whether a resolve succeeds.
pub(crate) fn dump_stage(dir: &Path, graph: &ExecutionGraph, stage: Stage) {
    use std::sync::atomic::Ordering;
    let resolve = if stage == Stage::Built {
        RESOLVE.fetch_add(1, Ordering::Relaxed)
    } else {
        RESOLVE.load(Ordering::Relaxed).saturating_sub(1)
    };
    if let Err(error) = std::fs::create_dir_all(dir) {
        tracing::warn!("dump_stages: {dir:?}: {error}");
        return;
    }
    let path = dir.join(format!("resolve{resolve:04}-{}.dot", stage.slug()));
    if let Err(error) = std::fs::write(&path, execution_graph_dot(graph, stage)) {
        tracing::warn!("dump_stages: {path:?}: {error}");
    }
}
