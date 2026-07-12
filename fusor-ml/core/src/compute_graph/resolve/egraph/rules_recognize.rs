//! Native egg recognition rules.
//!
//! Search is expressed through [`egg::Searcher`] and application through
//! [`egg::Applier`], so egg owns whole-egraph matching, saturation detection,
//! rebuilds, and scheduling.  Operation payloads remain Fusor's interned
//! objects; custom appliers rebuild concrete variants from matched e-classes.

use egg::{
    Applier, EGraph, Id, Language, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var,
};

use super::super::ExecutionVariant;
use super::super::recognize::{match_contraction, try_unflatten_matmul_input_with};
use super::analysis::FusorAnalysis;
use super::ingest::enode_for;
use super::interner::{rebind_variant_dependencies, variant_dependencies};
use super::lang::{FusorLang, Prov};
use crate::compute_graph::NodeIndex;
use crate::nary_wise::NaryExpr;

type FusorEGraph = EGraph<FusorLang, FusorAnalysis>;

#[derive(Debug, Clone, Copy)]
enum RecognitionKind {
    Contraction,
    QEmbedding,
}

impl RecognitionKind {
    fn root_matches(self, node: &FusorLang) -> bool {
        match self {
            Self::Contraction => matches!(node, FusorLang::Reduce(..)),
            Self::QEmbedding => matches!(node, FusorLang::Elementwise(..)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RecognitionSearcher(RecognitionKind);

impl Searcher<FusorLang, FusorAnalysis> for RecognitionSearcher {
    fn search_eclass_with_limit(
        &self,
        egraph: &FusorEGraph,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, FusorLang>> {
        if limit == 0
            || !egraph[egraph.find(eclass)]
                .nodes
                .iter()
                .any(|node| self.0.root_matches(node))
        {
            return None;
        }
        Some(SearchMatches {
            eclass: egraph.find(eclass),
            substs: vec![Subst::default()],
            ast: None,
        })
    }

    fn vars(&self) -> Vec<Var> {
        Vec::new()
    }
}

#[derive(Debug, Clone, Copy)]
struct RecognitionApplier(RecognitionKind);

impl Applier<FusorLang, FusorAnalysis> for RecognitionApplier {
    fn apply_one(
        &self,
        egraph: &mut FusorEGraph,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<FusorLang>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let alternatives = match self.0 {
            RecognitionKind::Contraction => contraction_alternatives(egraph, eclass),
            RecognitionKind::QEmbedding => qembedding_alternatives(egraph, eclass),
        };
        alternatives
            .into_iter()
            .filter_map(|(prov, variant)| add_alternative(egraph, eclass, prov, variant))
            .collect()
    }
}

pub(super) fn rules() -> Vec<Rewrite<FusorLang, FusorAnalysis>> {
    [
        ("recognize-contraction", RecognitionKind::Contraction),
        ("recognize-qembedding", RecognitionKind::QEmbedding),
    ]
    .into_iter()
    .map(|(name, kind)| {
        Rewrite::new(name, RecognitionSearcher(kind), RecognitionApplier(kind))
            .expect("recognition rewrite has matching empty substitutions")
    })
    .collect()
}

fn representative_inner(egraph: &FusorEGraph, class: Id) -> Option<NodeIndex> {
    let class = egraph.find(class);
    egraph[class]
        .nodes
        .iter()
        .map(|node| egraph.analysis.facts_of(node.prov()).inner)
        .min_by_key(|node| node.index())
}

fn variant_for_enode(egraph: &FusorEGraph, enode: &FusorLang) -> Option<ExecutionVariant> {
    let payload = enode.payload()?;
    let mut variant = egraph.analysis.payloads.get(payload).clone();
    let children = enode
        .children()
        .iter()
        .map(|&child| representative_inner(egraph, child))
        .collect::<Option<Vec<_>>>()?;
    rebind_variant_dependencies(&mut variant, &children);
    Some(variant)
}

fn class_for_inner(egraph: &FusorEGraph, inner: NodeIndex) -> Option<Id> {
    egraph
        .analysis
        .class_of_inner
        .get(&inner)
        .map(|&class| egraph.find(class))
}

fn is_cached(egraph: &FusorEGraph, inner: NodeIndex) -> bool {
    let Some(class) = class_for_inner(egraph, inner) else {
        return false;
    };
    egraph[class]
        .nodes
        .iter()
        .any(|node| egraph.analysis.facts_of(node.prov()).exec.is_none())
}

fn is_externally_live(egraph: &FusorEGraph, inner: NodeIndex) -> bool {
    let Some(class) = class_for_inner(egraph, inner) else {
        return false;
    };
    egraph[class]
        .nodes
        .iter()
        .any(|node| egraph.analysis.facts_of(node.prov()).externally_live)
}

fn contraction_alternatives(egraph: &FusorEGraph, root: Id) -> Vec<(Prov, ExecutionVariant)> {
    let root = egraph.find(root);
    let root_nodes = egraph[root].nodes.clone();
    let planner = egraph
        .analysis
        .planner
        .as_ref()
        .expect("ingested egraph has planner context")
        .clone();
    let mut alternatives = Vec::new();
    for root_node in root_nodes {
        let Some(ExecutionVariant::Reduce(reduce)) = variant_for_enode(egraph, &root_node) else {
            continue;
        };
        let Some(value) = reduce.plain_input() else {
            continue;
        };
        let Some(value_class) = class_for_inner(egraph, value) else {
            continue;
        };
        for value_node in egraph[value_class].nodes.clone() {
            let Some(ExecutionVariant::Elementwise(nary)) = variant_for_enode(egraph, &value_node)
            else {
                continue;
            };
            let Some(contraction) = match_contraction(&reduce, &nary) else {
                continue;
            };
            let variant = if let Some((operation, _)) =
                contraction.to_q_mat_mul(|node| planner.dequantize(node))
            {
                ExecutionVariant::QMatMul(Box::new(operation))
            } else if let Some((mut operation, _)) = contraction.to_mat_mul(&planner.device) {
                try_unflatten_matmul_input_with(
                    &mut operation,
                    &planner.device,
                    |node| is_cached(egraph, node),
                    |node| is_externally_live(egraph, node),
                    |node| planner.view(node).cloned(),
                );
                ExecutionVariant::MatMul(operation)
            } else {
                continue;
            };
            alternatives.push((root_node.prov(), variant));
        }
    }
    alternatives
}

fn qembedding_alternatives(egraph: &FusorEGraph, root: Id) -> Vec<(Prov, ExecutionVariant)> {
    let root = egraph.find(root);
    let root_nodes = egraph[root].nodes.clone();
    let planner = egraph
        .analysis
        .planner
        .as_ref()
        .expect("ingested egraph has planner context")
        .clone();
    let mut alternatives = Vec::new();
    for root_node in root_nodes {
        let Some(ExecutionVariant::Elementwise(nary)) = variant_for_enode(egraph, &root_node)
        else {
            continue;
        };
        if nary.inputs.len() != 2 || nary.shape.len() != 2 {
            continue;
        }
        let gather = match &nary.expression {
            NaryExpr::Op { children, function }
                if function.op == crate::nary_wise::NaryOp::Cast && children.len() == 1 =>
            {
                &children[0]
            }
            expr => expr,
        };
        let NaryExpr::IndexedInput {
            input_idx: 0,
            indices,
        } = gather
        else {
            continue;
        };
        let [row, NaryExpr::DimIndex(1)] = indices.as_slice() else {
            continue;
        };
        let NaryExpr::IndexedInput {
            input_idx: 1,
            indices: row_indices,
        } = row
        else {
            continue;
        };
        if row_indices.as_slice() != [NaryExpr::DimIndex(0)] {
            continue;
        }
        let Some(dequantize) = planner.dequantize(nary.inputs[0]) else {
            continue;
        };
        if crate::quantized::dequantize::quant_format(&dequantize.matrix).is_none()
            || dequantize.matrix.shape().len() != 2
            || dequantize.matrix.shape()[1] != nary.shape[1]
        {
            continue;
        }
        alternatives.push((
            root_node.prov(),
            ExecutionVariant::QEmbedding(crate::quantized::embedding::QEmbeddingOperation::new(
                nary.inputs[1],
                nary.shape[0],
                dequantize.matrix.clone(),
                nary.output_datatype,
            )),
        ));
    }
    alternatives
}

fn add_alternative(
    egraph: &mut FusorEGraph,
    root: Id,
    prov: Prov,
    variant: ExecutionVariant,
) -> Option<Id> {
    let dependencies = variant_dependencies(&variant);
    let children = dependencies
        .iter()
        .map(|dependency| class_for_inner(egraph, *dependency))
        .collect::<Option<Vec<_>>>()?;
    let enode = enode_for(&mut egraph.analysis, &variant, prov, children, true, None);
    let alternative = egraph.add(enode);
    egraph.union(root, alternative).then_some(alternative)
}
