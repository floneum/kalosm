pub mod analysis;
pub mod applier;
pub mod binding;
pub mod builders;
pub mod extractor;
pub mod language;
pub mod unroll;
pub mod types {
    pub use tensor_ir_frontend::types::*;
}

pub use analysis::{TensorAnalysis, TensorData};
pub use builders::IrBuilder;
pub use extractor::{
    BeamConfig, SyntheticCostModel, beam_extract, beam_extract_candidates, greedy_extract,
};
pub use language::{DispatchNode, HighLevelNode, SimdNode, TensorIr};
pub use tensor_ir_frontend::types::*;

use std::collections::HashMap;

use egg::RecExpr;
use tensor_ir_frontend::stages::{ExprId, TensorExprNode, TensorExprProgram};

/// Type alias for the tensor IR e-graph.
pub type TensorEGraph = egg::EGraph<TensorIr, TensorAnalysis>;

/// Type alias for tensor IR rewrites.
pub type TensorRewrite = egg::Rewrite<TensorIr, TensorAnalysis>;

/// Type alias for the tensor IR runner.
pub type TensorRunner = egg::Runner<TensorIr, TensorAnalysis>;

/// Add a node to the e-graph and register it in `chosen_nodes`.
///
/// This ensures extraction prefers the exact node we just inserted, even when
/// identity rewrites merge it into an existing e-class.
pub fn add_and_choose(
    egraph: &mut egg::EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<egg::Id, TensorIr>,
    node: TensorIr,
) -> egg::Id {
    let id = egraph.add(node.clone());
    let canonical = egraph.find(id);
    chosen.insert(canonical, node);
    id
}

/// Lower a `TensorExprProgram` to the low-level `TensorIr` `RecExpr` used by
/// the egraph + beam-search pipeline.
///
/// # Errors
///
/// Returns an error if the program fails validation.
pub fn tensor_expr_to_recexpr(expr: &TensorExprProgram) -> Result<RecExpr<TensorIr>, String> {
    fn lower_node(
        program: &TensorExprProgram,
        id: ExprId,
        builder: &mut IrBuilder,
        memo: &mut HashMap<ExprId, egg::Id>,
    ) -> Result<egg::Id, String> {
        if let Some(existing) = memo.get(&id) {
            return Ok(*existing);
        }

        let lowered = match program.node(id) {
            TensorExprNode::Input { id, shape, dtype } => builder.input(*id, shape.clone(), *dtype),
            TensorExprNode::Restride {
                expr: source,
                new_shape,
                strides,
                offset,
            } => {
                let lowered_expr = lower_node(program, *source, builder, memo)?;
                builder.restride_with_offset(
                    lowered_expr,
                    new_shape.clone(),
                    strides.clone(),
                    *offset,
                )
            }
            TensorExprNode::Elementwise {
                index_space,
                inputs,
                body,
            } => {
                let lowered_inputs = inputs
                    .iter()
                    .map(|input| lower_node(program, *input, builder, memo))
                    .collect::<Result<Vec<_>, _>>()?;
                let lowered_body = lower_node(program, *body, builder, memo)?;
                builder.elementwise(index_space.clone(), &lowered_inputs, lowered_body)
            }
            TensorExprNode::Reduce {
                expr: source,
                axis,
                op,
            } => {
                let lowered_expr = lower_node(program, *source, builder, memo)?;
                builder.reduce(lowered_expr, *axis, *op)
            }
            TensorExprNode::BinOp(op, children) => {
                let lhs = lower_node(program, children[0], builder, memo)?;
                let rhs = lower_node(program, children[1], builder, memo)?;
                builder.bin_op(*op, lhs, rhs)
            }
            TensorExprNode::UnOp(op, child) => {
                let lowered_child = lower_node(program, *child, builder, memo)?;
                builder.un_op(*op, lowered_child)
            }
            TensorExprNode::TernOp(op, children) => {
                let a = lower_node(program, children[0], builder, memo)?;
                let b = lower_node(program, children[1], builder, memo)?;
                let c = lower_node(program, children[2], builder, memo)?;
                builder.tern_op(*op, a, b, c)
            }
            TensorExprNode::Const(value) => builder.scalar_lit(value.clone()),
            TensorExprNode::Arg(index) => builder.scalar_arg(*index),
        };

        memo.insert(id, lowered);
        Ok(lowered)
    }

    let mut ir_builder = IrBuilder::new();
    let mut memo = HashMap::new();
    expr.validate()?;
    let _ = lower_node(expr, expr.root(), &mut ir_builder, &mut memo)?;
    Ok(ir_builder.expr)
}
