//! Multi-output elementwise regions.
//!
//! A region is a topologically-ordered list of elementwise statements fused
//! into one kernel body: statement `k` may read statement `j < k`'s value as
//! a register instead of a materialized tensor, and every statement whose
//! value is externally live (a flush target or user-held node) writes its
//! own output buffer. This generalizes the sole-consumer nary fusion rule:
//! a producer with several consumers still fuses when *all* of its consumers
//! land in the same region — external liveness is satisfied by emitting the
//! value as one of the region's outputs rather than by blocking fusion.
//!
//! Register reads use the `extras` slot-overflow convention of
//! [`crate::nary_direct::eval_nary_expr`]: statement `j`'s value is read as
//! `NaryExpr::IndexedInput { input_idx: inputs.len() + j, indices: [] }`.
//!
//! Regions exist only inside the resolver, between `optimize_large_graph`'s
//! dense branch and lowering: the inner compute graph never contains one, so
//! the flush fingerprint recipe is unaffected by their formation.

use rustc_hash::FxHasher;
use std::hash::Hash;

use crate::DataTypeEnum;
use crate::compute_graph::NodeIndex;
use crate::nary_wise::{ElementwiseOperation, NaryExpr};

/// Statements per region: bounds live register values (each statement's
/// value is one bound register while later statements can read it).
pub(crate) const REGION_MAX_STATEMENTS: usize = 16;

#[derive(Debug, Clone)]
pub(crate) struct RegionStatement {
    /// Slots `0..inputs.len()` read region inputs (elementwise or
    /// custom-indexed, e.g. folded broadcast views). Slot `inputs.len() + j`
    /// reads statement `j`'s register value (elementwise only).
    pub(crate) expression: NaryExpr,
    pub(crate) datatype: DataTypeEnum,
    /// `Some(inner)` = externally live: stored to its own output buffer and
    /// cached under that inner-graph node. `None` = register-only.
    pub(crate) output: Option<NodeIndex>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InputReadSummary {
    pub(crate) last_reader: Option<usize>,
    pub(crate) identity_only: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ElementwiseRegionOperation {
    /// External producers, deduplicated; order defines input slots.
    pub(crate) inputs: Vec<NodeIndex>,
    /// Topologically ordered; the last statement is the region's sink and
    /// always has `output: Some(_)`.
    pub(crate) statements: Vec<RegionStatement>,
    /// Shared index space: every statement evaluates over this shape.
    pub(crate) shape: Box<[usize]>,
}

impl ElementwiseRegionOperation {
    /// A single elementwise operation as a one-statement region.
    pub(crate) fn from_nary(op: ElementwiseOperation, node: NodeIndex) -> Self {
        Self {
            inputs: op.inputs,
            shape: op.shape,
            statements: vec![RegionStatement {
                expression: op.expression,
                datatype: op.output_datatype,
                output: Some(node),
            }],
        }
    }

    /// The inverse of [`Self::from_nary`], for lowering a lone
    /// single-statement region through the standalone elementwise path.
    pub(crate) fn into_nary(self) -> Option<ElementwiseOperation> {
        let mut statements = self.statements;
        if statements.len() != 1 {
            return None;
        }
        let statement = statements.pop().expect("length checked");
        statement.output?;
        Some(ElementwiseOperation {
            inputs: self.inputs,
            expression: statement.expression,
            shape: self.shape,
            output_datatype: statement.datatype,
        })
    }

    pub(crate) fn output_count(&self) -> usize {
        self.statements
            .iter()
            .filter(|statement| statement.output.is_some())
            .count()
    }

    /// Storage bindings a merged kernel declares for this region.
    pub(crate) fn binding_count(&self) -> usize {
        self.inputs.len() + self.output_count()
    }

    pub(crate) fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        for &input in &self.inputs {
            f(input);
        }
    }

    pub(crate) fn name(&self) -> String {
        format!("region_x{}", self.statements.len())
    }

    /// Per input slot: the last statement index that reads it from memory
    /// and whether every such read is identity-indexed (element `i` of the
    /// input read exactly at output coordinate `i`). An output statement may
    /// write its buffer over an input's only when all reads of that input
    /// are identity (threads own disjoint elements, and each thread loads
    /// before it stores) and none happen after the writing statement.
    pub(crate) fn input_read_summary(&self) -> Vec<InputReadSummary> {
        let rank = self.shape.len();
        let mut summary = vec![
            InputReadSummary {
                last_reader: None,
                identity_only: true,
            };
            self.inputs.len()
        ];
        for (position, statement) in self.statements.iter().enumerate() {
            Self::scan_reads(
                &statement.expression,
                self.inputs.len(),
                rank,
                position,
                false,
                &mut summary,
            );
        }
        summary
    }

    fn scan_reads(
        expr: &NaryExpr,
        input_count: usize,
        rank: usize,
        position: usize,
        in_index: bool,
        summary: &mut [InputReadSummary],
    ) {
        match expr {
            NaryExpr::Op { children, .. } => {
                for child in children {
                    Self::scan_reads(child, input_count, rank, position, in_index, summary);
                }
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                if *input_idx < input_count {
                    let entry = &mut summary[*input_idx];
                    entry.last_reader = Some(entry.last_reader.map_or(position, |last| last.max(position)));
                    let identity = !in_index
                        && indices.len() == rank
                        && indices
                            .iter()
                            .enumerate()
                            .all(|(dim, index)| matches!(index, NaryExpr::DimIndex(d) if *d == dim));
                    if !identity {
                        entry.identity_only = false;
                    }
                }
                for index in indices {
                    Self::scan_reads(index, input_count, rank, position, true, summary);
                }
            }
            NaryExpr::DimIndex(_) | NaryExpr::Scalar(_) => {}
        }
    }

    /// Hash every field that affects the generated kernel body.
    pub(crate) fn hash_kernel_fields(&self, state: &mut FxHasher) {
        self.inputs.len().hash(state);
        self.shape.hash(state);
        self.statements.len().hash(state);
        for statement in &self.statements {
            statement.expression.hash(state);
            statement.datatype.hash(state);
            statement.output.is_some().hash(state);
        }
    }
}
