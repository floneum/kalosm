//! Shared field traversal for symbol binding and structural cache keys.

use crate::carrier::{Carrier, SlotTy};
use crate::ir::Op;
use crate::ir::launch::{AccessPlan, Launch, Operand};
use crate::ir::logical::{LeafKind, Logical};
use crate::scalar::ScalarExpr;
use crate::shape::Dim;

/// Scalar hooks receive complete bodies so callers can memoize shared trees.
/// Operand and leaf hooks run after their nested fields have been visited.
pub trait VisitMut {
    fn dim(&mut self, _dim: &mut Dim) {}
    fn scalar(&mut self, _expr: &mut ScalarExpr) {}
    fn operand(&mut self, _operand: &mut Operand) {}
    fn leaf(&mut self, _leaf: &mut LeafKind) {}
}

fn dims(dims: &mut [Dim], visitor: &mut impl VisitMut) {
    for dim in dims {
        visitor.dim(dim);
    }
}

fn carrier(carrier: &mut Carrier, visitor: &mut impl VisitMut) {
    for slot in &mut carrier.slots {
        if let SlotTy::Vector(dim) = slot {
            visitor.dim(dim);
        }
    }
    for expr in carrier.lift.iter_mut().chain(&mut carrier.merge) {
        visitor.scalar(expr);
    }
}

fn operands(operands: &mut [Operand], visitor: &mut impl VisitMut) {
    for operand in operands {
        operand.layout.visit_dims_mut(&mut |d| visitor.dim(d));
        if let AccessPlan::Pack { into } = &mut operand.access {
            into.visit_dims_mut(&mut |d| visitor.dim(d));
        }
        visitor.operand(operand);
    }
}

impl Op {
    pub fn visit_mut(&mut self, visitor: &mut impl VisitMut) {
        match self {
            Op::Union(..) => {}
            Op::Logical(op) => match op {
                Logical::Leaf(leaf) => {
                    match leaf {
                        LeafKind::Buffer { shape, .. }
                        | LeafKind::Param { shape, .. }
                        | LeafKind::Const { shape, .. } => dims(shape, visitor),
                        LeafKind::Quantized { shape, .. } => dims(shape, visitor),
                        LeafKind::Uniform { .. } => {}
                    }
                    visitor.leaf(leaf);
                }
                Logical::Map { expr, .. } => visitor.scalar(expr),
                Logical::Fold { carrier: c, .. } => carrier(c, visitor),
                Logical::Restride { specs, .. } => {
                    for spec in specs {
                        visitor.dim(&mut spec.size);
                        visitor.dim(&mut spec.offset);
                    }
                }
                Logical::Contract { .. }
                | Logical::Window { .. }
                | Logical::Gather { .. }
                | Logical::Scatter { .. }
                | Logical::Dequant { .. }
                | Logical::Project { .. } => {}
            },
            Op::Launch(op) => launch(op, visitor, None),
        }
    }
}

fn launch(op: &mut Launch, visitor: &mut impl VisitMut, generated: Option<usize>) {
    match op {
        Launch::Map {
            space, body, ops, ..
        } => {
            dims(&mut space.dims, visitor);
            visitor.scalar(body);
            operands(ops, visitor);
        }
        Launch::Fold {
            space,
            carrier: c,
            post,
            ops,
            ..
        } => {
            dims(&mut space.dims, visitor);
            carrier(c, visitor);
            for expr in post {
                visitor.scalar(expr);
            }
            for (i, operand) in ops.iter_mut().enumerate() {
                if Some(i) == generated {
                    operand.layout.visit_dims_mut(&mut |d| visitor.dim(d));
                } else {
                    operands(std::slice::from_mut(operand), visitor);
                }
            }
        }
        Launch::StreamFold {
            producer,
            fold,
            operand,
            ..
        } => {
            launch(producer, visitor, None);
            launch(fold, visitor, Some(*operand as usize));
        }
        Launch::Contract {
            output,
            m,
            n,
            k,
            batch,
            post,
            a,
            b,
            ..
        } => {
            dims(&mut output.dims, visitor);
            for dim in [m, n, k, batch] {
                visitor.dim(dim);
            }
            for side in [a, b] {
                visitor.scalar(&mut side.pre);
                operands(&mut side.ops, visitor);
            }
            visitor.scalar(post);
        }
        Launch::Gather { space, ops, .. } | Launch::Scatter { space, ops, .. } => {
            dims(&mut space.dims, visitor);
            operands(ops, visitor);
        }
        Launch::Slab { .. } | Launch::Group { .. } => {}
    }
}
