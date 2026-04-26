use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::TensorIr;
use crate::types::{BinaryOp, DType, ScalarValue};

pub(super) fn build_all() -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    vec![
        Rewrite::new(
            "arith-identity",
            SimpleEclassSearcher::new(|egraph, eclass| {
                if egraph[egraph.find(eclass)].data.dtype != Some(DType::U32) {
                    return false;
                }
                egraph[eclass].iter().any(|node| {
                    let TensorIr::BinOp(name, args) = node else {
                        return false;
                    };
                    if args.len() != 2 {
                        return false;
                    }
                    let lc = match &egraph[args[0]].data.constant {
                        Some(ScalarValue::U32(v)) => Some(*v),
                        _ => None,
                    };
                    let rc = match &egraph[args[1]].data.constant {
                        Some(ScalarValue::U32(v)) => Some(*v),
                        _ => None,
                    };
                    match name {
                        BinaryOp::Add => lc == Some(0) || rc == Some(0),
                        BinaryOp::Mul => {
                            lc == Some(0) || rc == Some(0) || lc == Some(1) || rc == Some(1)
                        }
                        BinaryOp::Div | BinaryOp::Mod => rc == Some(1) || lc == Some(0),
                        _ => false,
                    }
                })
            }),
            crate::applier::AdaptedApplier(ArithIdentityApplier),
        )
        .unwrap(),
    ]
}

struct ArithIdentityApplier;

impl crate::applier::TypedApplier for ArithIdentityApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        if egraph[egraph.find(eclass)].data.dtype != Some(DType::U32) {
            return vec![];
        }

        let node = egraph[eclass]
            .iter()
            .find(|n| matches!(n, TensorIr::BinOp(_, args) if args.len() == 2))
            .cloned();

        let Some(TensorIr::BinOp(name, args)) = node else {
            return vec![];
        };

        let lc = match &egraph[args[0]].data.constant {
            Some(ScalarValue::U32(v)) => Some(*v),
            _ => None,
        };
        let rc = match &egraph[args[1]].data.constant {
            Some(ScalarValue::U32(v)) => Some(*v),
            _ => None,
        };

        let simplified = match name {
            BinaryOp::Add => {
                if lc == Some(0) {
                    Some(args[1])
                } else if rc == Some(0) {
                    Some(args[0])
                } else {
                    None
                }
            }
            BinaryOp::Mul => {
                if lc == Some(0) || rc == Some(0) {
                    let zero = egraph.add(TensorIr::Const(ScalarValue::U32(0)));
                    Some(zero)
                } else if lc == Some(1) {
                    Some(args[1])
                } else if rc == Some(1) {
                    Some(args[0])
                } else {
                    None
                }
            }
            BinaryOp::Div => {
                if rc == Some(1) {
                    Some(args[0])
                } else if lc == Some(0) {
                    let zero = egraph.add(TensorIr::Const(ScalarValue::U32(0)));
                    Some(zero)
                } else {
                    None
                }
            }
            BinaryOp::Mod => {
                if rc == Some(1) || lc == Some(0) {
                    let zero = egraph.add(TensorIr::Const(ScalarValue::U32(0)));
                    Some(zero)
                } else {
                    None
                }
            }
            _ => None,
        };

        simplified.map_or_else(Vec::new, |result| {
            egraph.union(eclass, result);
            vec![result]
        })
    }
}
