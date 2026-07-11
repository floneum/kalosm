//! The e-graph term language over execution-graph nodes.
//!
//! Every e-node is salted with [`Prov`], the ingestion position of the
//! execution node it denotes. Provenance participates in equality and
//! hashing, so hash-consing never unifies e-nodes minted for distinct
//! execution nodes: an e-class is exactly one execution node's set of
//! alternative forms. Congruence closure is intentionally inert across
//! nodes; only explicit `union` calls (rule appliers joining an alternative
//! into its root's class) merge, and idempotent rule re-application dedups
//! through the payload interner.
//!
//! Payloads are complete [`ExecutionVariant`]s held in the driver's
//! [`super::interner::PayloadTable`], referenced by [`PayloadId`]. Children
//! are the operand e-classes in the payload's dependency order
//! (`visit_dependencies` order), kept in lockstep with the payload's
//! `inputs` vector by construction.
//!
//! [`ExecutionVariant`]: super::super::ExecutionVariant

use egg::{Id, Language};

/// Ingestion position of the execution node an e-node denotes. Dense,
/// deterministic (DFS discovery order), and unique per execution node and
/// per cached-boundary leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Prov(pub(super) u32);

/// Index into the driver's payload table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct PayloadId(pub(super) u32);

/// One alternative form of one execution node.
///
/// The variants mirror `ExecutionVariant`, plus `Boundary` for inputs that
/// were already cached when the resolve started (the `resolved_set`): those
/// are opaque leaves exactly like `build_execution_graph` excluding them —
/// no rule may see through a cached boundary.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum FusorLang {
    /// Concrete tensor data already bound to a buffer.
    TensorLeaf(Prov),
    /// A node cached before this resolve began; contents opaque.
    Boundary(Prov),
    /// Quantized-matrix leaf (`DequantizeOperation` payload).
    QMatrixLeaf(Prov, PayloadId),
    Elementwise(Prov, PayloadId, Box<[Id]>),
    Reduce(Prov, PayloadId, Box<[Id]>),
    View(Prov, PayloadId, [Id; 1]),
    Assign(Prov, PayloadId, [Id; 2]),
    MatMul(Prov, PayloadId, [Id; 2]),
    QMatMul(Prov, PayloadId, Box<[Id]>),
    QEmbedding(Prov, PayloadId, [Id; 1]),
    /// Opaque fused program (attention row program and future GraphOps).
    GraphOp(Prov, PayloadId, Box<[Id]>),
    /// Multi-output elementwise region (only after region formation).
    Region(Prov, PayloadId, Box<[Id]>),
}

impl FusorLang {
    pub(super) fn prov(&self) -> Prov {
        match self {
            Self::TensorLeaf(prov)
            | Self::Boundary(prov)
            | Self::QMatrixLeaf(prov, _)
            | Self::Elementwise(prov, _, _)
            | Self::Reduce(prov, _, _)
            | Self::View(prov, _, _)
            | Self::Assign(prov, _, _)
            | Self::MatMul(prov, _, _)
            | Self::QMatMul(prov, _, _)
            | Self::QEmbedding(prov, _, _)
            | Self::GraphOp(prov, _, _)
            | Self::Region(prov, _, _) => *prov,
        }
    }

    pub(super) fn payload(&self) -> Option<PayloadId> {
        match self {
            Self::TensorLeaf(_) | Self::Boundary(_) => None,
            Self::QMatrixLeaf(_, payload)
            | Self::Elementwise(_, payload, _)
            | Self::Reduce(_, payload, _)
            | Self::View(_, payload, _)
            | Self::Assign(_, payload, _)
            | Self::MatMul(_, payload, _)
            | Self::QMatMul(_, payload, _)
            | Self::QEmbedding(_, payload, _)
            | Self::GraphOp(_, payload, _)
            | Self::Region(_, payload, _) => Some(*payload),
        }
    }
}

impl Language for FusorLang {
    fn matches(&self, other: &Self) -> bool {
        // Same operator: everything except children. Children are compared
        // separately by the e-graph.
        std::mem::discriminant(self) == std::mem::discriminant(other)
            && self.prov() == other.prov()
            && self.payload() == other.payload()
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::TensorLeaf(_) | Self::Boundary(_) | Self::QMatrixLeaf(_, _) => &[],
            Self::Elementwise(_, _, children)
            | Self::Reduce(_, _, children)
            | Self::QMatMul(_, _, children)
            | Self::GraphOp(_, _, children)
            | Self::Region(_, _, children) => children,
            Self::View(_, _, children) | Self::QEmbedding(_, _, children) => children,
            Self::Assign(_, _, children) | Self::MatMul(_, _, children) => children,
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::TensorLeaf(_) | Self::Boundary(_) | Self::QMatrixLeaf(_, _) => &mut [],
            Self::Elementwise(_, _, children)
            | Self::Reduce(_, _, children)
            | Self::QMatMul(_, _, children)
            | Self::GraphOp(_, _, children)
            | Self::Region(_, _, children) => children,
            Self::View(_, _, children) | Self::QEmbedding(_, _, children) => children,
            Self::Assign(_, _, children) | Self::MatMul(_, _, children) => children,
        }
    }
}
