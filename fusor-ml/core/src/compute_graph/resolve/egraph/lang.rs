//! The e-graph term language over execution-graph nodes.
//!
//! Pure operators compare by semantic payload and child e-classes and carry
//! no observation metadata. Concrete and cached leaves compare by
//! [`AllocationId`]. Effectful assignments and multi-output regions retain
//! observation identity through the [`Prov`] they were ingested for.
//!
//! Payloads are complete [`ExecutionVariant`]s held in the driver's
//! [`super::interner::PayloadTable`], referenced by [`PayloadId`]. Children
//! are the operand e-classes in the payload's dependency order
//! (`visit_dependencies` order), kept in lockstep with the payload's
//! `inputs` vector by construction.
//!
//! [`ExecutionVariant`]: super::super::ExecutionVariant

use egg::{Id, Language};
use std::hash::{Hash, Hasher};

/// Dense observation id assigned to an execution `NodeIndex`. This indexes
/// liveness and target facts; it is not pure value identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Prov(pub(super) u32);

/// Index into the driver's payload table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct PayloadId(pub(super) u32);

/// Identity of an already-existing storage allocation. Allocation-backed
/// leaves are equal only when they name the same buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct AllocationId(pub(super) usize);

/// One alternative form of one execution node.
///
/// The variants mirror `ExecutionVariant`, plus `Boundary` for inputs that
/// were already cached when the resolve started (the `resolved_set`): those
/// are opaque leaves exactly like `build_execution_graph` excluding them —
/// no rule may see through a cached boundary.
#[derive(Debug, Clone)]
pub(super) enum FusorLang {
    /// Concrete tensor data already bound to a buffer.
    TensorLeaf(AllocationId),
    /// A node cached before this resolve began; contents opaque.
    Boundary(AllocationId),
    /// Quantized-matrix leaf (`DequantizeOperation` payload).
    QMatrixLeaf(AllocationId, PayloadId),
    Elementwise(PayloadId, Box<[Id]>),
    Reduce(PayloadId, Box<[Id]>),
    View(PayloadId, [Id; 1]),
    Assign(Prov, PayloadId, [Id; 2]),
    MatMul(PayloadId, [Id; 2]),
    QMatMul(PayloadId, Box<[Id]>),
    QEmbedding(PayloadId, [Id; 1]),
    /// Structurally comparable fused row program, including attention.
    RowProgram(PayloadId, Box<[Id]>),
    /// Multi-output elementwise region (only after region formation).
    Region(Prov, PayloadId, Box<[Id]>),
}

impl PartialEq for FusorLang {
    fn eq(&self, other: &Self) -> bool {
        use FusorLang::*;
        match (self, other) {
            (TensorLeaf(a), TensorLeaf(b)) | (Boundary(a), Boundary(b)) => a == b,
            (QMatrixLeaf(aa, a), QMatrixLeaf(ba, b)) => aa == ba && a == b,
            (Elementwise(a, ac), Elementwise(b, bc))
            | (Reduce(a, ac), Reduce(b, bc))
            | (QMatMul(a, ac), QMatMul(b, bc))
            | (RowProgram(a, ac), RowProgram(b, bc)) => a == b && ac == bc,
            (View(a, ac), View(b, bc)) | (QEmbedding(a, ac), QEmbedding(b, bc)) => {
                a == b && ac == bc
            }
            (MatMul(a, ac), MatMul(b, bc)) => a == b && ac == bc,
            (Assign(ap, a, ac), Assign(bp, b, bc)) => ap == bp && a == b && ac == bc,
            (Region(ap, a, ac), Region(bp, b, bc)) => ap == bp && a == b && ac == bc,
            _ => false,
        }
    }
}

impl Eq for FusorLang {}

impl PartialOrd for FusorLang {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FusorLang {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let tag = |node: &Self| match node {
            Self::TensorLeaf(..) => 0u8,
            Self::Boundary(..) => 1,
            Self::QMatrixLeaf(..) => 2,
            Self::Elementwise(..) => 3,
            Self::Reduce(..) => 4,
            Self::View(..) => 5,
            Self::Assign(..) => 6,
            Self::MatMul(..) => 7,
            Self::QMatMul(..) => 8,
            Self::QEmbedding(..) => 9,
            Self::RowProgram(..) => 10,
            Self::Region(..) => 11,
        };
        tag(self)
            .cmp(&tag(other))
            .then_with(|| match (self, other) {
                (Self::TensorLeaf(a), Self::TensorLeaf(b))
                | (Self::Boundary(a), Self::Boundary(b)) => a.cmp(b),
                (Self::QMatrixLeaf(aa, a), Self::QMatrixLeaf(ba, b)) => {
                    aa.cmp(ba).then_with(|| a.cmp(b))
                }
                (Self::Assign(ap, a, ac), Self::Assign(bp, b, bc)) => {
                    ap.cmp(bp).then_with(|| a.cmp(b)).then_with(|| ac.cmp(bc))
                }
                (Self::Region(ap, a, ac), Self::Region(bp, b, bc)) => {
                    ap.cmp(bp).then_with(|| a.cmp(b)).then_with(|| ac.cmp(bc))
                }
                _ => self
                    .payload()
                    .cmp(&other.payload())
                    .then_with(|| self.children().cmp(other.children())),
            })
    }
}

impl Hash for FusorLang {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::TensorLeaf(allocation) | Self::Boundary(allocation) => allocation.hash(state),
            Self::QMatrixLeaf(allocation, payload) => {
                allocation.hash(state);
                payload.hash(state);
            }
            Self::Assign(prov, payload, children) => {
                prov.hash(state);
                payload.hash(state);
                children.hash(state);
            }
            Self::Region(prov, payload, children) => {
                prov.hash(state);
                payload.hash(state);
                children.hash(state);
            }
            _ => {
                self.payload().hash(state);
                self.children().hash(state);
            }
        }
    }
}

impl FusorLang {
    pub(super) fn payload(&self) -> Option<PayloadId> {
        match self {
            Self::TensorLeaf(..) | Self::Boundary(..) => None,
            Self::QMatrixLeaf(_, payload)
            | Self::Elementwise(payload, _)
            | Self::Reduce(payload, _)
            | Self::View(payload, _)
            | Self::Assign(_, payload, _)
            | Self::MatMul(payload, _)
            | Self::QMatMul(payload, _)
            | Self::QEmbedding(payload, _)
            | Self::RowProgram(payload, _)
            | Self::Region(_, payload, _) => Some(*payload),
        }
    }
}

impl Language for FusorLang {
    /// Everything `matches` compares below the children: variant, leaf
    /// allocation, payload, and observation identity for effectful nodes.
    /// Discriminant equality must coincide with `matches`.
    type Discriminant = (
        std::mem::Discriminant<FusorLang>,
        Option<AllocationId>,
        Option<PayloadId>,
        Option<Prov>,
    );

    fn discriminant(&self) -> Self::Discriminant {
        let allocation = match self {
            Self::TensorLeaf(allocation)
            | Self::Boundary(allocation)
            | Self::QMatrixLeaf(allocation, _) => Some(*allocation),
            _ => None,
        };
        let prov = match self {
            Self::Assign(prov, _, _) | Self::Region(prov, _, _) => Some(*prov),
            _ => None,
        };
        (
            std::mem::discriminant(self),
            allocation,
            self.payload(),
            prov,
        )
    }

    fn matches(&self, other: &Self) -> bool {
        use FusorLang::*;
        match (self, other) {
            (TensorLeaf(a), TensorLeaf(b)) | (Boundary(a), Boundary(b)) => a == b,
            (QMatrixLeaf(aa, a), QMatrixLeaf(ba, b)) => aa == ba && a == b,
            (Assign(ap, a, _), Assign(bp, b, _)) | (Region(ap, a, _), Region(bp, b, _)) => {
                ap == bp && a == b
            }
            _ => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
                    && self.payload() == other.payload()
            }
        }
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::TensorLeaf(..) | Self::Boundary(..) | Self::QMatrixLeaf(..) => &[],
            Self::Elementwise(_, children)
            | Self::Reduce(_, children)
            | Self::QMatMul(_, children)
            | Self::RowProgram(_, children)
            | Self::Region(_, _, children) => children,
            Self::View(_, children) | Self::QEmbedding(_, children) => children,
            Self::MatMul(_, children) => children,
            Self::Assign(_, _, children) => children,
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::TensorLeaf(..) | Self::Boundary(..) | Self::QMatrixLeaf(..) => &mut [],
            Self::Elementwise(_, children)
            | Self::Reduce(_, children)
            | Self::QMatMul(_, children)
            | Self::RowProgram(_, children)
            | Self::Region(_, _, children) => children,
            Self::View(_, children) | Self::QEmbedding(_, children) => children,
            Self::MatMul(_, children) => children,
            Self::Assign(_, _, children) => children,
        }
    }
}
