//! The e-graph term language over execution-graph nodes.
//!
//! Pure operators compare by semantic payload and child e-classes; `Prov` is
//! observation metadata and does not participate in their identity. Concrete
//! and cached leaves compare by [`AllocationId`]. Effectful assignments and
//! multi-output regions retain observation identity.
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
    TensorLeaf(Prov, AllocationId),
    /// A node cached before this resolve began; contents opaque.
    Boundary(Prov, AllocationId),
    /// Quantized-matrix leaf (`DequantizeOperation` payload).
    QMatrixLeaf(Prov, AllocationId, PayloadId),
    Elementwise(Prov, PayloadId, Box<[Id]>),
    Reduce(Prov, PayloadId, Box<[Id]>),
    View(Prov, PayloadId, [Id; 1]),
    Assign(Prov, PayloadId, [Id; 2]),
    MatMul(Prov, PayloadId, [Id; 2]),
    QMatMul(Prov, PayloadId, Box<[Id]>),
    QEmbedding(Prov, PayloadId, [Id; 1]),
    /// Structurally comparable fused row program, including attention.
    RowProgram(Prov, PayloadId, Box<[Id]>),
    /// Multi-output elementwise region (only after region formation).
    Region(Prov, PayloadId, Box<[Id]>),
}

impl PartialEq for FusorLang {
    fn eq(&self, other: &Self) -> bool {
        use FusorLang::*;
        match (self, other) {
            (TensorLeaf(_, a), TensorLeaf(_, b)) | (Boundary(_, a), Boundary(_, b)) => a == b,
            (QMatrixLeaf(_, aa, a), QMatrixLeaf(_, ba, b)) => aa == ba && a == b,
            (Elementwise(_, a, ac), Elementwise(_, b, bc))
            | (Reduce(_, a, ac), Reduce(_, b, bc))
            | (QMatMul(_, a, ac), QMatMul(_, b, bc))
            | (RowProgram(_, a, ac), RowProgram(_, b, bc)) => a == b && ac == bc,
            (View(_, a, ac), View(_, b, bc)) | (QEmbedding(_, a, ac), QEmbedding(_, b, bc)) => {
                a == b && ac == bc
            }
            (MatMul(_, a, ac), MatMul(_, b, bc)) => a == b && ac == bc,
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
                (Self::TensorLeaf(_, a), Self::TensorLeaf(_, b))
                | (Self::Boundary(_, a), Self::Boundary(_, b)) => a.cmp(b),
                (Self::QMatrixLeaf(_, aa, a), Self::QMatrixLeaf(_, ba, b)) => {
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
            Self::TensorLeaf(_, allocation) | Self::Boundary(_, allocation) => {
                allocation.hash(state)
            }
            Self::QMatrixLeaf(_, allocation, payload) => {
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
    pub(super) fn prov(&self) -> Prov {
        match self {
            Self::TensorLeaf(prov, _)
            | Self::Boundary(prov, _)
            | Self::QMatrixLeaf(prov, _, _)
            | Self::Elementwise(prov, _, _)
            | Self::Reduce(prov, _, _)
            | Self::View(prov, _, _)
            | Self::Assign(prov, _, _)
            | Self::MatMul(prov, _, _)
            | Self::QMatMul(prov, _, _)
            | Self::QEmbedding(prov, _, _)
            | Self::RowProgram(prov, _, _)
            | Self::Region(prov, _, _) => *prov,
        }
    }

    pub(super) fn payload(&self) -> Option<PayloadId> {
        match self {
            Self::TensorLeaf(..) | Self::Boundary(..) => None,
            Self::QMatrixLeaf(_, _, payload)
            | Self::Elementwise(_, payload, _)
            | Self::Reduce(_, payload, _)
            | Self::View(_, payload, _)
            | Self::Assign(_, payload, _)
            | Self::MatMul(_, payload, _)
            | Self::QMatMul(_, payload, _)
            | Self::QEmbedding(_, payload, _)
            | Self::RowProgram(_, payload, _)
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
            Self::TensorLeaf(_, allocation)
            | Self::Boundary(_, allocation)
            | Self::QMatrixLeaf(_, allocation, _) => Some(*allocation),
            _ => None,
        };
        let prov = match self {
            Self::Assign(prov, _, _) | Self::Region(prov, _, _) => Some(*prov),
            _ => None,
        };
        (std::mem::discriminant(self), allocation, self.payload(), prov)
    }

    fn matches(&self, other: &Self) -> bool {
        use FusorLang::*;
        match (self, other) {
            (TensorLeaf(_, a), TensorLeaf(_, b)) | (Boundary(_, a), Boundary(_, b)) => a == b,
            (QMatrixLeaf(_, aa, a), QMatrixLeaf(_, ba, b)) => aa == ba && a == b,
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
            Self::Elementwise(_, _, children)
            | Self::Reduce(_, _, children)
            | Self::QMatMul(_, _, children)
            | Self::RowProgram(_, _, children)
            | Self::Region(_, _, children) => children,
            Self::View(_, _, children) | Self::QEmbedding(_, _, children) => children,
            Self::Assign(_, _, children) | Self::MatMul(_, _, children) => children,
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::TensorLeaf(..) | Self::Boundary(..) | Self::QMatrixLeaf(..) => &mut [],
            Self::Elementwise(_, _, children)
            | Self::Reduce(_, _, children)
            | Self::QMatMul(_, _, children)
            | Self::RowProgram(_, _, children)
            | Self::Region(_, _, children) => children,
            Self::View(_, _, children) | Self::QEmbedding(_, _, children) => children,
            Self::Assign(_, _, children) | Self::MatMul(_, _, children) => children,
        }
    }
}
