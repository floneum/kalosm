//! Small actual KernelIr programs for comparing memory layouts.
use crate::{Error, Result, require};
use fusor_ir::{dtype::NumericContract, ir::kernel::*, scalar::BinOp};
use std::sync::Arc;

pub const NAMES: &[&str] = &[
    "coalesced",
    "strided",
    "broadcast",
    "reuse",
    "bank-conflict",
    "padded",
];

pub fn kernel(name: &str, block: u32) -> Result<KernelIr> {
    require(
        (1..=1024).contains(&block),
        "demo workgroup size must be 1..=1024",
    )?;
    let name = NAMES
        .iter()
        .copied()
        .find(|n| *n == name)
        .ok_or_else(|| Error(format!("unknown demo {name}; choose {}", NAMES.join(", "))))?;
    let elem = ScalarElement::U32.element();
    let buffer = Arc::new(BufferDecl {
        binding: 1,
        element: elem,
        layout: TileLayout::contiguous(MemoryLevel::Storage, &[65536]),
        access: BufferAccess::Read,
    });
    let view = StorageView {
        buffer: buffer.clone(),
        offset: 0,
        layout: buffer.layout.clone(),
    };
    let lane = TileExpr::new(TileExprKind::Builtin(Builtin::Lane), elem);
    let sg_lane = TileExpr::new(TileExprKind::Builtin(Builtin::SubgroupLane), elem);
    let local = Arc::new(LocalDecl::new(elem));
    let load = |index: TileExpr| {
        TileExpr::new(
            TileExprKind::Load {
                src: Source::Storage(view.clone()),
                addr: Box::new(Addr::Linear(index)),
                mask: TileExpr::new(
                    TileExprKind::Literal(TileLiteral::Bool(true)),
                    ScalarElement::Bool.element(),
                ),
                fill: lit(0),
            },
            elem,
        )
    };
    let read = |index| Stmt::StoreLocal {
        dst: local.clone(),
        value: load(index),
    };
    let body = match name {
        "coalesced" => vec![read(lane)],
        "strided" => vec![read(mul(lane, lit(32)))],
        "broadcast" => vec![read(lit(0))],
        "reuse" => vec![read(sg_lane.clone()), read(sg_lane)],
        "bank-conflict" | "padded" => {
            let stride = if name == "padded" { 33 } else { 32 };
            let tile = Arc::new(TileDecl::new(
                elem,
                TileLayout::contiguous(MemoryLevel::Workgroup, &[block * stride]),
                "shared_tile",
            ));
            let index = mul(lane.clone(), lit(stride));
            vec![
                Stmt::StoreTile {
                    dst: tile.clone(),
                    index: index.clone(),
                    value: load(lane),
                },
                Stmt::Barrier,
                Stmt::StoreLocal {
                    dst: local,
                    value: TileExpr::new(TileExprKind::LoadTile { tile, index }, elem),
                },
            ]
        }
        _ => unreachable!(),
    };
    Ok(KernelIr {
        buffers: vec![buffer],
        grid: [2, 1, 1],
        block,
        body,
        byte_arena: None,
        name,
    })
}

fn lit(v: u32) -> TileExpr {
    TileExpr::new(
        TileExprKind::Literal(TileLiteral::U32(v)),
        ScalarElement::U32.element(),
    )
}
fn mul(a: TileExpr, b: TileExpr) -> TileExpr {
    TileExpr::new(
        TileExprKind::Binary {
            op: BinOp::Mul,
            left: a,
            right: b,
            numeric: NumericContract::STRICT,
        },
        ScalarElement::U32.element(),
    )
}
