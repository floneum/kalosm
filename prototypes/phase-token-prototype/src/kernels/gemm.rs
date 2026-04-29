use crate::{Clean, Numeric, Phase, ReadyTile, RegTile, Shape, TileLevel};

/// Concrete tiling plan for userland tiled GEMM.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GemmTilePlan {
    pub subgroup_m: u32,
    pub subgroup_n: u32,
    pub subgroup_k: u32,
    pub thread_m: u32,
    pub thread_n: u32,
    pub thread_k: u32,
}

impl GemmTilePlan {
    /// Match the conservative portable tiling previously used by the built-in GEMM op.
    pub fn portable(m: u32, n: u32, k: u32) -> Self {
        let subgroup_m = m.min(16);
        let subgroup_n = n.min(16);
        let subgroup_k = k;
        Self {
            subgroup_m,
            subgroup_n,
            subgroup_k,
            thread_m: subgroup_m.min(4),
            thread_n: subgroup_n.min(4),
            thread_k: subgroup_k,
        }
    }

    fn validate(self, m: u32, n: u32, k: u32) {
        assert!(self.subgroup_m > 0, "subgroup_m must be non-zero");
        assert!(self.subgroup_n > 0, "subgroup_n must be non-zero");
        assert!(self.subgroup_k > 0, "subgroup_k must be non-zero");
        assert!(self.thread_m > 0, "thread_m must be non-zero");
        assert!(self.thread_n > 0, "thread_n must be non-zero");
        assert!(self.thread_k > 0, "thread_k must be non-zero");
        assert_eq!(self.subgroup_k, k, "subgroup_k must cover full K");
        assert_eq!(self.thread_k, k, "thread_k must cover full K");
        assert_eq!(m % self.subgroup_m, 0, "M must divide subgroup_m");
        assert_eq!(n % self.subgroup_n, 0, "N must divide subgroup_n");
        assert_eq!(
            self.subgroup_m % self.thread_m,
            0,
            "subgroup_m must divide thread_m"
        );
        assert_eq!(
            self.subgroup_n % self.thread_n,
            0,
            "subgroup_n must divide thread_n"
        );
    }
}

/// Emit a tiled matrix multiply-accumulate from primitive partition and MMA ops.
pub fn tiled<'cx, 'k, 'flow, TA, TB, TC>(
    phase: &mut Phase<'cx, 'k, 'flow, Clean>,
    a: &ReadyTile<'k, '_, TA>,
    b: &ReadyTile<'k, '_, TB>,
    acc: &mut RegTile<'k, TC>,
    plan: GemmTilePlan,
) where
    TA: Numeric,
    TB: Numeric,
    TC: Numeric,
{
    let [m, k] = phase.tile_matrix_shape(a.tile);
    let [k_b, n] = phase.tile_matrix_shape(b.tile);
    let [m_acc, n_acc] = phase.tile_matrix_shape(acc.tile);
    assert_eq!(k, k_b, "gemm K dimensions must match");
    assert_eq!(m, m_acc, "gemm M dimension must match accumulator");
    assert_eq!(n, n_acc, "gemm N dimension must match accumulator");
    plan.validate(m, n, k);

    for subgroup_m in (0..m).step_by(plan.subgroup_m as usize) {
        for subgroup_n in (0..n).step_by(plan.subgroup_n as usize) {
            let mut acc_parent = *acc;
            phase.partition_at(
                a,
                TileLevel::Subgroup,
                Shape::new([plan.subgroup_m, plan.subgroup_k]),
                [subgroup_m, 0],
                |phase, a_subgroup| {
                    phase.partition_at(
                        b,
                        TileLevel::Subgroup,
                        Shape::new([plan.subgroup_k, plan.subgroup_n]),
                        [0, subgroup_n],
                        |phase, b_subgroup| {
                            phase.partition_private_at(
                                &mut acc_parent,
                                TileLevel::Subgroup,
                                Shape::new([plan.subgroup_m, plan.subgroup_n]),
                                [subgroup_m, subgroup_n],
                                |phase, acc_subgroup| {
                                    emit_thread_mmas(
                                        phase,
                                        &a_subgroup,
                                        &b_subgroup,
                                        acc_subgroup,
                                        plan,
                                    );
                                },
                            );
                        },
                    );
                },
            );
        }
    }
}

fn emit_thread_mmas<'cx, 'k, 'flow, TA, TB, TC>(
    phase: &mut Phase<'cx, 'k, 'flow, Clean>,
    a_subgroup: &ReadyTile<'k, '_, TA>,
    b_subgroup: &ReadyTile<'k, '_, TB>,
    acc_subgroup: RegTile<'k, TC>,
    plan: GemmTilePlan,
) where
    TA: Numeric,
    TB: Numeric,
    TC: Numeric,
{
    for thread_m in (0..plan.subgroup_m).step_by(plan.thread_m as usize) {
        for thread_n in (0..plan.subgroup_n).step_by(plan.thread_n as usize) {
            let mut acc_for_thread = acc_subgroup;
            phase.partition_at(
                a_subgroup,
                TileLevel::Thread,
                Shape::new([plan.thread_m, plan.thread_k]),
                [thread_m, 0],
                |phase, a_thread| {
                    phase.partition_at(
                        b_subgroup,
                        TileLevel::Thread,
                        Shape::new([plan.thread_k, plan.thread_n]),
                        [0, thread_n],
                        |phase, b_thread| {
                            phase.partition_private_at(
                                &mut acc_for_thread,
                                TileLevel::Thread,
                                Shape::new([plan.thread_m, plan.thread_n]),
                                [thread_m, thread_n],
                                |phase, mut acc_thread| {
                                    phase.mma(&a_thread, &b_thread, &mut acc_thread);
                                },
                            );
                        },
                    );
                },
            );
        }
    }
}
