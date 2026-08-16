//! [`Roofline`] — the [`CostModel`] implementation.
//!
//! Per launch:
//! `launch_ps + max(dram_ps, occupancy * (math_ps + wg_ps)) + occupancy *
//! drain_ps + combine_ps`.
//!
//! T1 and T2 are **summed** inside the `max` because they contend for the
//! same per-core issue and load/store slots; DRAM overlaps them; the combine
//! dispatch sits behind its own barrier and adds.
//!
//! One scalar. Precision is a verifier property (`NumericContract`), not a
//! cost term, because a time-only model eliminates f32 everywhere.

use crate::terms;
use fusor2_ir::cost::{CostModel, DeviceFacts, LaunchPlan, MacUnit, Picoseconds, ShapeStats};
use fusor2_ir::dtype::Dtype;
use fusor2_ir::extract::{Extraction, PlanHash};
use fusor2_ir::facts::ValueFacts;
use fusor2_ir::ir::launch::SchedPoint;
use fusor2_ir::ir::Node;
use fusor2_ir::shape::Dim;
use parking_lot::RwLock;
use rustc_hash::{FxHashSet, FxHasher};
use std::hash::{Hash, Hasher};

/// The one cost model. `score_fs` maps onto its terms one for one:
/// T1 -> math, T2 -> wg, T3 -> drain, T4 -> the `max`, T5 -> the combine
/// launch.
pub struct Roofline {
    facts: DeviceFacts,
    stats: RwLock<ShapeStats>,
}

/// The schedule-dependent inputs one launch's terms need, decoded from its
/// root's [`SchedPoint`].
#[derive(Copy, Clone, Debug)]
struct Sched {
    unit: MacUnit,
    /// 1 loses the load/MMA overlap the threadgroup rate was fitted on.
    staging: u8,
    splits: u32,
    /// Emitting subgroups per workgroup — the epilogue drain is per element
    /// *and* per subgroup.
    subgroups: u32,
}

impl Default for Sched {
    fn default() -> Self {
        Self {
            unit: MacUnit::Fma,
            staging: 2,
            splits: 1,
            subgroups: 1,
        }
    }
}

impl Sched {
    fn of(theta: Option<SchedPoint>) -> Self {
        match theta {
            Some(SchedPoint::Coop {
                geom,
                splits,
                staging,
            }) => Self {
                unit: MacUnit::Coop,
                staging,
                splits,
                subgroups: (geom.rg * geom.cg).max(1),
            },
            Some(SchedPoint::Sgemm(p)) => Self {
                staging: if p.double_buffer { 2 } else { 1 },
                ..Self::default()
            },
            Some(SchedPoint::Sgemv(p)) => Self {
                subgroups: p.subgroups.max(1),
                ..Self::default()
            },
            _ => Self::default(),
        }
    }
}

impl Roofline {
    pub fn new(facts: DeviceFacts) -> Self {
        Self {
            facts,
            stats: RwLock::new(ShapeStats::new()),
        }
    }

    /// Record that this plan ran at this dim binding, and return how many
    /// times that pair has now been seen.
    ///
    /// Also bumps a plan-level counter (the empty binding), which is what
    /// [`CostModel::total`] amortizes compilation against — a plan is
    /// compiled once per plan, not once per binding.
    pub fn observe_binding(&self, plan: PlanHash, binding: &[Dim]) -> u32 {
        let mut stats = self.stats.write();
        stats.observe(plan, &[]);
        stats.observe(plan, binding)
    }

    /// How many times this plan has been seen at any binding. `1` on first
    /// sighting, so nothing compiles speculatively and the generic symbolic
    /// variant wins outright.
    pub fn expected_reuse(&self, plan: PlanHash) -> u32 {
        self.stats.read().expected_reuse(plan, &[])
    }

    /// The compile identity of one launch: its root, its members, the
    /// schedule points they resolved to, whether the root is materialized,
    /// and the device fingerprint.
    ///
    /// This is a *stand-in*. The authoritative `PlanHash` is
    /// `plan::plan_hash` over the whole realized term; a launch alone
    /// cannot see that term.
    pub fn launch_plan_hash(&self, launch: &LaunchPlan<'_>, materialized: bool) -> PlanHash {
        let mut h = FxHasher::default();
        self.facts.fingerprint().hash(&mut h);
        launch.root.hash(&mut h);
        materialized.hash(&mut h);
        for id in launch.members {
            id.hash(&mut h);
            launch.theta.get(id).hash(&mut h);
        }
        launch.grid.hash(&mut h);
        let lo = h.finish();
        // A second lane so a 64-bit collision is not a plan collision.
        let mut h2 = FxHasher::default();
        (lo, 0x9e37_79b9_7f4a_7c15u64).hash(&mut h2);
        launch.work.hash(&mut h2);
        PlanHash((u128::from(h2.finish()) << 64) | u128::from(lo))
    }

    /// [`CostModel::launch_cost`] at an explicit operand dtype.
    ///
    /// [`LaunchPlan`] carries no dtype, so the trait method assumes f32.
    /// Callers that know better — every real lowering does — should come
    /// through here, so an f16 contraction is priced at the f16 MAC rate.
    pub fn launch_cost_at(&self, launch: &LaunchPlan<'_>, dtype: Dtype) -> Picoseconds {
        let f = &self.facts;
        let sched = Sched::of(launch.theta.get(&launch.root).copied());
        let elem_bytes = dtype.byte_size().max(1);

        let math = terms::math_ps(f, launch.work, sched.unit, dtype);
        let wg = terms::wg_ps(f, launch.work.wg_bytes, sched.staging);
        let (num, den) = terms::occupancy_scale_num_den(f, launch.resident_lanes);
        let issue = terms::scaled(math + wg, num, den);

        // `writes` is the padded output the launch actually stores,
        // including every split's full tile — exactly the reference's
        // `workgroups * bm * bn` once divided by the element size.
        let padded_out_elems = launch.writes / elem_bytes;
        let drain = terms::scaled(
            terms::drain_ps(
                f,
                padded_out_elems,
                sched.subgroups,
                u32::try_from(launch.wg_bytes).unwrap_or(u32::MAX),
                f.caps.limits.max_compute_workgroup_storage_size,
            ),
            num,
            den,
        );

        let dram = terms::dram_ps(f, launch.reads, launch.writes);
        // One split's padded output; `(splits + 1)` then counts reading
        // every partial and writing the result.
        let combine = terms::combine_ps(
            f,
            sched.splits,
            launch.writes / u64::from(sched.splits.max(1)),
        );

        Picoseconds(f.launch_ps) + dram.max(issue) + drain + combine
    }
}

/// Which functional unit and dtype a node's MACs issue on.
fn unit_and_dtype(
    ins: &[ValueFacts],
    out: &ValueFacts,
    theta: Option<SchedPoint>,
) -> (MacUnit, Dtype) {
    // MACs issue at the operand dtype; `acc` is a separate attribute and
    // does not set the issue rate.
    let dtype = ins.first().map_or(out.dtype, |f| f.dtype);
    match theta {
        Some(SchedPoint::Coop { .. }) => (MacUnit::Coop, dtype),
        _ => (MacUnit::Fma, dtype),
    }
}

impl CostModel for Roofline {
    fn facts(&self) -> &DeviceFacts {
        &self.facts
    }

    fn launch_cost(&self, launch: &LaunchPlan<'_>) -> Picoseconds {
        self.launch_cost_at(launch, Dtype::F32)
    }

    fn node_math(
        &self,
        node: &Node,
        ins: &[ValueFacts],
        out: &ValueFacts,
        theta: Option<SchedPoint>,
    ) -> Picoseconds {
        let (unit, dtype) = unit_and_dtype(ins, out, theta);
        let mut work = fusor2_ir::semantics::work::work_of(&node.op, ins, out);
        // A tiled point issues MACs on the *padded* tile. Padding is real
        // work the theta performs — the kernels stage zero-filled tiles and
        // run the whole tile's MACs — so charging it keeps the bound
        // admissible.
        if let (
            Some(tile),
            fusor2_ir::ir::Op::Launch(fusor2_ir::ir::launch::Launch::Contract {
                m, n, k, batch, ..
            }),
        ) = (
            theta.and_then(|t| match t {
                SchedPoint::Coop { geom, .. } => Some((geom.bm, geom.bn)),
                SchedPoint::Sgemm(p) => Some((p.bm, p.bn)),
                _ => None,
            }),
            &node.op,
        )
        {
            let geom_bm = tile.0;
            let geom_bn = tile.1;
            let priced = |d: &Dim| d.as_const().unwrap_or(1).max(1);
            let (m, n, k, batch) = (priced(m), priced(n), priced(k), priced(batch));
            let m_pad = m
                .div_ceil(u64::from(geom_bm.max(1)))
                .saturating_mul(u64::from(geom_bm.max(1)));
            let n_pad = n
                .div_ceil(u64::from(geom_bn.max(1)))
                .saturating_mul(u64::from(geom_bn.max(1)));
            let extra = m_pad
                .saturating_mul(n_pad)
                .saturating_sub(m.saturating_mul(n))
                .saturating_mul(k)
                .saturating_mul(batch);
            work.macs = work.macs.saturating_add(extra);
        }
        // Zero traffic, no occupancy scaling. The admissible lower bound is
        // built from this, and either addition would break admissibility.
        terms::math_ps(&self.facts, work, unit, dtype)
    }

    fn traffic(&self, bytes: u64, rereads: u32) -> Picoseconds {
        terms::dram_ps(&self.facts, &[(bytes, rereads)], 0)
    }

    fn compile_amortized(&self, plan: PlanHash, expected_reuse: u32) -> Picoseconds {
        let _ = plan;
        Picoseconds(self.facts.compile_ps_per_kernel / u64::from(expected_reuse.max(1)))
    }

    fn total(&self, extraction: &Extraction, launches: &[LaunchPlan<'_>]) -> Picoseconds {
        let mut total = Picoseconds(0);
        let mut compiled = FxHashSet::default();
        for launch in launches {
            total += self.launch_cost(launch);
            let hash = self.launch_plan_hash(launch, extraction.is_materialized(launch.root));
            if compiled.insert(hash) {
                total += self.compile_amortized(hash, self.expected_reuse(hash));
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::seed_facts;
    use crate::facts::tests::gpu_caps;
    use fusor2_ir::egraph::Id;
    use fusor2_ir::facts::Work;
    use fusor2_ir::ir::launch::{CoopGeom, SgemvParams};
    use rustc_hash::FxHashMap;
    use std::cmp::Reverse;

    const SUBGROUP_WIDTH: u32 = 32;
    const COOP_DIM: u32 = 8;

    /// One row of the reference's `COOP_TILE_TABLE`.
    #[derive(Copy, Clone, Debug)]
    struct Profile {
        bm: u32,
        bn: u32,
        bk: u32,
        subgroups: u32,
        n_passes: u32,
    }

    impl Profile {
        const fn new(bm: u32, bn: u32, bk: u32, subgroups: u32, n_passes: u32) -> Self {
            Self {
                bm,
                bn,
                bk,
                subgroups,
                n_passes,
            }
        }
        fn geom(self) -> CoopGeom {
            let (rg, cg) = CoopGeom::subgroup_split(self.bm, self.bn, self.n_passes, self.subgroups)
                .unwrap_or((1, self.subgroups));
            CoopGeom {
                bm: self.bm,
                bn: self.bn,
                bk: self.bk,
                n_passes: self.n_passes,
                subgroups: self.subgroups,
                rg,
                cg,
            }
        }
        /// `DenseCoopMatmulTile::stage_pair_elements`, verbatim.
        fn stage_pair_elements(self) -> u64 {
            let bn_pass = u64::from(self.bn / self.n_passes);
            let a_tile = u64::from(self.bm) * (u64::from(self.bk) + 1) - 1;
            let b_tile = u64::from(self.bk) * (bn_pass + 1) - 1;
            a_tile + b_tile
        }
        fn arena_bytes(self, elem_bytes: u64, staging: u8) -> u64 {
            self.stage_pair_elements() * elem_bytes * u64::from(staging)
        }
    }

    /// The five profiles the reference's doc header measures.
    const P128X64: Profile = Profile::new(128, 64, 16, 8, 1);
    const P64X64: Profile = Profile::new(64, 64, 16, 4, 1);
    const P128X128: Profile = Profile::new(128, 128, 16, 8, 2);
    const P64X16: Profile = Profile::new(64, 16, 16, 4, 1);
    const P16X64: Profile = Profile::new(16, 64, 16, 4, 1);
    const ANCHOR_PROFILES: [(&str, Profile); 5] = [
        ("128x64", P128X64),
        ("64x64", P64X64),
        ("128x128", P128X128),
        ("64x16", P64X16),
        ("16x64", P16X64),
    ];

    /// A launch's owned backing, so `LaunchPlan`'s borrows have somewhere to
    /// point.
    struct Case {
        members: Vec<Id>,
        theta: FxHashMap<Id, SchedPoint>,
        reads: Vec<(u64, u32)>,
        writes: u64,
        work: Work,
        resident_lanes: u64,
        wg_bytes: u64,
        grid: [u32; 3],
    }

    impl Case {
        fn plan(&self) -> LaunchPlan<'_> {
            LaunchPlan {
                members: &self.members,
                root: self.members[0],
                theta: &self.theta,
                reads: &self.reads,
                writes: self.writes,
                work: self.work,
                resident_lanes: self.resident_lanes,
                wg_bytes: self.wg_bytes,
                grid: self.grid,
            }
        }
    }

    /// The launch a cooperative contraction at one schedule point produces,
    /// with every quantity derived exactly the way `score_fs` derives its
    /// own. Reproducing the arithmetic of `work()` and `realize()` here means
    /// the anchors test measures the cost model rather than a stub.
    #[allow(clippy::too_many_arguments)]
    fn coop_case(
        p: Profile,
        m: u32,
        k: u32,
        n: u32,
        batch: u32,
        segments: u32,
        splits: u32,
        staging: u8,
        elem_bytes: u64,
    ) -> Case {
        let geom = p.geom();
        let (bm, bn, bk) = (u64::from(p.bm), u64::from(p.bn), u64::from(p.bk));
        let n_passes = u64::from(p.n_passes);
        let bn_pass = bn / n_passes;
        let tr = bm / u64::from(COOP_DIM * geom.rg);
        let tc = bn_pass / u64::from(COOP_DIM * geom.cg);
        let subgroups = u64::from(geom.rg) * u64::from(geom.cg);
        let threads = subgroups * u64::from(SUBGROUP_WIDTH);

        let tiles_m = u64::from(m.div_ceil(p.bm));
        let tiles_n = u64::from(n.div_ceil(p.bn));
        let k_iterations = u64::from(k.div_ceil(p.bk));
        let span_iterations = k_iterations.div_ceil(u64::from(splits));
        let workgroups =
            tiles_m * tiles_n * u64::from(batch) * u64::from(splits) * u64::from(segments);
        let per_workgroup = workgroups * span_iterations;
        let m_padded = tiles_m * bm;
        let n_padded = tiles_n * bn;

        // T1's MAC count, on the *padded* tile. There is no routing guard:
        // an over-padded candidate simply prices high here.
        let macs = per_workgroup * bm * bn * bk;
        let fragment_bytes =
            n_passes * subgroups * (tr + tc) * (bk / u64::from(COOP_DIM)) * 64 * elem_bytes;
        let stage_bytes = n_passes * (bm * bk + bk * bn_pass) * elem_bytes;
        let wg_traffic = per_workgroup * (fragment_bytes + stage_bytes);

        let operand_bytes = u64::from(segments)
            * elem_bytes
            * u64::from(batch)
            * (u64::from(m) * u64::from(k) + u64::from(k) * u64::from(n));
        let writes = u64::from(segments)
            * u64::from(splits)
            * u64::from(batch)
            * m_padded
            * n_padded
            * elem_bytes;

        let root = Id(1);
        let mut theta = FxHashMap::default();
        theta.insert(
            root,
            SchedPoint::Coop {
                geom,
                splits,
                staging,
            },
        );
        Case {
            members: vec![root],
            theta,
            reads: vec![(operand_bytes, 1)],
            writes,
            work: Work {
                macs,
                transcendentals: 0,
                index_ops: 0,
                wg_bytes: wg_traffic,
            },
            resident_lanes: workgroups * threads,
            wg_bytes: p.arena_bytes(elem_bytes, staging),
            grid: [u32::try_from(workgroups).unwrap_or(u32::MAX), 1, 1],
        }
    }

    /// The reference's `split_candidates`: never splitting, plus every
    /// divisor of the K loop leaving two iterations per workgroup, capped at
    /// 64.
    fn split_candidates(k_iterations: u32) -> Vec<u32> {
        let limit = (k_iterations / 2).clamp(1, 64);
        (1..=limit)
            .filter(|d| *d == 1 || k_iterations.is_multiple_of(*d))
            .collect()
    }

    /// Best cost of one profile on one contraction, minimized over its legal
    /// split counts and staging depths.
    fn best_cost(model: &Roofline, p: Profile, m: u32, k: u32, n: u32, elem_bytes: u64) -> u64 {
        let max_storage = model.facts.caps.limits.max_compute_workgroup_storage_size;
        let mut best = u64::MAX;
        for splits in split_candidates(k.div_ceil(p.bk)) {
            let depths: &[u8] = if splits > 1 { &[1] } else { &[1, 2] };
            for &staging in depths {
                if p.arena_bytes(elem_bytes, staging) > u64::from(max_storage) {
                    continue;
                }
                let case = coop_case(p, m, k, n, 1, 1, splits, staging, elem_bytes);
                best = best.min(model.launch_cost(&case.plan()).0);
            }
        }
        best
    }

    fn apple() -> Roofline {
        Roofline::new(seed_facts(&gpu_caps("anchor")))
    }

    /// Test 1. The five profiles of the reference's doc header over its five
    /// measured contractions. Measured per-entry minima in ms:
    ///
    /// | m x k x n        | 128x64 | 64x64 | 128x128 | 64x16 | 16x64 |
    /// |------------------|--------|-------|---------|-------|-------|
    /// | 16384x384x384    | 0.799  | 0.822 | 1.011   | 0.990 | 1.197 |
    /// | 16384x384x1536   | 2.705  | 2.893 | 3.456   | 3.823 | 3.996 |
    /// | 16384x1536x384   | 2.698  | 2.885 | 3.646   | 3.836 | 4.023 |
    /// | 16384x3072x1536  | 20.41  | 21.16 | 28.30   | 30.58 | 32.44 |
    /// | 1024x1024x1024   | 0.445  | 0.445 | 0.577   | 0.446 | 0.634 |
    ///
    /// The model is a *ranking* function; its absolute magnitude is not
    /// calibrated (the rates are fitted for argmin inside a documented box).
    /// What must hold is the argmin, and that the four K-deep shapes take a
    /// 128-wide profile.
    #[test]
    fn apple_seed_reproduces_score_fs_anchors() {
        /// One measured row: a contraction, and the profile names that tied
        /// for fastest on the bench.
        struct Anchor {
            m: u32,
            k: u32,
            n: u32,
            fastest: &'static [&'static str],
        }
        const fn anchor(m: u32, k: u32, n: u32, fastest: &'static [&'static str]) -> Anchor {
            Anchor { m, k, n, fastest }
        }

        let model = apple();
        // The 1024-cube row measures a dead heat between 128x64 and 64x64.
        let anchors = [
            anchor(16_384, 384, 384, &["128x64"]),
            anchor(16_384, 384, 1_536, &["128x64"]),
            anchor(16_384, 1_536, 384, &["128x64"]),
            anchor(16_384, 3_072, 1_536, &["128x64"]),
            anchor(1_024, 1_024, 1_024, &["128x64", "64x64"]),
        ];
        for Anchor { m, k, n, fastest } in anchors {
            let mut scored: Vec<(&str, u64, Profile)> = ANCHOR_PROFILES
                .iter()
                .map(|&(name, p)| (name, best_cost(&model, p, m, k, n, 4), p))
                .collect();
            // The reference's tie-break chain, reached and therefore
            // load-bearing: the score is invariant to `n_passes` by
            // construction (a p-pass profile does p times the per-workgroup
            // work over a p-times-smaller grid), and every recorded sweep
            // says fewer passes wins, so it leads. The two `Reverse` levels
            // keep the wider tile on an exact tie.
            scored.sort_by_key(|&(_, cost, p)| {
                (cost, p.n_passes, Reverse(p.bm), Reverse(p.bn))
            });
            let (winner, _, profile) = scored[0];
            assert!(
                fastest.contains(&winner),
                "{m}x{k}x{n}: picked {winner}, measured fastest is {fastest:?}; \
                 full ranking {:?}",
                scored.iter().map(|&(n, c, _)| (n, c)).collect::<Vec<_>>()
            );
            if m == 16_384 {
                assert_eq!(profile.bm, 128, "{m}x{k}x{n} must take a 128-wide profile");
            }
        }
    }

    /// Test 2. A fusion that removes one launch but adds 64 MiB of DRAM
    /// traffic must cost strictly more. Under the reference's lexicographic
    /// `(dispatches, bytes, work)` the dispatch saving is worth unbounded
    /// bandwidth; here the launch is worth 1 us and the traffic 177 us.
    #[test]
    fn scalar_cost_lets_traffic_outweigh_a_dispatch() {
        let model = apple();
        let facts = &model.facts;
        assert_eq!(facts.launch_ps, 1_000_000, "one launch is one microsecond");

        let extra = 64u64 << 20;
        let added = terms::dram_ps(facts, &[(extra, 1)], 0).0;
        let delta = added - facts.launch_ps;
        assert!(
            delta as f64 > 1.7e8,
            "64 MiB of traffic must dwarf a dispatch; the fused form costs \
             {delta} ps more"
        );
    }

    /// Test 5. At `1x4096x4096` f32 — the reference's pinned golden row,
    /// where its family selector picks Coop, its tile scorer declines on the
    /// padding gate, and production silently runs a third path — every legal
    /// cooperative schedule point must simply *price above* the sgemv
    /// candidate. No guard is invoked, because none exists.
    #[test]
    fn padded_coop_loses_to_sgemv_on_cost() {
        let model = apple();
        let (m, k, n) = (1u32, 4_096u32, 4_096u32);
        let elem = 4u64;

        // The sgemv candidate: useful MACs only, one row of A, all of B, one
        // row of output, no workgroup staging, no split.
        let root = Id(1);
        let mut theta = FxHashMap::default();
        theta.insert(
            root,
            SchedPoint::Sgemv(SgemvParams {
                vector: 4,
                subgroups: 4,
                cols: 1,
                parts: 1,
                gap: 0,
            }),
        );
        let rows = u64::from(n).div_ceil(64);
        let sgemv = Case {
            members: vec![root],
            theta,
            reads: vec![(
                (u64::from(m) * u64::from(k) + u64::from(k) * u64::from(n)) * elem,
                1,
            )],
            writes: u64::from(m) * u64::from(n) * elem,
            work: Work {
                macs: u64::from(m) * u64::from(k) * u64::from(n),
                transcendentals: 0,
                index_ops: 0,
                wg_bytes: 0,
            },
            resident_lanes: rows * 128,
            wg_bytes: 0,
            grid: [u32::try_from(rows).unwrap_or(u32::MAX), 1, 1],
        };
        let sgemv_cost = model.launch_cost(&sgemv.plan()).0;

        for (name, p) in ANCHOR_PROFILES {
            let coop = best_cost(&model, p, m, k, n, elem);
            assert!(
                coop > sgemv_cost,
                "coop {name} priced {coop} against sgemv {sgemv_cost}"
            );
        }
        // And the reason is padding, priced through `Work::macs` rather than
        // vetoed: the 128-wide profile pads m from 1 to 128.
        let padded = coop_case(P128X64, m, k, n, 1, 1, 1, 2, elem);
        assert_eq!(padded.work.macs, 128 * 4_096 * 4_096);
        assert_eq!(sgemv.work.macs, 4_096 * 4_096);
    }

    /// Test 7. The split-K sweep is a lane-count dial: more splits means
    /// more resident lanes *and* more redundant padded-tile writes plus a
    /// bigger combine. The reference measures 64x2048x64 at the 64x64
    /// profile running 0.202 ms with 20,480 lanes resident and 0.310 ms with
    /// 81,920 — the starved grid is 0.65x, where a linear occupancy law says
    /// it should have been 3.2x slower.
    ///
    /// The reference records the lane counts but not the batch they were
    /// measured at. With this profile's 128 threads and a single output
    /// tile, `resident = batch * splits * 128`, so 20,480 and 81,920 are
    /// reachable at batch 10 with splits 16 and 64 — both legal divisors of
    /// the 128-iteration K loop, which is what keeps T1 and T2 constant
    /// across the pair.
    #[test]
    fn occupancy_cube_root_matches_split_k_sweep() {
        let model = apple();
        let low = coop_case(P64X64, 64, 2_048, 64, 10, 1, 16, 1, 4);
        let high = coop_case(P64X64, 64, 2_048, 64, 10, 1, 64, 1, 4);
        assert_eq!(low.resident_lanes, 20_480);
        assert_eq!(high.resident_lanes, 81_920);
        // The dial moves lanes, not arithmetic.
        assert_eq!(low.work.macs, high.work.macs);
        assert_eq!(low.work.wg_bytes, high.work.wg_bytes);

        let ratio =
            model.launch_cost(&low.plan()).0 as f64 / model.launch_cost(&high.plan()).0 as f64;
        assert!(
            (0.55..=0.75).contains(&ratio),
            "predicted {ratio}, measured 0.202/0.310 = 0.652"
        );

        // A linear law would scale the starved grid by
        // `saturation_lanes / resident` = 3.2 and land nowhere near it.
        let f = &model.facts;
        let linear = |case: &Case, scale: f64| {
            let sched = Sched::of(case.theta.get(&case.members[0]).copied());
            let math = terms::math_ps(f, case.work, sched.unit, Dtype::F32).0 as f64;
            let wg = terms::wg_ps(f, case.work.wg_bytes, sched.staging).0 as f64;
            let drain = terms::drain_ps(
                f,
                case.writes / 4,
                sched.subgroups,
                case.wg_bytes as u32,
                f.caps.limits.max_compute_workgroup_storage_size,
            )
            .0 as f64;
            let dram = terms::dram_ps(f, &case.reads, case.writes).0 as f64;
            let combine =
                terms::combine_ps(f, sched.splits, case.writes / u64::from(sched.splits)).0 as f64;
            f.launch_ps as f64 + (scale * (math + wg)).max(dram) + scale * drain + combine
        };
        let linear_scale = f64::from(f.saturation_lanes) / 20_480.0;
        assert!((linear_scale - 3.2).abs() < 1e-9);
        let linear_ratio = linear(&low, linear_scale) / linear(&high, 1.0);
        assert!(
            !(0.55..=0.75).contains(&linear_ratio),
            "a linear law must fail this assert, got {linear_ratio}"
        );
        assert!(linear_ratio > 1.5, "{linear_ratio}");
    }

    /// Test 9. On first sighting the generic symbolic variant wins outright
    /// — nothing compiles per length bucket. Once a binding recurs, a
    /// specialization that saves real time wins.
    #[test]
    fn compile_amortizes_to_generic_on_first_sighting() {
        let model = apple();
        let h = PlanHash(0xdead_beef);
        assert_eq!(model.compile_amortized(h, 1).0, 1_000_000_000);
        assert_eq!(model.compile_amortized(h, 64).0, 15_625_000);
        assert_eq!(model.compile_amortized(h, 0).0, 1_000_000_000);

        // The trainer's modal bucket: 768 units at batch 128 through the
        // 256->96 dense head. The symbolic variant reads its extents from
        // binding 0 and pays index arithmetic for it; the specialized
        // variant bakes them and does not.
        let rows = 128u64 * 768;
        let mk = |index_ops: u64, root: Id| {
            let mut theta = FxHashMap::default();
            theta.insert(root, SchedPoint::Point);
            Case {
                members: vec![root],
                theta,
                reads: vec![(rows * 256 * 4, 1)],
                writes: rows * 96 * 4,
                work: Work {
                    macs: rows * 256 * 96,
                    transcendentals: 0,
                    index_ops,
                    wg_bytes: 0,
                },
                resident_lanes: 1 << 20,
                wg_bytes: 0,
                grid: [1_024, 1, 1],
            }
        };
        let symbolic = mk(rows * 96, Id(1));
        let specialized = mk(0, Id(2));

        let extraction = Extraction::default();
        let sym_hash = model.launch_plan_hash(&symbolic.plan(), false);
        let spec_hash = model.launch_plan_hash(&specialized.plan(), false);
        assert_ne!(sym_hash, spec_hash);

        // The symbolic plan already serves every other bucket.
        for _ in 0..64 {
            model.observe_binding(sym_hash, &[Dim::Const(768)]);
        }
        let sym_total = model.total(&extraction, &[symbolic.plan()]).0;
        let spec_total = model.total(&extraction, &[specialized.plan()]).0;
        assert!(
            sym_total < spec_total,
            "on first sighting the generic variant must win: {sym_total} vs {spec_total}"
        );

        // After the binding recurs, the specialization pays for itself.
        for _ in 0..64 {
            model.observe_binding(spec_hash, &[Dim::Const(768)]);
        }
        let sym_total = model.total(&extraction, &[symbolic.plan()]).0;
        let spec_total = model.total(&extraction, &[specialized.plan()]).0;
        assert!(
            spec_total < sym_total,
            "at reuse 64 the specialization must win: {spec_total} vs {sym_total}"
        );
    }

    /// `total` charges compilation once per distinct plan, not once per
    /// launch, and is a plain sum otherwise.
    #[test]
    fn total_is_a_plain_sum_with_one_compile_per_plan() {
        let model = apple();
        let case = coop_case(P64X64, 512, 512, 512, 1, 1, 1, 2, 4);
        let extraction = Extraction::default();
        let one = model.launch_cost(&case.plan()).0;
        let hash = model.launch_plan_hash(&case.plan(), false);
        let compile = model.compile_amortized(hash, 1).0;

        let plans = [case.plan(), case.plan(), case.plan()];
        assert_eq!(model.total(&extraction, &plans).0, 3 * one + compile);
        assert_eq!(model.total(&extraction, &[]).0, 0);
    }

    /// Extraction holds the model as `&dyn CostModel`, so it has to stay
    /// object-safe.
    #[test]
    fn roofline_is_object_safe() {
        let model = apple();
        let erased: &dyn CostModel = &model;
        assert_eq!(erased.facts().launch_ps, 1_000_000);
        let boxed: Box<dyn CostModel> = Box::new(apple());
        assert_eq!(boxed.traffic(1 << 30, 1).0, erased.traffic(1 << 30, 1).0);
        // And `Send + Sync`, because kernel building runs on worker threads.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Roofline>();
    }

    /// `traffic` is the one-operand form of the DRAM term, continuous
    /// through the cache watermark.
    #[test]
    fn traffic_is_the_single_operand_dram_term() {
        let model = apple();
        let bytes = 3u64 << 20;
        assert_eq!(
            model.traffic(bytes, 1),
            terms::dram_ps(&model.facts, &[(bytes, 1)], 0)
        );
        // Inside the cache, rereads are free.
        assert_eq!(model.traffic(bytes, 1), model.traffic(bytes, 8));
    }

    /// Staging depth and workgroup width both move the cost, which is what
    /// makes them real decisions rather than table columns.
    #[test]
    fn staging_depth_and_subgroup_count_move_the_cost() {
        let model = apple();
        let double = coop_case(P128X64, 4_096, 4_096, 4_096, 1, 1, 1, 2, 4);
        let single = coop_case(P128X64, 4_096, 4_096, 4_096, 1, 1, 1, 1, 4);
        assert!(
            model.launch_cost(&single.plan()) > model.launch_cost(&double.plan()),
            "one staged pair loses the load/MMA overlap"
        );
        // ... but it halves the arena, so a core holds more of them.
        assert!(single.wg_bytes < double.wg_bytes);
    }

    /// The invariant `lower_bound::best_math`'s domain dedupe rests on:
    /// `node_math` depends on the schedule point only through the MAC unit
    /// and the padded tile. Points sharing `(family, bm, bn)` must price
    /// identically — split, staging, subgroup, `tm`/`tn`/`bk` and every
    /// sgemv axis never move the term. And a tiled point at `m = n = 1`
    /// must charge its padded tile.
    #[test]
    fn node_math_is_a_function_of_unit_and_padded_tile() {
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::facts::ValueFacts;
        use fusor2_ir::ir::launch::{
            AccessPlan, ContractSide, Family, Launch, Operand, ScheduleDomain, SgemmParams,
        };
        use fusor2_ir::ir::{Level, Node, Op};
        use fusor2_ir::scalar::ScalarExpr;
        use fusor2_ir::shape::{Dim, Layout};

        let model = apple();
        let dot = |src: Id| Operand {
            src,
            layout: Layout::contiguous(&[Dim::from(1u64), Dim::from(1u64), Dim::from(4_096u64)]),
            access: AccessPlan::Alias,
        };
        let ident = ScalarExpr::arg(0, Dtype::F32);
        let node = Node {
            op: Op::Launch(Launch::Contract {
                m: Dim::from(1u64),
                n: Dim::from(1u64),
                k: Dim::from(4_096u64),
                batch: Dim::from(1u64),
                family: Family::Coop,
                post: ident.clone(),
                acc: Dtype::F32,
                a: ContractSide::one(ident.clone(), dot(Id(0))),
                b: ContractSide::one(ident, dot(Id(0))),
                sched: ScheduleDomain::Point,
            }),
            level: Level::Launch,
            children: [Id(0), Id(0)].into_iter().collect(),
        };
        let facts = ValueFacts::new(
            Dtype::F32,
            [Dim::from(1u64), Dim::from(1u64), Dim::from(4_096u64)],
        );
        let out = ValueFacts::new(Dtype::F32, [Dim::from(1u64), Dim::from(1u64)]);
        let ins = [facts.clone(), facts];
        let price = |theta: SchedPoint| model.node_math(&node, &ins, &out, Some(theta)).0;

        // Coop: splits and staging never move the term at fixed (bm, bn).
        let p16 = Profile::new(16, 16, 8, 1, 1);
        let coop = |splits, staging| {
            price(SchedPoint::Coop {
                geom: p16.geom(),
                splits,
                staging,
            })
        };
        assert_eq!(coop(1, 1), coop(8, 2));

        // Sgemm: tm/tn/bk/double_buffer never move the term at fixed
        // (bm, bn) — and the padded tile is charged: 16*32 tile on a 1x1x4k
        // dot prices well above the un-padded sgemv/fold spelling.
        let sgemm = |bk, tm, tn, double_buffer| {
            price(SchedPoint::Sgemm(SgemmParams {
                double_buffer,
                bm: 16,
                bn: 32,
                bk,
                tm,
                tn,
            }))
        };
        assert_eq!(sgemm(8, 2, 2, false), sgemm(16, 4, 4, true));

        // Sgemv: every axis is pure schedule.
        let sgemv = |vector, subgroups, cols| {
            price(SchedPoint::Sgemv(SgemvParams {
                vector,
                subgroups,
                cols,
                parts: 1,
                gap: 0,
            }))
        };
        assert_eq!(sgemv(32, 1, 2), sgemv(16, 8, 16));

        // The padding charge orders the families truthfully at m = n = 1.
        assert!(sgemm(8, 2, 2, false) > sgemv(32, 1, 2));
        assert!(coop(1, 1) > sgemv(32, 1, 2));
    }
}
