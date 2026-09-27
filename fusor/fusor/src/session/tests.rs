use super::*;
use crate::graph::Graph;
use fusor_ir::ir::launch::SchedPoint;

/// Plan `roots` selecting exactly `members`, with `launch` at `point` or, by
/// default, its first schedule point.
fn replan_with(
    session: &Session,
    g: &EGraph,
    roots: &[Id],
    members: &[Id],
    launch: Id,
    point: Option<SchedPoint>,
) -> Plan {
    let mut extraction = fusor_ir::extract::Extraction::default();
    for &id in members {
        extraction.sigma.insert(g.class_of(id), id);
    }
    let point = point.unwrap_or_else(|| {
        let Op::Launch(op) = &g.node(launch).op else {
            unreachable!()
        };
        op.schedule().unwrap().iter().next().unwrap()
    });
    extraction.theta.insert(launch, point);
    LocalSearch::new(Arc::new(Planner::new()), session.caps())
        .replan(
            g,
            roots,
            &mut extraction,
            session.inner.cost.as_ref(),
            &mut fusor_cost::realize::NodeCache::new(g.len()),
        )
        .unwrap()
}

/// Run `plan` and read `out` back as f32.
fn run_f32(session: &Session, h: &GraphRef, plan: &Plan, out: &Tensor) -> Vec<f32> {
    let resolving = h.state().resolve_lock.lock();
    session.run(h, plan, std::slice::from_ref(out)).unwrap();
    let bytes = session.read_bytes_locked(&resolving, h, out.id).unwrap();
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

#[test]
#[cfg(all(feature = "cpu", not(target_arch = "wasm32")))]
fn streamed_reductions_execute_without_the_producer_buffer() {
    use fusor_ir::ir::launch::Launch;

    let backend = test_backend();
    let session = Session::new(backend).unwrap();
    for producer_axis in [1, 2] {
        let graph = Graph::new(&session);
        let h = graph.handle();
        let symbols = [h.fresh_sym(), h.fresh_sym(), h.fresh_sym()];
        for (sym, value) in symbols.iter().zip([2, 17, 33]) {
            h.bind_dim(*sym, value);
        }
        let shape = if producer_axis == 2 {
            symbols
        } else {
            [symbols[0], symbols[2], symbols[1]]
        };
        let input = graph
            .leaf("x", &shape.map(Dim::Sym), crate::Dtype::F32)
            .unwrap();
        let out = input
            .sum(producer_axis)
            .unwrap()
            .add_scalar(0.125)
            .unwrap()
            .sqr()
            .unwrap()
            .sum(1)
            .unwrap();
        let plan = {
            let mut g = h.state().egraph.lock();
            g.add_root(out.id);
            Driver::new()
                .saturate(
                    &mut g,
                    &session.caps(),
                    &session.inner.rules,
                    Default::default(),
                )
                .unwrap();
            let stream = g
                .members(g.class_of(out.id))
                .into_iter()
                .find(|id| {
                    matches!(g.node(*id).op, Op::Launch(Launch::StreamFold { .. }))
                        && g.node(*id).children.iter().all(|child| *child == input.id)
                })
                .expect("ordinary nested reductions must offer a streaming candidate");
            replan_with(&session, &g, &[out.id], &[input.id, stream], stream, None)
        };
        assert_eq!(plan.launches.len(), 1);
        for (step, [rows, columns, width]) in [[2, 1, 17], [3, 17, 1], [2, 65, 33], [1, 17, 65]]
            .into_iter()
            .enumerate()
        {
            for (sym, value) in symbols.iter().zip([rows, columns, width]) {
                h.bind_dim(*sym, value);
            }
            let values: Vec<f32> = (0..rows * columns * width)
                .map(|i| ((i * 13 + step as u64 * 7) % 61) as f32 / 32.0 - 1.0)
                .collect();
            input
                .set_bytes(bytemuck::cast_slice(&values).to_vec())
                .unwrap();
            let actual = run_f32(&session, h, &plan, &out);
            assert_eq!(actual.len(), rows as usize);
            for (row, got) in actual.iter().enumerate() {
                let expected: f64 = (0..columns)
                    .map(|column| {
                        let sum: f64 = (0..width)
                            .map(|k| {
                                let offset = if producer_axis == 2 {
                                    column * width + k
                                } else {
                                    k * columns + column
                                };
                                f64::from(
                                    values[row * (columns * width) as usize + offset as usize],
                                )
                            })
                            .sum();
                        (sum + 0.125).powi(2)
                    })
                    .sum();
                assert!(
                    (f64::from(*got) - expected).abs() < 2e-5 * expected.abs().max(1.0),
                    "shape=[{rows},{columns},{width}] row={row}: {got} != {expected}"
                );
            }
        }
    }
}

#[test]
#[cfg(all(feature = "cpu", not(target_arch = "wasm32")))]
fn ordinary_attention_discovers_and_executes_a_streamed_weighted_reduction() {
    use fusor_ir::carrier::SlotTy;
    use fusor_ir::ir::launch::Launch;

    let backend = test_backend();
    let session = Session::new(backend).unwrap();
    let graph = Graph::new(&session);
    let h = graph.handle();
    let seq = h.fresh_sym();
    h.bind_dim(seq, 7);
    let q = graph
        .leaf("q", &[Dim::Const(2), Dim::Const(4)], crate::Dtype::F32)
        .unwrap();
    let k = graph
        .leaf("k", &[Dim::Sym(seq), Dim::Const(4)], crate::Dtype::F32)
        .unwrap();
    let v = graph
        .leaf("v", &[Dim::Sym(seq), Dim::Const(4)], crate::Dtype::F32)
        .unwrap();
    let out = q
        .matmul_t(&k)
        .unwrap()
        .softmax_last_dim()
        .unwrap()
        .matmul(&v)
        .unwrap();
    let (stream, plan) = {
        let mut g = h.state().egraph.lock();
        g.add_root(out.id);
        Driver::new()
            .saturate(
                &mut g,
                &session.caps(),
                &session.inner.rules,
                Default::default(),
            )
            .unwrap();
        let reachable = g.reachable_from_roots();
        let stream = reachable
            .ones()
            .map(|i| Id(i as u32))
            .find(|id| {
                let Op::Launch(Launch::StreamFold { producer, fold, .. }) = &g.node(*id).op else {
                    return false;
                };
                let Launch::Fold { carrier, .. } = fold.as_ref() else {
                    return false;
                };
                let Launch::Fold { ops, .. } = producer.as_ref() else {
                    return false;
                };
                carrier.slots.as_slice()
                    == [
                        SlotTy::Scalar,
                        SlotTy::Scalar,
                        SlotTy::Vector(Dim::Const(4)),
                    ]
                    && ops.iter().any(|o| o.src == q.id)
                    && ops.iter().any(|o| o.src == k.id)
                    && g.node(*id)
                        .children
                        .iter()
                        .copied()
                        .collect::<FxHashSet<_>>()
                        == [q.id, k.id, v.id].into_iter().collect()
            })
            .expect("ordinary QK, softmax and PV must derive a single streamed weighted reduction");
        g.clear_roots();
        g.add_root(stream);
        let members = [q.id, k.id, v.id, stream];
        let plan = replan_with(&session, &g, &[stream], &members, stream, None);
        (stream, plan)
    };
    assert_eq!(plan.launches.len(), 1);
    let state = h.tensor(stream);
    let queries = [1.0f32, 0.25, 0.5, 0.75, 0.5, 0.75, 0.25, 1.0];
    q.set_bytes(bytemuck::cast_slice(&queries).to_vec())
        .unwrap();
    for (length, nonfinite) in [
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 4),
        (1, 0),
        (0, 0),
        (9, 0),
    ] {
        h.bind_dim(seq, length);
        let mut keys: Vec<f32> = (0..length * 4)
            .map(|i| (i * 11 % 19) as f32 / 13.0 - 0.5)
            .collect();
        let values: Vec<f32> = (0..length * 4)
            .map(|i| (i * 7 % 23) as f32 / 11.0 - 0.75)
            .collect();
        for row in 0..length as usize {
            if nonfinite == 2 || nonfinite == 1 && row % 2 == 0 {
                keys[row * 4] = f32::NEG_INFINITY;
            }
        }
        if nonfinite == 3 {
            keys[0] = f32::INFINITY;
        }
        if nonfinite == 4 {
            keys[0] = f32::NAN;
        }
        k.set_bytes(bytemuck::cast_slice(&keys).to_vec()).unwrap();
        v.set_bytes(bytemuck::cast_slice(&values).to_vec()).unwrap();
        let actual = run_f32(&session, h, &plan, &state);
        assert_eq!(actual.len(), 12);
        for row in 0..2 {
            let scores: Vec<f64> = keys
                .as_chunks::<4>()
                .0
                .iter()
                .map(|key| {
                    (0..4)
                        .map(|d| f64::from(queries[row * 4 + d]) * f64::from(key[d]))
                        .sum()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for d in 0..4 {
                let expected: f64 = weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| w / sum * f64::from(values[i * 4 + d]))
                    .sum();
                let got = if length == 0 {
                    assert_eq!(actual[row * 6 + 1], 0.0);
                    actual[row * 6 + 2 + d]
                } else {
                    actual[row * 6 + 2 + d] / actual[row * 6 + 1]
                };
                assert!(
                    expected.is_nan() && got.is_nan()
                        || (f64::from(got) - expected).abs() < 3e-5 * expected.abs().max(1.0),
                    "length={length} nonfinite={nonfinite} row={row} d={d}: {got} != {expected}"
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "cpu")]
fn shape_family_replay_preserves_inputs_and_prior_results() {
    let session = Session::new(Backend::cpu().unwrap()).unwrap();
    let graph = Graph::new(&session);
    let x = graph
        .leaf("x", &[Dim::Const(2)], crate::Dtype::F32)
        .unwrap();
    let y = graph
        .leaf("y", &[Dim::Const(2)], crate::Dtype::F32)
        .unwrap();
    x.set_bytes(bytemuck::cast_slice(&[1.0f32, 3.0]).to_vec())
        .unwrap();
    y.set_bytes(bytemuck::cast_slice(&[10.0f32, 20.0]).to_vec())
        .unwrap();
    let first = x.sub(&y).unwrap();
    assert_eq!(first.to_vec_f32().unwrap(), [-9.0, -17.0]);
    let swapped = y.sub(&x).unwrap();
    assert_eq!(swapped.to_vec_f32().unwrap(), [9.0, 17.0]);
    assert_eq!(first.to_vec_f32().unwrap(), [-9.0, -17.0]);
    assert_eq!(x.to_vec_f32().unwrap(), [1.0, 3.0]);
    assert_eq!(y.to_vec_f32().unwrap(), [10.0, 20.0]);
}

#[test]
#[cfg(feature = "cpu")]
fn dead_input_buffers_live_until_their_last_reader_drops() {
    let session = Session::new(Backend::cpu().unwrap()).unwrap();
    let graph = Graph::new(&session);
    let h = graph.handle();
    let input = Tensor::from_elements(h, &[Dim::Const(2)], &[1.0f32, 2.0]).unwrap();
    session.upload_leaf(&input).unwrap();
    let input_id = input.id;
    let reader = input.add_scalar(1.0).unwrap();
    let descendant = reader.add_scalar(2.0).unwrap();
    assert!(h.device_buf(reader.id).is_none());
    assert!(h.device_buf(descendant.id).is_none());
    drop(input);
    h.reap_dead();
    assert!(h.device_buf(input_id).is_some());
    h.reap_dead();
    assert!(h.device_buf(input_id).is_some());
    drop(reader);
    h.reap_dead();
    assert!(h.device_buf(input_id).is_some());
    drop(descendant);
    h.reap_dead();
    assert!(h.device_buf(input_id).is_none());
}

#[test]
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn symbolic_contractions_reuse_pipelines_across_tile_boundaries() {
    use fusor_ir::dtype::Dtype;
    use fusor_ir::ir::launch::{Family, Launch};
    use fusor_tile::rules::contract::{lower_coop, lower_sgemm, lower_sgemv};

    let backend = match Backend::gpu_blocking() {
        Ok(backend) => backend,
        Err(error) => {
            let required = std::env::var("FUSOR_CONFORMANCE_REQUIRE_GPU").is_ok_and(|value| {
                !matches!(value.trim(), "" | "0") && !value.trim().eq_ignore_ascii_case("false")
            });
            assert!(!required, "GPU regression is required: {error}");
            eprintln!("symbolic contraction regression skipped: {error}");
            return;
        }
    };
    let session = Session::new(backend).unwrap();
    let target = match &session.inner.device {
        Backend::Gpu(target) => target,
        #[cfg(feature = "cpu")]
        Backend::Cpu(_) => unreachable!(),
    };
    let caps = session.caps();
    eprintln!("symbolic contraction regression on {}", caps.name);
    let mut executions = 0;
    for (family, varying_k, columns, kv_cache) in [
        (Family::Sgemm, false, false, false),
        (Family::Sgemm, true, false, false),
        (Family::Sgemv, false, false, false),
        (Family::Sgemv, false, true, false),
        (Family::Sgemv, true, false, false),
        (Family::Sgemv, true, true, false),
        (Family::Coop, true, false, false),
        (Family::Sgemv, false, false, true),
        (Family::Sgemv, false, true, true),
        (Family::Sgemv, true, false, true),
        (Family::Sgemv, true, true, true),
        (Family::Coop, true, false, true),
    ] {
        if (family == Family::Coop && caps.coop_for(Dtype::F32, Dtype::F32).is_none())
            || (columns && !caps.subgroups.is_some_and(|s| s.is_fixed()))
        {
            continue;
        }
        let graph = Graph::new(&session);
        let h = graph.handle();
        let sym = h.fresh_sym();
        h.bind_dim(sym, 1);
        let (batch, rows, fixed_n, fixed_k) = if kv_cache {
            (8, 4, 128, 128)
        } else {
            (2, 3, 7, 19)
        };
        let capacity_sym = h.fresh_sym();
        h.bind_dim(capacity_sym, 64);
        let capacity_dim = Dim::Sym(capacity_sym);
        let n = if varying_k {
            Dim::Const(fixed_n)
        } else {
            Dim::Sym(sym)
        };
        let k = if varying_k {
            Dim::Sym(sym)
        } else {
            Dim::Const(fixed_k)
        };
        let a = graph
            .leaf("a", &[Dim::Const(batch), Dim::Const(rows), k], Dtype::F32)
            .unwrap();
        let b_shape = if kv_cache {
            [Dim::Const(batch), capacity_dim, Dim::Const(128)]
        } else {
            [Dim::Const(batch), n, k]
        };
        let b = graph.leaf("b", &b_shape, Dtype::F32).unwrap();
        let viewed = kv_cache.then(|| {
            use fusor_ir::shape::StrideSpec;
            b.restride(&[
                StrideSpec::dim(0, Dim::Const(batch)),
                StrideSpec::dim(1, Dim::Sym(sym)),
                StrideSpec::dim(2, Dim::Const(128)),
            ])
            .unwrap()
        });
        let out = if let Some(viewed) = &viewed {
            if varying_k {
                a.matmul(viewed).unwrap()
            } else {
                a.matmul_t(viewed).unwrap()
            }
        } else {
            a.matmul_t(&b).unwrap()
        };
        let plan = {
            let mut g = h.state().egraph.lock();
            let node = g.node(out.id).clone();
            let facts = g.facts_view(out.id, &caps);
            let lower = match family {
                Family::Sgemm => lower_sgemm,
                Family::Sgemv => lower_sgemv,
                Family::Coop => lower_coop,
            };
            let mut variant =
                lower(&mut g.builder(&caps), out.id, &node, &facts).unwrap_or_else(|| {
                    panic!("no {family:?} varying_k={varying_k} kv_cache={kv_cache}")
                });
            if kv_cache {
                // Read the live KV prefix at its physical head stride.
                let mut op = g.node(variant).op.clone();
                let Op::Launch(Launch::Contract { b: side, .. }) = &mut op else {
                    unreachable!()
                };
                side.ops[0].src = b.id;
                side.ops[0].layout = fusor_ir::shape::Layout::from_parts(
                    Dim::Const(0),
                    &[Dim::Const(batch), k, n],
                    &[
                        capacity_dim * Dim::Const(128),
                        Dim::Const(if varying_k { 128 } else { 1 }),
                        Dim::Const(if varying_k { 1 } else { 128 }),
                    ],
                )
                .unwrap();
                variant = g.add(op).unwrap();
                g.union(out.id, variant).unwrap();
            }
            let Op::Launch(Launch::Contract { sched, .. }) = &g.node(variant).op else {
                unreachable!()
            };
            let point = sched
                .iter()
                .find(|point| match point {
                    SchedPoint::Sgemv(p) if kv_cache => {
                        *p == if columns {
                            fusor_ir::ir::launch::SgemvParams {
                                vector: 32,
                                subgroups: 2,
                                cols: 4,
                                parts: 4,
                                gap: 32,
                            }
                        } else {
                            fusor_ir::ir::launch::SgemvParams {
                                vector: 16,
                                subgroups: 4,
                                cols: 1,
                                parts: 1,
                                gap: 0,
                            }
                        }
                    }
                    SchedPoint::Sgemv(p) => p.cols == if columns { 4 } else { 1 },
                    SchedPoint::Sgemm(p) => p.tn > 1,
                    _ => true,
                })
                .unwrap();
            let members = [a.id, b.id, variant];
            replan_with(&session, &g, &[out.id], &members, variant, Some(point))
        };
        assert_eq!(plan.launches.len(), 1);
        let resolving = h.state().resolve_lock.lock();
        h.bind_dim(sym, 33);
        let dispatches = target.launcher().dispatch_count();
        let before = target.launcher().pipeline_compiles();
        let compiled = before + 1;
        let lengths = [33, 1, 15, 16, 17, 65]
            .into_iter()
            .chain([512].into_iter().filter(|_| kv_cache))
            .chain(
                [1024, 2048, 2049]
                    .into_iter()
                    .filter(|_| family == Family::Sgemv && varying_k && !kv_cache),
            );
        for len in lengths {
            h.bind_dim(sym, len);
            let capacity = len.next_power_of_two().max(64);
            h.bind_dim(capacity_sym, capacity);
            let (n, k) = if varying_k {
                (fixed_n, len)
            } else {
                (len, fixed_k)
            };
            let mut av: Vec<f32> = (0..batch * rows * k)
                .map(|i| ((i * 7 % 17) as f32 - 8.0) / 8.0)
                .collect();
            let b_elements = if kv_cache {
                batch * capacity * 128
            } else {
                batch * n * k
            };
            let bv: Vec<f32> = (0..b_elements)
                .map(|i| ((i * 11 % 19) as f32 - 9.0) / 8.0)
                .collect();
            a.set_bytes(bytemuck::cast_slice(&av).to_vec()).unwrap();
            b.set_bytes(bytemuck::cast_slice(&bv).to_vec()).unwrap();
            if len == 33 {
                let a_bytes = h.leaf_bytes_shared(a.id).unwrap();
                let b_bytes = h.leaf_bytes_shared(b.id).unwrap();
                let buffers = target
                    .prepare_resources(
                        &plan,
                        &h.state().egraph.lock(),
                        &h.dim_bindings(),
                        &[0],
                        vec![Arc::clone(&a_bytes), Arc::clone(&b_bytes)],
                    )
                    .unwrap()
                    .unwrap()
                    .join()
                    .unwrap()
                    .unwrap();
                assert_eq!(target.launcher().pipeline_compiles(), compiled);
                assert_eq!(target.launcher().dispatch_count(), dispatches);
                h.bind_prepared_leaf(a.id, &a_bytes, buffers[0].clone());
                h.bind_prepared_leaf(b.id, &b_bytes, buffers[1].clone());
                assert_eq!(h.device_buf(a.id).unwrap().addr(), buffers[0].addr());
                assert_eq!(h.device_buf(b.id).unwrap().addr(), buffers[1].addr());
                if family == Family::Sgemm && !varying_k {
                    av[0] += 1.0;
                    a.set_bytes(bytemuck::cast_slice(&av).to_vec()).unwrap();
                    h.bind_prepared_leaf(a.id, &a_bytes, buffers[0].clone());
                    assert!(h.device_buf(a.id).is_none());
                    h.bind_prepared_leaf(b.id, &b_bytes, buffers[0].clone());
                    assert_eq!(h.device_buf(b.id).unwrap().addr(), buffers[1].addr());
                }
            }
            session.run(h, &plan, std::slice::from_ref(&out)).unwrap_or_else(|error| {
                panic!("{family:?} varying_k={varying_k} columns={columns} kv_cache={kv_cache} length={len}: {error}")
            });
            let bytes = session.read_bytes_locked(&resolving, h, out.id).unwrap();
            let actual = bytemuck::cast_slice::<u8, f32>(&bytes);
            let (batch, rows, n, k, capacity) = (
                batch as usize,
                rows as usize,
                n as usize,
                k as usize,
                capacity as usize,
            );
            assert_eq!(actual.len(), batch * rows * n);
            for row in 0..batch * rows {
                for col in 0..n {
                    let expected: f32 = (0..k)
                        .map(|i| {
                            let b_index = if kv_cache {
                                (row / rows) * capacity * 128
                                    + if varying_k {
                                        i * 128 + col
                                    } else {
                                        col * 128 + i
                                    }
                            } else {
                                ((row / rows) * n + col) * k + i
                            };
                            av[row * k + i] * bv[b_index]
                        })
                        .sum();
                    assert!(
                        (actual[row * n + col] - expected).abs() < 1e-4,
                        "{family:?} k={k} n={n} at [{row}, {col}]: {} != {expected}",
                        actual[row * n + col]
                    );
                }
            }
            executions += 1;
            let count = target.launcher().pipeline_compiles();
            assert_eq!(compiled, count, "{family:?} recompiled at length {len}");
        }
    }
    eprintln!("verified {executions} matrix executions and pipeline reuse");

    // A persisted winner must be the first compiled production plan.
    let graph = Graph::new(&session);
    let h = graph.handle();
    let a = Tensor::from_elements(h, &[Dim::ONE, Dim::Const(521)], &vec![0.125f32; 521]).unwrap();
    let b = Tensor::from_elements(
        h,
        &[Dim::Const(1009), Dim::Const(521)],
        &vec![0.25f32; 1009 * 521],
    )
    .unwrap();
    let out = a.matmul_t(&b).unwrap();
    let (base, key, field, base_label, winner) = {
        use fusor_cost::extract::{incumbent_signature, launch_signature};
        let mut g = h.state().egraph.lock();
        let node = g.node(out.id).clone();
        let facts = g.facts_view(out.id, &caps);
        let variant = lower_sgemv(&mut g.builder(&caps), out.id, &node, &facts).unwrap();
        let members = [a.id, b.id, variant];
        let base = Arc::new(replan_with(
            &session,
            &g,
            &[out.id],
            &members,
            variant,
            None,
        ));
        let field = launch_signature(&g, &base.launches[0]);
        let label = incumbent_signature(&g, &base, 0).unwrap();
        let winner = session
            .inner
            .extractor
            .launch_variant_labels(
                &g,
                &[out.id],
                &base,
                0,
                session.inner.cost.as_ref(),
                flags().autotune_min_macs,
            )
            .into_iter()
            .next()
            .unwrap()
            .0;
        let key = ReplayKey {
            l0_term: fusor_cost::replay::l0_term_hash(&g, &[out.id]),
            device: session.inner.cost.facts().fingerprint(),
        };
        (base, key, field, label, winner)
    };
    for _ in 0..2 {
        session.inner.tune.observe(&field, &base_label, 100);
        session.inner.tune.observe(&field, &winner, 1);
    }
    let before = target.launcher().pipeline_compiles();
    let selected = session.explore_prior(h, &[out.id], key, base);
    assert_eq!(target.launcher().pipeline_compiles(), before);
    assert_eq!(
        fusor_cost::extract::incumbent_signature(&h.state().egraph.lock(), &selected, 0),
        Some(winner)
    );
    assert_eq!(run_f32(&session, h, &selected, &out), [16.28125; 1009]);
    assert_eq!(target.launcher().pipeline_compiles(), before + 1);
    eprintln!("verified persisted winner executes with one pipeline compile");
}

/// `sum(a @ b + 1, 1)` over fixed `[2, 3]` and `[3, 2]` operands, which is
/// `[124, 295]`. Returns `(a, out)`.
#[cfg(feature = "cpu")]
fn matmul_row_sums(h: &GraphRef) -> (Tensor, Tensor) {
    let a = Tensor::from_elements(
        h,
        &[Dim::Const(2), Dim::Const(3)],
        &[1.0f32, 2., 3., 4., 5., 6.],
    )
    .unwrap();
    let b = Tensor::from_elements(
        h,
        &[Dim::Const(3), Dim::Const(2)],
        &[7.0f32, 8., 9., 10., 11., 12.],
    )
    .unwrap();
    let out = a.matmul(&b).unwrap().add_scalar(1.0).unwrap();
    (a, out.sum(1).unwrap())
}

/// Saturate `out`'s closure alone under `budget`, which must stop short of
/// saturation, then extract a plan from what it reached.
#[cfg(feature = "cpu")]
fn plan_under_budget(
    session: &Session,
    h: &GraphRef,
    out: &Tensor,
    budget: SaturationBudget,
) -> (fusor_ir::egraph::SaturationReport, Plan) {
    let mut g = h.state().egraph.lock();
    g.clear_roots();
    g.add_root(out.id);
    let report = Driver::new()
        .saturate(&mut g, &session.caps(), &session.inner.rules, budget)
        .unwrap();
    assert!(!report.saturated);
    let plan = session
        .inner
        .extractor
        .extract(
            &g,
            &[out.id],
            session.inner.cost.as_ref(),
            ExtractBudget::default(),
        )
        .unwrap();
    (report, plan)
}

#[test]
#[cfg(feature = "cpu")]
fn saturation_budget_ignores_completed_graph_history() {
    let mut applications = Vec::new();
    for budget in [
        SaturationBudget {
            node_slope: 2,
            node_slack: 0,
            ..Default::default()
        },
        SaturationBudget {
            node_slope: 256,
            max_applications: 0,
            application_slope: 1,
            ..Default::default()
        },
        SaturationBudget {
            max_applications: 0,
            ..Default::default()
        },
    ] {
        let mut expansions = Vec::new();
        for history in [0, 32] {
            let session = Session::new(Backend::cpu().unwrap()).unwrap();
            let graph = Graph::new(&session);
            let h = graph.handle();
            if history > 0 {
                let input = Tensor::from_elements(h, &[Dim::ONE], &[2.0f32]).unwrap();
                let old: Vec<_> = (0..history)
                    .map(|i| input.add_scalar(i as f32).unwrap())
                    .collect();
                let mut g = h.state().egraph.lock();
                for value in &old {
                    g.add_root(value.id);
                }
                Driver::new()
                    .saturate(
                        &mut g,
                        &session.caps(),
                        &session.inner.rules,
                        Default::default(),
                    )
                    .unwrap();
            }
            let (_, out) = matmul_row_sums(h);
            let (report, plan) = plan_under_budget(&session, h, &out, budget);
            expansions.push((
                report.final_nodes - report.initial_nodes,
                report.applications,
            ));
            assert_eq!(run_f32(&session, h, &plan, &out), [124.0, 295.0]);
        }
        assert_eq!(expansions[0], expansions[1]);
        applications.push(expansions[0].1);
    }
    assert!(applications[1] > applications[2]);
}

#[test]
#[cfg(feature = "cpu")]
fn exhausted_saturation_still_executes_the_lowering_floor() {
    let session = Session::new(Backend::cpu().unwrap()).unwrap();
    let graph = Graph::new(&session);
    let h = graph.handle();
    let (a, out) = matmul_row_sums(h);
    let unrelated = a.add_scalar(5.0).unwrap();
    let execute_floor = |out: &Tensor| {
        let budget = SaturationBudget {
            max_applications: 0,
            ..SaturationBudget::default()
        };
        let (_, plan) = plan_under_budget(&session, h, out, budget);
        run_f32(&session, h, &plan, out)
    };
    assert_eq!(execute_floor(&out), [124.0, 295.0]);
    {
        let mut g = h.state().egraph.lock();
        g.add_root(a.id);
        let before = g.len();
        let report = Driver::new()
            .saturate(
                &mut g,
                &session.caps(),
                &session.inner.rules,
                SaturationBudget::default(),
            )
            .unwrap();
        assert_eq!(
            report.applications, 0,
            "revisited an exhausted root closure"
        );
        assert_eq!(g.len(), before);
    }
    assert_eq!(execute_floor(&unrelated), [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
}

/// The same expression rebuilt over a fresh input leaf must run the
/// recorded plan again rather than extract another: the graph grew, so
/// the replay key cannot hit, and the structural memo is what remains.
fn a_fresh_step_leaf_reuses_the_plan(session: Session) {
    let graph = Graph::new(&session);
    let run = |values: &[f32]| {
        let x = Tensor::from_elements(graph.handle(), &[Dim::Const(values.len() as u64)], values)
            .unwrap();
        let y = x.add_scalar(1.0).unwrap();
        let bytes = graph.handle().read_back(y.id).unwrap();
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    };

    assert_eq!(run(&[1.0, 2.0, 3.0]), vec![2.0, 3.0, 4.0]);
    assert_eq!(session.inner.families.lock().len(), 1);

    assert_eq!(run(&[10.0, 11.0, 12.0]), vec![11.0, 12.0, 13.0]);
    assert_eq!(
        session.inner.families.lock().len(),
        1,
        "a replay hit must not extract and record another plan"
    );
}

/// A `Coop` contraction pads its output to whole blocks, and a view of
/// that output is served by cutting the graph at the bound buffer. The
/// cut mints an external leaf, which carries no `BufferPlan` for
/// `repad_index` to correct the read with, so the view must not be cut
/// there — it read the padding as data.
#[test]
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn a_view_of_a_padded_contraction_reads_the_value_not_its_padding() {
    let Ok(backend) = Backend::gpu_blocking() else {
        return;
    };
    let session = Session::new(backend).unwrap();
    let graph = Graph::new(&session);
    let h = graph.handle();
    // `n = 130` is not a multiple of any block width, so every geometry
    // the extractor can pick pads it; the extents are large enough that
    // it picks a cooperative one.
    const T: u64 = 512;
    const K: u64 = 512;
    const N: u64 = 130;
    let xs: Vec<f32> = (0..T * K)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let ws: Vec<f32> = (0..N * K)
        .map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0)
        .collect();
    let x = Tensor::from_elements(h, &[Dim::Const(T), Dim::Const(K)], &xs).unwrap();
    let w = Tensor::from_elements(h, &[Dim::Const(N), Dim::Const(K)], &ws).unwrap();
    let y = x.matmul_t(&w).unwrap();
    let flat = y
        .reshape_dims(&[Dim::Const(1), Dim::Const(T), Dim::Const(N)])
        .unwrap();

    let full = h.read_back(y.id).unwrap();
    let full = bytemuck::cast_slice::<u8, f32>(&full).to_vec();
    let view = h.read_back(flat.id).unwrap();
    let view = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    assert_eq!(full.len(), view.len());
    let worst = full
        .iter()
        .zip(&view)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "a view of the contraction differs by {worst}");
}

/// A model step rebuilt with a longer cache every call (a `cat` onto a
/// re-leafed cache, a view at a moving offset, a contraction over the
/// cache length): from the second call on the session plans a symbolic
/// twin, and every call's values must equal the host's.
#[test]
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn a_shape_family_twin_computes_what_its_members_do() {
    let Ok(backend) = Backend::gpu_blocking() else {
        return;
    };
    let session = Session::new(backend).unwrap();
    let graph = Graph::new(&session);
    let h = graph.handle();
    const D: usize = 8;
    let w_host: Vec<f32> = (0..D * D)
        .map(|i| ((i * 7) % 11) as f32 * 0.1 - 0.4)
        .collect();
    let w =
        Tensor::from_elements(h, &[Dim::Const(D as u64), Dim::Const(D as u64)], &w_host).unwrap();
    let mut cache_host: Vec<f32> = Vec::new();
    for step in 0..(CONCRETE_SHAPES + 4) {
        let new_row: Vec<f32> = (0..D).map(|j| (step * D + j) as f32 * 0.05 - 0.3).collect();
        // The step: cache' = cat(cache, row); q = row @ w; scores =
        // q @ cache'^T (contraction over D, N = len); s = softmax(scores);
        // out = s @ cache' (contraction over len); tail = cache' narrowed
        // at the moving offset `step`.
        let row =
            Tensor::from_elements(h, &[Dim::Const(1), Dim::Const(D as u64)], &new_row).unwrap();
        let cache = if cache_host.is_empty() {
            row.clone()
        } else {
            let prev = Tensor::from_elements(
                h,
                &[
                    Dim::Const((cache_host.len() / D) as u64),
                    Dim::Const(D as u64),
                ],
                &cache_host,
            )
            .unwrap();
            Tensor::cat(&[prev, row.clone()], 0).unwrap()
        };
        cache_host.extend_from_slice(&new_row);
        let len = cache_host.len() / D;
        let q = row.matmul(&w).unwrap();
        let scores = q.matmul(&cache.t().unwrap()).unwrap();
        let s = scores.softmax(1).unwrap();
        let out = s.matmul(&cache).unwrap();
        let tail = cache.narrow(0, step, 1).unwrap();
        let got_out: Vec<f32> = bytemuck::cast_slice(&h.read_back(out.id).unwrap()).to_vec();
        let got_tail: Vec<f32> = bytemuck::cast_slice(&h.read_back(tail.id).unwrap()).to_vec();
        let got_cache: Vec<f32> = bytemuck::cast_slice(&h.read_back(cache.id).unwrap()).to_vec();

        // Host reference.
        let q_h: Vec<f32> = (0..D)
            .map(|j| (0..D).map(|k| new_row[k] * w_host[k * D + j]).sum())
            .collect();
        let sc: Vec<f32> = (0..len)
            .map(|r| (0..D).map(|k| q_h[k] * cache_host[r * D + k]).sum())
            .collect();
        let m = sc.iter().cloned().fold(f32::MIN, f32::max);
        let e: Vec<f32> = sc.iter().map(|v| (v - m).exp()).collect();
        let z: f32 = e.iter().sum();
        let out_h: Vec<f32> = (0..D)
            .map(|j| (0..len).map(|r| e[r] / z * cache_host[r * D + j]).sum())
            .collect();
        let close = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4)
        };
        assert!(
            close(&got_cache, &cache_host),
            "step {step}: cache {got_cache:?}"
        );
        assert!(
            close(&got_tail, &cache_host[step * D..(step + 1) * D]),
            "step {step}: tail {got_tail:?}"
        );
        assert!(
            close(&got_out, &out_h),
            "step {step}: out {got_out:?} vs {out_h:?}"
        );
    }
    assert!(
        session
            .inner
            .families
            .lock()
            .values()
            .any(|f| f.symbolic.is_some()),
        "the step's shape family never went symbolic"
    );
}

/// The smallest moving-offset view: a fresh `[len, D]` leaf narrowed at
/// row `step`. The twin's view offset is a derived symbol.
fn shape_family_symbolic_offset(session: Session) {
    let graph = Graph::new(&session);
    let h = graph.handle();
    const D: usize = 4;
    for step in 0..(CONCRETE_SHAPES + 4) {
        let len = step + 2;
        let host: Vec<f32> = (0..len * D).map(|i| i as f32).collect();
        let x = Tensor::from_elements(h, &[Dim::Const(len as u64), Dim::Const(D as u64)], &host)
            .unwrap();
        let tail = x.narrow(0, step, 1).unwrap().add_scalar(0.0).unwrap();
        let got: Vec<f32> = bytemuck::cast_slice(&h.read_back(tail.id).unwrap()).to_vec();
        assert_eq!(got, host[step * D..(step + 1) * D], "step {step}");
    }
    assert!(
        session
            .inner
            .families
            .lock()
            .values()
            .any(|family| family.symbolic.is_some() && !family.blocked),
        "symbolic offsets must execute without falling back to concrete plans"
    );
}

/// One test per backend over a `fn(Session)` body; a missing GPU skips.
macro_rules! on_each_backend {
    ($body:ident: $cpu:ident, $gpu:ident) => {
        #[test]
        #[cfg(feature = "cpu")]
        fn $cpu() {
            $body(Session::new(Backend::cpu().unwrap()).unwrap());
        }

        #[test]
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        fn $gpu() {
            let Ok(backend) = Backend::gpu_blocking() else {
                return;
            };
            $body(Session::new(backend).unwrap());
        }
    };
}

on_each_backend!(
    shape_family_symbolic_offset: a_shape_family_twin_reads_a_cpu_view_at_a_symbolic_offset,
    a_shape_family_twin_reads_a_view_at_a_symbolic_offset
);
on_each_backend!(
    a_fresh_step_leaf_reuses_the_plan: a_fresh_step_leaf_reuses_the_cpu_plan_and_executable,
    a_fresh_step_leaf_reuses_the_gpu_plan
);
