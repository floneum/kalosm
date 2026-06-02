use super::*;

impl Resolver {
    pub(super) fn try_fuse_paired_qmatmul(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        // The paired-mode QMatMul kernel emitted by this rewrite uses
        // `subgroup_id` / `subgroup_reduce_*` and has no workgroup-only
        // fallback. The resolver later panics if `build_direct_kernel`
        // returns None for any operation it scheduled, so refuse the
        // rewrite up-front on adapters without a trusted subgroup path. The
        // unfused source still resolves via the regular qmatmul + epilogue
        // kernels.
        let device = graph.device();
        if !device.subgroups_supported() {
            return false;
        }
        let ComputeGraphNodeVariant::Nary(nary) = self.execution_graph[node_idx].variant.clone()
        else {
            return false;
        };
        let Some(split) = nary.try_extract_paired_split() else {
            return false;
        };

        // Classify each Nary input as either a matmul-view (MapLayout
        // whose source is the same QMatMul) or an extra (broadcast bias
        // vector). Broadcasts in fusor are emitted as `MapLayout(rank-1
        // leaf)` with stride-0 dims, so a MapLayout can appear for either
        // role; disambiguate by the source-node kind. The original (un-
        // broadcasted) NodeIndex is stored for extras so the kernel can
        // load the rank-1 tensor directly with `bias[col]` semantics.
        // Each matmul-view slot carries the chain of MapLayouts that wrap the
        // common qmatmul (innermost-first), so we can compose them onto the
        // qmatmul's base layout to recover the view's effective layout.
        let mut matmul_view_slots: Vec<(
            usize,
            ExecutionNodeIndex,
            Vec<crate::map_layout::MapLayoutOperation>,
        )> = Vec::new();
        let mut extra_slots: Vec<(usize, NodeIndex)> = Vec::new();
        let mut common_qmatmul_inner: Option<NodeIndex> = None;
        for (slot_idx, input_inner) in split.inputs.iter().enumerate() {
            if !split.inputs_seen[slot_idx] {
                return false;
            }
            let Some(exec_idx) = self.get_input_node_in_exec_graph(*input_inner) else {
                return false;
            };
            match self.execution_graph[exec_idx].variant.clone() {
                ComputeGraphNodeVariant::MapLayout(_map) => {
                    // Walk the MapLayout chain to the ultimate source. fusor
                    // can stack MapLayouts (e.g. narrow + broadcast_as), and
                    // any QMatMul-rooted chain is treated as a matmul-view
                    // (its composed layout closure is invoked later).
                    let (source_inner, composed) = match self.walk_map_layout_chain(*input_inner) {
                        Some(walked) => walked,
                        None => return false,
                    };
                    let source_exec = self.get_input_node_in_exec_graph(source_inner);
                    let source_is_qmatmul = source_exec
                        .map(|e| {
                            matches!(
                                self.execution_graph[e].variant,
                                ComputeGraphNodeVariant::QMatMul(_)
                            )
                        })
                        .unwrap_or(false);
                    if source_is_qmatmul {
                        match common_qmatmul_inner {
                            None => common_qmatmul_inner = Some(source_inner),
                            Some(existing) if existing == source_inner => {}
                            Some(_) => {
                                return false;
                            }
                        }
                        matmul_view_slots.push((slot_idx, exec_idx, composed));
                    } else {
                        // Broadcast-extra path: chain bottoms at a leaf
                        // tensor (e.g. bias). Use the source NodeIndex so
                        // the kernel loads the un-broadcasted rank-1
                        // tensor as `Storage<F32, 1>`.
                        extra_slots.push((slot_idx, source_inner));
                    }
                }
                _ => {
                    extra_slots.push((slot_idx, *input_inner));
                }
            }
        }
        if matmul_view_slots.len() != 2 {
            return false;
        }
        let Some(qmatmul_inner) = common_qmatmul_inner else {
            return false;
        };

        // Common source must be an uncached QMatMul.
        if self.check_cached(graph, qmatmul_inner) {
            return false;
        }
        let Some(qmatmul_exec) = self.get_input_node_in_exec_graph(qmatmul_inner) else {
            return false;
        };
        let ComputeGraphNodeVariant::QMatMul(qmatmul_op) =
            self.execution_graph[qmatmul_exec].variant.clone()
        else {
            return false;
        };
        let Some(&m_size) = qmatmul_op
            .in_shape
            .get(qmatmul_op.in_shape.len().saturating_sub(2))
        else {
            return false;
        };
        if m_size != 1 {
            return false;
        }
        if qmatmul_op.matrix.datatype() != GgmlType::Q4K {
            return false;
        }

        // The qmatmul produces a contiguous output of its declared out_shape;
        // apply each MapLayout closure to that base layout to recover the
        // gate/up views.
        let qmatmul_layout = Layout::contiguous(&qmatmul_op.out_shape);
        let (a_slot, a_exec, a_map) = matmul_view_slots[0].clone();
        let (b_slot, b_exec, b_map) = matmul_view_slots[1].clone();
        let a_layout = Self::apply_map_layout_chain(&qmatmul_layout, &a_map);
        let b_layout = Self::apply_map_layout_chain(&qmatmul_layout, &b_map);

        let q_shape = qmatmul_layout.shape();
        let a_shape = a_layout.shape();
        let b_shape = b_layout.shape();
        if q_shape.len() != a_shape.len() || q_shape.len() != b_shape.len() || q_shape.is_empty() {
            return false;
        }
        let last = q_shape.len() - 1;
        let pair_len = a_shape[last];
        if pair_len == 0 || b_shape[last] != pair_len || q_shape[last] != pair_len * 2 {
            return false;
        }
        for axis in 0..last {
            if a_shape[axis] != q_shape[axis] || b_shape[axis] != q_shape[axis] {
                return false;
            }
        }
        let q_strides = qmatmul_layout.strides();
        if a_layout.strides() != q_strides || b_layout.strides() != q_strides {
            return false;
        }
        let last_stride = q_strides[last];
        let q_offset = qmatmul_layout.offset();
        let Some(shift) = pair_len.checked_mul(last_stride) else {
            return false;
        };
        let Some(b_expected_offset) = q_offset.checked_add(shift) else {
            return false;
        };
        // Determine which slot is the gate (offset 0) and which is up
        // (offset pair_len). The kernel always passes (gate, up, extras...)
        // to the epilogue closure, but the captured NaryExpr's
        // `IndexedInput(i, _)` slots refer to the original Nary input
        // ordering. Build a permutation: gate_input_idx -> slot 0,
        // up_input_idx -> slot 1, extra_k's input_idx -> slot 2+k.
        let (gate_input_idx, up_input_idx) =
            if a_layout.offset() == q_offset && b_layout.offset() == b_expected_offset {
                (a_slot, b_slot)
            } else if b_layout.offset() == q_offset && a_layout.offset() == b_expected_offset {
                (b_slot, a_slot)
            } else {
                return false;
            };

        // Permutation: for each captured-NaryExpr input slot, the closure
        // tile-slice index it maps to.
        let total_inputs = split.inputs.len();
        let mut input_slot_to_tile_idx = vec![usize::MAX; total_inputs];
        input_slot_to_tile_idx[gate_input_idx] = 0;
        input_slot_to_tile_idx[up_input_idx] = 1;
        let extras_order: Vec<NodeIndex> = extra_slots
            .iter()
            .enumerate()
            .map(|(k, (slot_idx, inner))| {
                input_slot_to_tile_idx[*slot_idx] = 2 + k;
                *inner
            })
            .collect();
        let extras_count = extras_order.len();

        // Synthesize the epilogue closure. Captures the recorded NaryExpr
        // and re-emits it at tile-IR level inside the qgemv kernel,
        // substituting each input slot with the corresponding tile from the
        // closure's slice via `input_slot_to_tile_idx`.
        let expression = split.expression.clone();
        let datatype = qmatmul_op.input_datatype;
        let permutation = input_slot_to_tile_idx.clone();
        let epilogue = fusor_tile_ir_kernels::PairedEpilogue::with_extras(
            "autofused_with_extras",
            extras_count,
            move |tiles| {
                // Re-order the closure's tile slice to match the captured
                // NaryExpr's input numbering.
                let inputs: Vec<(fusor_tile_ir::tile::Tile, DataTypeEnum)> = permutation
                    .iter()
                    .map(|&tile_idx| (tiles[tile_idx].clone(), datatype))
                    .collect();
                let (tile, _) = eval_nary_expr_on_tiles(&expression, &inputs);
                tile
            },
        );

        let paired_op = QMatMulOperation::new_paired(
            qmatmul_op.input_datatype,
            &qmatmul_op.in_shape,
            qmatmul_op.input,
            qmatmul_op.matrix.clone(),
            pair_len,
            epilogue,
            extras_order,
        );

        // Rewrite this Nary's variant in place; the produced shape and edges
        // already match the new operation.
        self.execution_graph[node_idx].variant =
            ComputeGraphNodeVariant::QMatMul(Box::new(paired_op));

        // Re-wire incoming dependency edges. The new op consumes the qmatmul
        // input *and* every extra; the matmul views and the qmatmul itself
        // are now dead.
        let qmatmul_input_inner = qmatmul_op.input;
        let mut new_deps = vec![qmatmul_input_inner];
        for (_, extra_inner) in &extra_slots {
            new_deps.push(*extra_inner);
        }
        for dep in &new_deps {
            if let Some(input_exec) = self.get_input_node_in_exec_graph(*dep)
                && self
                    .execution_graph
                    .find_edge(input_exec, node_idx)
                    .is_none()
            {
                self.execution_graph.add_edge(input_exec, node_idx, ());
            }
        }
        for stale in [a_exec, b_exec] {
            if let Some(edge) = self.execution_graph.find_edge(stale, node_idx) {
                self.execution_graph.remove_edge(edge);
            }
        }
        self.add_physical_dependencies(graph, node_idx, &new_deps);
        for stale in [a_exec, b_exec, qmatmul_exec] {
            self.remove_node_if_dead(stale);
        }
        true
    }
}
