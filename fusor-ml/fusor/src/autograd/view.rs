use std::ops::Range;

use crate::{Dim, Layout};
use fusor_types::StrideSpec;

use super::*;

impl<const R: usize> Tensor<R> {
    pub fn reshape<const OUT: usize>(&self, shape: [usize; OUT]) -> Tensor<OUT> {
        let input_shape = self.shape();
        let value = self.value.reshape(shape).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "reshape")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.reshape(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let value = self.value.transpose(dim0, dim1).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "transpose")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.transpose(dim0, dim1).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn permute(&self, axes: [usize; R]) -> Self {
        let value = self.value.permute(axes).to_concrete();
        let input_id = self.handle.id;
        let mut inverse = [0usize; R];
        for (index, axis) in axes.iter().copied().enumerate() {
            inverse[axis] = index;
        }
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "permute")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.permute(inverse).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn slice(&self, slices: [Range<usize>; R]) -> Self {
        let input_shape = self.shape();
        let value = self.value.slice(slices.clone()).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "slice")?;
            let zeros = RawTensor::zeros(&gradient.device(), input_shape);
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(zeros.slice_assign(slices.clone(), &gradient).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn broadcast_as<const OUT: usize>(&self, shape: [usize; OUT]) -> Tensor<OUT> {
        let input_shape = self.shape();
        let value = self.value.broadcast_as(shape).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "broadcast_as")?;
            let reduced = reduce_broadcast_gradient(gradient, input_shape)?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: reduced,
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn expand<const OUT: usize>(&self, shape: [usize; OUT]) -> Tensor<OUT> {
        self.broadcast_as(shape)
    }

    pub fn flatten_all(&self) -> Tensor<1> {
        self.reshape([self.shape().iter().product()])
    }

    pub fn flatten_last_n<const FROM_END: usize, const OUT: usize>(&self) -> Tensor<OUT>
    where
        crate::gpu::Tensor<R, f32>: crate::gpu::SmallerRank<FROM_END, OUT, f32>,
    {
        let shape = self.shape();
        let new_shape: [usize; OUT] = std::array::from_fn(|i| {
            if i < R - 1 - FROM_END {
                shape[i]
            } else if i == R - 1 - FROM_END {
                shape[R - 1 - FROM_END..].iter().product()
            } else {
                1
            }
        });
        self.reshape(new_shape)
    }

    pub fn flatten_first_n<const FROM_START: usize, const OUT: usize>(&self) -> Tensor<OUT>
    where
        crate::gpu::Tensor<R, f32>: crate::gpu::SmallerRank<FROM_START, OUT, f32>,
    {
        let shape = self.shape();
        let new_shape: [usize; OUT] = std::array::from_fn(|i| {
            if i == 0 {
                shape[..=FROM_START].iter().product()
            } else {
                shape[i + FROM_START]
            }
        });
        self.reshape(new_shape)
    }

    pub fn narrow(&self, dim: impl Dim<R>, start: usize, length: usize) -> Self {
        let dim = dim.resolve();
        let shape = self.shape();
        let slices: [Range<usize>; R] = std::array::from_fn(|axis| {
            if axis == dim {
                start..start + length
            } else {
                0..shape[axis]
            }
        });
        self.slice(slices)
    }

    pub fn chunk(&self, chunks: usize, dim: impl Dim<R>) -> Vec<Self> {
        let dim = dim.resolve();
        let shape = self.shape();
        let dim_size = shape[dim];
        let chunk_size = dim_size.div_ceil(chunks);

        let mut result = Vec::with_capacity(chunks);
        let mut start = 0;
        while start < dim_size {
            let length = chunk_size.min(dim_size - start);
            result.push(self.narrow(dim, start, length));
            start += length;
        }
        result
    }

    pub fn repeat(&self, repeats: [usize; R]) -> Self {
        let input_shape = self.shape();
        let value = self.value.repeat(repeats).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "repeat")?;
            let total: usize = gradient.shape().iter().product();
            let mut flat = gradient.reshape([total]).to_concrete();
            for axis in (0..R).rev() {
                if repeats[axis] == 1 {
                    continue;
                }
                let before: usize = (0..axis)
                    .map(|dim| repeats[dim] * input_shape[dim])
                    .product();
                let after: usize = input_shape[axis + 1..].iter().product();
                flat = flat
                    .reshape([before, repeats[axis], input_shape[axis], after])
                    .to_concrete()
                    .sum::<3>(1)
                    .reshape([before * input_shape[axis] * after])
                    .to_concrete();
            }
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(flat.reshape(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn resize(&self, new_shape: [usize; R]) -> Self {
        let input_shape = self.shape();
        let value = self.value.resize(new_shape).to_concrete();
        let input_id = self.handle.id;
        let copy_shape = std::array::from_fn(|axis| input_shape[axis].min(new_shape[axis]));
        let copy_slices = copy_shape.map(|size| 0..size);
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "resize")?;
            let patch = gradient.slice(copy_slices.clone()).to_concrete();
            let zeros = RawTensor::zeros(&gradient.device(), input_shape);
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(zeros.slice_assign(copy_slices.clone(), &patch).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn restride<const OUT: usize>(&self, specs: [StrideSpec; OUT]) -> Tensor<OUT> {
        let input_shape = self.shape();
        let value = self.value.restride(specs).to_concrete();
        let input_id = self.handle.id;
        let output_shape: [usize; OUT] = specs.map(|spec| spec.size);
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "restride")?;
            let reduced = reduce_restride_gradient(&gradient, &specs, [0; R], input_shape)
                .unwrap_or_else(|| {
                    scatter_restride_gradient(&gradient, output_shape, input_shape, |output_index| {
                        restride_input_index(specs, output_index)
                    })
                });
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(reduced),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn restride_layout<const OUT: usize>(&self, new_layout: Layout) -> Tensor<OUT> {
        assert_eq!(new_layout.rank(), OUT, "restride_layout rank mismatch");
        let input_shape = self.shape();
        let value = self.value.restride_layout(new_layout.clone()).to_concrete();
        let input_id = self.handle.id;
        let output_shape: [usize; OUT] = std::array::from_fn(|axis| new_layout.shape()[axis]);
        let input_strides = Layout::continuous_strides(&input_shape);
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "restride_layout")?;
            let reduced = layout_restride_specs(&new_layout, input_shape)
                .and_then(|(specs, offsets)| {
                    reduce_restride_gradient(&gradient, &specs, offsets, input_shape)
                })
                .unwrap_or_else(|| {
                    scatter_restride_gradient(&gradient, output_shape, input_shape, |output_index| {
                        let linear = new_layout.linear_index(&output_index);
                        contiguous_index_from_linear::<R>(linear, &input_strides)
                    })
                });
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(reduced),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn squeeze_dims<const DIFF: usize, const OUT: usize>(
        &self,
        axes: [usize; DIFF],
    ) -> Tensor<OUT>
    where
        crate::gpu::Tensor<R, f32>: crate::gpu::SmallerRank<DIFF, OUT, f32>,
    {
        let shape = self.shape();
        for &axis in &axes {
            assert_eq!(shape[axis], 1, "Squeeze dimension {} must have size 1", axis);
        }
        let mut sorted_axes = axes;
        sorted_axes.sort_unstable();
        let mut input_axis = 0;
        let mut axis_index = 0;
        let specs: [StrideSpec; OUT] = std::array::from_fn(|_| {
            while axis_index < DIFF && input_axis == sorted_axes[axis_index] {
                input_axis += 1;
                axis_index += 1;
            }
            let spec = StrideSpec::dim(input_axis, shape[input_axis]);
            input_axis += 1;
            spec
        });
        self.restride(specs)
    }

    pub fn unsqueeze_dims<const DIFF: usize, const OUT: usize>(
        &self,
        axes: [usize; DIFF],
    ) -> Tensor<OUT>
    where
        crate::gpu::Tensor<R, f32>: crate::gpu::LargerRank<DIFF, OUT, f32>,
    {
        let shape = self.shape();
        let mut sorted_axes = axes;
        sorted_axes.sort_unstable();
        let mut input_axis = 0;
        let mut axis_index = 0;
        let specs: [StrideSpec; OUT] = std::array::from_fn(|output_axis| {
            if axis_index < DIFF && output_axis == sorted_axes[axis_index] {
                axis_index += 1;
                StrideSpec::dim_with(0, 1, 0)
            } else {
                let spec = StrideSpec::dim(input_axis, shape[input_axis]);
                input_axis += 1;
                spec
            }
        });
        self.restride(specs)
    }

    pub fn slice_assign(&self, slices: [Range<usize>; R], value: &Self) -> Self {
        assert_same_graph(self, value);

        let output = self.value.slice_assign(slices.clone(), &value.value).to_concrete();
        let input_id = self.handle.id;
        let value_id = value.handle.id;
        let slice_shape = slices
            .clone()
            .map(|range| range.end.saturating_sub(range.start));
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "slice_assign")?;
            let zeros = RawTensor::zeros(&gradient.device(), slice_shape);
            Ok(vec![
                BackwardTarget {
                    node: input_id,
                    gradient: Box::new(gradient.slice_assign(slices.clone(), &zeros).to_concrete()),
                },
                BackwardTarget {
                    node: value_id,
                    gradient: Box::new(gradient.slice(slices.clone()).to_concrete()),
                },
            ])
        });
        self.from_op(
            output,
            vec![self.handle.clone(), value.handle.clone()],
            Some(backward),
        )
    }

    pub fn stack<const OUT: usize>(tensors: impl IntoIterator<Item = Self>, dim: usize) -> Tensor<OUT>
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LargerRank<OUT, 1, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LargerRank<1, OUT, f32>,
    {
        let tensors: Vec<Self> = tensors.into_iter().collect();
        assert!(!tensors.is_empty(), "stack requires at least one tensor");

        let graph = tensors[0].handle.graph.clone();
        let input_shape = tensors[0].shape();
        let raw = tensors
            .iter()
            .map(|tensor| {
                assert!(
                    Arc::ptr_eq(&graph, &tensor.handle.graph),
                    "cannot mix autograd tensors from different graphs"
                );
                assert_eq!(tensor.shape(), input_shape, "stack requires matching shapes");
                tensor.value.unsqueeze_dims::<1, OUT>([dim]).to_concrete()
            })
            .collect::<Vec<_>>();
        let value = RawTensor::cat(raw, dim);
        let parents = tensors
            .iter()
            .map(|tensor| tensor.handle.clone())
            .collect::<Vec<_>>();
        let parent_ids = parents.iter().map(|parent| parent.id).collect::<Vec<_>>();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "stack")?;
            let mut targets = Vec::with_capacity(parent_ids.len());
            for (index, &parent_id) in parent_ids.iter().enumerate() {
                let slices: [Range<usize>; OUT] = std::array::from_fn(|axis| {
                    if axis == dim {
                        index..index + 1
                    } else {
                        0..gradient.shape()[axis]
                    }
                });
                let grad = gradient.slice(slices).reshape(input_shape).to_concrete();
                targets.push(BackwardTarget {
                    node: parent_id,
                    gradient: Box::new(grad),
                });
            }
            Ok(targets)
        });
        let id = graph.add_node(
            parents.iter().map(|parent| parent.id).collect(),
            Some(backward),
            parents
                .iter()
                .any(|parent| parent.graph.requires_grad(parent.id)),
        );
        Tensor {
            value,
            handle: NodeHandle { graph, id },
        }
    }
}

impl Tensor<1> {
    pub fn unsqueeze(&self, dim: usize) -> Tensor<2> {
        self.unsqueeze_dims::<1, 2>([dim])
    }
}

impl Tensor<2> {
    pub fn squeeze(&self, dim: usize) -> Tensor<1> {
        self.squeeze_dims::<1, 1>([dim])
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor<3> {
        self.unsqueeze_dims::<1, 3>([dim])
    }
}

impl Tensor<3> {
    pub fn squeeze(&self, dim: usize) -> Tensor<2> {
        self.squeeze_dims::<1, 2>([dim])
    }

    pub fn cat(tensors: Vec<Tensor<3>>, dim: usize) -> Tensor<3> {
        assert!(!tensors.is_empty(), "cat requires at least one tensor");
        let graph = tensors[0].handle.graph.clone();
        let raw = tensors
            .iter()
            .map(|tensor| tensor.value.clone())
            .collect::<Vec<_>>();
        let value = RawTensor::cat(raw, dim);
        let parents = tensors
            .iter()
            .map(|tensor| tensor.handle.clone())
            .collect::<Vec<_>>();
        let parent_ids = parents.iter().map(|parent| parent.id).collect::<Vec<_>>();
        let slices = tensors
            .iter()
            .scan(0usize, |offset, tensor| {
                let start = *offset;
                let length = tensor.shape()[dim];
                *offset += length;
                Some(start..start + length)
            })
            .collect::<Vec<_>>();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "cat")?;
            let mut targets = Vec::with_capacity(parent_ids.len());
            for (&parent_id, slice) in parent_ids.iter().zip(slices.iter()) {
                let grad_slice = match dim {
                    0 => gradient.slice([
                        slice.clone(),
                        0..gradient.shape()[1],
                        0..gradient.shape()[2],
                    ]),
                    1 => gradient.slice([
                        0..gradient.shape()[0],
                        slice.clone(),
                        0..gradient.shape()[2],
                    ]),
                    2 => gradient.slice([
                        0..gradient.shape()[0],
                        0..gradient.shape()[1],
                        slice.clone(),
                    ]),
                    _ => panic!("invalid cat dim"),
                }
                .to_concrete();
                targets.push(BackwardTarget {
                    node: parent_id,
                    gradient: Box::new(grad_slice),
                });
            }
            Ok(targets)
        });
        let id = graph.add_node(
            parents.iter().map(|parent| parent.id).collect(),
            Some(backward),
            parents
                .iter()
                .any(|parent| parent.graph.requires_grad(parent.id)),
        );
        Tensor {
            value,
            handle: NodeHandle { graph, id },
        }
    }
}

fn for_each_index<const R: usize>(limits: [usize; R], mut visitor: impl FnMut([usize; R])) {
    if limits.contains(&0) {
        return;
    }

    let mut index = [0; R];
    loop {
        visitor(index);

        let mut axis = R;
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            index[axis] += 1;
            if index[axis] < limits[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

fn restride_input_index<const R: usize, const OUT: usize>(
    specs: [StrideSpec; OUT],
    output_index: [usize; OUT],
) -> [usize; R] {
    let mut input_index = [0; R];
    for axis in 0..OUT {
        let spec = specs[axis];
        input_index[spec.input_dim] += spec.offset + output_index[axis] * spec.multiplier;
    }
    input_index
}

fn contiguous_index_from_linear<const R: usize>(
    mut linear: usize,
    strides: &[usize],
) -> [usize; R] {
    let mut input_index = [0; R];
    for axis in 0..R {
        input_index[axis] = linear / strides[axis];
        linear %= strides[axis];
    }
    input_index
}

/// Grouped view of a restride's specs for one input axis: at most one strided
/// "position" run and one unit-stride "window" run plus a constant offset, so
/// the forward map factors per input axis as
/// `input = offset + position * step + window`.
#[derive(Clone, Copy)]
struct RestrideRuns {
    offset: usize,
    /// `(output_axis, step, count)` of the strided run.
    position: Option<(usize, usize, usize)>,
    /// `(output_axis, size)` of the unit-stride run.
    window: Option<(usize, usize)>,
}

impl RestrideRuns {
    fn counts(&self) -> (usize, usize, usize) {
        let (_, step, positions) = self.position.unwrap_or((0, 1, 1));
        let (_, window) = self.window.unwrap_or((0, 1));
        (positions, step, window)
    }

    fn output_len(&self) -> usize {
        let (positions, _, window) = self.counts();
        positions * window
    }

    fn fold_len(&self) -> usize {
        let (positions, step, window) = self.counts();
        (positions - 1) * step + window
    }

    /// True when the runs enumerate the input axis exactly once in row-major
    /// order, so the gradient maps back with a plain reshape.
    fn is_reshape(&self, size: usize) -> bool {
        let (positions, step, window) = self.counts();
        self.offset == 0 && self.fold_len() == size && (positions == 1 || step == window)
    }
}

fn group_restride_runs<const IN: usize>(
    specs: &[StrideSpec],
    base_offsets: [usize; IN],
    input_shape: [usize; IN],
) -> Option<[RestrideRuns; IN]> {
    let mut runs: [RestrideRuns; IN] = std::array::from_fn(|axis| RestrideRuns {
        offset: base_offsets[axis],
        position: None,
        window: None,
    });
    for (output_axis, spec) in specs.iter().enumerate() {
        let axis = &mut runs[spec.input_dim];
        axis.offset += spec.offset;
        if spec.size == 1 {
            continue;
        }
        if spec.multiplier == 1 && axis.window.is_none() {
            axis.window = Some((output_axis, spec.size));
        } else if spec.multiplier >= 1 && axis.position.is_none() {
            axis.position = Some((output_axis, spec.multiplier, spec.size));
        } else {
            return None;
        }
    }
    runs.iter()
        .zip(&input_shape)
        .all(|(axis, &size)| axis.offset + axis.fold_len() <= size)
        .then_some(runs)
}

/// Backward of a position/window restride as compiled graph ops: group the
/// output axes per input axis, canonicalize their order with one permute,
/// then fold each axis with [`fold_restride_axis`]. Returns `None` for
/// restrides that do not factor into per-axis runs; those go through the
/// host-loop fallback.
fn reduce_restride_gradient<const IN: usize, const OUT: usize>(
    gradient: &RawTensor<OUT, f32>,
    specs: &[StrideSpec],
    base_offsets: [usize; IN],
    input_shape: [usize; IN],
) -> Option<RawTensor<IN, f32>> {
    let output_shape = gradient.shape();
    if input_shape.contains(&0) || output_shape.contains(&0) {
        return None;
    }
    let runs = group_restride_runs(specs, base_offsets, input_shape)?;

    let mut order = Vec::with_capacity(OUT);
    for axis in &runs {
        if let Some((output_axis, _, _)) = axis.position {
            order.push(output_axis);
        }
        if let Some((output_axis, _)) = axis.window {
            order.push(output_axis);
        }
    }
    // Size-1 output axes reshape away wherever they sit, so only the run axes
    // decide whether the gradient needs a permute into canonical order.
    let canonical = if order.windows(2).all(|pair| pair[0] < pair[1]) {
        gradient.clone()
    } else {
        let mut in_runs = [false; OUT];
        for &axis in &order {
            in_runs[axis] = true;
        }
        order.extend((0..OUT).filter(|&axis| !in_runs[axis]));
        let permutation: [usize; OUT] = order
            .as_slice()
            .try_into()
            .expect("every output axis appears in the permutation once");
        gradient.permute(permutation).to_concrete()
    };

    let mut flat = canonical
        .reshape([output_shape.iter().product()])
        .to_concrete();
    let mut after = 1usize;
    for dim in (0..IN).rev() {
        let size = input_shape[dim];
        if !runs[dim].is_reshape(size) {
            let before = runs[..dim].iter().map(RestrideRuns::output_len).product();
            let (positions, step, window) = runs[dim].counts();
            flat = fold_restride_axis(
                flat,
                before,
                positions,
                step,
                window,
                runs[dim].offset,
                size,
                after,
            );
        }
        after *= size;
    }
    Some(flat.reshape(input_shape).to_concrete())
}

/// Scatter-add one folded axis of the gradient with padded views and a
/// reduce: `out[offset + p*step + w] += g[.., p, w, ..]` for every position
/// `p` and window element `w`, with the surrounding axes flattened into
/// `before` and `after` batch extents.
#[allow(clippy::too_many_arguments)]
fn fold_restride_axis(
    flat: RawTensor<1, f32>,
    before: usize,
    positions: usize,
    step: usize,
    window: usize,
    offset: usize,
    size: usize,
    after: usize,
) -> RawTensor<1, f32> {
    let fold_len = (positions - 1) * step + window;
    let block = flat
        .reshape([before, positions, window, after])
        .to_concrete();
    let folded: RawTensor<3, f32> = if positions == 1 {
        block.reshape([before, window, after]).to_concrete()
    } else if step >= window {
        // Injective: interleave the windows with zeros and trim the overhang.
        block
            .pad_with_zeros(2, 0, step - window)
            .reshape([before, positions * step, after])
            .to_concrete()
            .narrow(1usize, 0, fold_len)
            .to_concrete()
    } else {
        // Overlapping: reverse the window axis (`u = window - 1 - w`),
        // right-pad each window row to `step * window` elements and left-pad
        // `(window - 1) * window` zeros; the affine view
        // `f(v, u) = v*window + u*(window + 1)` then reads `g[p, w]` exactly
        // when `p*step + w == v` and a zero cell otherwise, so one reduce
        // over `u` folds every overlapping window.
        let reversed: Vec<u32> = (0..window as u32).rev().collect();
        let indices = RawTensor::from_slice(&block.device(), [window], &reversed);
        block
            .index_select(2, &indices)
            .pad_with_zeros(2, 0, (step - 1) * window)
            .reshape([before, positions * step * window, after])
            .to_concrete()
            .pad_with_zeros(1, (window - 1) * window, window * (window - step))
            .restride([
                StrideSpec::dim(0, before),
                StrideSpec::dim_with(1, fold_len, window),
                StrideSpec::dim_with(1, window, window + 1),
                StrideSpec::dim(2, after),
            ])
            .to_concrete()
            .sum::<3>(2)
    };
    folded
        .pad_with_zeros(1, offset, size - offset - fold_len)
        .reshape([before * size * after])
        .to_concrete()
}

/// Express a layout over a contiguous input as per-output-axis stride specs
/// plus per-input-axis base offsets. Returns `None` when a stride does not
/// decompose into a single input dimension or when an axis' reach could carry
/// into the next dimension, where per-axis factoring would diverge from the
/// layout's linear indexing.
fn layout_restride_specs<const IN: usize>(
    layout: &Layout,
    input_shape: [usize; IN],
) -> Option<(Vec<StrideSpec>, [usize; IN])> {
    if input_shape.contains(&0) {
        return None;
    }
    let input_strides = Layout::continuous_strides(&input_shape);
    let mut reach = [0usize; IN];
    let mut specs = Vec::with_capacity(layout.rank());
    for (&size, &stride) in layout.shape().iter().zip(layout.strides()) {
        if size == 1 || stride == 0 {
            specs.push(StrideSpec::dim_with(0, size, 0));
            continue;
        }
        let dim = (0..IN).find(|&dim| input_strides[dim] <= stride)?;
        if stride % input_strides[dim] != 0 {
            return None;
        }
        let multiplier = stride / input_strides[dim];
        reach[dim] += (size - 1) * multiplier;
        specs.push(StrideSpec::dim_with(dim, size, multiplier));
    }
    let mut offsets = [0usize; IN];
    let mut offset = layout.offset();
    for dim in 0..IN {
        offsets[dim] = offset / input_strides[dim];
        offset %= input_strides[dim];
        reach[dim] += offsets[dim];
    }
    reach
        .iter()
        .zip(&input_shape)
        .all(|(&reach, &size)| reach < size)
        .then_some((specs, offsets))
}

/// Host-loop fallback for the restride patterns [`reduce_restride_gradient`]
/// cannot factor into per-axis runs.
fn scatter_restride_gradient<const IN: usize, const OUT: usize>(
    gradient: &RawTensor<OUT, f32>,
    output_shape: [usize; OUT],
    input_shape: [usize; IN],
    input_index: impl Fn([usize; OUT]) -> [usize; IN],
) -> RawTensor<IN, f32> {
    let mut input_gradient = RawTensor::zeros(&gradient.device(), input_shape);
    for_each_index(output_shape, |output_index| {
        let input_index = input_index(output_index);
        let output_slices: [Range<usize>; OUT] =
            std::array::from_fn(|axis| output_index[axis]..output_index[axis] + 1);
        let patch = gradient.slice(output_slices).reshape([1; IN]).to_concrete();
        let target: [Range<usize>; IN] =
            std::array::from_fn(|axis| input_index[axis]..input_index[axis] + 1);
        let current = input_gradient.slice(target.clone()).to_concrete();
        let updated = (current + patch).to_concrete();
        input_gradient = input_gradient.slice_assign(target, &updated).to_concrete();
    });
    input_gradient
}

fn reduce_broadcast_gradient<const IN: usize, const OUT: usize>(
    gradient: RawTensor<OUT, f32>,
    input_shape: [usize; IN],
) -> Result<Box<dyn AnyTensorValue>> {
    let output_shape = gradient.shape();
    let mut aligned_input_shape = [1usize; OUT];
    for axis in 0..IN {
        aligned_input_shape[OUT - IN + axis] = input_shape[axis];
    }

    for axis in 0..OUT {
        let output_dim = output_shape[axis];
        let input_dim = aligned_input_shape[axis];
        if input_dim != 1 && input_dim != output_dim {
            return Err(Error::msg("incompatible broadcast gradient shape"));
        }
    }

    if aligned_input_shape == output_shape {
        if IN == OUT {
            return Ok(Box::new(gradient));
        }
        return Ok(Box::new(gradient.reshape(input_shape).to_concrete()));
    }

    // Sum the axes the forward broadcast expanded, one compiled reduce per axis.
    let mut remaining = output_shape;
    let mut flat = gradient
        .reshape([output_shape.iter().product()])
        .to_concrete();
    for axis in 0..OUT {
        if aligned_input_shape[axis] != 1 || remaining[axis] == 1 {
            continue;
        }
        let before: usize = remaining[..axis].iter().product();
        let after: usize = remaining[axis + 1..].iter().product();
        flat = flat
            .reshape([before, remaining[axis], after])
            .to_concrete()
            .sum::<2>(1)
            .reshape([before * after])
            .to_concrete();
        remaining[axis] = 1;
    }
    Ok(Box::new(flat.reshape(input_shape).to_concrete()))
}
