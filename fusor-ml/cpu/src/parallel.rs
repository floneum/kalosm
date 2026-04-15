//! Parallel execution utilities using std::thread::scope
//!
//! This module provides structured parallelism for CPU tensor operations.
//! Work is split evenly among threads once and then joined, which is
//! better suited for predictable linear algebra workloads than work-stealing.

/// Get the number of threads to use for parallel operations.
///
/// Returns 1 on wasm32 targets (no threading support).
/// On other targets, returns the available parallelism or 1 if unavailable.
#[inline]
pub fn num_threads() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

/// Execute a function in parallel over pairs of input/output chunks.
///
/// Useful for operations like softmax where each input row maps to an output row.
///
/// # Arguments
/// * `input` - Input data slice
/// * `output` - Output data slice (must be same length as input)
/// * `chunk_size` - Size of each chunk to process together
/// * `f` - Function receiving (chunk_index, input_chunk, output_chunk)
#[inline]
pub fn parallel_zip_chunks_mut<T, U, F>(input: &[T], output: &mut [U], chunk_size: usize, f: F)
where
    T: Sync,
    U: Send,
    F: Fn(usize, &[T], &mut [U]) + Send + Sync,
{
    assert_eq!(
        input.len(),
        output.len(),
        "Input and output must have same length"
    );

    if input.is_empty() || chunk_size == 0 {
        return;
    }

    let n_threads = num_threads();
    let total_chunks = input.len().div_ceil(chunk_size);

    // If single-threaded or very small workload, run sequentially
    if n_threads == 1 || total_chunks <= 1 {
        for (i, (in_chunk, out_chunk)) in input
            .chunks(chunk_size)
            .zip(output.chunks_mut(chunk_size))
            .enumerate()
        {
            f(i, in_chunk, out_chunk);
        }
        return;
    }

    // Distribute chunks evenly among threads
    let chunks_per_thread = total_chunks.div_ceil(n_threads);
    let elements_per_thread = chunks_per_thread * chunk_size;

    std::thread::scope(|scope| {
        let mut remaining_in = input;
        let mut remaining_out = output;
        let mut chunk_offset = 0;

        for thread_id in 0..n_threads {
            if remaining_in.is_empty() {
                break;
            }

            let this_size = if thread_id == n_threads - 1 {
                remaining_in.len()
            } else {
                elements_per_thread.min(remaining_in.len())
            };

            let (thread_in, rest_in) = remaining_in.split_at(this_size);
            let (thread_out, rest_out) = remaining_out.split_at_mut(this_size);
            remaining_in = rest_in;
            remaining_out = rest_out;

            let current_chunk_offset = chunk_offset;
            chunk_offset += this_size.div_ceil(chunk_size);

            let f_ref = &f;
            scope.spawn(move || {
                for (i, (in_chunk, out_chunk)) in thread_in
                    .chunks(chunk_size)
                    .zip(thread_out.chunks_mut(chunk_size))
                    .enumerate()
                {
                    f_ref(current_chunk_offset + i, in_chunk, out_chunk);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_threads() {
        let n = num_threads();
        assert!(n >= 1);
    }

    #[test]
    fn test_parallel_zip_chunks_mut() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 100];

        parallel_zip_chunks_mut(&input, &mut output, 10, |_, in_chunk, out_chunk| {
            for (i, o) in in_chunk.iter().zip(out_chunk.iter_mut()) {
                *o = *i * 2.0;
            }
        });

        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, (i as f32) * 2.0);
        }
    }
}
