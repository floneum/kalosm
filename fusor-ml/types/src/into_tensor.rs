/// Trait for tensor types that can be created from array-like data.
///
/// This trait is generic over:
/// - `R` - tensor rank (const generic)
/// - `D` - data type (f32, i32, etc.)
/// - `T` - input data type (array, slice, iterator, etc.)
/// - `Dev` - device type
///
/// By having the tensor type implement this trait (rather than the input type),
/// we satisfy Rust's orphan rules and can implement this for generic input types.
pub trait FromArray<const R: usize, D, T, Dev> {
    fn from_array(data: T, device: &Dev) -> Self;
}

/// Flattened tensor data with the shape inferred from nested array-like input.
pub struct FlatArray<D, const R: usize> {
    pub shape: [usize; R],
    pub data: Vec<D>,
}

/// Converts nested array-like data into a flat row-major buffer and shape.
///
/// Implementations require exact-size iterators at each rank so tensor backends
/// can share one construction path while preserving rectangular shape checks.
pub trait IntoFlatArray<D, const R: usize> {
    fn into_flat_array(self) -> FlatArray<D, R>;
}

impl<D> IntoFlatArray<D, 0> for () {
    fn into_flat_array(self) -> FlatArray<D, 0> {
        FlatArray {
            shape: [],
            data: Vec::new(),
        }
    }
}

impl<'a, I, D: Copy + 'a> IntoFlatArray<D, 1> for I
where
    I: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_flat_array(self) -> FlatArray<D, 1> {
        let iter = self.into_iter();
        let shape = [iter.len()];
        let data = iter.copied().collect();
        FlatArray { shape, data }
    }
}

impl<'a, I, I2, D: Copy + 'a> IntoFlatArray<D, 2> for I
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_flat_array(self) -> FlatArray<D, 2> {
        let mut iter = self.into_iter().map(IntoIterator::into_iter).peekable();
        let size = iter.len();
        let second_size = iter.peek().map(ExactSizeIterator::len).unwrap_or_default();
        let data = iter
            .flat_map(|i| {
                let size = i.len();
                if size != second_size {
                    panic!(
                        "expected a rectangular matrix. The first inner iterator size was {second_size}, but another inner iterator size was {size}"
                    );
                }
                i.copied()
            })
            .collect();

        FlatArray {
            shape: [size, second_size],
            data,
        }
    }
}

impl<'a, I, I2, I3, D: Copy + 'a> IntoFlatArray<D, 3> for I
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_flat_array(self) -> FlatArray<D, 3> {
        let mut iter = self
            .into_iter()
            .map(|i| i.into_iter().map(IntoIterator::into_iter).peekable())
            .peekable();
        let mut shape = [iter.len(), 0, 0];
        if let Some(iter) = iter.peek_mut() {
            let size = iter.len();
            shape[1] = size;
            if let Some(iter) = iter.peek() {
                let size = iter.len();
                shape[2] = size;
            }
        }

        let data = iter
            .flat_map(|i| {
                let size = i.len();
                let required_size = shape[1];
                if size != required_size {
                    panic!(
                        "expected a rectangular matrix. The first inner iterator size was {required_size}, but another inner iterator size was {size}"
                    );
                }
                i.flat_map(|i| {
                    let size = i.len();
                    let required_size = shape[2];
                    if size != required_size {
                        panic!(
                            "expected a rectangular matrix. The first inner inner iterator size was {required_size}, but another inner inner iterator size was {size}"
                        );
                    }
                    i.copied()
                })
            })
            .collect();

        FlatArray { shape, data }
    }
}

impl<'a, I, I2, I3, I4, D: Copy + 'a> IntoFlatArray<D, 4> for I
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = I4, IntoIter: ExactSizeIterator>,
    I4: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_flat_array(self) -> FlatArray<D, 4> {
        let mut iter = self
            .into_iter()
            .map(|i| {
                i.into_iter()
                    .map(|i| i.into_iter().map(IntoIterator::into_iter).peekable())
                    .peekable()
            })
            .peekable();
        let mut shape = [iter.len(), 0, 0, 0];
        if let Some(iter) = iter.peek_mut() {
            let size = iter.len();
            shape[1] = size;
            if let Some(iter) = iter.peek_mut() {
                let size = iter.len();
                shape[2] = size;
                if let Some(iter) = iter.peek() {
                    let size = iter.len();
                    shape[3] = size;
                }
            }
        }

        let data = iter
            .flat_map(|i| {
                let size = i.len();
                let required_size = shape[1];
                if size != required_size {
                    panic!(
                        "expected a rectangular matrix. The first inner iterator size was {required_size}, but another inner iterator size was {size}"
                    );
                }
                i.flat_map(|i| {
                    let size = i.len();
                    let required_size = shape[2];
                    if size != required_size {
                        panic!(
                            "expected a rectangular matrix. The first inner inner iterator size was {required_size}, but another inner inner iterator size was {size}"
                        );
                    }
                    i.flat_map(|i| {
                        let size = i.len();
                        let required_size = shape[3];
                        if size != required_size {
                            panic!(
                                "expected a rectangular matrix. The first inner inner inner iterator size was {required_size}, but another inner inner inner iterator size was {size}"
                            );
                        }
                        i.copied()
                    })
                })
            })
            .collect();

        FlatArray { shape, data }
    }
}

impl<'a, I, I2, I3, I4, I5, D: Copy + 'a> IntoFlatArray<D, 5> for I
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = I4, IntoIter: ExactSizeIterator>,
    I4: IntoIterator<Item = I5, IntoIter: ExactSizeIterator>,
    I5: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_flat_array(self) -> FlatArray<D, 5> {
        let mut iter = self
            .into_iter()
            .map(|i| {
                i.into_iter()
                    .map(|i| {
                        i.into_iter()
                            .map(|i| i.into_iter().map(IntoIterator::into_iter).peekable())
                            .peekable()
                    })
                    .peekable()
            })
            .peekable();
        let mut shape = [iter.len(), 0, 0, 0, 0];
        if let Some(iter) = iter.peek_mut() {
            let size = iter.len();
            shape[1] = size;
            if let Some(iter) = iter.peek_mut() {
                let size = iter.len();
                shape[2] = size;
                if let Some(iter) = iter.peek_mut() {
                    let size = iter.len();
                    shape[3] = size;
                    if let Some(iter) = iter.peek() {
                        let size = iter.len();
                        shape[4] = size;
                    }
                }
            }
        }

        let data = iter
            .flat_map(|i| {
                let size = i.len();
                let required_size = shape[1];
                if size != required_size {
                    panic!(
                        "expected a rectangular matrix. The first inner iterator size was {required_size}, but another inner iterator size was {size}"
                    );
                }
                i.flat_map(|i| {
                    let size = i.len();
                    let required_size = shape[2];
                    if size != required_size {
                        panic!(
                            "expected a rectangular matrix. The first inner inner iterator size was {required_size}, but another inner inner iterator size was {size}"
                        );
                    }
                    i.flat_map(|i| {
                        let size = i.len();
                        let required_size = shape[3];
                        if size != required_size {
                            panic!(
                                "expected a rectangular matrix. The first inner inner inner iterator size was {required_size}, but another inner inner inner iterator size was {size}"
                            );
                        }
                        i.flat_map(|i| {
                            let size = i.len();
                            let required_size = shape[4];
                            if size != required_size {
                                panic!(
                                    "expected a rectangular matrix. The first inner inner inner inner iterator size was {required_size}, but another inner inner inner inner iterator size was {size}"
                                );
                            }
                            i.copied()
                        })
                    })
                })
            })
            .collect();

        FlatArray { shape, data }
    }
}

#[cfg(test)]
mod tests {
    use super::IntoFlatArray;

    #[test]
    fn flattens_rank_2_arrays_with_shape() {
        let data = [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let flat = <&[[f32; 2]; 3] as IntoFlatArray<f32, 2>>::into_flat_array(&data);

        assert_eq!(flat.shape, [3, 2]);
        assert_eq!(flat.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn flattens_rank_5_arrays_with_shape() {
        let data = [[[[[1u32, 2], [3, 4]]]]];
        let flat = <&[[[[[u32; 2]; 2]; 1]; 1]; 1] as IntoFlatArray<u32, 5>>::into_flat_array(&data);

        assert_eq!(flat.shape, [1, 1, 1, 2, 2]);
        assert_eq!(flat.data, vec![1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(
        expected = "expected a rectangular matrix. The first inner iterator size was 1, but another inner iterator size was 2"
    )]
    fn rejects_ragged_rank_2_inputs() {
        let data = vec![vec![1.0f32], vec![2.0, 3.0]];
        let _ = <&Vec<Vec<f32>> as IntoFlatArray<f32, 2>>::into_flat_array(&data);
    }
}
