use super::*;

#[test]
fn layout_is_structured_shape_strides_and_memory_level() {
    let shape = Shape::new([4, 8]);
    let layout = Layout::contiguous(MemoryLevel::Storage, shape.clone());

    assert_eq!(layout.shape(), &shape);
    assert_eq!(layout.affine_strides(), vec![8, 1]);
    assert_eq!(layout.memory_level(), MemoryLevel::Storage);
    assert_eq!(layout.element_count().get(), 32);
}
