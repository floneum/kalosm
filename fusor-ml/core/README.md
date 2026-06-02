# Fusor ML

This is a WGPU ML runtime with kernel fusion for ergonomic high performance custom operations. This will hopefully serve as the web and amd runtime for [kalosm](https://crates.io/crates/kalosm) once it is stable enough.

## Status

Basic operations are working and simple kernel fusion is implemented, but this is **not production ready yet**. 

Features:

- [x] Elementwise ops
- [x] Fuse Elementwise ops together
- [x] MatMul
- [x] Reduce ops
- [x] Fuse Elementwise ops into Reduce ops
- [x] PairWise ops
- [x] Fuse Elementwise ops into PairWise ops
- [x] Analyze buffer usage for in-place ops
- [x] Memory move/cat/etc ops
- [x] Cast ops
- [ ] Fuse PairWise ops together?
- [ ] Fuse parallel Reduce ops?
- [ ] Fuse PairWise ops with two of the same input into an elementwise op
- [ ] Dynamically apply fusion based on runtime throughput data

Operations required for a Llama implementation:

- [x] RmsNorm
- [x] Matmul
- [x] Rope
- [x] Unqueeze
- [x] Cat
- [x] Reshape
- [x] Transpose
- [x] Softmax
- [x] narraw
- [x] silu
- [x] arange
- [x] sin
- [x] cos

## Resources

- https://github.com/googlefonts/compute-shader-101
- https://siboehm.com/articles/22/CUDA-MMM
