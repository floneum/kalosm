# Feather 8x8 Pipeline

This crate builds a strict, conservative Feather-to-8x8 dataset. It parses the full SVG corpus with `usvg`, flattens every supported shape into absolute polylines, fits each icon to its tight geometry bounding box, and quantizes onto an 8x8 lattice with unit 8-neighbor moves only.

The default filter is intentionally biased toward rejection. An icon is rejected if any of the following are true:

- mean symmetric error `> 0.20` grid cells
- max symmetric error `> 0.40` grid cells
- turn inflation `> 1.35x`
- path length inflation `> 1.50x`
- repeated internal lattice points introduced by quantization
- repeated edges introduced by quantization
- new self-intersections introduced by quantization
- disconnected or ambiguous closed-contour graphs
- source geometry is curve-heavy by default: circles, ellipses, rounded rectangles, or any path that resolves to quadratic/cubic segments

Run it with:

```bash
cargo run -p fusor-feather-pipeline -- --input /path/to/feather/icons --output /path/to/report
```

The output directory contains `icons_all.json`, `icons_clean.json`, `icons_rejected.json`, `summary.json`, `gallery.html`, and preview SVGs for accepted and rejected icons.
