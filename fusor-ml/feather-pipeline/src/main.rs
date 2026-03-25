use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
    env,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use kurbo::{PathEl, flatten};
use serde::Serialize;
use usvg::{Node, Options, Tree, tiny_skia_path::PathSegment};

const DEFAULT_GRID_SIZE: usize = 8;
const FIT_MODE: &str = "tight_bbox_centered_contain";
const FLATTEN_TOLERANCE: f64 = 0.05;
const ERROR_SAMPLE_DENSITY: f64 = 32.0;
const RENDER_RESOLUTION: usize = 64;
const RENDER_STROKE_RADIUS: f64 = 0.48;
const ALIGNMENT_RENDER_CANDIDATES: usize = 9;

const MEAN_ERROR_THRESHOLD: f64 = 0.20;
const MAX_ERROR_THRESHOLD: f64 = 0.40;
const TURN_INFLATION_THRESHOLD: f64 = 1.35;
const PATH_LENGTH_INFLATION_THRESHOLD: f64 = 1.50;
const RENDER_F1_THRESHOLD: f64 = 0.83;
const RENDER_IOU_THRESHOLD: f64 = 0.70;
const RESCUE_MEAN_ERROR_THRESHOLD: f64 = 0.35;
const RESCUE_MAX_ERROR_THRESHOLD: f64 = 1.00;
const RESCUE_TURN_INFLATION_THRESHOLD: f64 = 1.75;
const RESCUE_PATH_LENGTH_INFLATION_THRESHOLD: f64 = 1.18;
const RESCUE_RENDER_F1_THRESHOLD: f64 = 0.89;
const RESCUE_RENDER_IOU_THRESHOLD: f64 = 0.76;
const CURVED_RESCUE_RENDER_F1_THRESHOLD: f64 = 0.91;
const CURVED_RESCUE_RENDER_IOU_THRESHOLD: f64 = 0.80;
const CIRCLE_RESCUE_RENDER_F1_THRESHOLD: f64 = 0.95;
const CIRCLE_RESCUE_RENDER_IOU_THRESHOLD: f64 = 0.87;
const ORTHOGONAL_RESCUE_MEAN_ERROR_THRESHOLD: f64 = 0.23;
const ORTHOGONAL_RESCUE_MAX_ERROR_THRESHOLD: f64 = 0.60;
const ORTHOGONAL_RESCUE_RENDER_F1_THRESHOLD: f64 = 0.73;
const ORTHOGONAL_RESCUE_RENDER_IOU_THRESHOLD: f64 = 0.58;
const SYMMETRY_BONUS_START: f64 = 0.90;
const SYMMETRY_BONUS_RANGE: f64 = 0.10;
const SYMMETRY_MEAN_ERROR_BONUS: f64 = 0.06;
const SYMMETRY_MAX_ERROR_BONUS: f64 = 0.25;
const SYMMETRY_RENDER_F1_BONUS: f64 = 0.18;
const SYMMETRY_RENDER_IOU_BONUS: f64 = 0.20;
const SYMMETRY_TURN_BONUS: f64 = 1.10;

const PREVIEW_SIZE: f64 = 256.0;
const PREVIEW_PADDING: f64 = 36.0;

fn main() -> Result<()> {
    let config = Config::from_env()?;
    let svg_root = config.resolve_input()?;
    let output_dir = config.output_dir.clone();

    fs::create_dir_all(output_dir.join("previews_clean"))
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    fs::create_dir_all(output_dir.join("previews_rejected"))
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let svg_files = collect_svg_files(&svg_root)?;
    if svg_files.is_empty() {
        bail!("no SVG files found under {}", svg_root.display());
    }

    let mut processed = Vec::with_capacity(svg_files.len());
    for svg_path in svg_files {
        let source = load_source_icon(&svg_path)?;
        let icon = process_icon(&source, &config);
        processed.push(icon);
    }

    processed.sort_by(|a, b| a.record.name.cmp(&b.record.name));

    for icon in &mut processed {
        let (dir_name, relative_dir) = if icon.record.accepted {
            ("previews_clean", "previews_clean")
        } else {
            ("previews_rejected", "previews_rejected")
        };
        let preview_name = format!("{}.svg", icon.record.name);
        let preview_path = output_dir.join(dir_name).join(&preview_name);
        let preview_svg = render_preview(icon);
        fs::write(&preview_path, preview_svg)
            .with_context(|| format!("failed to write {}", preview_path.display()))?;
        let relative_preview = format!("{relative_dir}/{preview_name}");
        icon.record.preview_path = relative_preview;
    }

    let all_records = processed
        .iter()
        .map(|icon| icon.record.clone())
        .collect::<Vec<_>>();
    let clean_records = all_records
        .iter()
        .filter(|icon| icon.accepted)
        .cloned()
        .collect::<Vec<_>>();
    let rejected_records = all_records
        .iter()
        .filter(|icon| !icon.accepted)
        .cloned()
        .collect::<Vec<_>>();

    write_json(&output_dir.join("icons_all.json"), &all_records)?;
    write_json(&output_dir.join("icons_clean.json"), &clean_records)?;
    write_json(&output_dir.join("icons_rejected.json"), &rejected_records)?;

    let summary = build_summary(&processed, &config, &svg_root);
    write_json(&output_dir.join("summary.json"), &summary)?;

    let gallery = render_gallery(&processed, &summary);
    fs::write(output_dir.join("gallery.html"), gallery).with_context(|| {
        format!(
            "failed to write {}",
            output_dir.join("gallery.html").display()
        )
    })?;

    fs::write(
        output_dir.join("README.md"),
        render_methodology_readme(&summary),
    )
    .with_context(|| format!("failed to write {}", output_dir.join("README.md").display()))?;

    Ok(())
}

#[derive(Clone, Debug)]
struct Config {
    input: Option<PathBuf>,
    output_dir: PathBuf,
    grid_size: usize,
    allow_curved_source: bool,
}

impl Config {
    fn from_env() -> Result<Self> {
        let mut input = None;
        let mut output_dir = PathBuf::from("fusor-ml/feather-pipeline/output");
        let mut grid_size = DEFAULT_GRID_SIZE;
        let mut allow_curved_source = false;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow!("missing value after --input"))?;
                    input = Some(PathBuf::from(value));
                }
                "--output" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow!("missing value after --output"))?;
                    output_dir = PathBuf::from(value);
                }
                "--grid" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow!("missing value after --grid"))?;
                    grid_size = value
                        .parse::<usize>()
                        .context("failed to parse --grid as an integer")?;
                    if grid_size < 2 {
                        bail!("--grid must be at least 2");
                    }
                }
                "--allow-curved-source" => allow_curved_source = true,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => bail!("unknown argument: {other}"),
            }
        }

        Ok(Self {
            input,
            output_dir,
            grid_size,
            allow_curved_source,
        })
    }

    fn resolve_input(&self) -> Result<PathBuf> {
        if let Some(path) = &self.input {
            return normalize_input_root(path);
        }

        let candidates = [
            "feather",
            "feather/icons",
            "../feather",
            "../feather/icons",
            "../../feather",
            "../../feather/icons",
            "/Users/evanalmloff/Desktop/Github/feather",
            "/Users/evanalmloff/Desktop/Github/feather/icons",
        ];

        for candidate in candidates {
            let path = PathBuf::from(candidate);
            if let Ok(root) = normalize_input_root(&path) {
                return Ok(root);
            }
        }

        bail!(
            "could not locate a local Feather corpus; pass --input /path/to/feather/icons or a Feather repo root"
        );
    }
}

fn print_help() {
    eprintln!(
        "Usage: cargo run -p fusor-feather-pipeline -- --input /path/to/feather/icons [--output report_dir] [--grid 8] [--allow-curved-source]"
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct PointI {
    x: i32,
    y: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PointF {
    x: f64,
    y: f64,
}

impl PointF {
    fn distance(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Direction8 {
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
}

impl Direction8 {
    fn from_step(dx: i32, dy: i32) -> Option<Self> {
        match (dx, dy) {
            (0, -1) => Some(Self::N),
            (1, -1) => Some(Self::NE),
            (1, 0) => Some(Self::E),
            (1, 1) => Some(Self::SE),
            (0, 1) => Some(Self::S),
            (-1, 1) => Some(Self::SW),
            (-1, 0) => Some(Self::W),
            (-1, -1) => Some(Self::NW),
            _ => None,
        }
    }

    fn from_vector(dx: f64, dy: f64) -> Option<Self> {
        if dx.abs() < 1e-9 && dy.abs() < 1e-9 {
            return None;
        }

        let angle = dy.atan2(dx);
        let octant = ((angle / (std::f64::consts::FRAC_PI_4)).round() as i32).rem_euclid(8);
        match octant {
            0 => Some(Self::E),
            1 => Some(Self::SE),
            2 => Some(Self::S),
            3 => Some(Self::SW),
            4 => Some(Self::W),
            5 => Some(Self::NW),
            6 => Some(Self::N),
            7 => Some(Self::NE),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::N => "N",
            Self::NE => "NE",
            Self::E => "E",
            Self::SE => "SE",
            Self::S => "S",
            Self::SW => "SW",
            Self::W => "W",
            Self::NW => "NW",
        }
    }
}

#[derive(Clone, Debug)]
struct BoundingBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl BoundingBox {
    fn new() -> Self {
        Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }

    fn include(&mut self, point: PointF) {
        self.min_x = self.min_x.min(point.x);
        self.min_y = self.min_y.min(point.y);
        self.max_x = self.max_x.max(point.x);
        self.max_y = self.max_y.max(point.y);
    }

    fn is_valid(&self) -> bool {
        self.min_x.is_finite()
            && self.min_y.is_finite()
            && self.max_x.is_finite()
            && self.max_y.is_finite()
    }

    fn width(&self) -> f64 {
        (self.max_x - self.min_x).max(0.0)
    }

    fn height(&self) -> f64 {
        (self.max_y - self.min_y).max(0.0)
    }
}

#[derive(Clone, Debug)]
struct SourceProfile {
    element_kinds: BTreeSet<String>,
    has_circle: bool,
    has_ellipse: bool,
    has_rounded_rect: bool,
}

#[derive(Clone, Debug)]
struct SourceIcon {
    name: String,
    original_svg_path: PathBuf,
    subpaths: Vec<FlattenedSubpath>,
    bbox: BoundingBox,
    profile: SourceProfile,
    has_curve_segments: bool,
}

#[derive(Clone, Debug)]
struct FlattenedSubpath {
    source_path_index: usize,
    is_closed: bool,
    has_curve: bool,
    points: Vec<PointF>,
}

#[derive(Clone, Debug)]
struct PreparedSubpath {
    source_path_index: usize,
    is_closed: bool,
    points: Vec<PointF>,
}

#[derive(Clone, Debug)]
struct QuantizedSubpath {
    is_closed: bool,
    points: Vec<PointI>,
    dirs: Vec<Direction8>,
    turn_points: Vec<PointI>,
}

#[derive(Clone, Copy, Debug)]
struct RenderStats {
    precision: f64,
    recall: f64,
    f1: f64,
    iou: f64,
}

#[derive(Clone, Debug)]
struct ProcessedIcon {
    record: IconRecord,
    fitted_subpaths: Vec<PreparedSubpath>,
    quantized_subpaths: Vec<QuantizedSubpath>,
    source_flags: SourceFlags,
    mixed_metric_rescued: bool,
}

#[derive(Clone, Debug)]
struct SourceFlags {
    source_kinds: Vec<String>,
    has_curved_source: bool,
    has_circle: bool,
    has_ellipse: bool,
    has_rounded_rect: bool,
}

#[derive(Clone, Debug)]
struct ClosedQuantization {
    points: Vec<PointI>,
    repeated_edges: usize,
    disconnected_components: usize,
    ambiguous_branches: usize,
}

#[derive(Clone, Debug, Serialize)]
struct IconRecord {
    name: String,
    original_svg_path: String,
    is_closed: bool,
    original_subpaths: Vec<OriginalSubpathRecord>,
    quantized_points: Vec<Vec<[i32; 2]>>,
    dirs: Vec<Vec<String>>,
    turn_points: Vec<Vec<[i32; 2]>>,
    grid_size: usize,
    fit_mode: String,
    mean_error: f64,
    max_error: f64,
    render_precision: f64,
    render_recall: f64,
    render_f1: f64,
    render_iou: f64,
    symmetry_score: f64,
    turn_inflation: f64,
    self_intersections: usize,
    revisited_points: usize,
    repeated_edges: usize,
    accepted: bool,
    rejection_reasons: Vec<String>,
    path_length_inflation: f64,
    direction_sequence_length: usize,
    source_kinds: Vec<String>,
    has_curved_source: bool,
    preview_path: String,
}

#[derive(Clone, Debug, Serialize)]
struct OriginalSubpathRecord {
    source_path_index: usize,
    is_closed: bool,
    has_curve: bool,
    points: Vec<[f64; 2]>,
}

#[derive(Clone, Serialize)]
struct SummaryRecord {
    input_root: String,
    output_root: String,
    grid_size: usize,
    fit_mode: String,
    thresholds: Thresholds,
    totals: Totals,
    metrics: SummaryMetrics,
    rejection_reasons: BTreeMap<String, usize>,
    curved_source: CurvedSourceSummary,
    source_geometry: SourceGeometrySummary,
    mixed_metric_rescues: usize,
}

#[derive(Clone, Serialize)]
struct Thresholds {
    mean_error_max: f64,
    max_error_max: f64,
    turn_inflation_max: f64,
    path_length_inflation_max: f64,
    render_f1_min: f64,
    render_iou_min: f64,
    rescue_render_f1_min: f64,
    rescue_render_iou_min: f64,
    circle_ellipse_rejected_by_default: bool,
}

#[derive(Clone, Serialize)]
struct Totals {
    icons: usize,
    accepted: usize,
    rejected: usize,
    acceptance_rate: f64,
}

#[derive(Clone, Serialize)]
struct SummaryMetrics {
    mean_error: MetricSummary,
    max_error: MetricSummary,
    render_f1: MetricSummary,
    render_iou: MetricSummary,
    symmetry_score: MetricSummary,
    turn_inflation: MetricSummary,
    path_length_inflation: MetricSummary,
}

#[derive(Clone, Serialize)]
struct MetricSummary {
    min: f64,
    max: f64,
    avg: f64,
}

#[derive(Clone, Serialize)]
struct CurvedSourceSummary {
    icons: usize,
    accepted: usize,
    rejected: usize,
}

#[derive(Clone, Serialize)]
struct SourceGeometrySummary {
    circles: usize,
    ellipses: usize,
    rounded_rects: usize,
}

fn normalize_input_root(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }

    if !path.exists() {
        bail!("input path does not exist: {}", path.display());
    }

    let icons_dir = path.join("icons");
    if icons_dir.is_dir() && directory_contains_svg(&icons_dir)? {
        return Ok(icons_dir);
    }

    let dist_icons_dir = path.join("dist").join("icons");
    if dist_icons_dir.is_dir() && directory_contains_svg(&dist_icons_dir)? {
        return Ok(dist_icons_dir);
    }

    if path.is_dir() && directory_contains_svg(path)? {
        return Ok(path.to_path_buf());
    }

    bail!(
        "input path does not contain a Feather SVG corpus: {}",
        path.display()
    );
}

fn directory_contains_svg(path: &Path) -> Result<bool> {
    for entry in fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("svg"))
            .unwrap_or(false)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn collect_svg_files(root: &Path) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }

    let mut files = Vec::new();
    collect_svg_files_recursive(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_svg_files_recursive(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_svg_files_recursive(&path, files)?;
            continue;
        }

        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("svg"))
            .unwrap_or(false)
        {
            files.push(path);
        }
    }
    Ok(())
}

fn load_source_icon(svg_path: &Path) -> Result<SourceIcon> {
    let data =
        fs::read(svg_path).with_context(|| format!("failed to read {}", svg_path.display()))?;
    let profile = inspect_source_document(&data)?;

    let mut options = Options::default();
    options.resources_dir = svg_path.parent().map(Path::to_path_buf);
    let tree = Tree::from_data(&data, &options)
        .with_context(|| format!("failed to parse {}", svg_path.display()))?;

    let mut subpaths = Vec::new();
    let mut next_path_index = 0usize;
    let mut has_curve_segments = false;
    collect_tree_subpaths(
        tree.root(),
        &mut next_path_index,
        &mut subpaths,
        &mut has_curve_segments,
    )?;

    if subpaths.is_empty() {
        bail!("no path geometry found in {}", svg_path.display());
    }

    let mut bbox = BoundingBox::new();
    for subpath in &subpaths {
        for point in &subpath.points {
            bbox.include(*point);
        }
    }

    if !bbox.is_valid() {
        bail!("invalid geometry bounding box in {}", svg_path.display());
    }

    Ok(SourceIcon {
        name: svg_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("unknown")
            .to_string(),
        original_svg_path: svg_path.to_path_buf(),
        subpaths,
        bbox,
        profile,
        has_curve_segments,
    })
}

fn inspect_source_document(data: &[u8]) -> Result<SourceProfile> {
    let text = std::str::from_utf8(data).context("SVG was not valid UTF-8")?;
    let document =
        usvg::roxmltree::Document::parse(text).context("failed to parse SVG source XML")?;
    let mut profile = SourceProfile {
        element_kinds: BTreeSet::new(),
        has_circle: false,
        has_ellipse: false,
        has_rounded_rect: false,
    };

    for node in document.descendants().filter(|node| node.is_element()) {
        let name = node.tag_name().name();
        if matches!(
            name,
            "path" | "line" | "polyline" | "polygon" | "rect" | "circle" | "ellipse"
        ) {
            profile.element_kinds.insert(name.to_string());
        }

        match name {
            "circle" => profile.has_circle = true,
            "ellipse" => profile.has_ellipse = true,
            "rect" => {
                let rx = node
                    .attribute("rx")
                    .and_then(parse_nonzero_number)
                    .unwrap_or(0.0);
                let ry = node
                    .attribute("ry")
                    .and_then(parse_nonzero_number)
                    .unwrap_or(0.0);
                if rx > 0.0 || ry > 0.0 {
                    profile.has_rounded_rect = true;
                }
            }
            _ => {}
        }
    }

    Ok(profile)
}

fn parse_nonzero_number(value: &str) -> Option<f64> {
    let trimmed = value.trim_end_matches(|character: char| character.is_ascii_alphabetic());
    let parsed = trimmed.parse::<f64>().ok()?;
    if parsed > 0.0 { Some(parsed) } else { None }
}

fn collect_tree_subpaths(
    group: &usvg::Group,
    next_path_index: &mut usize,
    subpaths: &mut Vec<FlattenedSubpath>,
    has_curve_segments: &mut bool,
) -> Result<()> {
    for child in group.children() {
        match child {
            Node::Group(nested) => {
                collect_tree_subpaths(nested, next_path_index, subpaths, has_curve_segments)?;
            }
            Node::Path(path) => {
                let mut transformed = path.data().clone();
                if !path.abs_transform().is_identity() {
                    transformed = transformed
                        .transform(path.abs_transform())
                        .ok_or_else(|| anyhow!("failed to apply path transform"))?;
                }

                let mut elements = Vec::with_capacity(transformed.len());
                let mut segments = transformed.segments();
                segments.set_auto_close(false);
                let mut has_curve = false;
                for segment in segments {
                    match segment {
                        PathSegment::MoveTo(point) => {
                            elements.push(PathEl::MoveTo(kurbo::Point::new(
                                point.x as f64,
                                point.y as f64,
                            )));
                        }
                        PathSegment::LineTo(point) => {
                            elements.push(PathEl::LineTo(kurbo::Point::new(
                                point.x as f64,
                                point.y as f64,
                            )));
                        }
                        PathSegment::QuadTo(control, point) => {
                            has_curve = true;
                            elements.push(PathEl::QuadTo(
                                kurbo::Point::new(control.x as f64, control.y as f64),
                                kurbo::Point::new(point.x as f64, point.y as f64),
                            ));
                        }
                        PathSegment::CubicTo(control1, control2, point) => {
                            has_curve = true;
                            elements.push(PathEl::CurveTo(
                                kurbo::Point::new(control1.x as f64, control1.y as f64),
                                kurbo::Point::new(control2.x as f64, control2.y as f64),
                                kurbo::Point::new(point.x as f64, point.y as f64),
                            ));
                        }
                        PathSegment::Close => elements.push(PathEl::ClosePath),
                    }
                }

                let path_index = *next_path_index;
                *next_path_index += 1;
                if has_curve {
                    *has_curve_segments = true;
                }

                let mut flattened = Vec::new();
                flatten(elements, FLATTEN_TOLERANCE, |element| {
                    flattened.push(element)
                });
                let mut current_points = Vec::new();
                let mut current_closed = false;

                for element in flattened {
                    match element {
                        PathEl::MoveTo(point) => {
                            finish_flattened_subpath(
                                subpaths,
                                path_index,
                                current_closed,
                                has_curve,
                                &mut current_points,
                            );
                            current_points.push(PointF {
                                x: point.x,
                                y: point.y,
                            });
                            current_closed = false;
                        }
                        PathEl::LineTo(point) => current_points.push(PointF {
                            x: point.x,
                            y: point.y,
                        }),
                        PathEl::ClosePath => {
                            current_closed = true;
                            finish_flattened_subpath(
                                subpaths,
                                path_index,
                                current_closed,
                                has_curve,
                                &mut current_points,
                            );
                            current_closed = false;
                        }
                        _ => {}
                    }
                }

                finish_flattened_subpath(
                    subpaths,
                    path_index,
                    current_closed,
                    has_curve,
                    &mut current_points,
                );
            }
            Node::Image(_) | Node::Text(_) => {}
        }
    }

    Ok(())
}

fn finish_flattened_subpath(
    subpaths: &mut Vec<FlattenedSubpath>,
    source_path_index: usize,
    is_closed: bool,
    has_curve: bool,
    current_points: &mut Vec<PointF>,
) {
    if current_points.is_empty() {
        return;
    }

    let mut points = Vec::with_capacity(current_points.len());
    for point in current_points.drain(..) {
        if points.last().copied() != Some(point) {
            points.push(point);
        }
    }

    if is_closed && points.len() > 1 && approx_eq_point(points[0], *points.last().unwrap(), 1e-9) {
        points.pop();
    }

    if points.len() >= 2 {
        subpaths.push(FlattenedSubpath {
            source_path_index,
            is_closed,
            has_curve,
            points,
        });
    }
}

fn process_icon(source: &SourceIcon, config: &Config) -> ProcessedIcon {
    let fitted_subpaths = source
        .subpaths
        .iter()
        .map(|subpath| PreparedSubpath {
            source_path_index: subpath.source_path_index,
            is_closed: subpath.is_closed,
            points: subpath.points.clone(),
        })
        .collect::<Vec<_>>();

    let fitted_subpaths = fit_subpaths_to_grid(&fitted_subpaths, &source.bbox, config.grid_size);

    let mut quantized_subpaths = Vec::with_capacity(fitted_subpaths.len());
    let mut rejection_reasons = BTreeSet::new();
    let mut graph_repeated_edges = 0usize;

    for subpath in &fitted_subpaths {
        if subpath.points.len() < 2 {
            rejection_reasons.insert("degenerate_subpath".to_string());
            quantized_subpaths.push(QuantizedSubpath {
                is_closed: subpath.is_closed,
                points: Vec::new(),
                dirs: Vec::new(),
                turn_points: Vec::new(),
            });
            continue;
        }

        if subpath.is_closed {
            let closed = quantize_closed_subpath(&subpath.points, config.grid_size);
            graph_repeated_edges += closed.repeated_edges;
            if closed.disconnected_components > 1 {
                rejection_reasons.insert("closed_disconnected_components".to_string());
            }
            if closed.ambiguous_branches > 0 {
                rejection_reasons.insert("closed_ambiguous_branches".to_string());
            }
            quantized_subpaths.push(build_quantized_subpath(closed.points, true));
        } else {
            let open = quantize_open_subpath(&subpath.points, config.grid_size);
            quantized_subpaths.push(build_quantized_subpath(open, false));
        }
    }

    let source_flags = SourceFlags {
        source_kinds: source.profile.element_kinds.iter().cloned().collect(),
        has_curved_source: source.has_curve_segments
            || source.profile.has_circle
            || source.profile.has_ellipse
            || source.profile.has_rounded_rect,
        has_circle: source.profile.has_circle,
        has_ellipse: source.profile.has_ellipse,
        has_rounded_rect: source.profile.has_rounded_rect,
    };

    if !config.allow_curved_source {
        if source_flags.has_circle {
            rejection_reasons.insert("circle_source_blacklist".to_string());
        }
        if source_flags.has_ellipse {
            rejection_reasons.insert("ellipse_source_blacklist".to_string());
        }
    }

    let metric_bundle =
        compute_metric_bundle(&fitted_subpaths, &quantized_subpaths, config.grid_size);
    let revisited_points = quantized_subpaths
        .iter()
        .map(|subpath| revisited_point_count(&subpath.points))
        .sum::<usize>();
    let introduced_revisited_points =
        revisited_points.saturating_sub(metric_bundle.source_intersections);
    let repeated_edges = graph_repeated_edges
        + quantized_subpaths
            .iter()
            .map(|subpath| repeated_edge_count(&subpath.points, subpath.is_closed))
            .sum::<usize>();

    if metric_bundle.mean_error > MEAN_ERROR_THRESHOLD {
        rejection_reasons.insert("mean_error".to_string());
    }
    if metric_bundle.max_error > MAX_ERROR_THRESHOLD {
        rejection_reasons.insert("max_error".to_string());
    }
    if metric_bundle.turn_inflation > TURN_INFLATION_THRESHOLD {
        rejection_reasons.insert("turn_inflation".to_string());
    }
    if metric_bundle.path_length_inflation > PATH_LENGTH_INFLATION_THRESHOLD {
        rejection_reasons.insert("path_length_inflation".to_string());
    }
    if metric_bundle.render_f1 < RENDER_F1_THRESHOLD {
        rejection_reasons.insert("render_f1".to_string());
    }
    if metric_bundle.render_iou < RENDER_IOU_THRESHOLD {
        rejection_reasons.insert("render_iou".to_string());
    }
    if introduced_revisited_points > 0 {
        rejection_reasons.insert("revisited_points".to_string());
    }
    if repeated_edges > 0 {
        rejection_reasons.insert("repeated_edges".to_string());
    }
    if metric_bundle.self_intersections > 0 {
        rejection_reasons.insert("self_intersections".to_string());
    }

    let direction_sequence_length = quantized_subpaths
        .iter()
        .map(|subpath| subpath.dirs.len())
        .sum();
    let mixed_metric_rescued = passes_mixed_metric_rescue(
        &source_flags,
        &quantized_subpaths,
        &rejection_reasons,
        &metric_bundle,
        introduced_revisited_points,
        repeated_edges,
    );
    let accepted = mixed_metric_rescued || rejection_reasons.is_empty();
    let rejection_reasons = if accepted {
        Vec::new()
    } else {
        rejection_reasons.into_iter().collect()
    };

    let record = IconRecord {
        name: source.name.clone(),
        original_svg_path: source.original_svg_path.display().to_string(),
        is_closed: fitted_subpaths.iter().all(|subpath| subpath.is_closed),
        original_subpaths: source
            .subpaths
            .iter()
            .map(|subpath| OriginalSubpathRecord {
                source_path_index: subpath.source_path_index,
                is_closed: subpath.is_closed,
                has_curve: subpath.has_curve,
                points: subpath
                    .points
                    .iter()
                    .map(|point| [round_float(point.x, 4), round_float(point.y, 4)])
                    .collect(),
            })
            .collect(),
        quantized_points: quantized_subpaths
            .iter()
            .map(|subpath| points_for_output(&subpath.points, subpath.is_closed))
            .collect(),
        dirs: quantized_subpaths
            .iter()
            .map(|subpath| {
                subpath
                    .dirs
                    .iter()
                    .map(|direction| direction.as_str().to_string())
                    .collect()
            })
            .collect(),
        turn_points: quantized_subpaths
            .iter()
            .map(|subpath| {
                subpath
                    .turn_points
                    .iter()
                    .map(|point| [point.x, point.y])
                    .collect()
            })
            .collect(),
        grid_size: config.grid_size,
        fit_mode: FIT_MODE.to_string(),
        mean_error: round_float(metric_bundle.mean_error, 4),
        max_error: round_float(metric_bundle.max_error, 4),
        render_precision: round_float(metric_bundle.render_precision, 4),
        render_recall: round_float(metric_bundle.render_recall, 4),
        render_f1: round_float(metric_bundle.render_f1, 4),
        render_iou: round_float(metric_bundle.render_iou, 4),
        symmetry_score: round_float(metric_bundle.symmetry_score, 4),
        turn_inflation: round_float(metric_bundle.turn_inflation, 4),
        self_intersections: metric_bundle.self_intersections,
        revisited_points,
        repeated_edges,
        accepted,
        rejection_reasons,
        path_length_inflation: round_float(metric_bundle.path_length_inflation, 4),
        direction_sequence_length,
        source_kinds: source_flags.source_kinds.clone(),
        has_curved_source: source_flags.has_curved_source,
        preview_path: String::new(),
    };

    ProcessedIcon {
        record,
        fitted_subpaths: translate_prepared_subpaths(
            &fitted_subpaths,
            metric_bundle.alignment_dx,
            metric_bundle.alignment_dy,
        ),
        quantized_subpaths,
        source_flags,
        mixed_metric_rescued,
    }
}

fn passes_mixed_metric_rescue(
    source_flags: &SourceFlags,
    quantized_subpaths: &[QuantizedSubpath],
    rejection_reasons: &BTreeSet<String>,
    metric_bundle: &MetricBundle,
    revisited_points: usize,
    repeated_edges: usize,
) -> bool {
    if revisited_points > 0 || repeated_edges > 0 || metric_bundle.self_intersections > 0 {
        return false;
    }

    if !rejection_reasons.iter().all(|reason| {
        matches!(
            reason.as_str(),
            "curved_source_blacklist"
                | "circle_source_blacklist"
                | "ellipse_source_blacklist"
                | "mean_error"
                | "max_error"
                | "turn_inflation"
                | "render_f1"
                | "render_iou"
        )
    }) {
        return false;
    }

    let symmetry_bonus = symmetry_bonus(metric_bundle.symmetry_score);
    let effective_mean_error =
        (metric_bundle.mean_error - symmetry_bonus * SYMMETRY_MEAN_ERROR_BONUS).max(0.0);
    let effective_max_error =
        (metric_bundle.max_error - symmetry_bonus * SYMMETRY_MAX_ERROR_BONUS).max(0.0);
    let effective_render_f1 =
        (metric_bundle.render_f1 + symmetry_bonus * SYMMETRY_RENDER_F1_BONUS).min(1.0);
    let effective_render_iou =
        (metric_bundle.render_iou + symmetry_bonus * SYMMETRY_RENDER_IOU_BONUS).min(1.0);
    let effective_turn_inflation =
        (metric_bundle.turn_inflation - symmetry_bonus * SYMMETRY_TURN_BONUS).max(0.0);
    let cardinal_only = quantized_uses_only_cardinal_dirs(quantized_subpaths);

    if !source_flags.has_curved_source
        && cardinal_only
        && symmetry_bonus >= 0.5
        && metric_bundle.turn_inflation <= 1.05
        && (0.92..=1.08).contains(&metric_bundle.path_length_inflation)
        && effective_mean_error <= ORTHOGONAL_RESCUE_MEAN_ERROR_THRESHOLD
        && effective_max_error <= ORTHOGONAL_RESCUE_MAX_ERROR_THRESHOLD
        && effective_render_f1 >= ORTHOGONAL_RESCUE_RENDER_F1_THRESHOLD
        && effective_render_iou >= ORTHOGONAL_RESCUE_RENDER_IOU_THRESHOLD
    {
        return true;
    }

    let (render_f1_floor, render_iou_floor, max_error_limit, turn_limit) =
        if source_flags.has_circle || source_flags.has_ellipse {
            (
                CIRCLE_RESCUE_RENDER_F1_THRESHOLD,
                CIRCLE_RESCUE_RENDER_IOU_THRESHOLD,
                0.55,
                RESCUE_TURN_INFLATION_THRESHOLD,
            )
        } else if source_flags.has_curved_source {
            (
                CURVED_RESCUE_RENDER_F1_THRESHOLD,
                CURVED_RESCUE_RENDER_IOU_THRESHOLD,
                RESCUE_MAX_ERROR_THRESHOLD,
                RESCUE_TURN_INFLATION_THRESHOLD,
            )
        } else {
            (
                RESCUE_RENDER_F1_THRESHOLD,
                RESCUE_RENDER_IOU_THRESHOLD,
                RESCUE_MAX_ERROR_THRESHOLD,
                RESCUE_TURN_INFLATION_THRESHOLD,
            )
        };

    if effective_mean_error > RESCUE_MEAN_ERROR_THRESHOLD
        || effective_max_error > max_error_limit
        || effective_turn_inflation > turn_limit
        || metric_bundle.path_length_inflation > RESCUE_PATH_LENGTH_INFLATION_THRESHOLD
        || effective_render_f1 < render_f1_floor
        || effective_render_iou < render_iou_floor
    {
        return false;
    }

    if source_flags.has_circle || source_flags.has_ellipse {
        metric_bundle.mean_error <= 0.22
            && metric_bundle.path_length_inflation <= 1.14
            && symmetry_bonus >= 0.45
    } else {
        true
    }
}

fn fit_subpaths_to_grid(
    subpaths: &[PreparedSubpath],
    bbox: &BoundingBox,
    grid_size: usize,
) -> Vec<PreparedSubpath> {
    let target = (grid_size.saturating_sub(1)) as f64;
    let center_snap = (grid_size / 2) as f64;
    let width = bbox.width();
    let height = bbox.height();
    let epsilon = 1e-9;
    let scale_x = if width > 1e-9 {
        target / width
    } else {
        f64::INFINITY
    };
    let scale_y = if height > 1e-9 {
        target / height
    } else {
        f64::INFINITY
    };
    let scale = match (scale_x.is_finite(), scale_y.is_finite()) {
        (true, true) => scale_x.min(scale_y),
        (true, false) => scale_x,
        (false, true) => scale_y,
        (false, false) => 1.0,
    };

    let scaled_width = width * scale;
    let scaled_height = height * scale;
    let offset_x = if width <= epsilon {
        center_snap - bbox.min_x * scale
    } else {
        (target - scaled_width) / 2.0 - bbox.min_x * scale
    };
    let offset_y = if height <= epsilon {
        center_snap - bbox.min_y * scale
    } else {
        (target - scaled_height) / 2.0 - bbox.min_y * scale
    };

    subpaths
        .iter()
        .map(|subpath| PreparedSubpath {
            source_path_index: subpath.source_path_index,
            is_closed: subpath.is_closed,
            points: subpath
                .points
                .iter()
                .map(|point| PointF {
                    x: clamp(point.x * scale + offset_x, 0.0, target),
                    y: clamp(point.y * scale + offset_y, 0.0, target),
                })
                .collect(),
        })
        .collect()
}

fn quantize_open_subpath(points: &[PointF], grid_size: usize) -> Vec<PointI> {
    let mut walk = Vec::new();
    for window in points.windows(2) {
        let segment = rasterize_segment(window[0], window[1], grid_size);
        append_walk(&mut walk, &segment);
    }
    dedupe_consecutive_points(&walk)
}

fn quantize_closed_subpath(points: &[PointF], grid_size: usize) -> ClosedQuantization {
    let mut raw_walk = Vec::new();
    let mut edge_counts = HashMap::<EdgeKey, usize>::new();
    let mut adjacency = BTreeMap::<PointI, BTreeSet<PointI>>::new();

    for (start, end) in float_segments(points, true) {
        let segment = rasterize_segment(start, end, grid_size);
        append_walk(&mut raw_walk, &segment);
        for edge in segment.windows(2) {
            if edge[0] == edge[1] {
                continue;
            }
            let key = EdgeKey::new(edge[0], edge[1]);
            *edge_counts.entry(key).or_insert(0) += 1;
            adjacency.entry(edge[0]).or_default().insert(edge[1]);
            adjacency.entry(edge[1]).or_default().insert(edge[0]);
        }
    }

    raw_walk = dedupe_consecutive_points(&raw_walk);
    if raw_walk.len() > 1 && raw_walk.first() == raw_walk.last() {
        raw_walk.pop();
    }

    let repeated_edges = edge_counts
        .values()
        .filter(|count| **count > 1)
        .map(|count| count - 1)
        .sum();
    let disconnected_components = connected_components(&adjacency);
    let ambiguous_branches = adjacency
        .values()
        .filter(|neighbors| neighbors.len() != 2)
        .count();

    let points = if disconnected_components == 1 && ambiguous_branches == 0 && !adjacency.is_empty()
    {
        trace_closed_cycle(&adjacency, points).unwrap_or_else(|| raw_walk.clone())
    } else {
        raw_walk.clone()
    };

    ClosedQuantization {
        points,
        repeated_edges,
        disconnected_components,
        ambiguous_branches,
    }
}

fn build_quantized_subpath(points: Vec<PointI>, is_closed: bool) -> QuantizedSubpath {
    let points = if is_closed {
        normalize_closed_points(points)
    } else {
        dedupe_consecutive_points(&points)
    };
    let dirs = discrete_dirs(&points, is_closed);
    let turn_points = discrete_turn_points(&points, &dirs, is_closed);
    QuantizedSubpath {
        is_closed,
        points,
        dirs,
        turn_points,
    }
}

fn normalize_closed_points(mut points: Vec<PointI>) -> Vec<PointI> {
    points = dedupe_consecutive_points(&points);
    if points.len() > 1 && points.first() == points.last() {
        points.pop();
    }
    points
}

fn append_walk(target: &mut Vec<PointI>, segment: &[PointI]) {
    if target.is_empty() {
        target.extend_from_slice(segment);
        return;
    }

    for point in segment {
        if target.last() != Some(point) {
            target.push(*point);
        }
    }
}

fn rasterize_segment(start: PointF, end: PointF, grid_size: usize) -> Vec<PointI> {
    let start_i = round_to_grid(start, grid_size);
    let end_i = round_to_grid(end, grid_size);
    if start.distance(end) <= 1e-9 {
        return vec![start_i];
    }
    bresenham_line(start_i, end_i)
}

fn bresenham_line(start: PointI, end: PointI) -> Vec<PointI> {
    let mut points = Vec::new();
    let mut x0 = start.x;
    let mut y0 = start.y;
    let x1 = end.x;
    let y1 = end.y;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        points.push(PointI { x: x0, y: y0 });
        if x0 == x1 && y0 == y1 {
            break;
        }

        let twice_err = err * 2;
        if twice_err >= dy {
            err += dy;
            x0 += sx;
        }
        if twice_err <= dx {
            err += dx;
            y0 += sy;
        }
    }

    points
}

fn round_to_grid(point: PointF, grid_size: usize) -> PointI {
    let max = grid_size.saturating_sub(1) as f64;
    PointI {
        x: clamp(point.x.round(), 0.0, max) as i32,
        y: clamp(point.y.round(), 0.0, max) as i32,
    }
}

fn dedupe_consecutive_points(points: &[PointI]) -> Vec<PointI> {
    let mut deduped = Vec::with_capacity(points.len());
    for point in points {
        if deduped.last() != Some(point) {
            deduped.push(*point);
        }
    }
    deduped
}

fn trace_closed_cycle(
    adjacency: &BTreeMap<PointI, BTreeSet<PointI>>,
    original_points: &[PointF],
) -> Option<Vec<PointI>> {
    let start = *adjacency.keys().next()?;
    let neighbors = adjacency.get(&start)?.iter().copied().collect::<Vec<_>>();
    if neighbors.len() != 2 {
        return None;
    }

    let candidate_a = trace_cycle_with_first_neighbor(adjacency, start, neighbors[0])?;
    let candidate_b = trace_cycle_with_first_neighbor(adjacency, start, neighbors[1])?;
    let original_sign = polygon_signed_area_f64(original_points).signum();
    let sign_a = polygon_signed_area_i(&candidate_a).signum();
    let sign_b = polygon_signed_area_i(&candidate_b).signum();

    match (
        (sign_a * original_sign).partial_cmp(&0.0),
        (sign_b * original_sign).partial_cmp(&0.0),
    ) {
        (Some(Ordering::Greater), _) => Some(candidate_a),
        (_, Some(Ordering::Greater)) => Some(candidate_b),
        _ => Some(candidate_a),
    }
}

fn trace_cycle_with_first_neighbor(
    adjacency: &BTreeMap<PointI, BTreeSet<PointI>>,
    start: PointI,
    first_neighbor: PointI,
) -> Option<Vec<PointI>> {
    let mut cycle = vec![start];
    let mut previous = start;
    let mut current = first_neighbor;

    while current != start {
        cycle.push(current);
        let neighbors = adjacency.get(&current)?.iter().copied().collect::<Vec<_>>();
        if neighbors.len() != 2 {
            return None;
        }
        let next = if neighbors[0] == previous {
            neighbors[1]
        } else {
            neighbors[0]
        };
        previous = current;
        current = next;
        if cycle.len() > adjacency.len() + 1 {
            return None;
        }
    }

    if cycle.len() == adjacency.len() {
        Some(cycle)
    } else {
        None
    }
}

fn connected_components(adjacency: &BTreeMap<PointI, BTreeSet<PointI>>) -> usize {
    let mut visited = BTreeSet::new();
    let mut components = 0usize;

    for point in adjacency.keys().copied() {
        if visited.contains(&point) {
            continue;
        }

        components += 1;
        let mut stack = vec![point];
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            if let Some(neighbors) = adjacency.get(&current) {
                stack.extend(neighbors.iter().copied());
            }
        }
    }

    components
}

fn discrete_dirs(points: &[PointI], is_closed: bool) -> Vec<Direction8> {
    let mut dirs = Vec::new();
    for (start, end) in int_segments(points, is_closed) {
        if let Some(direction) = Direction8::from_step(end.x - start.x, end.y - start.y) {
            dirs.push(direction);
        }
    }
    dirs
}

fn discrete_turn_points(points: &[PointI], dirs: &[Direction8], is_closed: bool) -> Vec<PointI> {
    let mut turns = Vec::new();
    if points.is_empty() || dirs.len() < 2 {
        return turns;
    }

    if is_closed {
        for index in 0..dirs.len() {
            let next = (index + 1) % dirs.len();
            if dirs[index] != dirs[next] {
                turns.push(points[next]);
            }
        }
    } else {
        for index in 0..dirs.len() - 1 {
            if dirs[index] != dirs[index + 1] {
                turns.push(points[index + 1]);
            }
        }
    }
    turns
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct EdgeKey(PointI, PointI);

impl EdgeKey {
    fn new(a: PointI, b: PointI) -> Self {
        if a <= b { Self(a, b) } else { Self(b, a) }
    }
}

#[derive(Clone, Debug)]
struct MetricBundle {
    mean_error: f64,
    max_error: f64,
    render_precision: f64,
    render_recall: f64,
    render_f1: f64,
    render_iou: f64,
    symmetry_score: f64,
    alignment_dx: f64,
    alignment_dy: f64,
    source_intersections: usize,
    turn_inflation: f64,
    path_length_inflation: f64,
    self_intersections: usize,
}

#[derive(Clone, Copy, Debug)]
struct AlignmentMetrics {
    dx: f64,
    dy: f64,
    mean_error: f64,
    max_error: f64,
    render: RenderStats,
    symmetry_score: f64,
    blended_score: f64,
}

#[derive(Clone, Copy, Debug)]
struct ErrorStats {
    mean_error: f64,
    max_error: f64,
}

fn compute_metric_bundle(
    original: &[PreparedSubpath],
    quantized: &[QuantizedSubpath],
    grid_size: usize,
) -> MetricBundle {
    let mut original_turns = 0usize;
    let mut quantized_turns = 0usize;
    let mut original_length = 0.0;
    let mut quantized_length = 0.0;
    let alignment = best_alignment_metrics(original, quantized, grid_size);

    for (source_subpath, quantized_subpath) in original.iter().zip(quantized.iter()) {
        if source_subpath.points.len() < 2 || quantized_subpath.points.len() < 2 {
            continue;
        }

        original_turns += turn_count_float(&source_subpath.points, source_subpath.is_closed);
        quantized_turns +=
            turn_count_quantized(&quantized_subpath.points, quantized_subpath.is_closed);
        original_length += polyline_length_float(&source_subpath.points, source_subpath.is_closed);
        quantized_length +=
            polyline_length_int(&quantized_subpath.points, quantized_subpath.is_closed);
    }

    let original_intersections = count_self_intersections_float(original);
    let quantized_intersections = count_self_intersections_int(quantized);
    let introduced_intersections = quantized_intersections.saturating_sub(original_intersections);

    MetricBundle {
        mean_error: alignment.mean_error,
        max_error: alignment.max_error,
        render_precision: alignment.render.precision,
        render_recall: alignment.render.recall,
        render_f1: alignment.render.f1,
        render_iou: alignment.render.iou,
        symmetry_score: alignment.symmetry_score,
        alignment_dx: alignment.dx,
        alignment_dy: alignment.dy,
        source_intersections: original_intersections,
        turn_inflation: if original_turns == 0 {
            if quantized_turns == 0 {
                1.0
            } else {
                quantized_turns as f64
            }
        } else {
            quantized_turns as f64 / original_turns as f64
        },
        path_length_inflation: if original_length <= 1e-9 {
            1.0
        } else {
            quantized_length / original_length
        },
        self_intersections: introduced_intersections,
    }
}

fn best_alignment_metrics(
    original: &[PreparedSubpath],
    quantized: &[QuantizedSubpath],
    grid_size: usize,
) -> AlignmentMetrics {
    let mut candidates = Vec::new();
    let shifts = [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5];
    let quantized_mask = render_quantized_mask(quantized, grid_size, RENDER_RESOLUTION);
    let quantized_ink = quantized_mask.iter().filter(|value| **value).count();
    let symmetry_score = mask_symmetry_score(&quantized_mask, RENDER_RESOLUTION);

    for dx in shifts {
        for dy in shifts {
            let error = translated_error_stats(original, quantized, dx, dy);
            candidates.push(AlignmentMetrics {
                dx,
                dy,
                mean_error: error.mean_error,
                max_error: error.max_error,
                render: RenderStats {
                    precision: 0.0,
                    recall: 0.0,
                    f1: 0.0,
                    iou: 0.0,
                },
                symmetry_score,
                blended_score: f64::NEG_INFINITY,
            });
        }
    }

    candidates.sort_by(|left, right| {
        left.mean_error
            .partial_cmp(&right.mean_error)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left.max_error
                    .partial_cmp(&right.max_error)
                    .unwrap_or(Ordering::Equal)
            })
    });

    let mut best = None;

    for candidate in candidates
        .into_iter()
        .take(ALIGNMENT_RENDER_CANDIDATES.max(1))
    {
        let render = translated_render_stats(
            original,
            &quantized_mask,
            quantized_ink,
            candidate.dx,
            candidate.dy,
            grid_size,
        );
        let candidate = AlignmentMetrics {
            render,
            blended_score: blended_alignment_score(
                candidate.mean_error,
                candidate.max_error,
                render,
            ),
            ..candidate
        };

        let is_better = match best {
            None => true,
            Some(current) => alignment_is_better(candidate, current),
        };

        if is_better {
            best = Some(candidate);
        }
    }

    best.unwrap_or(AlignmentMetrics {
        dx: 0.0,
        dy: 0.0,
        mean_error: 0.0,
        max_error: 0.0,
        render: RenderStats {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            iou: 0.0,
        },
        symmetry_score,
        blended_score: 0.0,
    })
}

fn translated_error_stats(
    original: &[PreparedSubpath],
    quantized: &[QuantizedSubpath],
    dx: f64,
    dy: f64,
) -> ErrorStats {
    let mut distance_sum = 0.0;
    let mut distance_count = 0usize;
    let mut max_error = 0.0f64;

    for (source_subpath, quantized_subpath) in original.iter().zip(quantized.iter()) {
        if source_subpath.points.len() < 2 || quantized_subpath.points.len() < 2 {
            continue;
        }

        let translated_original = source_subpath
            .points
            .iter()
            .map(|point| PointF {
                x: point.x + dx,
                y: point.y + dy,
            })
            .collect::<Vec<_>>();
        let quantized_polyline = quantized_subpath
            .points
            .iter()
            .map(|point| PointF {
                x: point.x as f64,
                y: point.y as f64,
            })
            .collect::<Vec<_>>();

        let original_samples =
            sample_float_polyline(&translated_original, source_subpath.is_closed);
        let quantized_samples =
            sample_float_polyline(&quantized_polyline, quantized_subpath.is_closed);

        for sample in &original_samples {
            let distance = point_to_polyline_distance(
                *sample,
                &quantized_polyline,
                quantized_subpath.is_closed,
            );
            distance_sum += distance;
            distance_count += 1;
            max_error = max_error.max(distance);
        }

        for sample in &quantized_samples {
            let distance =
                point_to_polyline_distance(*sample, &translated_original, source_subpath.is_closed);
            distance_sum += distance;
            distance_count += 1;
            max_error = max_error.max(distance);
        }
    }

    if distance_count == 0 {
        ErrorStats {
            mean_error: 0.0,
            max_error: 0.0,
        }
    } else {
        ErrorStats {
            mean_error: distance_sum / distance_count as f64,
            max_error,
        }
    }
}

fn blended_alignment_score(mean_error: f64, max_error: f64, render: RenderStats) -> f64 {
    let mean_score = 1.0 - clamp(mean_error / 0.6, 0.0, 1.0);
    let max_score = 1.0 - clamp(max_error / 1.2, 0.0, 1.0);
    render.f1 * 0.60 + render.iou * 0.20 + mean_score * 0.15 + max_score * 0.05
}

fn alignment_is_better(candidate: AlignmentMetrics, current: AlignmentMetrics) -> bool {
    if candidate.blended_score > current.blended_score + 1e-9 {
        return true;
    }
    if (candidate.blended_score - current.blended_score).abs() > 1e-9 {
        return false;
    }
    if candidate.render.f1 > current.render.f1 + 1e-9 {
        return true;
    }
    if (candidate.render.f1 - current.render.f1).abs() > 1e-9 {
        return false;
    }
    if candidate.render.iou > current.render.iou + 1e-9 {
        return true;
    }
    if (candidate.render.iou - current.render.iou).abs() > 1e-9 {
        return false;
    }
    if candidate.mean_error < current.mean_error - 1e-9 {
        return true;
    }
    if (candidate.mean_error - current.mean_error).abs() > 1e-9 {
        return false;
    }
    candidate.max_error < current.max_error - 1e-9
}

fn translate_prepared_subpaths(
    subpaths: &[PreparedSubpath],
    dx: f64,
    dy: f64,
) -> Vec<PreparedSubpath> {
    subpaths
        .iter()
        .map(|subpath| PreparedSubpath {
            source_path_index: subpath.source_path_index,
            is_closed: subpath.is_closed,
            points: subpath
                .points
                .iter()
                .map(|point| PointF {
                    x: point.x + dx,
                    y: point.y + dy,
                })
                .collect(),
        })
        .collect()
}

fn translated_render_stats(
    original: &[PreparedSubpath],
    quantized_mask: &[bool],
    quantized_ink: usize,
    dx: f64,
    dy: f64,
    grid_size: usize,
) -> RenderStats {
    let translated = translate_prepared_subpaths(original, dx, dy);
    let original_mask = render_prepared_mask(&translated, grid_size, RENDER_RESOLUTION);
    compare_masks(&original_mask, quantized_mask, quantized_ink)
}

fn render_prepared_mask(
    subpaths: &[PreparedSubpath],
    grid_size: usize,
    resolution: usize,
) -> Vec<bool> {
    let mut mask = vec![false; resolution * resolution];
    let scale = render_scale(grid_size, resolution);
    let radius = render_radius(grid_size, resolution);
    for subpath in subpaths {
        rasterize_float_polyline_to_mask(
            &subpath.points,
            subpath.is_closed,
            scale,
            radius,
            resolution,
            &mut mask,
        );
    }
    mask
}

fn render_quantized_mask(
    subpaths: &[QuantizedSubpath],
    grid_size: usize,
    resolution: usize,
) -> Vec<bool> {
    let mut mask = vec![false; resolution * resolution];
    let scale = render_scale(grid_size, resolution);
    let radius = render_radius(grid_size, resolution);
    for subpath in subpaths {
        let points = subpath
            .points
            .iter()
            .map(|point| PointF {
                x: point.x as f64,
                y: point.y as f64,
            })
            .collect::<Vec<_>>();
        rasterize_float_polyline_to_mask(
            &points,
            subpath.is_closed,
            scale,
            radius,
            resolution,
            &mut mask,
        );
    }
    mask
}

fn render_scale(grid_size: usize, resolution: usize) -> f64 {
    if grid_size <= 1 || resolution <= 1 {
        1.0
    } else {
        (resolution - 1) as f64 / (grid_size - 1) as f64
    }
}

fn render_radius(grid_size: usize, resolution: usize) -> f64 {
    RENDER_STROKE_RADIUS * render_scale(grid_size, resolution)
}

fn rasterize_float_polyline_to_mask(
    points: &[PointF],
    is_closed: bool,
    scale: f64,
    radius: f64,
    resolution: usize,
    mask: &mut [bool],
) {
    if points.is_empty() {
        return;
    }

    if points.len() == 1 {
        rasterize_disc_to_mask(
            PointF {
                x: points[0].x * scale,
                y: points[0].y * scale,
            },
            radius,
            resolution,
            mask,
        );
        return;
    }

    for (start, end) in float_segments(points, is_closed) {
        rasterize_segment_to_mask(
            PointF {
                x: start.x * scale,
                y: start.y * scale,
            },
            PointF {
                x: end.x * scale,
                y: end.y * scale,
            },
            radius,
            resolution,
            mask,
        );
    }
}

fn rasterize_disc_to_mask(center: PointF, radius: f64, resolution: usize, mask: &mut [bool]) {
    let max = resolution.saturating_sub(1) as f64;
    let min_x = clamp((center.x - radius).floor(), 0.0, max) as usize;
    let max_x = clamp((center.x + radius).ceil(), 0.0, max) as usize;
    let min_y = clamp((center.y - radius).floor(), 0.0, max) as usize;
    let max_y = clamp((center.y + radius).ceil(), 0.0, max) as usize;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let point = PointF {
                x: x as f64,
                y: y as f64,
            };
            if point.distance(center) <= radius {
                mask[y * resolution + x] = true;
            }
        }
    }
}

fn rasterize_segment_to_mask(
    start: PointF,
    end: PointF,
    radius: f64,
    resolution: usize,
    mask: &mut [bool],
) {
    let max = resolution.saturating_sub(1) as f64;
    let min_x = clamp((start.x.min(end.x) - radius).floor(), 0.0, max) as usize;
    let max_x = clamp((start.x.max(end.x) + radius).ceil(), 0.0, max) as usize;
    let min_y = clamp((start.y.min(end.y) - radius).floor(), 0.0, max) as usize;
    let max_y = clamp((start.y.max(end.y) + radius).ceil(), 0.0, max) as usize;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let point = PointF {
                x: x as f64,
                y: y as f64,
            };
            if point_to_segment_distance(point, start, end) <= radius {
                mask[y * resolution + x] = true;
            }
        }
    }
}

fn compare_masks(
    original_mask: &[bool],
    quantized_mask: &[bool],
    quantized_ink: usize,
) -> RenderStats {
    let original_ink = original_mask.iter().filter(|value| **value).count();
    let overlap = original_mask
        .iter()
        .zip(quantized_mask.iter())
        .filter(|(left, right)| **left && **right)
        .count();
    let union = original_mask
        .iter()
        .zip(quantized_mask.iter())
        .filter(|(left, right)| **left || **right)
        .count();

    if original_ink == 0 && quantized_ink == 0 {
        return RenderStats {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
            iou: 1.0,
        };
    }

    let precision = if quantized_ink == 0 {
        0.0
    } else {
        overlap as f64 / quantized_ink as f64
    };
    let recall = if original_ink == 0 {
        0.0
    } else {
        overlap as f64 / original_ink as f64
    };
    let f1 = if precision + recall <= 1e-12 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    let iou = if union == 0 {
        0.0
    } else {
        overlap as f64 / union as f64
    };

    RenderStats {
        precision,
        recall,
        f1,
        iou,
    }
}

fn mask_symmetry_score(mask: &[bool], resolution: usize) -> f64 {
    [
        mirror_iou(mask, resolution, MirrorAxis::Vertical),
        mirror_iou(mask, resolution, MirrorAxis::Horizontal),
        mirror_iou(mask, resolution, MirrorAxis::Diagonal),
        mirror_iou(mask, resolution, MirrorAxis::AntiDiagonal),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

#[derive(Clone, Copy)]
enum MirrorAxis {
    Vertical,
    Horizontal,
    Diagonal,
    AntiDiagonal,
}

fn mirror_iou(mask: &[bool], resolution: usize, axis: MirrorAxis) -> f64 {
    let mut overlap = 0usize;
    let mut union = 0usize;

    for y in 0..resolution {
        for x in 0..resolution {
            let index = y * resolution + x;
            let mirror_index = mirrored_index(x, y, resolution, axis);
            let left = mask[index];
            let right = mask[mirror_index];
            if left || right {
                union += 1;
                if left && right {
                    overlap += 1;
                }
            }
        }
    }

    if union == 0 {
        1.0
    } else {
        overlap as f64 / union as f64
    }
}

fn mirrored_index(x: usize, y: usize, resolution: usize, axis: MirrorAxis) -> usize {
    let last = resolution.saturating_sub(1);
    let (mx, my) = match axis {
        MirrorAxis::Vertical => (last.saturating_sub(x), y),
        MirrorAxis::Horizontal => (x, last.saturating_sub(y)),
        MirrorAxis::Diagonal => (y, x),
        MirrorAxis::AntiDiagonal => (last.saturating_sub(y), last.saturating_sub(x)),
    };
    my * resolution + mx
}

fn symmetry_bonus(symmetry_score: f64) -> f64 {
    clamp(
        (symmetry_score - SYMMETRY_BONUS_START) / SYMMETRY_BONUS_RANGE,
        0.0,
        1.0,
    )
}

fn quantized_uses_only_cardinal_dirs(subpaths: &[QuantizedSubpath]) -> bool {
    subpaths
        .iter()
        .flat_map(|subpath| subpath.dirs.iter())
        .all(|dir| {
            matches!(
                dir,
                Direction8::N | Direction8::S | Direction8::E | Direction8::W
            )
        })
}

fn polyline_length_float(points: &[PointF], is_closed: bool) -> f64 {
    float_segments(points, is_closed)
        .map(|(start, end)| start.distance(end))
        .sum()
}

fn polyline_length_int(points: &[PointI], is_closed: bool) -> f64 {
    int_segments(points, is_closed)
        .map(|(start, end)| {
            PointF {
                x: start.x as f64,
                y: start.y as f64,
            }
            .distance(PointF {
                x: end.x as f64,
                y: end.y as f64,
            })
        })
        .sum()
}

fn sample_float_polyline(points: &[PointF], is_closed: bool) -> Vec<PointF> {
    let mut samples = Vec::new();
    for (segment_index, (start, end)) in float_segments(points, is_closed).enumerate() {
        let steps = ((start.distance(end) * ERROR_SAMPLE_DENSITY).ceil() as usize).max(1);
        for index in 0..=steps {
            if segment_index > 0 && index == 0 {
                continue;
            }
            let t = index as f64 / steps as f64;
            samples.push(start.lerp(end, t));
        }
    }
    samples
}

fn point_to_polyline_distance(point: PointF, polyline: &[PointF], is_closed: bool) -> f64 {
    float_segments(polyline, is_closed)
        .map(|(start, end)| point_to_segment_distance(point, start, end))
        .fold(f64::INFINITY, f64::min)
}

fn point_to_segment_distance(point: PointF, start: PointF, end: PointF) -> f64 {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let length_sq = dx * dx + dy * dy;
    if length_sq <= 1e-12 {
        return point.distance(start);
    }
    let t = clamp(
        ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq,
        0.0,
        1.0,
    );
    let projection = PointF {
        x: start.x + dx * t,
        y: start.y + dy * t,
    };
    point.distance(projection)
}

fn turn_count_float(points: &[PointF], is_closed: bool) -> usize {
    let dirs = compress_dirs(
        float_segments(points, is_closed)
            .filter_map(|(start, end)| Direction8::from_vector(end.x - start.x, end.y - start.y))
            .collect(),
        is_closed,
    );
    if is_closed {
        dirs.len()
    } else {
        dirs.len().saturating_sub(1)
    }
}

fn turn_count_quantized(points: &[PointI], is_closed: bool) -> usize {
    let dirs = compress_dirs(discrete_dirs(points, is_closed), is_closed);
    if is_closed {
        dirs.len()
    } else {
        dirs.len().saturating_sub(1)
    }
}

fn compress_dirs(mut dirs: Vec<Direction8>, is_closed: bool) -> Vec<Direction8> {
    let mut compressed = Vec::new();
    for direction in dirs.drain(..) {
        if compressed.last() != Some(&direction) {
            compressed.push(direction);
        }
    }

    if is_closed && compressed.len() > 1 && compressed.first() == compressed.last() {
        compressed.pop();
    }
    compressed
}

fn revisited_point_count(points: &[PointI]) -> usize {
    let mut counts = HashMap::<PointI, usize>::new();
    for point in points {
        *counts.entry(*point).or_insert(0) += 1;
    }
    counts
        .values()
        .filter(|count| **count > 1)
        .map(|count| count - 1)
        .sum()
}

fn repeated_edge_count(points: &[PointI], is_closed: bool) -> usize {
    let mut counts = HashMap::<EdgeKey, usize>::new();
    for (start, end) in int_segments(points, is_closed) {
        *counts.entry(EdgeKey::new(start, end)).or_insert(0) += 1;
    }
    counts
        .values()
        .filter(|count| **count > 1)
        .map(|count| count - 1)
        .sum()
}

fn count_self_intersections_float(subpaths: &[PreparedSubpath]) -> usize {
    let segments = build_float_segment_index(subpaths);
    count_nontrivial_intersections(
        segments
            .iter()
            .map(|segment| SegmentRef {
                path_index: segment.path_index,
                segment_index: segment.segment_index,
                segment_count: segment.segment_count,
                start: segment.start,
                end: segment.end,
                is_closed: segment.is_closed,
            })
            .collect::<Vec<_>>()
            .as_slice(),
    )
}

fn count_self_intersections_int(subpaths: &[QuantizedSubpath]) -> usize {
    let segments = build_int_segment_index(subpaths);
    count_nontrivial_intersections(
        segments
            .iter()
            .map(|segment| SegmentRef {
                path_index: segment.path_index,
                segment_index: segment.segment_index,
                segment_count: segment.segment_count,
                start: PointF {
                    x: segment.start.x as f64,
                    y: segment.start.y as f64,
                },
                end: PointF {
                    x: segment.end.x as f64,
                    y: segment.end.y as f64,
                },
                is_closed: segment.is_closed,
            })
            .collect::<Vec<_>>()
            .as_slice(),
    )
}

#[derive(Clone, Copy)]
struct SegmentRef {
    path_index: usize,
    segment_index: usize,
    segment_count: usize,
    start: PointF,
    end: PointF,
    is_closed: bool,
}

fn count_nontrivial_intersections(segments: &[SegmentRef]) -> usize {
    let mut count = 0usize;
    for index in 0..segments.len() {
        for other_index in index + 1..segments.len() {
            let left = segments[index];
            let right = segments[other_index];

            if left.path_index == right.path_index
                && segments_are_adjacent(
                    left.segment_index,
                    right.segment_index,
                    left.segment_count,
                    left.is_closed,
                )
            {
                continue;
            }

            if segments_intersect_nontrivially(left.start, left.end, right.start, right.end) {
                count += 1;
            }
        }
    }
    count
}

fn segments_are_adjacent(
    left_index: usize,
    right_index: usize,
    segment_count: usize,
    is_closed: bool,
) -> bool {
    if left_index.abs_diff(right_index) <= 1 {
        return true;
    }

    is_closed
        && segment_count > 1
        && ((left_index == 0 && right_index + 1 == segment_count)
            || (right_index == 0 && left_index + 1 == segment_count))
}

fn segments_intersect_nontrivially(a1: PointF, a2: PointF, b1: PointF, b2: PointF) -> bool {
    let epsilon = 1e-9;
    let shared_endpoint = approx_eq_point(a1, b1, epsilon)
        || approx_eq_point(a1, b2, epsilon)
        || approx_eq_point(a2, b1, epsilon)
        || approx_eq_point(a2, b2, epsilon);

    let o1 = orient(a1, a2, b1);
    let o2 = orient(a1, a2, b2);
    let o3 = orient(b1, b2, a1);
    let o4 = orient(b1, b2, a2);

    if o1 * o2 < -epsilon && o3 * o4 < -epsilon {
        return true;
    }

    for (point, start, end) in [(b1, a1, a2), (b2, a1, a2), (a1, b1, b2), (a2, b1, b2)] {
        if point_on_segment(point, start, end, epsilon) {
            let endpoint_only = shared_endpoint
                && (approx_eq_point(point, start, epsilon) || approx_eq_point(point, end, epsilon));
            if !endpoint_only {
                return true;
            }
        }
    }

    false
}

fn orient(a: PointF, b: PointF, c: PointF) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn point_on_segment(point: PointF, start: PointF, end: PointF, epsilon: f64) -> bool {
    orient(start, end, point).abs() <= epsilon
        && point.x >= start.x.min(end.x) - epsilon
        && point.x <= start.x.max(end.x) + epsilon
        && point.y >= start.y.min(end.y) - epsilon
        && point.y <= start.y.max(end.y) + epsilon
}

fn approx_eq_point(left: PointF, right: PointF, epsilon: f64) -> bool {
    (left.x - right.x).abs() <= epsilon && (left.y - right.y).abs() <= epsilon
}

fn build_float_segment_index(subpaths: &[PreparedSubpath]) -> Vec<IndexedFloatSegment> {
    let mut segments = Vec::new();
    for (path_index, subpath) in subpaths.iter().enumerate() {
        let segment_count = if subpath.is_closed {
            subpath.points.len()
        } else {
            subpath.points.len().saturating_sub(1)
        };
        for (segment_index, (start, end)) in
            float_segments(&subpath.points, subpath.is_closed).enumerate()
        {
            segments.push(IndexedFloatSegment {
                path_index,
                segment_index,
                segment_count,
                start,
                end,
                is_closed: subpath.is_closed,
            });
        }
    }
    segments
}

fn build_int_segment_index(subpaths: &[QuantizedSubpath]) -> Vec<IndexedIntSegment> {
    let mut segments = Vec::new();
    for (path_index, subpath) in subpaths.iter().enumerate() {
        let segment_count = if subpath.is_closed {
            subpath.points.len()
        } else {
            subpath.points.len().saturating_sub(1)
        };
        for (segment_index, (start, end)) in
            int_segments(&subpath.points, subpath.is_closed).enumerate()
        {
            segments.push(IndexedIntSegment {
                path_index,
                segment_index,
                segment_count,
                start,
                end,
                is_closed: subpath.is_closed,
            });
        }
    }
    segments
}

#[derive(Clone, Copy)]
struct IndexedFloatSegment {
    path_index: usize,
    segment_index: usize,
    segment_count: usize,
    start: PointF,
    end: PointF,
    is_closed: bool,
}

#[derive(Clone, Copy)]
struct IndexedIntSegment {
    path_index: usize,
    segment_index: usize,
    segment_count: usize,
    start: PointI,
    end: PointI,
    is_closed: bool,
}

fn float_segments(
    points: &[PointF],
    is_closed: bool,
) -> impl Iterator<Item = (PointF, PointF)> + '_ {
    points.iter().enumerate().filter_map(move |(index, point)| {
        let next_index = if index + 1 < points.len() {
            Some(index + 1)
        } else if is_closed && points.len() > 1 {
            Some(0)
        } else {
            None
        }?;
        Some((*point, points[next_index]))
    })
}

fn int_segments(points: &[PointI], is_closed: bool) -> impl Iterator<Item = (PointI, PointI)> + '_ {
    points.iter().enumerate().filter_map(move |(index, point)| {
        let next_index = if index + 1 < points.len() {
            Some(index + 1)
        } else if is_closed && points.len() > 1 {
            Some(0)
        } else {
            None
        }?;
        Some((*point, points[next_index]))
    })
}

fn polygon_signed_area_f64(points: &[PointF]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    for (start, end) in float_segments(points, true) {
        area += start.x * end.y - end.x * start.y;
    }
    area * 0.5
}

fn polygon_signed_area_i(points: &[PointI]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    for (start, end) in int_segments(points, true) {
        area += start.x as f64 * end.y as f64 - end.x as f64 * start.y as f64;
    }
    area * 0.5
}

fn points_for_output(points: &[PointI], is_closed: bool) -> Vec<[i32; 2]> {
    let mut output = points
        .iter()
        .map(|point| [point.x, point.y])
        .collect::<Vec<_>>();
    if is_closed && !points.is_empty() {
        output.push([points[0].x, points[0].y]);
    }
    output
}

fn build_summary(processed: &[ProcessedIcon], config: &Config, input_root: &Path) -> SummaryRecord {
    let accepted = processed.iter().filter(|icon| icon.record.accepted).count();
    let rejected = processed.len().saturating_sub(accepted);
    let acceptance_rate = if processed.is_empty() {
        0.0
    } else {
        accepted as f64 / processed.len() as f64
    };

    let mut rejection_reasons = BTreeMap::<String, usize>::new();
    for icon in processed {
        for reason in &icon.record.rejection_reasons {
            *rejection_reasons.entry(reason.clone()).or_insert(0) += 1;
        }
    }

    let curved_icons = processed
        .iter()
        .filter(|icon| icon.source_flags.has_curved_source)
        .count();
    let curved_accepted = processed
        .iter()
        .filter(|icon| icon.source_flags.has_curved_source && icon.record.accepted)
        .count();

    SummaryRecord {
        input_root: input_root.display().to_string(),
        output_root: config.output_dir.display().to_string(),
        grid_size: config.grid_size,
        fit_mode: FIT_MODE.to_string(),
        thresholds: Thresholds {
            mean_error_max: MEAN_ERROR_THRESHOLD,
            max_error_max: MAX_ERROR_THRESHOLD,
            turn_inflation_max: TURN_INFLATION_THRESHOLD,
            path_length_inflation_max: PATH_LENGTH_INFLATION_THRESHOLD,
            render_f1_min: RENDER_F1_THRESHOLD,
            render_iou_min: RENDER_IOU_THRESHOLD,
            rescue_render_f1_min: RESCUE_RENDER_F1_THRESHOLD,
            rescue_render_iou_min: RESCUE_RENDER_IOU_THRESHOLD,
            circle_ellipse_rejected_by_default: !config.allow_curved_source,
        },
        totals: Totals {
            icons: processed.len(),
            accepted,
            rejected,
            acceptance_rate: round_float(acceptance_rate, 4),
        },
        metrics: SummaryMetrics {
            mean_error: summarize_metric(processed.iter().map(|icon| icon.record.mean_error)),
            max_error: summarize_metric(processed.iter().map(|icon| icon.record.max_error)),
            render_f1: summarize_metric(processed.iter().map(|icon| icon.record.render_f1)),
            render_iou: summarize_metric(processed.iter().map(|icon| icon.record.render_iou)),
            symmetry_score: summarize_metric(
                processed.iter().map(|icon| icon.record.symmetry_score),
            ),
            turn_inflation: summarize_metric(
                processed.iter().map(|icon| icon.record.turn_inflation),
            ),
            path_length_inflation: summarize_metric(
                processed
                    .iter()
                    .map(|icon| icon.record.path_length_inflation),
            ),
        },
        rejection_reasons,
        curved_source: CurvedSourceSummary {
            icons: curved_icons,
            accepted: curved_accepted,
            rejected: curved_icons.saturating_sub(curved_accepted),
        },
        source_geometry: SourceGeometrySummary {
            circles: processed
                .iter()
                .filter(|icon| icon.source_flags.has_circle)
                .count(),
            ellipses: processed
                .iter()
                .filter(|icon| icon.source_flags.has_ellipse)
                .count(),
            rounded_rects: processed
                .iter()
                .filter(|icon| icon.source_flags.has_rounded_rect)
                .count(),
        },
        mixed_metric_rescues: processed
            .iter()
            .filter(|icon| icon.mixed_metric_rescued)
            .count(),
    }
}

fn summarize_metric(values: impl Iterator<Item = f64>) -> MetricSummary {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return MetricSummary {
            min: 0.0,
            max: 0.0,
            avg: 0.0,
        };
    }

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let avg = values.iter().sum::<f64>() / values.len() as f64;

    MetricSummary {
        min: round_float(min, 4),
        max: round_float(max, 4),
        avg: round_float(avg, 4),
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("failed to serialize JSON")?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn render_preview(icon: &ProcessedIcon) -> String {
    let status = if icon.record.accepted {
        "ACCEPTED"
    } else {
        "REJECTED"
    };
    let badge_fill = if icon.record.accepted {
        "#1d4d3f"
    } else {
        "#8b2e26"
    };
    let badge_bg = if icon.record.accepted {
        "#d9f2ea"
    } else {
        "#f9d8d1"
    };
    let max_grid = (icon.record.grid_size.saturating_sub(1)) as f64;
    let drawable = PREVIEW_SIZE - PREVIEW_PADDING * 2.0;
    let grid_step = if max_grid <= 0.0 {
        drawable
    } else {
        drawable / max_grid
    };

    let mut grid_lines = String::new();
    for index in 0..icon.record.grid_size {
        let coordinate = PREVIEW_PADDING + index as f64 * grid_step;
        let _ = write!(
            grid_lines,
            "<line x1=\"{coordinate:.2}\" y1=\"{pad:.2}\" x2=\"{coordinate:.2}\" y2=\"{end:.2}\" stroke=\"#d8d1c4\" stroke-width=\"1\"/>",
            pad = PREVIEW_PADDING,
            end = PREVIEW_SIZE - PREVIEW_PADDING,
        );
        let _ = write!(
            grid_lines,
            "<line x1=\"{pad:.2}\" y1=\"{coordinate:.2}\" x2=\"{end:.2}\" y2=\"{coordinate:.2}\" stroke=\"#d8d1c4\" stroke-width=\"1\"/>",
            pad = PREVIEW_PADDING,
            end = PREVIEW_SIZE - PREVIEW_PADDING,
        );
    }

    let original_paths = icon
        .fitted_subpaths
        .iter()
        .filter_map(|subpath| preview_path_f(&subpath.points, subpath.is_closed, grid_step))
        .map(|path| {
            format!(
                "<path d=\"{path}\" fill=\"none\" stroke=\"#1b6f84\" stroke-width=\"3.2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" opacity=\"0.45\"/>"
            )
        })
        .collect::<String>();

    let quantized_paths = icon
        .quantized_subpaths
        .iter()
        .filter_map(|subpath| {
            preview_path_i(
                &subpath.points,
                subpath.is_closed,
                grid_step,
                PREVIEW_PADDING,
                PREVIEW_SIZE,
            )
        })
        .map(|path| {
            format!(
                "<path d=\"{path}\" fill=\"none\" stroke=\"#e76f51\" stroke-width=\"5.2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>"
            )
        })
        .collect::<String>();

    let point_markers = icon
        .quantized_subpaths
        .iter()
        .flat_map(|subpath| subpath.points.iter())
        .map(|point| {
            let x = PREVIEW_PADDING + point.x as f64 * grid_step;
            let y = PREVIEW_PADDING + point.y as f64 * grid_step;
            format!(
                "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"3.6\" fill=\"#edb183\" stroke=\"#8a4f34\" stroke-width=\"1.2\"/>"
            )
        })
        .collect::<String>();

    let reasons = if icon.record.rejection_reasons.is_empty() {
        "none".to_string()
    } else {
        icon.record.rejection_reasons.join(", ")
    };

    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {size} {height}\"><rect width=\"{size}\" height=\"{height}\" fill=\"#f8f4eb\" rx=\"18\"/><rect x=\"18\" y=\"18\" width=\"{inner}\" height=\"{inner}\" fill=\"#fffdf9\" stroke=\"#d8d1c4\" stroke-width=\"2\" rx=\"16\"/>{grid_lines}{original_paths}{quantized_paths}{point_markers}<rect x=\"20\" y=\"20\" width=\"104\" height=\"28\" fill=\"{badge_bg}\" rx=\"14\"/><text x=\"30\" y=\"39\" font-size=\"15\" font-family=\"Georgia, serif\" font-weight=\"700\" fill=\"{badge_fill}\">{status}</text><text x=\"20\" y=\"286\" font-size=\"14\" font-family=\"Menlo, monospace\" fill=\"#3f372d\">mean {mean:.3}  max {max:.3}</text><text x=\"20\" y=\"306\" font-size=\"14\" font-family=\"Menlo, monospace\" fill=\"#3f372d\">f1 {render_f1:.3}  iou {render_iou:.3}</text><text x=\"20\" y=\"326\" font-size=\"14\" font-family=\"Menlo, monospace\" fill=\"#3f372d\">turn {turn:.3}x  len {length:.3}x</text><text x=\"20\" y=\"346\" font-size=\"13\" font-family=\"Menlo, monospace\" fill=\"#5d5146\">reasons: {reasons}</text></svg>",
        size = PREVIEW_SIZE,
        height = 360.0,
        inner = PREVIEW_SIZE - 36.0,
        mean = icon.record.mean_error,
        max = icon.record.max_error,
        render_f1 = icon.record.render_f1,
        render_iou = icon.record.render_iou,
        turn = icon.record.turn_inflation,
        length = icon.record.path_length_inflation,
    )
}

fn preview_path_f(points: &[PointF], is_closed: bool, grid_step: f64) -> Option<String> {
    let mut path = String::new();
    for (index, point) in points.iter().enumerate() {
        let command = if index == 0 { 'M' } else { 'L' };
        let x = PREVIEW_PADDING + point.x * grid_step;
        let y = PREVIEW_PADDING + point.y * grid_step;
        let _ = write!(path, "{command} {x:.2} {y:.2} ");
    }
    if is_closed {
        path.push('Z');
    }
    if path.is_empty() { None } else { Some(path) }
}

fn preview_path_i(
    points: &[PointI],
    is_closed: bool,
    grid_step: f64,
    padding: f64,
    _size: f64,
) -> Option<String> {
    if points.is_empty() {
        return None;
    }

    let mut path = String::new();
    for (index, point) in points.iter().enumerate() {
        let command = if index == 0 { 'M' } else { 'L' };
        let x = padding + point.x as f64 * grid_step;
        let y = padding + point.y as f64 * grid_step;
        let _ = write!(path, "{command} {x:.2} {y:.2} ");
    }
    if is_closed {
        path.push('Z');
    }
    Some(path)
}

fn render_gallery(processed: &[ProcessedIcon], summary: &SummaryRecord) -> String {
    let cards = processed
        .iter()
        .map(|icon| render_gallery_card(icon))
        .collect::<String>();
    let reason_options = summary
        .rejection_reasons
        .keys()
        .map(|reason| format!("<option value=\"{reason}\">{reason}</option>"))
        .collect::<String>();

    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Feather {grid}x{grid} Quantization Report</title><style>:root{{--bg:#f2ede1;--panel:#fffaf2;--ink:#30271f;--muted:#6b6055;--line:#d5c8b2;--accept:#204f41;--reject:#8d3329;--accent:#cb6d4d;}}body{{margin:0;background:radial-gradient(circle at top,#fcf6e9 0%,var(--bg) 62%,#e5d9c4 100%);color:var(--ink);font-family:\"Iowan Old Style\",\"Palatino Linotype\",serif;}}main{{max-width:1460px;margin:0 auto;padding:28px 18px 56px;}}header,section,.controls{{background:rgba(255,250,242,0.92);border:1px solid var(--line);border-radius:24px;box-shadow:0 18px 46px rgba(66,44,22,0.08);}}header{{padding:28px;}}h1,h2,h3,p{{margin:0;}}header p{{margin-top:10px;line-height:1.45;color:var(--muted);}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:18px;}}.stat{{padding:14px 16px;border-radius:16px;background:#fffdf8;border:1px solid var(--line);}}.stat strong{{display:block;font-size:30px;color:var(--accent);}}.controls{{margin-top:18px;padding:18px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;}}label{{display:block;font-size:14px;color:var(--muted);margin-bottom:6px;}}input,select{{width:100%;padding:11px 12px;border-radius:14px;border:1px solid var(--line);background:#fffdf8;color:var(--ink);font:inherit;}}section{{margin-top:22px;padding:20px;}}section h2{{font-size:28px;}}section p{{margin-top:8px;color:var(--muted);}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:18px;margin-top:18px;}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:14px;box-shadow:0 14px 34px rgba(66,44,22,0.08);}}.card img{{display:block;width:100%;border-radius:16px;border:1px solid var(--line);background:white;}}.card-head{{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-top:12px;}}.badge{{padding:6px 10px;border-radius:999px;font-size:12px;font-weight:700;letter-spacing:0.05em;}}.badge.accept{{background:#dcefe8;color:var(--accept);}}.badge.reject{{background:#f4d8d3;color:var(--reject);}}.metrics{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:12px;}}.metric{{padding:10px 12px;border-radius:14px;background:#fffdf8;border:1px solid var(--line);}}.metric .label{{font-size:12px;color:var(--muted);}}.metric .value{{margin-top:4px;font-size:18px;font-weight:700;color:var(--accent);}}.card p{{margin-top:10px;color:var(--muted);line-height:1.45;}}code{{font-family:\"SFMono-Regular\",Menlo,Consolas,monospace;font-size:12px;line-height:1.45;white-space:pre-wrap;word-break:break-word;}}@media (max-width: 720px){{header,.controls,section{{border-radius:20px;}}}}</style></head><body><main><header><h1>Feather {grid}x{grid} Quantization Report</h1><p>Every SVG under the input corpus was flattened, fitted to a tight geometry bounding box, quantized onto a {grid}x{grid} lattice with 8-neighbor moves, aligned over a small phase search, and filtered by a mix of geometric, rendered, and symmetry-aware similarity.</p><div class=\"stats\"><div class=\"stat\"><strong>{icons}</strong>icons processed</div><div class=\"stat\"><strong>{accepted}</strong>accepted</div><div class=\"stat\"><strong>{rejected}</strong>rejected</div><div class=\"stat\"><strong>{rate:.1}%</strong>acceptance rate</div></div></header><div class=\"controls\"><div><label for=\"name-filter\">Name filter</label><input id=\"name-filter\" type=\"search\" placeholder=\"search by icon name\"></div><div><label for=\"reason-filter\">Rejection reason</label><select id=\"reason-filter\"><option value=\"\">all reasons</option>{reason_options}</select></div><div><label for=\"sort-order\">Sort</label><select id=\"sort-order\"><option value=\"name\">name</option><option value=\"symmetry_score\">symmetry</option><option value=\"render_f1\">render f1</option><option value=\"render_iou\">render iou</option><option value=\"mean_error\">mean error</option><option value=\"max_error\">max error</option><option value=\"turn_inflation\">turn inflation</option><option value=\"direction_sequence_length\">direction length</option></select></div></div><section><h2>Accepted</h2><p>Accepted icons survived the hard topology checks and the mixed geometric plus render-similarity filter.</p><div class=\"grid\" id=\"accepted-grid\">{cards}</div></section><section><h2>Rejected</h2><p>Rejected icons remain visible with previews and explicit rejection reasons, including low rendered overlap when that was the main failure.</p><div class=\"grid\" id=\"rejected-grid\"></div></section></main><script>const allCards=[...document.querySelectorAll('.card')];const acceptedGrid=document.getElementById('accepted-grid');const rejectedGrid=document.getElementById('rejected-grid');for(const card of [...allCards]){{if(card.dataset.accepted==='false')rejectedGrid.appendChild(card);}}function applyFilters(){{const name=document.getElementById('name-filter').value.toLowerCase().trim();const reason=document.getElementById('reason-filter').value;const sort=document.getElementById('sort-order').value;const grids=[acceptedGrid,rejectedGrid];for(const grid of grids){{const cards=[...grid.children];cards.sort((a,b)=>{{if(sort==='name')return a.dataset.name.localeCompare(b.dataset.name);return parseFloat(b.dataset[sort])-parseFloat(a.dataset[sort]);}});if(sort==='mean_error'||sort==='max_error'||sort==='turn_inflation'||sort==='direction_sequence_length'){{cards.reverse();}}for(const card of cards){{const nameOk=!name||card.dataset.name.includes(name);const reasonOk=!reason||card.dataset.reasons.split('|').includes(reason);card.style.display=nameOk&&reasonOk?'block':'none';grid.appendChild(card);}}}}document.getElementById('name-filter').addEventListener('input',applyFilters);document.getElementById('reason-filter').addEventListener('change',applyFilters);document.getElementById('sort-order').addEventListener('change',applyFilters);applyFilters();</script></body></html>",
        grid = summary.grid_size,
        icons = summary.totals.icons,
        accepted = summary.totals.accepted,
        rejected = summary.totals.rejected,
        rate = summary.totals.acceptance_rate * 100.0,
    )
}

fn render_gallery_card(icon: &ProcessedIcon) -> String {
    let status_class = if icon.record.accepted {
        "accept"
    } else {
        "reject"
    };
    let status_label = if icon.record.accepted {
        "ACCEPTED"
    } else {
        "REJECTED"
    };
    let reasons = if icon.record.rejection_reasons.is_empty() {
        "none".to_string()
    } else {
        icon.record.rejection_reasons.join(", ")
    };
    let reasons_attr = if icon.record.rejection_reasons.is_empty() {
        String::new()
    } else {
        icon.record.rejection_reasons.join("|")
    };

    format!(
        "<article class=\"card\" data-name=\"{name}\" data-accepted=\"{accepted}\" data-mean_error=\"{mean}\" data-max_error=\"{max}\" data-render_f1=\"{render_f1}\" data-render_iou=\"{render_iou}\" data-symmetry_score=\"{symmetry}\" data-turn_inflation=\"{turn}\" data-direction_sequence_length=\"{dirs}\" data-reasons=\"{reasons_attr}\"><img src=\"{preview}\" alt=\"{name} quantization preview\"><div class=\"card-head\"><div><h3>{name}</h3><p><code>{path}</code></p></div><span class=\"badge {status_class}\">{status_label}</span></div><div class=\"metrics\"><div class=\"metric\"><div class=\"label\">Mean error</div><div class=\"value\">{mean:.3}</div></div><div class=\"metric\"><div class=\"label\">Max error</div><div class=\"value\">{max:.3}</div></div><div class=\"metric\"><div class=\"label\">Render F1</div><div class=\"value\">{render_f1:.3}</div></div><div class=\"metric\"><div class=\"label\">Render IoU</div><div class=\"value\">{render_iou:.3}</div></div><div class=\"metric\"><div class=\"label\">Symmetry</div><div class=\"value\">{symmetry:.3}</div></div><div class=\"metric\"><div class=\"label\">Turn inflation</div><div class=\"value\">{turn:.3}x</div></div><div class=\"metric\"><div class=\"label\">Dirs</div><div class=\"value\">{dirs}</div></div></div><p><strong>Reasons</strong><br><code>{reasons}</code></p><p><strong>Counts</strong><br><code>self_intersections={self_intersections} revisited_points={revisited_points} repeated_edges={repeated_edges}</code></p></article>",
        name = escape_html(&icon.record.name),
        accepted = icon.record.accepted,
        mean = icon.record.mean_error,
        max = icon.record.max_error,
        render_f1 = icon.record.render_f1,
        render_iou = icon.record.render_iou,
        symmetry = icon.record.symmetry_score,
        turn = icon.record.turn_inflation,
        dirs = icon.record.direction_sequence_length,
        reasons_attr = escape_html(&reasons_attr),
        preview = escape_html(&icon.record.preview_path),
        path = escape_html(&icon.record.original_svg_path),
        status_class = status_class,
        status_label = status_label,
        reasons = escape_html(&reasons),
        self_intersections = icon.record.self_intersections,
        revisited_points = icon.record.revisited_points,
        repeated_edges = icon.record.repeated_edges,
    )
}

fn render_methodology_readme(summary: &SummaryRecord) -> String {
    format!(
        "# Feather Grid Pipeline\n\nThis report was generated from the full SVG corpus under `{input}`. Each icon was parsed with `usvg`, flattened into absolute polylines, fitted to its tight geometry bounding box, quantized onto a `{grid}x{grid}` lattice, phase-aligned over a small translation search, and then filtered conservatively.\n\nFrozen thresholds:\n\n- mean symmetric error `<= {mean:.2}` grid cells\n- max symmetric error `<= {max:.2}` grid cells\n- turn inflation `<= {turn:.2}x`\n- path length inflation `<= {length:.2}x`\n- rendered overlap F1 `>= {render_f1:.2}`\n- rendered overlap IoU `>= {render_iou:.2}`\n- no introduced repeated internal lattice points beyond the source icon's own intersection topology\n- no repeated edges\n- no introduced self-intersections\n- no disconnected or ambiguous closed-contour graphs\n- circles and ellipses are rejected by default unless `--allow-curved-source` is used\n- acceptance uses a mixed score that combines line distance with rendered stroke overlap\n- a reproducible mixed-metric rescue can clear only soft failures (`mean_error`, `max_error`, `turn_inflation`, `render_f1`, `render_iou`, `circle_source_blacklist`, `ellipse_source_blacklist`) when rendered overlap is strong and topology remains clean\n\nRun command:\n\n```bash\ncargo run -p fusor-feather-pipeline -- --input /path/to/feather/icons --output /path/to/report --grid {grid}\n```\n\nCurrent run summary: `{accepted}` accepted, `{rejected}` rejected, `{rate:.1}%` acceptance. Mixed-metric rescues applied: `{rescues}`.\n",
        input = summary.input_root,
        grid = summary.grid_size,
        mean = summary.thresholds.mean_error_max,
        max = summary.thresholds.max_error_max,
        turn = summary.thresholds.turn_inflation_max,
        length = summary.thresholds.path_length_inflation_max,
        render_f1 = summary.thresholds.render_f1_min,
        render_iou = summary.thresholds.render_iou_min,
        accepted = summary.totals.accepted,
        rejected = summary.totals.rejected,
        rate = summary.totals.acceptance_rate * 100.0,
        rescues = summary.mixed_metric_rescues,
    )
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

fn round_float(value: f64, decimals: usize) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    (value * factor).round() / factor
}
