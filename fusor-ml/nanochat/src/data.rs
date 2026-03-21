use std::{
    cmp::Reverse,
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use fusor::VarBuilder;
use fusor_gguf::GgufValue;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Deserialize;

use crate::config::RuntimeConfig;

const BOS_TOKEN: &str = "<bos>";
const EOT_TOKEN: &str = "<eot>";
const DEFAULT_MAX_COUNT: usize = 8;
const CANVAS_SIZE: f32 = 128.0;
const CANVAS_PADDING: f32 = 14.0;
pub const ACTION_MODE_COUNT: usize = 3;
pub const ACTION_DIRECTION_COUNT: usize = 8;
const MAX_STROKE_PERMUTATIONS: usize = 24;

#[derive(Clone)]
pub struct StrokeTokenizer {
    tokens: Vec<String>,
    bos_token: u32,
    eot_token: u32,
    grid_size: usize,
    max_count: usize,
    travel_start: u32,
    draw_start: u32,
}

fn axis_limit(position: i32, delta: i32, max: i32) -> usize {
    match delta.cmp(&0) {
        std::cmp::Ordering::Greater => (max - position).max(0) as usize,
        std::cmp::Ordering::Less => position.max(0) as usize,
        std::cmp::Ordering::Equal => usize::MAX,
    }
}

fn parse_segment_count_token(token: &str) -> Option<usize> {
    token.rsplit('_').next()?.parse::<usize>().ok()
}

fn inferred_max_count(tokens: &[String]) -> usize {
    tokens
        .iter()
        .filter(|token| token.starts_with("TRAVEL_") || token.starts_with("DRAW_"))
        .filter_map(|token| parse_segment_count_token(token))
        .max()
        .unwrap_or(DEFAULT_MAX_COUNT)
}

pub struct SourceDataset {
    files: Vec<SourceFile>,
}

#[derive(Clone)]
pub struct SourceFile {
    path: String,
    tokens: Vec<u32>,
    prompt_tokens: Vec<u32>,
    target_tokens: Vec<u32>,
}

pub struct DatasetSplit {
    pub train: SourceDataset,
    pub validation: SourceDataset,
    pub test: SourceDataset,
}

pub struct DatasetSource {
    pub label: String,
    pub tokenizer: StrokeTokenizer,
    pub split: DatasetSplit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanvasStateSpec {
    pub coordinate_vocab_size: usize,
    pub coordinate_offset: i32,
}

pub struct CanvasStateIndexes {
    pub cursor_x: Vec<Vec<u32>>,
    pub cursor_y: Vec<Vec<u32>>,
    pub pen_state: Vec<Vec<u32>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Direction {
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind {
    Bos,
    Eot,
    Segment(ActionMode, Direction, usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActionMode {
    Travel,
    Draw,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodedActionToken {
    pub mode_index: u32,
    pub direction_index: Option<u32>,
    pub count: Option<usize>,
    pub normalized_count: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrokePath {
    pub points: Vec<(i32, i32)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrokeScene {
    pub prompt_strokes: Vec<StrokePath>,
    pub continuation_strokes: Vec<StrokePath>,
}

#[derive(Deserialize)]
struct FeatherIconRecord {
    name: String,
    #[serde(rename = "original_svg_path")]
    _original_svg_path: String,
    quantized_points: Vec<Vec<[i32; 2]>>,
    #[serde(rename = "dirs")]
    _dirs: Vec<Vec<String>>,
    grid_size: usize,
    #[serde(default = "default_true")]
    accepted: bool,
}

#[derive(Clone, Copy)]
enum FeatherSplit {
    Train,
    Validation,
    Test,
}

#[derive(Clone, Copy)]
enum FeatherAugmentation {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    MirrorX,
    MirrorY,
    MirrorMainDiagonal,
    MirrorAntiDiagonal,
}

impl FeatherAugmentation {
    fn suffix(self) -> &'static str {
        match self {
            Self::Identity => "",
            Self::Rotate90 => "rot90",
            Self::Rotate180 => "rot180",
            Self::Rotate270 => "rot270",
            Self::MirrorX => "mirrorx",
            Self::MirrorY => "mirrory",
            Self::MirrorMainDiagonal => "diag",
            Self::MirrorAntiDiagonal => "antidiag",
        }
    }
}

#[derive(Clone)]
struct RenderState {
    cursor: (i32, i32),
    pen_down: bool,
    current_stroke: Vec<(i32, i32)>,
    completed_strokes: Vec<Vec<(i32, i32)>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FeatherStroke {
    points: Vec<(i32, i32)>,
}

#[derive(Clone, Copy)]
struct CanvasCursorState {
    cursor: (i32, i32),
    pen_down: bool,
}

#[derive(Clone, Copy)]
enum ShapeKind {
    Circle,
    Square,
    Rectangle,
    Triangle,
    RightTriangle,
    Diamond,
    Hexagon,
    Octagon,
    House,
    Trapezoid,
    Parallelogram,
    Elbow,
    Shield,
    Spiral,
    Kite,
    Tag,
    Notch,
    Bookmark,
    Stair,
    Capsule,
    Vase,
    Bolt,
    Tab,
    Cup,
    Cross,
    Star,
    Arrow,
    Pentagon,
}

pub struct Batch {
    pub windows: Vec<Vec<u32>>,
    pub masks: Vec<Vec<f32>>,
    pub valid_tokens: f32,
    pub seq_len: usize,
}

impl Direction {
    fn all() -> [Self; 8] {
        [
            Self::N,
            Self::NE,
            Self::E,
            Self::SE,
            Self::S,
            Self::SW,
            Self::W,
            Self::NW,
        ]
    }

    fn index(self) -> usize {
        match self {
            Self::N => 0,
            Self::NE => 1,
            Self::E => 2,
            Self::SE => 3,
            Self::S => 4,
            Self::SW => 5,
            Self::W => 6,
            Self::NW => 7,
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

    fn delta(self) -> (i32, i32) {
        match self {
            Self::N => (0, -1),
            Self::NE => (1, -1),
            Self::E => (1, 0),
            Self::SE => (1, 1),
            Self::S => (0, 1),
            Self::SW => (-1, 1),
            Self::W => (-1, 0),
            Self::NW => (-1, -1),
        }
    }

    fn rotate(self, steps: usize) -> Self {
        Self::all()[(self.index() + steps) % Self::all().len()]
    }

    fn mirror_horizontal(self) -> Self {
        match self {
            Self::N => Self::N,
            Self::NE => Self::NW,
            Self::E => Self::W,
            Self::SE => Self::SW,
            Self::S => Self::S,
            Self::SW => Self::SE,
            Self::W => Self::E,
            Self::NW => Self::NE,
        }
    }

    fn opposite(self) -> Self {
        self.rotate(Direction::all().len() / 2)
    }
}

impl ActionMode {
    fn index(self) -> u32 {
        match self {
            Self::Travel => 0,
            Self::Draw => 1,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Travel => "TRAVEL",
            Self::Draw => "DRAW",
        }
    }
}

impl ShapeKind {
    fn all() -> [Self; 28] {
        [
            Self::Circle,
            Self::Square,
            Self::Rectangle,
            Self::Triangle,
            Self::RightTriangle,
            Self::Diamond,
            Self::Hexagon,
            Self::Octagon,
            Self::House,
            Self::Trapezoid,
            Self::Parallelogram,
            Self::Elbow,
            Self::Shield,
            Self::Spiral,
            Self::Kite,
            Self::Tag,
            Self::Notch,
            Self::Bookmark,
            Self::Stair,
            Self::Capsule,
            Self::Vase,
            Self::Bolt,
            Self::Tab,
            Self::Cup,
            Self::Cross,
            Self::Star,
            Self::Arrow,
            Self::Pentagon,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::Circle => "circle",
            Self::Square => "square",
            Self::Rectangle => "rectangle",
            Self::Triangle => "triangle",
            Self::RightTriangle => "righttriangle",
            Self::Diamond => "diamond",
            Self::Hexagon => "hexagon",
            Self::Octagon => "octagon",
            Self::House => "house",
            Self::Trapezoid => "trapezoid",
            Self::Parallelogram => "parallelogram",
            Self::Elbow => "elbow",
            Self::Shield => "shield",
            Self::Spiral => "spiral",
            Self::Kite => "kite",
            Self::Tag => "tag",
            Self::Notch => "notch",
            Self::Bookmark => "bookmark",
            Self::Stair => "stair",
            Self::Capsule => "capsule",
            Self::Vase => "vase",
            Self::Bolt => "bolt",
            Self::Tab => "tab",
            Self::Cup => "cup",
            Self::Cross => "cross",
            Self::Star => "star",
            Self::Arrow => "arrow",
            Self::Pentagon => "pentagon",
        }
    }

    fn is_closed(self) -> bool {
        true
    }
}

impl StrokeTokenizer {
    pub fn new() -> Self {
        Self::synthetic()
    }

    pub fn synthetic() -> Self {
        Self::with_max_count(0, DEFAULT_MAX_COUNT)
    }

    pub fn with_grid(grid_size: usize) -> Self {
        let max_count = grid_size.saturating_sub(1).max(1);
        Self::with_max_count(grid_size, max_count)
    }

    fn with_max_count(grid_size: usize, max_count: usize) -> Self {
        let mut tokens = vec![BOS_TOKEN.to_string(), EOT_TOKEN.to_string()];
        let bos_token = 0;
        let eot_token = 1;
        let travel_start = tokens.len() as u32;
        for direction in Direction::all() {
            for count in 1..=max_count {
                tokens.push(format!(
                    "{}_{}_{}",
                    ActionMode::Travel.as_str(),
                    direction.as_str(),
                    count
                ));
            }
        }
        let draw_start = tokens.len() as u32;
        for direction in Direction::all() {
            for count in 1..=max_count {
                tokens.push(format!(
                    "{}_{}_{}",
                    ActionMode::Draw.as_str(),
                    direction.as_str(),
                    count
                ));
            }
        }

        Self {
            tokens,
            bos_token,
            eot_token,
            grid_size,
            max_count,
            travel_start,
            draw_start,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn bos_token(&self) -> u32 {
        self.bos_token
    }

    pub fn eot_token(&self) -> u32 {
        self.eot_token
    }

    pub fn max_count(&self) -> usize {
        self.max_count
    }

    pub fn grid_size(&self) -> usize {
        self.grid_size
    }

    pub fn direction_count(&self) -> usize {
        ACTION_DIRECTION_COUNT
    }

    pub fn token_name(&self, token: u32) -> &str {
        self.tokens
            .get(token as usize)
            .map(String::as_str)
            .unwrap_or("<invalid>")
    }

    pub fn describe_tokens(&self, tokens: &[u32], limit: usize) -> String {
        let shown = tokens
            .iter()
            .take(limit)
            .map(|token| self.token_name(*token))
            .collect::<Vec<_>>()
            .join(" ");
        if tokens.len() > limit {
            format!("{shown} ...")
        } else {
            shown
        }
    }

    pub fn gguf_metadata(&self) -> Vec<(String, GgufValue)> {
        let token_types: Box<[GgufValue]> = self
            .tokens
            .iter()
            .enumerate()
            .map(|(index, _)| {
                if index as u32 == self.bos_token || index as u32 == self.eot_token {
                    GgufValue::U32(3)
                } else {
                    GgufValue::U32(1)
                }
            })
            .collect();

        vec![
            (
                "tokenizer.ggml.model".to_string(),
                GgufValue::String("word".into()),
            ),
            (
                "tokenizer.ggml.pre".to_string(),
                GgufValue::String("default".into()),
            ),
            (
                "tokenizer.ggml.add_bos_token".to_string(),
                GgufValue::Bool(false),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(
                    self.tokens
                        .iter()
                        .cloned()
                        .map(|token| GgufValue::String(token.into_boxed_str()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                GgufValue::Array(token_types),
            ),
            (
                "tokenizer.ggml.bos_token_id".to_string(),
                GgufValue::U32(self.bos_token),
            ),
            (
                "tokenizer.ggml.eos_token_id".to_string(),
                GgufValue::U32(self.eot_token),
            ),
            (
                "tokenizer.nanochat.grid_size".to_string(),
                GgufValue::U32(self.grid_size as u32),
            ),
            (
                "tokenizer.nanochat.max_count".to_string(),
                GgufValue::U32(self.max_count as u32),
            ),
        ]
    }

    pub fn from_var_builder(vb: &VarBuilder) -> fusor::Result<Self> {
        let tokens = vb
            .get_metadata("tokenizer.ggml.tokens")
            .ok_or_else(|| fusor::Error::msg("tokenizer.ggml.tokens metadata missing"))?
            .to_array()
            .map_err(|error| fusor::Error::msg(error.to_string()))?;
        let loaded_tokens = tokens
            .iter()
            .map(|token| {
                token
                    .to_string()
                    .map(String::from)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(fusor::Error::msg)?;
        let grid_size = vb
            .get_metadata("tokenizer.nanochat.grid_size")
            .and_then(|value| value.to_u32().ok())
            .map(|value| value as usize)
            .unwrap_or(0);
        let max_count = vb
            .get_metadata("tokenizer.nanochat.max_count")
            .and_then(|value| value.to_u32().ok())
            .map(|value| value as usize)
            .unwrap_or_else(|| inferred_max_count(&loaded_tokens));
        let tokenizer = Self::with_max_count(grid_size, max_count.max(1));
        if loaded_tokens != tokenizer.tokens {
            return Err(fusor::Error::msg(
                "GGUF tokenizer vocabulary does not match the expected stroke vocabulary",
            ));
        }

        let bos_token = vb
            .get_metadata("tokenizer.ggml.bos_token_id")
            .ok_or_else(|| fusor::Error::msg("tokenizer.ggml.bos_token_id metadata missing"))?
            .to_u32()
            .map_err(|error| fusor::Error::msg(error.to_string()))?;
        let eot_token = vb
            .get_metadata("tokenizer.ggml.eos_token_id")
            .ok_or_else(|| fusor::Error::msg("tokenizer.ggml.eos_token_id metadata missing"))?
            .to_u32()
            .map_err(|error| fusor::Error::msg(error.to_string()))?;
        if bos_token != tokenizer.bos_token || eot_token != tokenizer.eot_token {
            return Err(fusor::Error::msg(
                "GGUF tokenizer special token ids do not match the expected stroke tokenizer ids",
            ));
        }

        Ok(tokenizer)
    }

    pub fn same_vocabulary(&self, other: &Self) -> bool {
        self.tokens == other.tokens
            && self.bos_token == other.bos_token
            && self.eot_token == other.eot_token
    }

    fn decode_token_kind(&self, token: u32) -> TokenKind {
        if token == self.bos_token {
            return TokenKind::Bos;
        }
        if token == self.eot_token {
            return TokenKind::Eot;
        }

        if let Some(segment) = self.decode_segment_token(token) {
            return TokenKind::Segment(
                if segment.mode_index == ActionMode::Travel.index() {
                    ActionMode::Travel
                } else {
                    ActionMode::Draw
                },
                Direction::all()[segment.direction_index.unwrap() as usize],
                segment.count.unwrap(),
            );
        }

        panic!("unknown stroke token id {token}");
    }

    pub fn decode_segment_token(&self, token: u32) -> Option<DecodedActionToken> {
        if token == self.bos_token || token == self.eot_token {
            return None;
        }

        let segments_per_mode = (ACTION_DIRECTION_COUNT * self.max_count) as u32;
        let (mode, offset) =
            if token >= self.travel_start && token < self.travel_start + segments_per_mode {
                (ActionMode::Travel, token - self.travel_start)
            } else if token >= self.draw_start && token < self.draw_start + segments_per_mode {
                (ActionMode::Draw, token - self.draw_start)
            } else {
                return None;
            };

        let direction_index = offset / self.max_count as u32;
        let count = (offset % self.max_count as u32 + 1) as usize;
        Some(DecodedActionToken {
            mode_index: mode.index(),
            direction_index: Some(direction_index),
            count: Some(count),
            normalized_count: Some(self.normalize_count(count)),
        })
    }

    pub fn decode_training_target(&self, token: u32) -> DecodedActionToken {
        if token == self.eot_token {
            return DecodedActionToken {
                mode_index: 2,
                direction_index: None,
                count: None,
                normalized_count: None,
            };
        }
        self.decode_segment_token(token).unwrap_or_else(|| {
            panic!(
                "unsupported training target token {}",
                self.token_name(token)
            )
        })
    }

    fn travel_token(&self, direction: Direction, count: usize) -> u32 {
        self.segment_token(ActionMode::Travel, direction, count)
    }

    fn draw_token(&self, direction: Direction, count: usize) -> u32 {
        self.segment_token(ActionMode::Draw, direction, count)
    }

    pub fn token_from_components(
        &self,
        mode_index: u32,
        direction_index: u32,
        count: usize,
    ) -> u32 {
        let mode = match mode_index {
            0 => ActionMode::Travel,
            1 => ActionMode::Draw,
            2 => return self.eot_token,
            other => panic!("invalid action mode index {other}"),
        };
        let direction = Direction::all()
            .get(direction_index as usize)
            .copied()
            .unwrap_or_else(|| panic!("invalid direction index {direction_index}"));
        self.segment_token(mode, direction, count)
    }

    pub fn normalize_count(&self, count: usize) -> f32 {
        if self.max_count <= 1 {
            0.0
        } else {
            (count.saturating_sub(1)) as f32 / (self.max_count - 1) as f32
        }
    }

    pub fn expected_count_from_ordinal(&self, ordinal_probs: &[f32]) -> f32 {
        ordinal_probs.iter().copied().sum::<f32>() + 1.0
    }

    pub fn count_from_normalized(&self, normalized: f32) -> usize {
        if self.max_count <= 1 {
            return 1;
        }
        let scaled = normalized.clamp(0.0, 1.0) * (self.max_count - 1) as f32;
        1 + scaled.round() as usize
    }

    pub fn legal_count_limit(&self, cursor: (i32, i32), direction_index: u32) -> usize {
        if self.grid_size == 0 {
            return self.max_count;
        }
        let direction = Direction::all()[direction_index as usize];
        let (dx, dy) = direction.delta();
        let max = self.grid_size.saturating_sub(1) as i32;
        let x_limit = axis_limit(cursor.0, dx, max);
        let y_limit = axis_limit(cursor.1, dy, max);
        x_limit.min(y_limit).min(self.max_count)
    }

    pub fn cursor_after_tokens(&self, tokens: &[u32]) -> (i32, i32) {
        let mut state = CanvasCursorState::default();
        for &token in tokens {
            apply_token_to_canvas_state(self, token, &mut state);
        }
        state.cursor
    }

    fn segment_token(&self, mode: ActionMode, direction: Direction, count: usize) -> u32 {
        assert!(
            (1..=self.max_count).contains(&count),
            "invalid segment count {count}; max count is {}",
            self.max_count
        );
        let base = match mode {
            ActionMode::Travel => self.travel_start,
            ActionMode::Draw => self.draw_start,
        };
        base + direction.index() as u32 * self.max_count as u32 + (count - 1) as u32
    }
}

const fn default_true() -> bool {
    true
}

impl SourceDataset {
    pub fn num_docs(&self) -> usize {
        self.files.len()
    }

    pub fn num_tokens(&self) -> usize {
        self.files.iter().map(|file| file.tokens.len()).sum()
    }

    pub fn num_training_windows(&self, _block_size: usize) -> usize {
        self.files.len()
    }

    pub fn max_tokens_per_example(&self) -> usize {
        self.files
            .iter()
            .map(|file| file.tokens.len())
            .max()
            .unwrap_or(0)
    }

    pub fn files(&self) -> &[SourceFile] {
        &self.files
    }

    pub fn sample_batch(&self, rng: &mut StdRng, pad_token: u32, config: &RuntimeConfig) -> Batch {
        if self.files.is_empty() {
            return Batch {
                windows: Vec::new(),
                masks: Vec::new(),
                valid_tokens: 0.0,
                seq_len: 0,
            };
        }

        let mut sampled = (0..config.batch_size)
            .map(|_| rng.random_range(0..self.files.len()))
            .collect::<Vec<_>>();
        sampled.sort_by_key(|&index| Reverse(self.window_len_at(index, config.block_size)));

        self.build_batch(&sampled, pad_token, config.block_size)
    }

    pub fn epoch_batches(
        &self,
        rng: &mut StdRng,
        pad_token: u32,
        config: &RuntimeConfig,
    ) -> Vec<Batch> {
        if self.files.is_empty() {
            return Vec::new();
        }

        let mut indices: Vec<usize> = (0..self.files.len()).collect();
        indices.shuffle(rng);
        let batch_size = config.batch_size.max(1);
        let bucket_size = batch_size.saturating_mul(32).max(batch_size);

        let mut batches = Vec::with_capacity(indices.len().div_ceil(batch_size));
        for bucket in indices.chunks(bucket_size) {
            let mut sorted_bucket = bucket.to_vec();
            sorted_bucket
                .sort_by_key(|&index| Reverse(self.window_len_at(index, config.block_size)));
            for chunk in sorted_bucket.chunks(batch_size) {
                batches.push(self.build_batch(chunk, pad_token, config.block_size));
            }
        }
        batches.shuffle(rng);
        batches
    }

    pub fn steps_per_epoch(&self, _block_size: usize, batch_size: usize) -> usize {
        self.files.len().div_ceil(batch_size).max(1)
    }

    pub fn evaluation_batches(&self, pad_token: u32, config: &RuntimeConfig) -> Vec<Batch> {
        if self.files.is_empty() {
            return Vec::new();
        }

        let steps = config
            .eval_batches
            .saturating_mul(config.batch_size)
            .min(self.files.len());
        let batch_size = config.batch_size.max(1);
        let mut indices: Vec<usize> = (0..steps).collect();
        indices.sort_by_key(|&index| Reverse(self.window_len_at(index, config.block_size)));
        let mut batches = Vec::new();
        for chunk in indices.chunks(batch_size) {
            batches.push(self.build_batch(chunk, pad_token, config.block_size));
        }

        batches
    }

    fn build_batch(&self, file_indices: &[usize], pad_token: u32, block_size: usize) -> Batch {
        if file_indices.is_empty() {
            return Batch {
                windows: Vec::new(),
                masks: Vec::new(),
                valid_tokens: 0.0,
                seq_len: 0,
            };
        }

        let slices = file_indices
            .iter()
            .map(|&index| self.window_slice_at(index, block_size))
            .collect::<Vec<_>>();
        let seq_len = slices
            .iter()
            .map(|slice| slice.len().saturating_sub(1))
            .max()
            .unwrap_or(0);

        let mut windows = Vec::with_capacity(slices.len());
        let mut masks = Vec::with_capacity(slices.len());
        let mut valid_tokens = 0.0;
        for slice in slices {
            let mut window = vec![pad_token; seq_len + 1];
            let mut mask = vec![0.0; seq_len];
            window[..slice.len()].copy_from_slice(slice);

            let valid = slice.len().saturating_sub(1);
            for local_index in 0..valid {
                mask[local_index] = 1.0;
            }

            valid_tokens += valid as f32;
            windows.push(window);
            masks.push(mask);
        }

        Batch {
            windows,
            masks,
            valid_tokens,
            seq_len,
        }
    }

    fn window_slice_at(&self, file_index: usize, block_size: usize) -> &[u32] {
        let file = &self.files[file_index];
        let start = file.tokens.len().saturating_sub(block_size + 1);
        &file.tokens[start..]
    }

    fn window_len_at(&self, file_index: usize, block_size: usize) -> usize {
        self.window_slice_at(file_index, block_size).len()
    }
}

impl SourceFile {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn target_tokens(&self) -> &[u32] {
        &self.target_tokens
    }

    pub fn completion_prompt_tokens(
        &self,
        prompt_tokens: usize,
        block_size: usize,
        sample_tokens: usize,
    ) -> &[u32] {
        let max_prompt_tokens = prompt_tokens.min(block_size.saturating_sub(sample_tokens + 1));
        &self.prompt_tokens[..self.prompt_tokens.len().min(max_prompt_tokens)]
    }
}

pub fn load_dataset_source(runtime: &RuntimeConfig) -> DatasetSource {
    match (&runtime.dataset_path, runtime.include_synthetic_data) {
        (Some(dataset_path), true) => load_hybrid_dataset(dataset_path, runtime),
        (Some(dataset_path), false) => load_feather_dataset(dataset_path, runtime),
        (None, _) => synthetic_dataset_source(StrokeTokenizer::new(), runtime),
    }
}

fn synthetic_dataset_source(tokenizer: StrokeTokenizer, runtime: &RuntimeConfig) -> DatasetSource {
    let split = synthetic_dataset_split(&tokenizer, runtime);
    DatasetSource {
        label: "synthetic-stroke-autocomplete".to_string(),
        tokenizer,
        split,
    }
}

fn load_hybrid_dataset(dataset_path: &Path, runtime: &RuntimeConfig) -> DatasetSource {
    let real_source = load_feather_dataset(dataset_path, runtime);
    let real_label = real_source.label.clone();
    let real_train_examples = real_source.split.train.num_docs();
    let real_validation_examples = real_source.split.validation.num_docs();
    let real_test_examples = real_source.split.test.num_docs();
    let tokenizer = StrokeTokenizer::with_max_count(0, real_source.tokenizer.max_count());
    assert!(
        tokenizer.same_vocabulary(&real_source.tokenizer),
        "hybrid dataset requires synthetic and real vocabularies to match"
    );

    let synthetic_split = synthetic_dataset_split_with_sizes(
        &tokenizer,
        runtime.seed,
        real_train_examples,
        real_validation_examples,
        real_test_examples,
    );
    let total_train_examples = real_train_examples + synthetic_split.train.num_docs();
    let total_validation_examples =
        real_validation_examples + synthetic_split.validation.num_docs();
    let total_test_examples = real_test_examples + synthetic_split.test.num_docs();
    let DatasetSplit {
        train,
        validation,
        test,
    } = real_source.split;

    DatasetSource {
        label: format!(
            "hybrid-stroke-autocomplete real=[{real_label}] synthetic={}/{}/{} total={}/{}/{}",
            synthetic_split.train.num_docs(),
            synthetic_split.validation.num_docs(),
            synthetic_split.test.num_docs(),
            total_train_examples,
            total_validation_examples,
            total_test_examples
        ),
        tokenizer,
        split: DatasetSplit {
            train: merge_source_datasets(train, synthetic_split.train),
            validation: merge_source_datasets(validation, synthetic_split.validation),
            test: merge_source_datasets(test, synthetic_split.test),
        },
    }
}

fn load_feather_dataset(dataset_path: &Path, runtime: &RuntimeConfig) -> DatasetSource {
    let dataset_path = resolve_dataset_path(dataset_path);
    let bytes = fs::read(&dataset_path).unwrap_or_else(|error| {
        panic!(
            "failed to read Feather dataset {}: {error}",
            dataset_path.display()
        )
    });
    let mut icons: Vec<FeatherIconRecord> =
        serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            panic!(
                "failed to parse Feather dataset {}: {error}",
                dataset_path.display()
            )
        });
    icons.retain(|icon| icon.accepted);
    assert!(
        !icons.is_empty(),
        "Feather dataset {} did not contain any accepted icons",
        dataset_path.display()
    );

    icons.sort_by(|left, right| left.name.cmp(&right.name));
    let grid_size = icons[0].grid_size;
    assert!(
        icons.iter().all(|icon| icon.grid_size == grid_size),
        "Feather dataset {} mixes grid sizes",
        dataset_path.display()
    );

    let tokenizer = StrokeTokenizer::with_grid(grid_size);
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();

    for icon in icons {
        let split = feather_split_for_name(&icon.name);
        let files = feather_source_files(&tokenizer, &icon, split);
        match split {
            FeatherSplit::Train => train.extend(files),
            FeatherSplit::Validation => validation.extend(files),
            FeatherSplit::Test => test.extend(files),
        }
    }

    train.sort_by(|left, right| left.path.cmp(&right.path));
    validation.sort_by(|left, right| left.path.cmp(&right.path));
    test.sort_by(|left, right| left.path.cmp(&right.path));

    truncate_if_limited(&mut train, runtime.train_examples);
    truncate_if_limited(&mut validation, runtime.validation_examples);
    truncate_if_limited(&mut test, runtime.test_examples);

    DatasetSource {
        label: format!(
            "feather-icon-autocomplete grid={} source={} subset={}/{}/{}",
            grid_size,
            dataset_path.display(),
            train.len(),
            validation.len(),
            test.len()
        ),
        tokenizer,
        split: DatasetSplit {
            train: SourceDataset { files: train },
            validation: SourceDataset { files: validation },
            test: SourceDataset { files: test },
        },
    }
}

fn truncate_if_limited<T>(items: &mut Vec<T>, limit: usize) {
    if limit > 0 {
        items.truncate(limit.min(items.len()));
    }
}

fn feather_source_files(
    tokenizer: &StrokeTokenizer,
    icon: &FeatherIconRecord,
    split: FeatherSplit,
) -> Vec<SourceFile> {
    let base_strokes = feather_strokes(icon);
    let augmentations = split.augmentations();
    let mut seen = HashSet::new();
    let mut files = Vec::new();

    let permutations = match split {
        FeatherSplit::Train => stroke_order_permutations(&base_strokes, &icon.name),
        FeatherSplit::Validation | FeatherSplit::Test => vec![(0, base_strokes.clone())],
    };

    for (perm_index, permuted_strokes) in &permutations {
        for &augmentation in augmentations {
            let strokes =
                augment_feather_strokes(permuted_strokes, icon.grid_size, augmentation);
            let content_tokens = feather_content_tokens(tokenizer, &strokes, &icon.name);
            if !seen.insert(content_tokens.clone()) {
                continue;
            }

            let mut tokens = Vec::with_capacity(content_tokens.len() + 2);
            tokens.push(tokenizer.bos_token());
            tokens.extend_from_slice(&content_tokens);
            tokens.push(tokenizer.eot_token());

            let aug_suffix = augmentation.suffix();
            let perm_suffix = if *perm_index == 0 {
                String::new()
            } else {
                format!("perm{perm_index}")
            };

            let path = match (perm_suffix.is_empty(), aug_suffix.is_empty()) {
                (true, true) => format!("{}/{}.stroke", split.as_str(), icon.name),
                (false, true) => {
                    format!("{}/{}__{}.stroke", split.as_str(), icon.name, perm_suffix)
                }
                (true, false) => {
                    format!("{}/{}__{}.stroke", split.as_str(), icon.name, aug_suffix)
                }
                (false, false) => {
                    format!(
                        "{}/{}__{}__{}.stroke",
                        split.as_str(),
                        icon.name,
                        perm_suffix,
                        aug_suffix
                    )
                }
            };

            files.push(SourceFile {
                path,
                tokens,
                prompt_tokens: content_tokens.clone(),
                target_tokens: content_tokens,
            });
        }
    }

    files
}

fn feather_content_tokens(
    tokenizer: &StrokeTokenizer,
    strokes: &[FeatherStroke],
    icon_name: &str,
) -> Vec<u32> {
    let mut content_tokens = Vec::new();
    let mut cursor = (0i32, 0i32);

    for stroke in strokes {
        let points = &stroke.points;
        if points.is_empty() {
            continue;
        }
        let start = points[0];

        if cursor != start {
            append_segments(
                &mut content_tokens,
                tokenizer,
                cursor,
                start,
                icon_name,
                ActionMode::Travel,
            );
            cursor = start;
        }

        let dirs = feather_stroke_dirs(points, icon_name);
        for (direction, count) in compress_direction_runs(&dirs) {
            append_segment(
                &mut content_tokens,
                tokenizer,
                ActionMode::Draw,
                direction,
                count,
                icon_name,
            );
            cursor = apply_run(cursor, direction, count);
        }
    }

    content_tokens
}

fn feather_strokes(icon: &FeatherIconRecord) -> Vec<FeatherStroke> {
    icon.quantized_points
        .iter()
        .filter_map(|points| {
            let points = points.iter().map(|[x, y]| (*x, *y)).collect::<Vec<_>>();
            (points.len() >= 2).then_some(FeatherStroke { points })
        })
        .collect()
}

fn stroke_order_permutations(
    strokes: &[FeatherStroke],
    icon_name: &str,
) -> Vec<(usize, Vec<FeatherStroke>)> {
    let n = strokes.len();
    if n == 0 {
        return vec![(0, strokes.to_vec())];
    }
    if n == 1 {
        // Single stroke: identity + reversed = 2 variants.
        let mut reversed = strokes[0].clone();
        reversed.points.reverse();
        return vec![(0, strokes.to_vec()), (1, vec![reversed])];
    }

    // Total combinatorial space: N! orderings × 2^N reversal masks.
    let factorial = (1..=n)
        .try_fold(1usize, |acc, x| acc.checked_mul(x))
        .unwrap_or(usize::MAX);
    let total = factorial.saturating_mul(1usize << n.min(30));

    if total <= MAX_STROKE_PERMUTATIONS {
        all_order_and_reversal_variants(strokes)
    } else {
        sampled_order_and_reversal_variants(strokes, icon_name)
    }
}

fn apply_reversal_mask(strokes: &[FeatherStroke], mask: usize) -> Vec<FeatherStroke> {
    strokes
        .iter()
        .enumerate()
        .map(|(i, stroke)| {
            if mask & (1 << i) != 0 {
                let mut reversed = stroke.clone();
                reversed.points.reverse();
                reversed
            } else {
                stroke.clone()
            }
        })
        .collect()
}

fn all_order_and_reversal_variants(
    strokes: &[FeatherStroke],
) -> Vec<(usize, Vec<FeatherStroke>)> {
    let n = strokes.len();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut result = Vec::new();
    let mut variant_index = 0usize;
    let reversal_count = 1usize << n;

    loop {
        let permuted: Vec<FeatherStroke> = indices.iter().map(|&i| strokes[i].clone()).collect();
        for mask in 0..reversal_count {
            result.push((variant_index, apply_reversal_mask(&permuted, mask)));
            variant_index += 1;
        }

        if !next_permutation(&mut indices) {
            break;
        }
    }

    result
}

fn next_permutation(indices: &mut [usize]) -> bool {
    let n = indices.len();
    if n < 2 {
        return false;
    }
    let Some(i) = (0..n - 1).rev().find(|&i| indices[i] < indices[i + 1]) else {
        return false;
    };
    let j = (i + 1..n)
        .rev()
        .find(|&j| indices[i] < indices[j])
        .unwrap();
    indices.swap(i, j);
    indices[i + 1..].reverse();
    true
}

fn sampled_order_and_reversal_variants(
    strokes: &[FeatherStroke],
    icon_name: &str,
) -> Vec<(usize, Vec<FeatherStroke>)> {
    let seed = stable_name_hash(icon_name);
    let mut rng = StdRng::seed_from_u64(seed);

    let n = strokes.len();
    let reversal_count = 1usize << n.min(30);
    let mut result = Vec::with_capacity(MAX_STROKE_PERMUTATIONS);
    let mut seen: HashSet<(Vec<usize>, usize)> = HashSet::new();

    // Always include the identity (original order, no reversals).
    let identity: Vec<usize> = (0..n).collect();
    seen.insert((identity, 0));
    result.push((0, strokes.to_vec()));

    let mut variant_index = 1usize;
    let max_attempts = MAX_STROKE_PERMUTATIONS * 10;
    let mut attempts = 0;

    while result.len() < MAX_STROKE_PERMUTATIONS && attempts < max_attempts {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let mask = rng.random_range(0..reversal_count);
        attempts += 1;

        if seen.insert((indices.clone(), mask)) {
            let permuted: Vec<FeatherStroke> =
                indices.iter().map(|&i| strokes[i].clone()).collect();
            result.push((variant_index, apply_reversal_mask(&permuted, mask)));
            variant_index += 1;
        }
    }

    result
}

fn feather_stroke_dirs(points: &[(i32, i32)], icon_name: &str) -> Vec<Direction> {
    let mut dirs = Vec::new();
    for segment in points.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        if dx == 0 && dy == 0 {
            continue;
        }
        if dx.abs() > 1 || dy.abs() > 1 {
            panic!("icon {} contains non-unit quantized edges", icon_name);
        }
        dirs.push(direction_from_signs(dx, dy));
    }
    dirs
}

fn augment_feather_strokes(
    strokes: &[FeatherStroke],
    grid_size: usize,
    augmentation: FeatherAugmentation,
) -> Vec<FeatherStroke> {
    strokes
        .iter()
        .cloned()
        .map(|stroke| FeatherStroke {
            points: stroke
                .points
                .into_iter()
                .map(|point| transform_point(point, grid_size, augmentation))
                .collect(),
        })
        .collect()
}

fn transform_point(
    point: (i32, i32),
    grid_size: usize,
    augmentation: FeatherAugmentation,
) -> (i32, i32) {
    let max = grid_size.saturating_sub(1) as i32;
    let (x, y) = point;
    match augmentation {
        FeatherAugmentation::Identity => (x, y),
        FeatherAugmentation::Rotate90 => (max - y, x),
        FeatherAugmentation::Rotate180 => (max - x, max - y),
        FeatherAugmentation::Rotate270 => (y, max - x),
        FeatherAugmentation::MirrorX => (max - x, y),
        FeatherAugmentation::MirrorY => (x, max - y),
        FeatherAugmentation::MirrorMainDiagonal => (y, x),
        FeatherAugmentation::MirrorAntiDiagonal => (max - y, max - x),
    }
}

fn append_segments(
    tokens: &mut Vec<u32>,
    tokenizer: &StrokeTokenizer,
    from: (i32, i32),
    to: (i32, i32),
    icon_name: &str,
    mode: ActionMode,
) {
    let mut cursor = from;
    while cursor != to {
        let dx = to.0 - cursor.0;
        let dy = to.1 - cursor.1;
        let (direction, count) = if dx != 0 && dy != 0 {
            (
                direction_from_signs(dx.signum(), dy.signum()),
                dx.unsigned_abs().min(dy.unsigned_abs()) as usize,
            )
        } else if dx != 0 {
            (
                direction_from_signs(dx.signum(), 0),
                dx.unsigned_abs() as usize,
            )
        } else {
            (
                direction_from_signs(0, dy.signum()),
                dy.unsigned_abs() as usize,
            )
        };
        append_segment(tokens, tokenizer, mode, direction, count, icon_name);
        cursor = apply_run(cursor, direction, count);
    }
}

fn append_segment(
    tokens: &mut Vec<u32>,
    tokenizer: &StrokeTokenizer,
    mode: ActionMode,
    direction: Direction,
    count: usize,
    icon_name: &str,
) {
    assert!(
        (1..=tokenizer.max_count()).contains(&count),
        "icon {} requires count {} but tokenizer max count is {}",
        icon_name,
        count,
        tokenizer.max_count()
    );
    let token = match mode {
        ActionMode::Travel => tokenizer.travel_token(direction, count),
        ActionMode::Draw => tokenizer.draw_token(direction, count),
    };
    if tokenizer.decode_segment_token(token).is_none() {
        panic!(
            "icon {} produced an invalid segment token for {:?} {}",
            icon_name, direction, count
        );
    }
    tokens.push(token);
}

fn direction_from_signs(dx: i32, dy: i32) -> Direction {
    match (dx, dy) {
        (0, -1) => Direction::N,
        (1, -1) => Direction::NE,
        (1, 0) => Direction::E,
        (1, 1) => Direction::SE,
        (0, 1) => Direction::S,
        (-1, 1) => Direction::SW,
        (-1, 0) => Direction::W,
        (-1, -1) => Direction::NW,
        _ => panic!("invalid direction delta ({dx}, {dy})"),
    }
}

fn compress_direction_runs(dirs: &[Direction]) -> Vec<(Direction, usize)> {
    let mut runs = Vec::new();
    for &direction in dirs {
        if let Some((last_direction, count)) = runs.last_mut() {
            if *last_direction == direction {
                *count += 1;
                continue;
            }
        }
        runs.push((direction, 1));
    }
    runs
}

fn apply_run(cursor: (i32, i32), direction: Direction, count: usize) -> (i32, i32) {
    let (dx, dy) = direction.delta();
    (cursor.0 + dx * count as i32, cursor.1 + dy * count as i32)
}

fn resolve_dataset_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
    }
}

fn feather_split_for_name(name: &str) -> FeatherSplit {
    let bucket = (stable_name_hash(name) % 10) as usize;
    match bucket {
        0 => FeatherSplit::Validation,
        1 => FeatherSplit::Test,
        _ => FeatherSplit::Train,
    }
}

fn stable_name_hash(name: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in name.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

impl FeatherSplit {
    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "valid",
            Self::Test => "test",
        }
    }

    fn augmentations(self) -> &'static [FeatherAugmentation] {
        match self {
            Self::Train => &[
                FeatherAugmentation::Identity,
                FeatherAugmentation::Rotate90,
                FeatherAugmentation::Rotate180,
                FeatherAugmentation::Rotate270,
                FeatherAugmentation::MirrorX,
                FeatherAugmentation::MirrorY,
                FeatherAugmentation::MirrorMainDiagonal,
                FeatherAugmentation::MirrorAntiDiagonal,
            ],
            Self::Validation | Self::Test => &[FeatherAugmentation::Identity],
        }
    }
}

pub fn synthetic_dataset_split(
    tokenizer: &StrokeTokenizer,
    config: &RuntimeConfig,
) -> DatasetSplit {
    synthetic_dataset_split_with_sizes(
        tokenizer,
        config.seed,
        config.train_examples,
        config.validation_examples,
        config.test_examples,
    )
}

fn synthetic_dataset_split_with_sizes(
    tokenizer: &StrokeTokenizer,
    seed: u64,
    train_examples: usize,
    validation_examples: usize,
    test_examples: usize,
) -> DatasetSplit {
    DatasetSplit {
        train: generate_split(tokenizer, train_examples, seed ^ 0xA11CE, "train"),
        validation: generate_split(tokenizer, validation_examples, seed ^ 0xBEEFu64, "valid"),
        test: generate_split(tokenizer, test_examples, seed ^ 0xC0DEu64, "test"),
    }
}

fn merge_source_datasets(mut primary: SourceDataset, secondary: SourceDataset) -> SourceDataset {
    let SourceDataset {
        files: secondary_files,
    } = secondary;
    primary.files.extend(secondary_files);
    primary
        .files
        .sort_by(|left, right| left.path.cmp(&right.path));
    primary
}

pub fn write_tokens_to_svg_file(
    tokenizer: &StrokeTokenizer,
    prompt_tokens: &[u32],
    continuation_tokens: &[u32],
    path: &Path,
) {
    let svg = tokens_to_svg_string(tokenizer, prompt_tokens, continuation_tokens);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!(
                "failed to create SVG output directory {}: {error}",
                parent.display()
            )
        });
    }
    fs::write(path, svg)
        .unwrap_or_else(|error| panic!("failed to write SVG sample {}: {error}", path.display()));
}

pub fn tokens_to_svg_string(
    tokenizer: &StrokeTokenizer,
    prompt_tokens: &[u32],
    continuation_tokens: &[u32],
) -> String {
    let scene = tokens_to_stroke_scene(tokenizer, prompt_tokens, continuation_tokens);
    svg_document(&scene.prompt_strokes, &scene.continuation_strokes)
}

pub fn tokens_to_stroke_scene(
    tokenizer: &StrokeTokenizer,
    prompt_tokens: &[u32],
    continuation_tokens: &[u32],
) -> StrokeScene {
    let mut prompt_state = RenderState::default();
    render_tokens_into_state(tokenizer, prompt_tokens, &mut prompt_state);
    prompt_state.finish_current_stroke();
    let prompt_strokes = prompt_state.completed_strokes.clone();

    let mut continuation_state = RenderState {
        cursor: prompt_state.cursor,
        pen_down: prompt_state.pen_down,
        current_stroke: prompt_state.current_stroke,
        completed_strokes: Vec::new(),
    };
    render_tokens_into_state(tokenizer, continuation_tokens, &mut continuation_state);
    continuation_state.finish_current_stroke();

    StrokeScene {
        prompt_strokes: prompt_strokes
            .into_iter()
            .map(|points| StrokePath { points })
            .collect(),
        continuation_strokes: continuation_state
            .completed_strokes
            .into_iter()
            .map(|points| StrokePath { points })
            .collect(),
    }
}

pub fn windows_to_token_inputs(windows: &[Vec<u32>]) -> Vec<Vec<u32>> {
    windows
        .iter()
        .map(|window| window[..window.len().saturating_sub(1)].to_vec())
        .collect()
}

pub fn windows_to_token_targets(windows: &[Vec<u32>]) -> Vec<Vec<u32>> {
    windows.iter().map(|window| window[1..].to_vec()).collect()
}

pub fn autoregressive_context(
    tokens: &[u32],
    _pad_token: u32,
    block_size: usize,
) -> (Vec<u32>, usize) {
    let context = if tokens.len() > block_size {
        tokens[tokens.len() - block_size..].to_vec()
    } else {
        tokens.to_vec()
    };
    let last_index = context.len().saturating_sub(1);
    (context, last_index)
}

pub fn position_indexes(batch_size: usize, block_size: usize) -> Vec<Vec<u32>> {
    (0..batch_size)
        .map(|_| (0..block_size).map(|position| position as u32).collect())
        .collect()
}

pub fn canvas_state_spec(tokenizer: &StrokeTokenizer, block_size: usize) -> CanvasStateSpec {
    if tokenizer.grid_size() > 0 {
        return CanvasStateSpec {
            coordinate_vocab_size: tokenizer.grid_size(),
            coordinate_offset: 0,
        };
    }

    let max_abs_coordinate = tokenizer.max_count().saturating_mul(block_size) as i32;
    CanvasStateSpec {
        coordinate_vocab_size: max_abs_coordinate as usize * 2 + 1,
        coordinate_offset: max_abs_coordinate,
    }
}

pub fn canvas_state_indexes(
    tokenizer: &StrokeTokenizer,
    token_windows: &[Vec<u32>],
    spec: CanvasStateSpec,
) -> CanvasStateIndexes {
    let mut cursor_x = Vec::with_capacity(token_windows.len());
    let mut cursor_y = Vec::with_capacity(token_windows.len());
    let mut pen_state = Vec::with_capacity(token_windows.len());

    for tokens in token_windows {
        let mut state = CanvasCursorState::default();
        let mut x_row = Vec::with_capacity(tokens.len());
        let mut y_row = Vec::with_capacity(tokens.len());
        let mut pen_row = Vec::with_capacity(tokens.len());

        for &token in tokens {
            apply_token_to_canvas_state(tokenizer, token, &mut state);
            x_row.push(encode_canvas_coordinate(state.cursor.0, spec));
            y_row.push(encode_canvas_coordinate(state.cursor.1, spec));
            pen_row.push(u32::from(state.pen_down));
        }

        cursor_x.push(x_row);
        cursor_y.push(y_row);
        pen_state.push(pen_row);
    }

    CanvasStateIndexes {
        cursor_x,
        cursor_y,
        pen_state,
    }
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            cursor: (0, 0),
            pen_down: false,
            current_stroke: Vec::new(),
            completed_strokes: Vec::new(),
        }
    }
}

impl CanvasStateIndexes {
    pub fn zeros(batch_size: usize, seq_len: usize) -> Self {
        Self {
            cursor_x: vec![vec![0; seq_len]; batch_size],
            cursor_y: vec![vec![0; seq_len]; batch_size],
            pen_state: vec![vec![0; seq_len]; batch_size],
        }
    }
}

impl Default for CanvasCursorState {
    fn default() -> Self {
        Self {
            cursor: (0, 0),
            pen_down: false,
        }
    }
}

impl RenderState {
    fn finish_current_stroke(&mut self) {
        if self.current_stroke.len() > 1 {
            self.completed_strokes
                .push(std::mem::take(&mut self.current_stroke));
        } else {
            self.current_stroke.clear();
        }
    }
}

fn render_tokens_into_state(tokenizer: &StrokeTokenizer, tokens: &[u32], state: &mut RenderState) {
    for token in tokens {
        match tokenizer.decode_token_kind(*token) {
            TokenKind::Bos | TokenKind::Eot => {}
            TokenKind::Segment(ActionMode::Travel, direction, count) => {
                state.finish_current_stroke();
                state.pen_down = false;
                state.cursor = apply_run(state.cursor, direction, count);
            }
            TokenKind::Segment(ActionMode::Draw, direction, count) => {
                if state.pen_down && state.current_stroke.is_empty() {
                    state.current_stroke.push(state.cursor);
                } else if !state.pen_down {
                    state.current_stroke.push(state.cursor);
                    state.pen_down = true;
                }
                let (dx, dy) = direction.delta();
                for _ in 0..count {
                    state.cursor.0 += dx;
                    state.cursor.1 += dy;
                    if state.pen_down {
                        state.current_stroke.push(state.cursor);
                    }
                }
            }
        }
    }
}

fn apply_token_to_canvas_state(
    tokenizer: &StrokeTokenizer,
    token: u32,
    state: &mut CanvasCursorState,
) {
    match tokenizer.decode_token_kind(token) {
        TokenKind::Bos | TokenKind::Eot => {}
        TokenKind::Segment(ActionMode::Travel, direction, count) => {
            state.cursor = apply_run(state.cursor, direction, count);
            state.pen_down = false;
        }
        TokenKind::Segment(ActionMode::Draw, direction, count) => {
            state.cursor = apply_run(state.cursor, direction, count);
            state.pen_down = true;
        }
    }
}

fn encode_canvas_coordinate(value: i32, spec: CanvasStateSpec) -> u32 {
    let max_index = spec.coordinate_vocab_size.saturating_sub(1) as i32;
    value
        .saturating_add(spec.coordinate_offset)
        .clamp(0, max_index) as u32
}

fn svg_document(prompt_strokes: &[StrokePath], continuation_strokes: &[StrokePath]) -> String {
    let all_points = prompt_strokes
        .iter()
        .chain(continuation_strokes.iter())
        .flat_map(|stroke| stroke.points.iter().copied())
        .collect::<Vec<_>>();

    let (offset_x, offset_y, scale) = if all_points.is_empty() {
        (CANVAS_SIZE / 2.0, CANVAS_SIZE / 2.0, 1.0)
    } else {
        let min_x = all_points.iter().map(|(x, _)| *x).min().unwrap() as f32;
        let max_x = all_points.iter().map(|(x, _)| *x).max().unwrap() as f32;
        let min_y = all_points.iter().map(|(_, y)| *y).min().unwrap() as f32;
        let max_y = all_points.iter().map(|(_, y)| *y).max().unwrap() as f32;
        let width = (max_x - min_x).max(1.0);
        let height = (max_y - min_y).max(1.0);
        let scale = ((CANVAS_SIZE - CANVAS_PADDING * 2.0) / width)
            .min((CANVAS_SIZE - CANVAS_PADDING * 2.0) / height);
        let scaled_width = width * scale;
        let scaled_height = height * scale;
        let offset_x = CANVAS_PADDING + (CANVAS_SIZE - CANVAS_PADDING * 2.0 - scaled_width) / 2.0;
        let offset_y = CANVAS_PADDING + (CANVAS_SIZE - CANVAS_PADDING * 2.0 - scaled_height) / 2.0;
        (offset_x - min_x * scale, offset_y - min_y * scale, scale)
    };

    let prompt_paths = prompt_strokes
        .iter()
        .filter_map(|stroke| polyline_path(&stroke.points, offset_x, offset_y, scale))
        .map(|path| {
            format!(
                "<path d=\"{path}\" fill=\"none\" stroke=\"#264653\" stroke-width=\"5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" opacity=\"0.65\"/>"
            )
        })
        .collect::<Vec<_>>()
        .join("");
    let continuation_paths = continuation_strokes
        .iter()
        .filter_map(|stroke| polyline_path(&stroke.points, offset_x, offset_y, scale))
        .map(|path| {
            format!(
                "<path d=\"{path}\" fill=\"none\" stroke=\"#e76f51\" stroke-width=\"5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>"
            )
        })
        .collect::<Vec<_>>()
        .join("");

    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {CANVAS_SIZE} {CANVAS_SIZE}\"><rect width=\"{CANVAS_SIZE}\" height=\"{CANVAS_SIZE}\" rx=\"18\" fill=\"#fdf6e3\"/><rect x=\"6\" y=\"6\" width=\"{inner}\" height=\"{inner}\" rx=\"14\" fill=\"#fffaf0\" stroke=\"#e9dcc3\" stroke-width=\"2\"/>{prompt_paths}{continuation_paths}</svg>",
        inner = CANVAS_SIZE - 12.0,
    )
}

fn polyline_path(
    stroke: &[(i32, i32)],
    offset_x: f32,
    offset_y: f32,
    scale: f32,
) -> Option<String> {
    let mut points = stroke.iter();
    let &(first_x, first_y) = points.next()?;
    let mut path = format!(
        "M {:.2} {:.2}",
        offset_x + first_x as f32 * scale,
        offset_y + first_y as f32 * scale
    );
    for &(x, y) in points {
        path.push_str(&format!(
            " L {:.2} {:.2}",
            offset_x + x as f32 * scale,
            offset_y + y as f32 * scale
        ));
    }
    Some(path)
}

fn generate_split(
    tokenizer: &StrokeTokenizer,
    example_count: usize,
    seed: u64,
    split_name: &str,
) -> SourceDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut shapes = (0..example_count)
        .map(|index| ShapeKind::all()[index % ShapeKind::all().len()])
        .collect::<Vec<_>>();
    shapes.shuffle(&mut rng);
    let mut used_sequences = HashSet::with_capacity(example_count);

    let files = shapes
        .into_iter()
        .enumerate()
        .map(|(index, shape)| {
            generate_source_file(
                tokenizer,
                &mut rng,
                split_name,
                index,
                shape,
                &mut used_sequences,
            )
        })
        .collect();
    SourceDataset { files }
}

fn generate_source_file(
    tokenizer: &StrokeTokenizer,
    rng: &mut StdRng,
    split_name: &str,
    index: usize,
    shape: ShapeKind,
    used_sequences: &mut HashSet<Vec<u32>>,
) -> SourceFile {
    let mut content_tokens = shape_tokens(shape, tokenizer, rng);
    for _ in 0..255 {
        if used_sequences.insert(content_tokens.clone()) {
            break;
        }
        content_tokens = shape_tokens(shape, tokenizer, rng);
    }
    let mut tokens = Vec::with_capacity(content_tokens.len() + 2);
    tokens.push(tokenizer.bos_token());
    tokens.extend_from_slice(&content_tokens);
    tokens.push(tokenizer.eot_token());

    SourceFile {
        path: format!("{split_name}/{}-{index:03}.stroke", shape.name()),
        tokens,
        prompt_tokens: content_tokens.clone(),
        target_tokens: content_tokens,
    }
}

fn shape_tokens(shape: ShapeKind, tokenizer: &StrokeTokenizer, rng: &mut StdRng) -> Vec<u32> {
    let mut drawing_tokens = Vec::new();
    match shape {
        ShapeKind::Circle => {
            let straight = rng.random_range(1..=2);
            let corner = rng.random_range(1..=2);
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, straight),
                    (Direction::SE, corner),
                    (Direction::S, straight),
                    (Direction::SW, corner),
                    (Direction::W, straight),
                    (Direction::NW, corner),
                    (Direction::N, straight),
                    (Direction::NE, corner),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Square => {
            let side = rng.random_range(1..=2);
            let rotation = rng.random_range(0..Direction::all().len());
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, side),
                    (Direction::S, side),
                    (Direction::W, side),
                    (Direction::N, side),
                ],
                rotation,
                false,
            );
        }
        ShapeKind::Rectangle => {
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 2),
                    (Direction::S, 1),
                    (Direction::W, 2),
                    (Direction::N, 1),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Triangle => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[(Direction::E, 2), (Direction::NW, 1), (Direction::SW, 1)],
                rotation,
                mirror,
            );
        }
        ShapeKind::RightTriangle => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[(Direction::E, 2), (Direction::SW, 2), (Direction::N, 2)],
                rotation,
                mirror,
            );
        }
        ShapeKind::Diamond => {
            let edge = rng.random_range(1..=2);
            let rotation = rng.random_range(0..Direction::all().len());
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::NE, edge),
                    (Direction::SE, edge),
                    (Direction::SW, edge),
                    (Direction::NW, edge),
                ],
                rotation,
                false,
            );
        }
        ShapeKind::Hexagon => {
            let edge = rng.random_range(1..=2);
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, edge),
                    (Direction::SE, edge),
                    (Direction::SW, edge),
                    (Direction::W, edge),
                    (Direction::NW, edge),
                    (Direction::NE, edge),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Octagon => {
            let straight = rng.random_range(1..=2);
            let corner = rng.random_range(1..=2);
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, straight),
                    (Direction::SE, corner),
                    (Direction::S, straight),
                    (Direction::SW, corner),
                    (Direction::W, straight),
                    (Direction::NW, corner),
                    (Direction::N, straight),
                    (Direction::NE, corner),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::House => {
            // roof=1, base=2 (=roof*2), wall 1..=2. All segments ≤ 2.
            let wall = rng.random_range(1..=2);
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 2),
                    (Direction::N, wall),
                    (Direction::NW, 1),
                    (Direction::SW, 1),
                    (Direction::S, wall),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Trapezoid => {
            // Right trapezoid: bottom=2, top=1, one sloped side, one vertical.
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 2),
                    (Direction::NW, 1),
                    (Direction::W, 1),
                    (Direction::S, 1),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Parallelogram => {
            let width = rng.random_range(1..=2);
            let lean = rng.random_range(1..=2);
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, width),
                    (Direction::NE, lean),
                    (Direction::W, width),
                    (Direction::SW, lean),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Elbow => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 2),
                    (Direction::S, 1),
                    (Direction::W, 1),
                    (Direction::S, 1),
                    (Direction::W, 1),
                    (Direction::N, 2),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Shield => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 2),
                    (Direction::SE, 1),
                    (Direction::SW, 1),
                    (Direction::W, 2),
                    (Direction::NW, 1),
                    (Direction::NE, 1),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Spiral => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 2),
                    (Direction::S, 2),
                    (Direction::W, 1),
                    (Direction::N, 1),
                    (Direction::W, 2),
                    (Direction::N, 2),
                    (Direction::E, 1),
                    (Direction::S, 1),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Kite => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 1),
                    (Direction::SE, 1),
                    (Direction::SW, 1),
                    (Direction::W, 1),
                    (Direction::NW, 1),
                    (Direction::NE, 1),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Tag => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 2),
                    (Direction::SE, 1),
                    (Direction::SW, 1),
                    (Direction::W, 1),
                    (Direction::NW, 1),
                    (Direction::N, 1),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Notch => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 2),
                    (Direction::S, 2),
                    (Direction::W, 1),
                    (Direction::N, 1),
                    (Direction::W, 1),
                    (Direction::N, 1),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Bookmark => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 2),
                    (Direction::S, 2),
                    (Direction::NW, 1),
                    (Direction::SW, 1),
                    (Direction::N, 2),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Stair => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 1),
                    (Direction::S, 1),
                    (Direction::E, 1),
                    (Direction::S, 1),
                    (Direction::W, 2),
                    (Direction::N, 2),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Capsule => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::NE, 1),
                    (Direction::E, 2),
                    (Direction::SE, 1),
                    (Direction::SW, 1),
                    (Direction::W, 2),
                    (Direction::NW, 1),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Vase => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::NE, 1),
                    (Direction::N, 1),
                    (Direction::SE, 1),
                    (Direction::S, 2),
                    (Direction::SW, 1),
                    (Direction::NW, 1),
                    (Direction::N, 1),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Bolt => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 1),
                    (Direction::NE, 1),
                    (Direction::SE, 1),
                    (Direction::SW, 1),
                    (Direction::W, 2),
                    (Direction::N, 1),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Tab => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::NE, 1),
                    (Direction::E, 2),
                    (Direction::S, 1),
                    (Direction::SW, 1),
                    (Direction::W, 1),
                    (Direction::NW, 1),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Cup => {
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::S, 1),
                    (Direction::SE, 1),
                    (Direction::E, 1),
                    (Direction::N, 2),
                    (Direction::W, 2),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Cross => {
            // Plus/cross shape with 12 cardinal segments
            let arm = rng.random_range(1..=2);
            let width = 1;
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::N, arm),
                    (Direction::E, width),
                    (Direction::S, arm),
                    (Direction::E, arm),
                    (Direction::S, width),
                    (Direction::W, arm),
                    (Direction::S, arm),
                    (Direction::W, width),
                    (Direction::N, arm),
                    (Direction::W, arm),
                    (Direction::N, width),
                    (Direction::E, arm),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Star => {
            // 4-pointed star: diagonal spikes with cardinal returns.
            // All 8 directions appear with equal count, so any rotation is safe.
            let spike = rng.random_range(1..=2);
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::NE, spike),
                    (Direction::S, spike),
                    (Direction::SE, spike),
                    (Direction::W, spike),
                    (Direction::SW, spike),
                    (Direction::N, spike),
                    (Direction::NW, spike),
                    (Direction::E, spike),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Arrow => {
            // Symmetric arrow: rectangular shaft with centered triangular head.
            // shaft ≤ 2, head=1 so S(head*2)=S(2) ≤ 2.
            let shaft = rng.random_range(1..=2);
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, shaft),
                    (Direction::NE, 1),
                    (Direction::NW, 1),
                    (Direction::W, shaft),
                    (Direction::S, 2),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Pentagon => {
            // Irregular pentagon: edge=1 so edge*2=2. All segments ≤ 2.
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut drawing_tokens,
                tokenizer,
                &[
                    (Direction::E, 2),
                    (Direction::NW, 1),
                    (Direction::N, 1),
                    (Direction::SW, 1),
                    (Direction::S, 1),
                ],
                rotation,
                mirror,
            );
        }
    }
    if shape.is_closed() {
        apply_closed_loop_variant(tokenizer, rng, &mut drawing_tokens);
    }

    let tokens = drawing_tokens;

    debug_assert_eq!(
        token_displacement(tokenizer, &tokens),
        (0, 0),
        "{} does not close back to the origin",
        shape.name()
    );
    debug_assert!(
        max_straight_run(tokenizer, &tokens) <= 2,
        "{} has a straight run of {} (max 2)",
        shape.name(),
        max_straight_run(tokenizer, &tokens)
    );
    debug_assert!(
        !has_overlapping_points(tokenizer, &tokens),
        "{} has overlapping points",
        shape.name()
    );
    debug_assert!(
        !has_crossing_segments(tokenizer, &tokens),
        "{} has crossing line segments",
        shape.name()
    );
    tokens
}

fn token_displacement(tokenizer: &StrokeTokenizer, tokens: &[u32]) -> (i32, i32) {
    token_segments(tokenizer, tokens)
        .into_iter()
        .fold((0, 0), |cursor, (_, direction, count)| {
            apply_run(cursor, direction, count)
        })
}

/// Convert move tokens into the sequence of grid points visited by the path.
fn token_points(tokenizer: &StrokeTokenizer, tokens: &[u32]) -> Vec<(i32, i32)> {
    let mut points = vec![(0i32, 0i32)];
    let mut cursor = (0i32, 0i32);
    for (_, direction, count) in token_segments(tokenizer, tokens) {
        let (dx, dy) = direction.delta();
        for _ in 0..count {
            cursor.0 += dx;
            cursor.1 += dy;
            points.push(cursor);
        }
    }
    points
}

/// Returns the longest consecutive run of the same direction in the token sequence.
fn max_straight_run(tokenizer: &StrokeTokenizer, tokens: &[u32]) -> usize {
    let mut max_run = 0;
    let mut current_run = 0;
    let mut last_direction: Option<Direction> = None;
    for (_, direction, count) in token_segments(tokenizer, tokens) {
        if last_direction == Some(direction) {
            current_run += count;
        } else {
            current_run = count;
            last_direction = Some(direction);
        }
        max_run = max_run.max(current_run);
    }
    max_run
}

/// Returns `true` when the path visits any grid point more than once
/// (the first and last points are allowed to coincide for closed loops).
fn has_overlapping_points(tokenizer: &StrokeTokenizer, tokens: &[u32]) -> bool {
    let points = token_points(tokenizer, tokens);
    let mut seen = std::collections::HashSet::new();
    // Skip the last point — it may equal the first for a closed loop.
    for &point in &points[..points.len().saturating_sub(1)] {
        if !seen.insert(point) {
            return true;
        }
    }
    false
}

/// Returns `true` when any two non-adjacent segments in the path properly cross.
fn has_crossing_segments(tokenizer: &StrokeTokenizer, tokens: &[u32]) -> bool {
    /// Sign of the cross-product (p2 − p1) × (p3 − p1).
    fn cross(p1: (i32, i32), p2: (i32, i32), p3: (i32, i32)) -> i64 {
        (p2.0 - p1.0) as i64 * (p3.1 - p1.1) as i64 - (p2.1 - p1.1) as i64 * (p3.0 - p1.0) as i64
    }

    /// True when the interiors of segments (a1–a2) and (b1–b2) intersect.
    fn segments_properly_cross(
        a1: (i32, i32),
        a2: (i32, i32),
        b1: (i32, i32),
        b2: (i32, i32),
    ) -> bool {
        let d1 = cross(a1, a2, b1);
        let d2 = cross(a1, a2, b2);
        let d3 = cross(b1, b2, a1);
        let d4 = cross(b1, b2, a2);
        ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
    }

    let points = token_points(tokenizer, tokens);
    let n = points.len();
    if n < 4 {
        return false;
    }
    for i in 0..n - 1 {
        for j in i + 2..n - 1 {
            // First and last segments share the start/end point of closed loops.
            if i == 0 && j == n - 2 {
                continue;
            }
            if segments_properly_cross(points[i], points[i + 1], points[j], points[j + 1]) {
                return true;
            }
        }
    }
    false
}

fn transformed_direction(
    direction: Direction,
    rotation: usize,
    mirror_horizontal: bool,
) -> Direction {
    let direction = if mirror_horizontal {
        direction.mirror_horizontal()
    } else {
        direction
    };
    direction.rotate(rotation)
}

fn repeat_segment(
    tokens: &mut Vec<u32>,
    tokenizer: &StrokeTokenizer,
    mode: ActionMode,
    direction: Direction,
    count: usize,
) {
    let token = match mode {
        ActionMode::Travel => tokenizer.travel_token(direction, count),
        ActionMode::Draw => tokenizer.draw_token(direction, count),
    };
    tokens.push(token);
}

fn trace_moves(
    tokens: &mut Vec<u32>,
    tokenizer: &StrokeTokenizer,
    segments: &[(Direction, usize)],
    rotation: usize,
    mirror_horizontal: bool,
) {
    for &(direction, count) in segments {
        repeat_segment(
            tokens,
            tokenizer,
            ActionMode::Draw,
            transformed_direction(direction, rotation, mirror_horizontal),
            count,
        );
    }
}

fn apply_closed_loop_variant(tokenizer: &StrokeTokenizer, rng: &mut StdRng, tokens: &mut Vec<u32>) {
    if tokens.is_empty() {
        return;
    }

    let mut segments = token_segments(tokenizer, tokens);
    let start_offset = rng.random_range(0..segments.len());
    segments.rotate_left(start_offset);

    if rng.random_bool(0.5) {
        segments = segments
            .iter()
            .rev()
            .map(|&(_, direction, count)| (ActionMode::Draw, direction.opposite(), count))
            .collect();
    }

    let mut rebuilt = Vec::with_capacity(tokens.len());
    for (_, direction, count) in segments {
        repeat_segment(&mut rebuilt, tokenizer, ActionMode::Draw, direction, count);
    }
    *tokens = rebuilt;
}

fn token_segments(
    tokenizer: &StrokeTokenizer,
    tokens: &[u32],
) -> Vec<(ActionMode, Direction, usize)> {
    let mut segments = Vec::new();
    for &token in tokens {
        match tokenizer.decode_token_kind(token) {
            TokenKind::Segment(mode, direction, count) => {
                segments.push((mode, direction, count));
            }
            TokenKind::Bos | TokenKind::Eot => {}
        }
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RuntimeConfig, SaveQuantization};
    use fusor_gguf::{GgufMetadata, GgufVersion};
    use std::{collections::HashSet, io::Cursor};

    fn test_runtime(block_size: usize, batch_size: usize, eval_batches: usize) -> RuntimeConfig {
        RuntimeConfig {
            epochs: 1,
            warmup_steps: 1,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.1,
            log_every: 10,
            eval_batches,
            sample_tokens: 8,
            sample_prefix_tokens: 4,
            sample_temperature: 0.7,
            sample_top_k: 4,
            block_size,
            batch_size,
            n_embd: 32,
            n_head: 2,
            n_kv_head: 2,
            n_ff: 64,
            n_layer: 2,
            conv_kernel_size: 3,
            attention_period: 1,
            use_rope: false,
            rope_theta: 10_000.0,
            use_canvas_state_embeddings: true,
            use_extra_norms: false,
            eps: 1e-5,
            init_scale: 0.08,
            seed: 42,
            save_every_steps: 0,
            save_final_model: false,
            save_quantization: SaveQuantization::F32,
            train_examples: 0,
            validation_examples: 0,
            test_examples: 0,
            dataset_path: None,
            include_synthetic_data: false,
            gguf_path: "test.gguf".into(),
            sample_output_path: "sample.svg".into(),
        }
    }

    fn test_file(path: &str, len: usize) -> SourceFile {
        let tokens = (0..len as u32).collect::<Vec<_>>();
        SourceFile {
            path: path.to_string(),
            tokens: tokens.clone(),
            prompt_tokens: tokens.clone(),
            target_tokens: tokens,
        }
    }

    fn valid_len(mask: &[f32]) -> usize {
        mask.iter().filter(|&&value| value > 0.0).count()
    }

    #[test]
    fn tokenizer_survives_gguf_metadata_round_trip() {
        let tokenizer = StrokeTokenizer::new();
        let gguf = GgufMetadata {
            version: GgufVersion::V3,
            metadata: tokenizer
                .gguf_metadata()
                .into_iter()
                .map(|(key, value)| (key.into_boxed_str(), value))
                .collect(),
            tensor_infos: Default::default(),
            tensor_data_offset: 0,
        };
        let mut bytes = Cursor::new(Vec::new());
        gguf.write(&mut bytes, std::iter::empty::<(&str, &[u8])>())
            .unwrap();
        let mut reader = Cursor::new(bytes.into_inner());
        let vb = VarBuilder::from_gguf(&mut reader).unwrap();
        let restored = StrokeTokenizer::from_var_builder(&vb).unwrap();
        assert_eq!(restored.vocab_size(), tokenizer.vocab_size());
        assert_eq!(restored.bos_token(), tokenizer.bos_token());
        assert_eq!(restored.eot_token(), tokenizer.eot_token());
    }

    #[test]
    fn epoch_batches_pad_to_local_max_length() {
        let dataset = SourceDataset {
            files: vec![
                test_file("a.stroke", 4),
                test_file("b.stroke", 7),
                test_file("c.stroke", 5),
                test_file("d.stroke", 9),
            ],
        };
        let config = test_runtime(32, 2, 2);
        let mut rng = StdRng::seed_from_u64(7);
        let batches = dataset.epoch_batches(&mut rng, 99, &config);

        assert_eq!(batches.len(), 2);
        assert!(
            batches
                .iter()
                .all(|batch| batch.seq_len < config.block_size)
        );
        assert!(batches.iter().all(|batch| {
            batch
                .windows
                .iter()
                .all(|window| window.len() == batch.seq_len + 1)
                && batch.masks.iter().all(|mask| mask.len() == batch.seq_len)
        }));
        assert!(batches.iter().all(|batch| {
            let valid_lengths = batch
                .masks
                .iter()
                .map(|mask| valid_len(mask))
                .collect::<Vec<_>>();
            valid_lengths.windows(2).all(|pair| pair[0] >= pair[1])
        }));
    }

    #[test]
    fn evaluation_batches_sort_by_length_and_pad_locally() {
        let dataset = SourceDataset {
            files: vec![
                test_file("short.stroke", 4),
                test_file("long.stroke", 8),
                test_file("mid.stroke", 6),
            ],
        };
        let config = test_runtime(32, 2, 2);
        let batches = dataset.evaluation_batches(77, &config);

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].seq_len, 7);
        assert_eq!(valid_len(&batches[0].masks[0]), 7);
        assert_eq!(valid_len(&batches[0].masks[1]), 5);
        assert_eq!(batches[0].windows[1][6], 77);
        assert_eq!(batches[0].windows[1][7], 77);
        assert_eq!(batches[1].seq_len, 3);
        assert_eq!(valid_len(&batches[1].masks[0]), 3);
    }

    #[test]
    fn autoregressive_context_is_trimmed_not_padded() {
        let (full, full_last) = autoregressive_context(&[1, 2, 3], 99, 8);
        assert_eq!(full, vec![1, 2, 3]);
        assert_eq!(full_last, 2);

        let (trimmed, trimmed_last) = autoregressive_context(&[1, 2, 3, 4], 99, 2);
        assert_eq!(trimmed, vec![3, 4]);
        assert_eq!(trimmed_last, 1);
    }

    #[test]
    fn grid_tokenizer_round_trips_and_decodes_segment_tokens() {
        let tokenizer = StrokeTokenizer::with_grid(9);
        let gguf = GgufMetadata {
            version: GgufVersion::V3,
            metadata: tokenizer
                .gguf_metadata()
                .into_iter()
                .map(|(key, value)| (key.into_boxed_str(), value))
                .collect(),
            tensor_infos: Default::default(),
            tensor_data_offset: 0,
        };
        let mut bytes = Cursor::new(Vec::new());
        gguf.write(&mut bytes, std::iter::empty::<(&str, &[u8])>())
            .unwrap();
        let mut reader = Cursor::new(bytes.into_inner());
        let vb = VarBuilder::from_gguf(&mut reader).unwrap();
        let restored = StrokeTokenizer::from_var_builder(&vb).unwrap();
        assert!(restored.same_vocabulary(&tokenizer));
        let draw = restored.draw_token(Direction::SE, 4);
        assert_eq!(restored.token_name(draw), "DRAW_SE_4");
        assert!(matches!(
            restored.decode_token_kind(draw),
            TokenKind::Segment(ActionMode::Draw, Direction::SE, 4)
        ));
        assert!(matches!(
            restored.decode_token_kind(restored.travel_token(Direction::N, 2)),
            TokenKind::Segment(ActionMode::Travel, Direction::N, 2)
        ));
    }

    #[test]
    fn feather_dataset_loader_keeps_multistroke_icons() {
        let icon = FeatherIconRecord {
            name: "archive".to_string(),
            _original_svg_path: "/tmp/archive.svg".to_string(),
            quantized_points: vec![vec![[0, 0], [1, 0]], vec![[2, 2], [2, 3], [2, 4]]],
            _dirs: vec![
                vec!["E".to_string()],
                vec!["S".to_string(), "S".to_string()],
            ],
            grid_size: 9,
            accepted: true,
        };
        let tokenizer = StrokeTokenizer::with_grid(9);

        let train_files = feather_source_files(&tokenizer, &icon, FeatherSplit::Train);
        assert!(train_files.len() > 1);
        let base = train_files
            .iter()
            .find(|file| file.path() == "train/archive.stroke")
            .unwrap();
        let token_names = base
            .target_tokens()
            .iter()
            .map(|&token| tokenizer.token_name(token).to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            token_names,
            vec![
                "DRAW_E_1".to_string(),
                "TRAVEL_SE_1".to_string(),
                "TRAVEL_S_1".to_string(),
                "DRAW_S_2".to_string()
            ]
        );
        assert!(
            train_files
                .iter()
                .any(|file| file.path().contains("__rot90.stroke"))
        );

        let validation_files = feather_source_files(&tokenizer, &icon, FeatherSplit::Validation);
        assert_eq!(validation_files.len(), 1);
        assert_eq!(validation_files[0].path(), "valid/archive.stroke");
    }

    #[test]
    fn hybrid_dataset_source_includes_real_and_matching_synthetic_examples() {
        let train_name = (0..256)
            .map(|index| format!("hybrid-train-{index}"))
            .find(|name| matches!(feather_split_for_name(name), FeatherSplit::Train))
            .unwrap();
        let validation_name = (0..256)
            .map(|index| format!("hybrid-valid-{index}"))
            .find(|name| matches!(feather_split_for_name(name), FeatherSplit::Validation))
            .unwrap();
        let test_name = (0..256)
            .map(|index| format!("hybrid-test-{index}"))
            .find(|name| matches!(feather_split_for_name(name), FeatherSplit::Test))
            .unwrap();
        let dataset_path = std::env::temp_dir().join(format!(
            "nanochat-hybrid-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let icons = vec![
            serde_json::json!({
                "name": train_name,
                "original_svg_path": "/tmp/hybrid-train.svg",
                "quantized_points": [[[0, 0], [1, 0], [1, 1]]],
                "dirs": [["E", "S"]],
                "grid_size": 9,
                "accepted": true,
            }),
            serde_json::json!({
                "name": validation_name,
                "original_svg_path": "/tmp/hybrid-valid.svg",
                "quantized_points": [[[0, 0], [0, 1], [1, 1]]],
                "dirs": [["S", "E"]],
                "grid_size": 9,
                "accepted": true,
            }),
            serde_json::json!({
                "name": test_name,
                "original_svg_path": "/tmp/hybrid-test.svg",
                "quantized_points": [[[0, 0], [1, 1], [2, 1]]],
                "dirs": [["SE", "E"]],
                "grid_size": 9,
                "accepted": true,
            }),
        ];
        fs::write(&dataset_path, serde_json::to_vec(&icons).unwrap()).unwrap();

        let config = RuntimeConfig {
            epochs: 1,
            warmup_steps: 1,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.1,
            log_every: 10,
            eval_batches: 1,
            sample_tokens: 8,
            sample_prefix_tokens: 4,
            sample_temperature: 0.7,
            sample_top_k: 4,
            block_size: 48,
            batch_size: 4,
            n_embd: 32,
            n_head: 2,
            n_kv_head: 2,
            n_ff: 64,
            n_layer: 2,
            conv_kernel_size: 3,
            attention_period: 1,
            use_rope: false,
            rope_theta: 10_000.0,
            use_canvas_state_embeddings: true,
            use_extra_norms: false,
            eps: 1e-5,
            init_scale: 0.08,
            seed: 42,
            save_every_steps: 0,
            save_final_model: false,
            save_quantization: crate::config::SaveQuantization::F32,
            train_examples: 1,
            validation_examples: 1,
            test_examples: 1,
            dataset_path: Some(dataset_path.clone()),
            include_synthetic_data: true,
            gguf_path: "test.gguf".into(),
            sample_output_path: "sample.svg".into(),
        };

        let source = load_dataset_source(&config);
        assert_eq!(source.tokenizer.grid_size(), 0);
        assert_eq!(source.split.train.num_docs(), 2);
        assert_eq!(source.split.validation.num_docs(), 2);
        assert_eq!(source.split.test.num_docs(), 2);
        assert!(source.label.contains("hybrid-stroke-autocomplete"));
        assert!(
            source
                .split
                .train
                .files()
                .iter()
                .any(|file| file.path().contains("circle-000.stroke"))
        );
        assert!(
            source
                .split
                .train
                .files()
                .iter()
                .any(|file| file.path().contains("hybrid-train-"))
        );

        let _ = fs::remove_file(dataset_path);
    }

    #[test]
    fn canvas_state_indexes_track_cursor_and_pen_after_each_token() {
        let tokenizer = StrokeTokenizer::with_grid(9);
        let spec = canvas_state_spec(&tokenizer, 2);
        let tokens = vec![vec![
            tokenizer.draw_token(Direction::E, 2),
            tokenizer.travel_token(Direction::S, 1),
        ]];

        let states = canvas_state_indexes(&tokenizer, &tokens, spec);
        assert_eq!(states.cursor_x, vec![vec![2, 2]]);
        assert_eq!(states.cursor_y, vec![vec![0, 1]]);
        assert_eq!(states.pen_state, vec![vec![1, 0]]);
    }

    #[test]
    fn synthetic_dataset_split_uses_requested_sizes() {
        let tokenizer = StrokeTokenizer::new();
        let config = RuntimeConfig {
            epochs: 1,
            warmup_steps: 1,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.1,
            log_every: 10,
            eval_batches: 1,
            sample_tokens: 8,
            sample_prefix_tokens: 4,
            sample_temperature: 0.7,
            sample_top_k: 4,
            block_size: 48,
            batch_size: 4,
            n_embd: 32,
            n_head: 2,
            n_kv_head: 2,
            n_ff: 64,
            n_layer: 2,
            conv_kernel_size: 3,
            attention_period: 1,
            use_rope: false,
            rope_theta: 10_000.0,
            use_canvas_state_embeddings: true,
            use_extra_norms: false,
            eps: 1e-5,
            init_scale: 0.08,
            seed: 42,
            save_every_steps: 0,
            save_final_model: false,
            save_quantization: crate::config::SaveQuantization::F32,
            train_examples: 7,
            validation_examples: 3,
            test_examples: 2,
            dataset_path: None,
            include_synthetic_data: false,
            gguf_path: "test.gguf".into(),
            sample_output_path: "sample.svg".into(),
        };

        let split = synthetic_dataset_split(&tokenizer, &config);
        assert_eq!(split.train.num_docs(), 7);
        assert_eq!(split.validation.num_docs(), 3);
        assert_eq!(split.test.num_docs(), 2);
    }

    #[test]
    fn synthetic_dataset_stays_balanced_and_within_context_window() {
        let tokenizer = StrokeTokenizer::new();
        let config = RuntimeConfig {
            epochs: 1,
            warmup_steps: 1,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.1,
            log_every: 10,
            eval_batches: 1,
            sample_tokens: 8,
            sample_prefix_tokens: 4,
            sample_temperature: 0.7,
            sample_top_k: 4,
            block_size: 48,
            batch_size: 8,
            n_embd: 24,
            n_head: 2,
            n_kv_head: 2,
            n_ff: 48,
            n_layer: 1,
            conv_kernel_size: 3,
            attention_period: 1,
            use_rope: false,
            rope_theta: 10_000.0,
            use_canvas_state_embeddings: true,
            use_extra_norms: false,
            eps: 1e-5,
            init_scale: 0.1,
            seed: 42,
            save_every_steps: 0,
            save_final_model: false,
            save_quantization: crate::config::SaveQuantization::F32,
            train_examples: ShapeKind::all().len() * 2,
            validation_examples: ShapeKind::all().len(),
            test_examples: ShapeKind::all().len(),
            dataset_path: None,
            include_synthetic_data: false,
            gguf_path: "test.gguf".into(),
            sample_output_path: "sample.svg".into(),
        };

        let split = synthetic_dataset_split(&tokenizer, &config);
        let seen_shapes = split
            .train
            .files()
            .iter()
            .map(|file| {
                let filename = file.path().split('/').next_back().unwrap();
                filename
                    .rsplit_once('-')
                    .map(|(shape, _)| shape.to_string())
                    .unwrap()
            })
            .collect::<HashSet<_>>();

        assert_eq!(seen_shapes.len(), ShapeKind::all().len());
        let unique_token_sequences = split
            .train
            .files()
            .iter()
            .map(|file| file.target_tokens().to_vec())
            .collect::<HashSet<_>>();
        assert_eq!(unique_token_sequences.len(), split.train.files().len());
        assert!(
            split
                .train
                .files()
                .iter()
                .all(|file| file.tokens.len() <= config.block_size + 1)
        );
    }

    #[test]
    fn synthetic_shapes_are_closed_loops() {
        let tokenizer = StrokeTokenizer::new();
        for shape in ShapeKind::all() {
            for seed in 0..32 {
                let mut rng = StdRng::seed_from_u64(seed);
                let tokens = shape_tokens(shape, &tokenizer, &mut rng);
                assert_eq!(
                    token_displacement(&tokenizer, &tokens),
                    (0, 0),
                    "{} should close back to the origin",
                    shape.name()
                );
            }
        }
    }

    #[test]
    fn synthetic_shapes_have_no_long_straight_runs() {
        let tokenizer = StrokeTokenizer::new();
        for shape in ShapeKind::all() {
            for seed in 0..32 {
                let mut rng = StdRng::seed_from_u64(seed);
                let tokens = shape_tokens(shape, &tokenizer, &mut rng);
                let run = max_straight_run(&tokenizer, &tokens);
                assert!(
                    run <= 2,
                    "{} (seed {}) has a straight run of {} (max 2)",
                    shape.name(),
                    seed,
                    run,
                );
            }
        }
    }

    #[test]
    fn synthetic_shapes_do_not_overlap() {
        let tokenizer = StrokeTokenizer::new();
        for shape in ShapeKind::all() {
            for seed in 0..32 {
                let mut rng = StdRng::seed_from_u64(seed);
                let tokens = shape_tokens(shape, &tokenizer, &mut rng);
                assert!(
                    !has_overlapping_points(&tokenizer, &tokens),
                    "{} (seed {}) visits the same grid point twice",
                    shape.name(),
                    seed,
                );
            }
        }
    }

    #[test]
    fn synthetic_shapes_do_not_self_intersect() {
        let tokenizer = StrokeTokenizer::new();
        for shape in ShapeKind::all() {
            for seed in 0..32 {
                let mut rng = StdRng::seed_from_u64(seed);
                let tokens = shape_tokens(shape, &tokenizer, &mut rng);
                assert!(
                    !has_crossing_segments(&tokenizer, &tokens),
                    "{} (seed {}) has crossing line segments",
                    shape.name(),
                    seed,
                );
            }
        }
    }

    #[test]
    fn closed_loop_variants_shift_and_reverse_sequences() {
        let tokenizer = StrokeTokenizer::new();
        let base = vec![
            tokenizer.draw_token(Direction::E, 2),
            tokenizer.draw_token(Direction::S, 2),
            tokenizer.draw_token(Direction::W, 2),
            tokenizer.draw_token(Direction::N, 2),
        ];

        let variants = (0..32)
            .map(|seed| {
                let mut tokens = base.clone();
                apply_closed_loop_variant(
                    &tokenizer,
                    &mut StdRng::seed_from_u64(seed),
                    &mut tokens,
                );
                assert_eq!(token_displacement(&tokenizer, &tokens), (0, 0));
                tokens
            })
            .collect::<HashSet<_>>();

        assert!(
            variants.len() > 4,
            "expected shifted/reversed closed-loop variants"
        );
    }

    #[test]
    fn sample_export_produces_valid_svg_bytes() {
        let tokenizer = StrokeTokenizer::new();
        let tokens = vec![
            tokenizer.draw_token(Direction::E, 1),
            tokenizer.draw_token(Direction::SE, 1),
            tokenizer.draw_token(Direction::S, 1),
        ];
        let root = std::env::temp_dir().join("nanochat-stroke-sample.svg");
        write_tokens_to_svg_file(&tokenizer, &tokens[..1], &tokens[1..], &root);
        let svg = fs::read_to_string(&root).unwrap();
        assert!(svg.starts_with("<svg"));
        let _ = fs::remove_file(root);
    }

    #[test]
    fn stroke_scene_keeps_prompt_and_continuation_as_separate_layers() {
        let tokenizer = StrokeTokenizer::with_grid(5);
        let prompt = vec![tokenizer.draw_token(Direction::E, 1)];
        let continuation = vec![tokenizer.draw_token(Direction::S, 1)];

        let scene = tokens_to_stroke_scene(&tokenizer, &prompt, &continuation);

        assert_eq!(
            scene.prompt_strokes,
            vec![StrokePath {
                points: vec![(0, 0), (1, 0)],
            }]
        );
        assert_eq!(
            scene.continuation_strokes,
            vec![StrokePath {
                points: vec![(1, 0), (1, 1)],
            }]
        );
    }

    #[test]
    fn stroke_variants_single_stroke_produces_identity_and_reversed() {
        let strokes = vec![FeatherStroke {
            points: vec![(0, 0), (1, 0), (2, 0)],
        }];
        let perms = stroke_order_permutations(&strokes, "single");
        assert_eq!(perms.len(), 2);
        assert_eq!(perms[0].1, strokes);
        assert_eq!(perms[1].1[0].points, vec![(2, 0), (1, 0), (0, 0)]);
    }

    #[test]
    fn stroke_variants_two_strokes_all_order_and_reversal_combos() {
        let strokes = vec![
            FeatherStroke {
                points: vec![(0, 0), (1, 0)],
            },
            FeatherStroke {
                points: vec![(3, 3), (4, 3)],
            },
        ];
        // 2! orderings × 2^2 reversal masks = 8 variants
        let perms = stroke_order_permutations(&strokes, "two");
        assert_eq!(perms.len(), 8);
        // First variant is the identity (original order, no reversals)
        assert_eq!(perms[0].1[0].points, vec![(0, 0), (1, 0)]);
        assert_eq!(perms[0].1[1].points, vec![(3, 3), (4, 3)]);
        // All variants are unique
        let unique: HashSet<Vec<Vec<(i32, i32)>>> = perms
            .iter()
            .map(|(_, s)| s.iter().map(|stroke| stroke.points.clone()).collect())
            .collect();
        assert_eq!(unique.len(), 8);
    }

    #[test]
    fn stroke_variants_caps_at_max_for_many_strokes() {
        let strokes: Vec<FeatherStroke> = (0..4)
            .map(|i| FeatherStroke {
                points: vec![(i, 0), (i, 1)],
            })
            .collect();
        // 4! × 2^4 = 384 > MAX_STROKE_PERMUTATIONS, so sampled
        let perms = stroke_order_permutations(&strokes, "four");
        assert_eq!(perms.len(), MAX_STROKE_PERMUTATIONS);
        assert_eq!(perms[0].0, 0);
        // All variants are unique
        let unique: HashSet<Vec<Vec<(i32, i32)>>> = perms
            .iter()
            .map(|(_, s)| s.iter().map(|stroke| stroke.points.clone()).collect())
            .collect();
        assert_eq!(unique.len(), MAX_STROKE_PERMUTATIONS);
    }

    #[test]
    fn stroke_variants_include_reversals() {
        let strokes = vec![
            FeatherStroke {
                points: vec![(0, 0), (1, 0)],
            },
            FeatherStroke {
                points: vec![(3, 3), (4, 3)],
            },
        ];
        let perms = stroke_order_permutations(&strokes, "rev_check");
        // Should contain a variant where first stroke is reversed
        assert!(perms.iter().any(|(_, s)| s[0].points == vec![(1, 0), (0, 0)]));
        // Should contain a variant where second stroke is reversed
        assert!(perms.iter().any(|(_, s)| s[1].points == vec![(4, 3), (3, 3)]));
    }

    #[test]
    fn stroke_variants_are_deterministic() {
        let strokes: Vec<FeatherStroke> = (0..6)
            .map(|i| FeatherStroke {
                points: vec![(i, 0), (i, 1)],
            })
            .collect();
        let run1 = stroke_order_permutations(&strokes, "deterministic");
        let run2 = stroke_order_permutations(&strokes, "deterministic");
        assert_eq!(run1.len(), run2.len());
        for (a, b) in run1.iter().zip(run2.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    fn feather_source_files_generates_shuffle_variants_for_multistroke() {
        let icon = FeatherIconRecord {
            name: "archive".to_string(),
            _original_svg_path: "/tmp/archive.svg".to_string(),
            quantized_points: vec![vec![[0, 0], [1, 0]], vec![[2, 2], [2, 3], [2, 4]]],
            _dirs: vec![
                vec!["E".to_string()],
                vec!["S".to_string(), "S".to_string()],
            ],
            grid_size: 9,
            accepted: true,
        };
        let tokenizer = StrokeTokenizer::with_grid(9);

        let train_files = feather_source_files(&tokenizer, &icon, FeatherSplit::Train);
        assert!(
            train_files.iter().any(|f| f.path().contains("__perm1")),
            "train files should contain permutation variants"
        );
        assert!(
            train_files.len() > 8,
            "multi-stroke icon should produce more than 8 files (was {})",
            train_files.len()
        );

        let val_files = feather_source_files(&tokenizer, &icon, FeatherSplit::Validation);
        assert!(
            !val_files.iter().any(|f| f.path().contains("perm")),
            "validation files should not contain permutation variants"
        );
        assert_eq!(val_files.len(), 1);
    }
}
