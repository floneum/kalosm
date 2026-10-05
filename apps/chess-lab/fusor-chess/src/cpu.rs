//! CPU play: the rules, the model's value network as a compiled Fusor CPU
//! program, and an alpha-beta search, so a human's opponent searches far more
//! positions per second than GPU rounds allow. Positions, move words and
//! position keys match chess.wgsl exactly.
use fusor::{CpuProgram, CpuValue, Device, Tensor};
use std::cell::Cell;
use std::marker::PhantomData;

/// Game-state words, as chess.wgsl stores a game.
pub const WORDS: usize = 768;
const MATE: f32 = 2.;
const INF: f32 = 4.;

fn valid(s: i32) -> bool {
    (0..128).contains(&s) && s & 0x88 == 0
}
pub fn move_from(m: u32) -> i32 {
    (m & 127) as i32
}
pub fn move_to(m: u32) -> i32 {
    ((m >> 7) & 127) as i32
}
fn promotion(m: u32) -> i8 {
    ((m >> 14) & 7) as i8
}
fn flags(m: u32) -> u32 {
    (m >> 17) & 7
}
fn movement(f: i32, t: i32, p: i8, fl: u32) -> u32 {
    f as u32 | (t as u32) << 7 | (p as u32) << 14 | fl << 17
}
fn hash(x: u32) -> u32 {
    let mut v = x;
    v ^= v >> 16;
    v = v.wrapping_mul(2146121005);
    v ^= v >> 15;
    v = v.wrapping_mul(2221713035);
    v ^= v >> 16;
    v
}
const KNIGHT: [i32; 8] = [-33, -31, -18, -14, 14, 18, 31, 33];
const LINES: [i32; 8] = [-17, -15, 15, 17, -16, -1, 1, 16];

/// A position: 0x88 board (+white/-black; 1 pawn, 2 knight, 3 bishop, 4 rook,
/// 5 queen, 6 king), side to move, castling rights (bits K Q k q), en passant
/// square or -1, halfmove clock, ply and king squares.
#[derive(Clone)]
pub struct Position {
    pub board: [i8; 128],
    pub side: i8,
    pub rights: u32,
    pub ep: i32,
    pub halfmove: u32,
    pub ply: u32,
    pub kings: [i32; 2],
    /// XOR of the pieces' key hashes (position_key's per-piece terms), kept by
    /// make and unmake.
    pieces: [u32; 2],
}
fn piece_hash(p: i8, s: i32) -> [u32; 2] {
    let n = (p as i32 + 6) as u32 * 128 + s as u32;
    [hash(n + 101), hash(n + 98765)]
}
/// What `make` needs to restore.
pub struct Undo {
    captured: i8,
    rights: u32,
    ep: i32,
    halfmove: u32,
}
impl Position {
    pub fn from_state(s: &[u32]) -> Self {
        let mut board = [0i8; 128];
        for (i, b) in board.iter_mut().enumerate() {
            *b = s[i] as i32 as i8;
        }
        let mut pos = Self {
            board,
            side: s[128] as i32 as i8,
            rights: s[129],
            ep: s[130] as i32,
            halfmove: s[131],
            ply: s[132],
            kings: [s[133] as i32, s[134] as i32],
            pieces: [0; 2],
        };
        for sq in (0..128).filter(|&sq| valid(sq)) {
            let p = pos.board[sq as usize];
            if p != 0 {
                pos.toggle_hash(p, sq);
            }
        }
        pos
    }
    fn toggle_hash(&mut self, p: i8, s: i32) {
        let h = piece_hash(p, s);
        self.pieces[0] ^= h[0];
        self.pieces[1] ^= h[1];
    }
    /// Set square s to piece p (0 empties it), keeping the piece hash.
    fn set(&mut self, s: i32, p: i8) {
        let old = self.board[s as usize];
        if old != 0 {
            self.toggle_hash(old, s);
        }
        if p != 0 {
            self.toggle_hash(p, s);
        }
        self.board[s as usize] = p;
    }
    fn king(&self, side: i8) -> i32 {
        self.kings[usize::from(side != 1)]
    }
    /// Whether `by` attacks square `s`.
    pub fn attacked(&self, s: i32, by: i8) -> bool {
        let at = |t: i32| if valid(t) { self.board[t as usize] } else { 0 };
        let b = by as i32;
        if at(s - b * 16 - 1) == by || at(s - b * 16 + 1) == by {
            return true;
        }
        if KNIGHT.iter().any(|d| at(s + d) == by * 2) {
            return true;
        }
        for (i, d) in LINES.iter().enumerate() {
            let mut t = s + d;
            let mut distance = 1;
            while valid(t) {
                let p = self.board[t as usize];
                if p != 0 {
                    if p * by > 0 {
                        let kind = p.abs();
                        let slider = if i < 4 { 3 } else { 4 };
                        if kind == 5 || kind == slider || (kind == 6 && distance == 1) {
                            return true;
                        }
                    }
                    break;
                }
                t += d;
                distance += 1;
            }
        }
        false
    }
    pub fn in_check(&self) -> bool {
        self.attacked(self.king(self.side), -self.side)
    }
    pub fn make(&mut self, m: u32) -> Undo {
        let (f, t, fl) = (move_from(m), move_to(m), flags(m));
        let side = self.side;
        let piece = self.board[f as usize];
        let undo = Undo { captured: self.board[t as usize], rights: self.rights, ep: self.ep, halfmove: self.halfmove };
        let p = promotion(m);
        self.set(t, if p > 0 { side * p } else { piece });
        self.set(f, 0);
        if fl & 1 != 0 {
            self.set(t - side as i32 * 16, 0);
        }
        if fl & 2 != 0 {
            let (rf, rt) = if t > f { (f + 3, f + 1) } else { (f - 4, f - 1) };
            self.set(rt, self.board[rf as usize]);
            self.set(rf, 0);
        }
        if piece.abs() == 6 {
            if side == 1 {
                self.kings[0] = t;
                self.rights &= 12;
            } else {
                self.kings[1] = t;
                self.rights &= 3;
            }
        }
        for (corner, mask) in [(0, 13), (7, 14), (112, 7), (119, 11)] {
            if f == corner || t == corner {
                self.rights &= mask;
            }
        }
        self.ep = if fl & 4 != 0 { f + side as i32 * 16 } else { -1 };
        self.halfmove = if piece.abs() == 1 || undo.captured != 0 || fl & 1 != 0 { 0 } else { self.halfmove + 1 };
        self.ply += 1;
        self.side = -side;
        undo
    }
    pub fn unmake(&mut self, m: u32, undo: &Undo) {
        self.side = -self.side;
        self.ply -= 1;
        let side = self.side;
        let (f, t, fl) = (move_from(m), move_to(m), flags(m));
        let moved = self.board[t as usize];
        self.set(f, if promotion(m) > 0 { side } else { moved });
        self.set(t, undo.captured);
        if fl & 1 != 0 {
            self.set(t - side as i32 * 16, -side);
        }
        if fl & 2 != 0 {
            let (rf, rt) = if t > f { (f + 3, f + 1) } else { (f - 4, f - 1) };
            self.set(rf, self.board[rt as usize]);
            self.set(rt, 0);
        }
        if moved.abs() == 6 {
            self.kings[usize::from(side != 1)] = f;
        }
        self.rights = undo.rights;
        self.ep = undo.ep;
        self.halfmove = undo.halfmove;
    }
    /// Whether move m leaves the mover's king safe.
    /// Whether move m leaves the mover's king safe: the board alone is updated
    /// and restored (no key, rights or clock bookkeeping).
    fn legal(&mut self, m: u32) -> bool {
        let (f, t, fl) = (move_from(m), move_to(m), flags(m));
        let side = self.side;
        let piece = self.board[f as usize];
        let captured = self.board[t as usize];
        self.board[t as usize] = piece;
        self.board[f as usize] = 0;
        let passed = t - side as i32 * 16;
        let taken = if fl & 1 != 0 { std::mem::take(&mut self.board[passed as usize]) } else { 0 };
        let king = if piece.abs() == 6 { t } else { self.king(side) };
        let safe = !self.attacked(king, -side);
        if fl & 1 != 0 {
            self.board[passed as usize] = taken;
        }
        self.board[f as usize] = piece;
        self.board[t as usize] = captured;
        safe
    }
    /// The squares (as bits of their 0x88 index) of `side`'s pieces that stand
    /// alone between their king and an enemy slider on that line.
    fn pinned(&self, side: i8) -> u128 {
        let king = self.king(side);
        let mut pinned = 0u128;
        for (i, d) in LINES.iter().enumerate() {
            let mut t = king + d;
            let mut own = -1;
            while valid(t) {
                let p = self.board[t as usize];
                if p != 0 {
                    if p * side > 0 {
                        if own >= 0 {
                            break;
                        }
                        own = t;
                    } else {
                        let kind = p.abs();
                        if own >= 0 && (kind == 5 || kind == if i < 4 { 3 } else { 4 }) {
                            pinned |= 1 << own;
                        }
                        break;
                    }
                }
                t += d;
            }
        }
        pinned
    }
    /// Legal moves, as chess.wgsl generates them (kings are never captured).
    pub fn moves(&mut self) -> Vec<u32> {
        self.generate(false)
    }
    /// Whether any legal move exists (checkmate and stalemate tests).
    pub fn has_move(&mut self) -> bool {
        !self.generate(true).is_empty()
    }
    /// With `first`, stops after the first square that yields a legal move.
    fn generate(&mut self, first: bool) -> Vec<u32> {
        let side = self.side;
        let s = side as i32;
        let mut out = Vec::with_capacity(48);
        // Out of check, a piece that is not pinned to its king can move
        // anywhere without exposing it; only king moves, en passant and
        // pinned pieces need the attack test.
        let free = if self.in_check() { 0 } else { !self.pinned(side) };
        let king = self.king(side);
        let push = |pos: &mut Self, out: &mut Vec<u32>, f: i32, t: i32, p: i8, fl: u32| {
            let m = movement(f, t, p, fl);
            let skip = free >> f & 1 == 1 && f != king && fl & 1 == 0;
            debug_assert!(!skip || pos.legal(m), "a move taken as legal without the attack test is not");
            if skip || pos.legal(m) {
                out.push(m);
            }
        };
        for f in (0..128).filter(|&f| valid(f)) {
            if first && !out.is_empty() {
                break;
            }
            let piece = self.board[f as usize];
            if piece * side <= 0 {
                continue;
            }
            match piece.abs() {
                1 => {
                    let promotes = |t: i32| t >> 4 == 0 || t >> 4 == 7;
                    let t = f + s * 16;
                    if valid(t) && self.board[t as usize] == 0 {
                        if promotes(t) {
                            for p in 2..=5 {
                                push(self, &mut out, f, t, p, 0);
                            }
                        } else {
                            push(self, &mut out, f, t, 0, 0);
                        }
                        let t2 = t + s * 16;
                        if f >> 4 == if side == 1 { 1 } else { 6 } && self.board[t2 as usize] == 0 {
                            push(self, &mut out, f, t2, 0, 4);
                        }
                    }
                    for df in [-1, 1] {
                        let d = t + df;
                        if !valid(d) {
                            continue;
                        }
                        let target = self.board[d as usize];
                        if target * side < 0 && target.abs() != 6 {
                            if promotes(d) {
                                for p in 2..=5 {
                                    push(self, &mut out, f, d, p, 0);
                                }
                            } else {
                                push(self, &mut out, f, d, 0, 0);
                            }
                        } else if d == self.ep && self.board[(d - s * 16) as usize] == -side {
                            push(self, &mut out, f, d, 0, 1);
                        }
                    }
                }
                2 => {
                    for d in KNIGHT {
                        let t = f + d;
                        if valid(t) && self.board[t as usize] * side <= 0 && self.board[t as usize].abs() != 6 {
                            push(self, &mut out, f, t, 0, 0);
                        }
                    }
                }
                kind => {
                    for (i, d) in LINES.iter().enumerate() {
                        if (kind == 3 && i >= 4) || (kind == 4 && i < 4) {
                            continue;
                        }
                        let mut t = f + d;
                        while valid(t) {
                            let target = self.board[t as usize];
                            if target * side > 0 || target.abs() == 6 {
                                break;
                            }
                            push(self, &mut out, f, t, 0, 0);
                            if target != 0 || kind == 6 {
                                break;
                            }
                            t += d;
                        }
                    }
                    let home = if side == 1 { 4 } else { 116 };
                    if kind == 6 && f == home && !self.attacked(f, -side) {
                        let shift = if side == 1 { 0 } else { 2 };
                        let empty = |pos: &Self, sq: i32| pos.board[sq as usize] == 0;
                        if self.rights & (1 << shift) != 0
                            && self.board[(f + 3) as usize] == side * 4
                            && empty(self, f + 1)
                            && empty(self, f + 2)
                            && !self.attacked(f + 1, -side)
                            && !self.attacked(f + 2, -side)
                        {
                            push(self, &mut out, f, f + 2, 0, 2);
                        }
                        if self.rights & (2 << shift) != 0
                            && self.board[(f - 4) as usize] == side * 4
                            && empty(self, f - 1)
                            && empty(self, f - 2)
                            && empty(self, f - 3)
                            && !self.attacked(f - 1, -side)
                            && !self.attacked(f - 2, -side)
                        {
                            push(self, &mut out, f, f - 2, 0, 2);
                        }
                    }
                }
            }
        }
        out
    }
    /// chess.wgsl's position_key: repetition detection and the transposition table.
    pub fn key(&mut self) -> u64 {
        let base = hash(self.rights + 32 * u32::from(self.side != 1) + 77);
        let mut lo = base ^ self.pieces[0];
        let mut hi = hash(base.wrapping_add(17)) ^ self.pieces[1];
        if self.ep >= 0 {
            let s = self.side as i32;
            for df in [-1, 1] {
                let f = self.ep - s * 16 + df;
                if valid(f) && self.board[f as usize] == self.side && self.legal(movement(f, self.ep, 0, 1)) {
                    lo ^= hash(self.ep as u32 + 99999);
                    hi ^= hash(self.ep as u32 + 77777);
                    break;
                }
            }
        }
        (hi as u64) << 32 | lo as u64
    }
    /// Insufficient material, as chess.wgsl rules it.
    pub fn material_draw(&self) -> bool {
        let (mut pieces, mut bishops, mut color, mut same) = (0, 0, -1, true);
        for s in (0..128).filter(|&s| valid(s)) {
            let p = self.board[s as usize].abs();
            if p == 0 || p == 6 {
                continue;
            }
            pieces += 1;
            if p == 1 || p == 4 || p == 5 {
                return false;
            }
            if p == 3 {
                bishops += 1;
                let c = ((s >> 4) + (s & 7)) & 1;
                if color < 0 {
                    color = c;
                } else if c != color {
                    same = false;
                }
            }
        }
        pieces <= 1 || (bishops == pieces && same)
    }
}

/// Inputs one move can change, per perspective: the moved, captured, promoted
/// and castled pieces with their counts, the rights, en passant and the clocks.
const CHANGED: usize = 16;

/// The MLP family's value network on the CPU: the model's own graph, built on
/// Fusor's CPU device as two compiled programs. `update` moves one
/// perspective's accumulator (the first layer's sums, with the linear path as
/// one more column) by a sparse change of its inputs; from zero, that is the
/// whole first layer. `head` runs the rest of the network from an accumulator.
pub struct Net {
    config: crate::Config,
    update: Program<3>,
    head: Program<2>,
    /// The first layer transposed with the linear path as its last column,
    /// then the hidden layers and the value head.
    parameters: Vec<CpuValue>,
    /// The linear path's weights, read for move ordering (`gain`).
    linear: Vec<f32>,
    /// Scale of the piece-count inputs (see `count_scale`).
    count: f32,
    /// The change entries the running update set, to zero again after it.
    touched: std::cell::RefCell<Vec<u32>>,
    /// The programs' input buffers belong to one thread at a time.
    _single: PhantomData<Cell<()>>,
}
/// A compiled program with its inputs and, last, its output.
struct Program<const N: usize> {
    program: CpuProgram,
    values: [CpuValue; N],
}
impl<const N: usize> Program<N> {
    fn new(output: &fusor::tensor::Dyn, values: [&fusor::tensor::Dyn; N]) -> Self {
        let program = CpuProgram::compile(output).expect("the value network compiles for the CPU");
        let values = values.map(|v| program.value(v).expect("a program value's buffer"));
        Self { program, values }
    }
    fn run(&self) {
        self.program.run().expect("the value network runs");
    }
}
/// The inputs a move changes, per perspective (White's, then Black's): each
/// input's index and by how much.
#[derive(Default)]
struct Changes {
    list: [[(u32, f32); CHANGED]; 2],
    len: [usize; 2],
}
impl Changes {
    /// Hand perspective `k`'s changes to `poke`.
    fn poke(&self, k: usize, poke: &mut dyn FnMut(usize, f32)) {
        for &(i, x) in &self.list[k][..self.len[k]] {
            poke(i as usize, x);
        }
    }
    fn push(&mut self, perspective: usize, input: usize, by: f32) {
        self.list[perspective][self.len[perspective]] = (input as u32, by);
        self.len[perspective] += 1;
    }
}

impl Net {
    /// A zero-weight network of `config`'s shape.
    pub fn new(config: crate::Config) -> Self {
        let (width, layers, hidden) = (config.width, config.depth, config.hidden);
        let cpu = Device::cpu();
        let wide = width + 1;
        let zeros = |rows: usize, cols: usize| Tensor::<2, f32>::from_slice(&cpu, [rows, cols], &vec![0f32; rows * cols]);
        let mut parameters = vec![zeros(832, wide)];
        let mut last = width;
        for _ in 1..layers {
            parameters.push(zeros(hidden, last));
            last = hidden;
        }
        parameters.push(zeros(1, last));
        let first = &parameters[0];
        // update: an accumulator plus the first layer of the change in its
        // inputs (zero but for a few entries, which the contraction skips).
        let before = zeros(1, wide);
        let change = zeros(1, 832);
        let after = before.add(&change.matmul(first));
        let update = Program::new(after.as_dyn(), [before.as_dyn(), change.as_dyn(), after.as_dyn()]);
        // head: the network network.rs builds for training, from the first
        // layer's sums on.
        let acc = zeros(wide, 1);
        let mut h = acc.narrow(0usize, 0, width).relu();
        let mut running = width;
        for w in &parameters[1..layers] {
            let next = w.matmul(&h);
            h = if running == hidden {
                next.add(&h).mul_scalar(std::f32::consts::FRAC_1_SQRT_2).relu()
            } else {
                next.relu()
            };
            running = hidden;
        }
        let value = parameters[layers].matmul(&h).add(&acc.narrow(0usize, width, 1)).tanh();
        let head = Program::new(value.as_dyn(), [acc.as_dyn(), value.as_dyn()]);
        let parameters = parameters
            .iter()
            .map(|p| update.program.value(p.as_dyn()).expect("a parameter's buffer"))
            .collect();
        Self {
            config,
            update,
            head,
            parameters,
            linear: vec![0.; 832],
            count: crate::count_scale(),
            touched: Default::default(),
            _single: PhantomData,
        }
    }
    /// Numbers in one accumulator: the first layer's sums and the linear path's.
    fn wide(&self) -> usize {
        self.config.width + 1
    }
    /// Replace every weight, in network.rs order: first layer, hidden layers,
    /// value head, linear path.
    pub fn set_parameters(&mut self, parameters: &[Vec<f32>]) {
        let (width, layers) = (self.config.width, self.config.depth);
        assert_eq!(parameters.len(), layers + 2, "parameter count of the network's shape");
        let (wide, linear) = (self.wide(), &parameters[layers + 1]);
        let mut first = vec![0f32; 832 * wide];
        for i in 0..832 {
            for j in 0..width {
                first[i * wide + j] = parameters[0][j * 832 + i];
            }
            first[i * wide + width] = linear[i];
        }
        self.parameters[0].write(&first).expect("the first layer's weights");
        for (value, weights) in self.parameters[1..].iter().zip(&parameters[1..=layers]) {
            value.write(weights).expect("weights of the parameter's size");
        }
        self.linear = linear.clone();
    }
    /// Every weight, in network.rs order and layout.
    pub fn parameters(&self) -> Vec<Vec<f32>> {
        let wide = self.wide();
        let mut first = vec![0f32; 832 * wide];
        self.parameters[0].read_into(&mut first).expect("the first layer's weights");
        let width = self.config.width;
        let mut out = vec![vec![0f32; width * 832]];
        for i in 0..832 {
            for j in 0..width {
                out[0][j * 832 + i] = first[i * wide + j];
            }
        }
        for (value, size) in self.parameters[1..].iter().zip(&self.config.sizes()[1..]) {
            let mut weights = vec![0f32; *size];
            value.read_into(&mut weights).expect("the parameter's weights");
            out.push(weights);
        }
        out.push(self.linear.clone());
        out
    }
    /// From the parameters in network.rs order: first layer, hidden layers,
    /// value head, linear path.
    pub fn from_parameters(config: crate::Config, parameters: &[Vec<f32>]) -> Self {
        let mut net = Self::new(config);
        net.set_parameters(parameters);
        net
    }
    /// Replace every weight from the parameters concatenated in order
    /// (`Model::value_weights`).
    pub fn set_flat(&mut self, flat: &[f32]) {
        let mut parameters = Vec::new();
        let mut at = 0;
        for size in self.config.sizes() {
            parameters.push(flat[at..at + size].to_vec());
            at += size;
        }
        self.set_parameters(&parameters);
    }
    /// From a saved checkpoint (`Model::save`: step, then each parameter
    /// followed by its two optimizer moments), whose length names its size.
    pub fn from_checkpoint(state: &[f32]) -> Option<Self> {
        let config = crate::Config::all().find(|c| 1 + 3 * c.parameter_count() == state.len())?;
        let mut parameters = Vec::new();
        let mut at = 1;
        for size in config.sizes() {
            parameters.push(state[at..at + size].to_vec());
            at += 3 * size;
        }
        Some(Self::from_parameters(config, &parameters))
    }
    /// Diagnostic: evaluate with a different piece-count scale.
    pub fn with_count_scale(mut self, scale: f32) -> Self {
        self.count = scale;
        self
    }
    /// Input index of piece p on square s, seen by `perspective` (as chess.wgsl's
    /// feature: the viewer's pieces first, board mirrored for Black).
    fn feature(p: i8, s: i32, perspective: i8) -> usize {
        let relative = p * perspective;
        let plane = if relative > 0 { relative - 1 } else { 5 - relative } as usize;
        let oriented = if perspective == 1 { s } else { s ^ 112 };
        plane * 64 + ((oriented >> 4) * 8 + (oriented & 7)) as usize
    }
    /// The inputs of `pos` that are set for `perspective` to move, with their
    /// values (as `features` lays them out for the side to move).
    fn active(&self, pos: &Position, perspective: i8, mut push: impl FnMut(usize, f32)) {
        let mut counts = [0u8; 12];
        for s in (0..128).filter(|&s| valid(s)) {
            let p = pos.board[s as usize];
            if p != 0 {
                let f = Self::feature(p, s, perspective);
                push(f, 1.);
                counts[f / 64] += 1;
            }
        }
        for (plane, &c) in counts.iter().enumerate() {
            if c > 0 {
                push(782 + plane, f32::from(c) * self.count);
            }
        }
        for bit in 0..4 {
            if Self::rights(pos.rights, perspective) >> bit & 1 == 1 {
                push(768 + bit, 1.);
            }
        }
        if pos.ep >= 0 {
            push(772 + (pos.ep & 7) as usize, 1.);
        }
        push(780, Self::clock(pos.halfmove, 100.));
        push(781, Self::clock(pos.ply, 200.));
        push(831, 1.);
    }
    /// Castling rights as `perspective` sees them: its own first.
    fn rights(rights: u32, perspective: i8) -> u32 {
        if perspective == 1 { rights } else { ((rights & 3) << 2) | ((rights & 12) >> 2) }
    }
    fn clock(count: u32, scale: f32) -> f32 {
        (count as f32 / scale).min(1.)
    }
    /// Both perspectives' accumulators (White's, then Black's) of `pos`, from
    /// scratch: an update from zero by every active input.
    fn refresh(&self, pos: &Position, acc: &mut [f32]) {
        let wide = self.wide();
        let zero = vec![0f32; wide];
        for (k, perspective) in [1i8, -1].into_iter().enumerate() {
            self.update(&zero, &mut acc[k * wide..(k + 1) * wide], |poke| {
                self.active(pos, perspective, |i, x| poke(i, x));
            });
        }
    }
    /// `before` moved by the input changes `fill` pokes in (input, amount);
    /// the change buffer is zero again afterwards.
    fn update(&self, before: &[f32], after: &mut [f32], fill: impl FnOnce(&mut dyn FnMut(usize, f32))) {
        let [input, change, output] = &self.update.values;
        let mut touched = self.touched.borrow_mut();
        touched.clear();
        change.edit(|change: &mut [f32]| {
            // An input changed twice (a count, by two pieces) sums.
            fill(&mut |i, x| {
                change[i] += x;
                touched.push(i as u32);
            });
        });
        input.write(before).expect("the accumulator");
        self.update.run();
        output.read_into(after).expect("the accumulator");
        change.edit(|change: &mut [f32]| {
            for &i in touched.iter() {
                change[i as usize] = 0.;
            }
        });
    }
    /// The value for the side to move whose accumulator is `acc`.
    fn head(&self, acc: &[f32]) -> f32 {
        let [input, value] = &self.head.values;
        input.write(acc).expect("the accumulator");
        self.head.run();
        value.read_scalar().expect("the value")
    }
    /// Putting piece `p` on square `s` (`sign` 1) or lifting it (−1), as each
    /// perspective's inputs: the piece and its count.
    fn toggle(&self, changes: &mut Changes, p: i8, s: i32, sign: f32) {
        for (k, perspective) in [1i8, -1].into_iter().enumerate() {
            let f = Self::feature(p, s, perspective);
            changes.push(k, f, sign);
            changes.push(k, 782 + f / 64, sign * self.count);
        }
    }
    /// The pieces move `m` lifts and puts down in `pos` (before `make`). A
    /// piece that only changes square keeps its count.
    fn moved(&self, changes: &mut Changes, pos: &Position, m: u32) {
        let (f, t, fl) = (move_from(m), move_to(m), flags(m));
        let side = pos.side;
        let piece = pos.board[f as usize];
        let slide = |changes: &mut Changes, p: i8, from: i32, to: i32| {
            for (k, perspective) in [1i8, -1].into_iter().enumerate() {
                changes.push(k, Self::feature(p, from, perspective), -1.);
                changes.push(k, Self::feature(p, to, perspective), 1.);
            }
        };
        let captured = pos.board[t as usize];
        if captured != 0 {
            self.toggle(changes, captured, t, -1.);
        }
        let p = promotion(m);
        if p > 0 {
            self.toggle(changes, piece, f, -1.);
            self.toggle(changes, side * p, t, 1.);
        } else {
            slide(changes, piece, f, t);
        }
        if fl & 1 != 0 {
            self.toggle(changes, -side, t - side as i32 * 16, -1.);
        }
        if fl & 2 != 0 {
            let (rf, rt) = if t > f { (f + 3, f + 1) } else { (f - 4, f - 1) };
            slide(changes, side * 4, rf, rt);
        }
    }
    /// The non-piece inputs that differ between `before` (rights, en passant,
    /// halfmove, ply) and `pos`.
    fn drifted(changes: &mut Changes, before: (u32, i32, u32, u32), pos: &Position) {
        for (k, perspective) in [1i8, -1].into_iter().enumerate() {
            let (was, now) = (Self::rights(before.0, perspective), Self::rights(pos.rights, perspective));
            for bit in 0..4 {
                if (was ^ now) >> bit & 1 == 1 {
                    changes.push(k, 768 + bit, if now >> bit & 1 == 1 { 1. } else { -1. });
                }
            }
            if before.1 != pos.ep {
                if before.1 >= 0 {
                    changes.push(k, 772 + (before.1 & 7) as usize, -1.);
                }
                if pos.ep >= 0 {
                    changes.push(k, 772 + (pos.ep & 7) as usize, 1.);
                }
            }
            let halfmove = Self::clock(pos.halfmove, 100.) - Self::clock(before.2, 100.);
            if halfmove != 0. {
                changes.push(k, 780, halfmove);
            }
            let ply = Self::clock(pos.ply, 200.) - Self::clock(before.3, 200.);
            if ply != 0. {
                changes.push(k, 781, ply);
            }
        }
    }
    /// Diagnostic: the largest difference, over every legal move from `pos`,
    /// between the value from updated accumulators and one from scratch.
    pub fn incremental_error(&self, pos: &mut Position) -> f32 {
        let wide = self.wide();
        let mut before = vec![0f32; 2 * wide];
        self.refresh(pos, &mut before);
        let mut after = vec![0f32; wide];
        let mut worst = 0f32;
        for m in pos.moves() {
            let mut changes = Changes::default();
            self.moved(&mut changes, pos, m);
            let clocks = (pos.rights, pos.ep, pos.halfmove, pos.ply);
            let undo = pos.make(m);
            Self::drifted(&mut changes, clocks, pos);
            let k = usize::from(pos.side != 1);
            self.update(&before[k * wide..(k + 1) * wide], &mut after, |poke| changes.poke(k, poke));
            worst = worst.max((self.head(&after) - self.evaluate(pos)).abs());
            pos.unmake(m, &undo);
        }
        worst
    }
    /// The value of `pos` for its side to move, from scratch.
    pub fn evaluate(&self, pos: &Position) -> f32 {
        let wide = self.wide();
        let mut acc = vec![0f32; 2 * wide];
        self.refresh(pos, &mut acc);
        let at = if pos.side == 1 { 0 } else { wide };
        self.head(&acc[at..at + wide])
    }
    /// What the linear path says the mover gains by m: minus the change of the
    /// opponent's-perspective linear sum. Learned values only.
    fn gain(&self, pos: &Position, m: u32) -> f32 {
        let (f, t, fl) = (move_from(m), move_to(m), flags(m));
        let side = pos.side;
        let w = |p: i8, s: i32| {
            let f = Self::feature(p, s, -side);
            self.linear[f] + self.count * self.linear[782 + f / 64]
        };
        let piece = pos.board[f as usize];
        let captured = pos.board[t as usize];
        let p = promotion(m);
        let mut change = w(if p > 0 { side * p } else { piece }, t) - w(piece, f);
        if captured != 0 {
            change -= w(captured, t);
        }
        if fl & 1 != 0 {
            change -= w(-side, t - side as i32 * 16);
        }
        if fl & 2 != 0 {
            let (rf, rt) = if t > f { (f + 3, f + 1) } else { (f - 4, f - 1) };
            change += w(side * 4, rt) - w(side * 4, rf);
        }
        -change
    }
    /// The value of `pos` for its side to move.
    pub fn evaluate_position(&self, pos: &Position) -> f32 {
        self.evaluate(pos)
    }
    /// Diagnostic: every program's launches, each timed in microseconds.
    pub fn profile(&self) -> Vec<(&'static str, [u32; 3], f64)> {
        let mut out = Vec::new();
        for (name, program) in [("update", &self.update.program), ("head", &self.head.program)] {
            out.push((name, [0; 3], 0.));
            out.extend(program.profile().expect("the value network runs"));
        }
        out
    }
}

/// A copy with its own program, for another thread.
impl Clone for Net {
    fn clone(&self) -> Self {
        let mut net = Self::new(self.config);
        net.set_parameters(&self.parameters());
        net.count = self.count;
        net
    }
}
/// The model's 832 inputs for `pos`, from the side to move (chess.wgsl's feature).
pub fn features(pos: &Position) -> Vec<f32> {
    let mut x = vec![0f32; 832];
    for s in (0..128).filter(|&s| valid(s)) {
        let p = pos.board[s as usize];
        if p != 0 {
            let f = Net::feature(p, s, pos.side);
            x[f] = 1.;
            x[782 + f / 64] += crate::count_scale();
        }
    }
    let r = if pos.side == 1 { pos.rights } else { ((pos.rights & 3) << 2) | ((pos.rights & 12) >> 2) };
    for bit in 0..4 {
        x[768 + bit] = (r >> bit & 1) as f32;
    }
    if pos.ep >= 0 {
        x[772 + (pos.ep & 7) as usize] = 1.;
    }
    x[780] = (pos.halfmove as f32 / 100.).min(1.);
    x[781] = (pos.ply as f32 / 200.).min(1.);
    x[831] = 1.;
    x
}
/// Diagnostic: how well `net` tracks reference scores (centipawns for the side
/// to move): the correlation of its value with tanh(cp / 400), over all `rows`
/// and over the balanced ones (within three pawns), where material says little.
pub fn agreement(net: &Net, rows: &[(Vec<u32>, i32)]) -> (f32, f32) {
    let pairs: Vec<(f32, f32, bool)> = rows
        .iter()
        .map(|(state, cp)| {
            let mut padded = state.clone();
            padded.resize(WORDS, 0);
            let value = net.evaluate_position(&Position::from_state(&padded));
            (value, (*cp as f32 / 400.).tanh(), cp.abs() <= 300)
        })
        .collect();
    let correlation = |balanced: bool| {
        let chosen: Vec<(f32, f32)> = pairs.iter().filter(|p| !balanced || p.2).map(|p| (p.0, p.1)).collect();
        let n = chosen.len() as f32;
        let (mx, my) = (chosen.iter().map(|p| p.0).sum::<f32>() / n, chosen.iter().map(|p| p.1).sum::<f32>() / n);
        let (mut xy, mut xx, mut yy) = (0., 0., 0.);
        for (x, y) in chosen {
            xy += (x - mx) * (y - my);
            xx += (x - mx) * (x - mx);
            yy += (y - my) * (y - my);
        }
        xy / (xx * yy).sqrt().max(1e-12)
    };
    (correlation(false), correlation(true))
}
/// One searched move for the UI: the move, its search score for the mover and
/// its share of a softmax over the root scores.
pub struct Branch {
    pub movement: u32,
    pub score: f32,
    pub share: f32,
}
pub struct Found {
    pub movement: u32,
    pub value: f32,
    pub depth: u32,
    pub nodes: u64,
    pub branches: Vec<Branch>,
}
/// A transposition table, reusable across the moves of a game.
pub struct Table {
    entries: Vec<Entry>,
}
impl Table {
    pub fn new() -> Self {
        Self { entries: vec![Entry::default(); TABLE] }
    }
}
impl Default for Table {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    depth: i32,
    value: f32,
    bound: u8,
    movement: u32,
}
/// Deepest leaf extension: two plies won +147 Elo at equal time where eight
/// plies (too many nodes) lost 150; three plies another +34 over 600 games.
const QUIESCE_PLIES: i32 = 3;
/// Leaves extend along moves the model's linear values rate above this.
const QUIESCE_MARGIN: f32 = 0.05;
/// Frontier futility: a move is searched only if it could still reach alpha.
const FUTILITY_MARGIN: f32 = 0.1;
/// Transposition-table entries (a power of two).
const TABLE: usize = 1 << 18;
const EXACT: u8 = 0;
const LOWER: u8 = 1;
const UPPER: u8 = 2;

/// Iterative-deepening alpha-beta over `net`, as the GPU search does it: TT,
/// killers, history, PVS, rules for terminal positions, the model everywhere
/// else, and early stopping on the model's own root scores.
pub struct Search<'a> {
    net: &'a Net,
    pos: Position,
    /// Per ply of the line being searched (`top` is the current position):
    /// both perspectives' accumulators (White's, then Black's), which of them
    /// have been computed, and the input changes of the move that led there.
    /// An accumulator is computed when a position below it is evaluated.
    stack: Vec<f32>,
    computed: Vec<[bool; 2]>,
    changes: Vec<Changes>,
    top: usize,
    history_keys: Vec<u64>,
    path: Vec<u64>,
    table: Vec<Entry>,
    killers: Vec<[u32; 2]>,
    history: Vec<u32>,
    nodes: u64,
    deadline: f64,
    clock: &'a dyn Fn() -> f64,
    stopped: bool,
    max_depth: i32,
    node_limit: u64,
    limited: bool,
    margin: f32,
    lmr: bool,
    null_move: bool,
    last_null: bool,
    quiesce_plies: i32,
    early_stop: bool,
    aspiration: bool,
    gain_order: bool,
    check_extension: bool,
    futility: bool,
}
impl<'a> Search<'a> {
    /// `state` is a game as chess.wgsl stores it, including its key history.
    pub fn new(net: &'a Net, state: &[u32], clock: &'a dyn Fn() -> f64) -> Self {
        Self::with_table(net, state, clock, Table::new())
    }
    /// As `new`, reusing a transposition table from earlier searches: entries
    /// are keyed by position, so those of the same game stay valid.
    pub fn with_table(net: &'a Net, state: &[u32], clock: &'a dyn Fn() -> f64, table: Table) -> Self {
        let pos = Position::from_state(state);
        let history_keys = (0..=pos.ply.min(240) as usize)
            .map(|i| (state[225 + i * 2] as u64) << 32 | state[224 + i * 2] as u64)
            .collect();
        let wide = net.wide();
        let mut stack = vec![0f32; 2 * wide];
        net.refresh(&pos, &mut stack);
        Self {
            net,
            pos,
            stack,
            computed: vec![[true; 2]],
            changes: vec![Changes::default()],
            top: 0,
            history_keys,
            path: Vec::new(),
            table: table.entries,
            killers: vec![[0; 2]; 128],
            history: vec![0; 64 * 64],
            nodes: 0,
            deadline: 0.,
            clock,
            stopped: false,
            max_depth: 64,
            node_limit: u64::MAX,
            limited: false,
            lmr: true,
            null_move: true,
            last_null: false,
            quiesce_plies: QUIESCE_PLIES,
            // Measured at equal time: searching until time is up +38, ordering
            // quiet moves by the model's values +27, aspiration windows ±0.
            early_stop: false,
            aspiration: false,
            gain_order: true,
            // Check extension −17 at equal time (not adopted); frontier futility +81.
            check_extension: false,
            futility: true,
            margin: QUIESCE_MARGIN,
        }
    }
    fn history_slot(m: u32) -> usize {
        let square = |s: i32| ((s >> 4) * 8 + (s & 7)) as usize;
        square(move_from(m)) * 64 + square(move_to(m))
    }
    fn order(&self, moves: &mut [u32], ply: usize, tt: u32) {
        let key = |m: &u32| -> i64 {
            if *m == tt {
                i64::MAX
            } else if self.killers[ply][0] == *m {
                i64::MAX - 1
            } else if self.killers[ply][1] == *m {
                i64::MAX - 2
            } else if self.gain_order {
                // The model's own linear values first (what they reward sorts
                // first), history breaking ties.
                (self.net.gain(&self.pos, *m) * 1e8) as i64 + self.history[Self::history_slot(*m)] as i64
            } else {
                self.history[Self::history_slot(*m)] as i64
            }
        };
        moves.sort_by_cached_key(|m| (std::cmp::Reverse(key(m)), *m));
    }
    fn clocks(&self) -> (u32, i32, u32, u32) {
        (self.pos.rights, self.pos.ep, self.pos.halfmove, self.pos.ply)
    }
    /// Enter the position just reached: its input changes from the parent are
    /// `changes` and what else differs from `before`.
    fn advance(&mut self, mut changes: Changes, before: (u32, i32, u32, u32)) {
        Net::drifted(&mut changes, before, &self.pos);
        self.top += 1;
        if self.changes.len() == self.top {
            self.changes.push(changes);
            self.computed.push([false; 2]);
            self.stack.resize((self.top + 1) * 2 * self.net.wide(), 0.);
        } else {
            self.changes[self.top] = changes;
            self.computed[self.top] = [false; 2];
        }
    }
    fn make(&mut self, m: u32) -> Undo {
        let mut changes = Changes::default();
        self.net.moved(&mut changes, &self.pos, m);
        let before = self.clocks();
        let undo = self.pos.make(m);
        self.advance(changes, before);
        undo
    }
    fn unmake(&mut self, m: u32, undo: &Undo) {
        self.top -= 1;
        self.pos.unmake(m, undo);
    }
    /// The model's value of the current position for its side to move.
    fn value(&mut self) -> f32 {
        let wide = self.net.wide();
        let k = usize::from(self.pos.side != 1);
        let slot = |level: usize| (level * 2 + k) * wide;
        if !self.computed[self.top][k] {
            // Down from the nearest computed accumulator, a move at a time, so
            // every position on the line keeps its own for its other children.
            let from = (0..self.top).rev().find(|&level| self.computed[level][k]).unwrap_or(0);
            for level in from + 1..=self.top {
                let (below, above) = self.stack.split_at_mut(slot(level));
                let changes = &self.changes[level];
                self.net.update(&below[slot(level - 1)..slot(level - 1) + wide], &mut above[..wide], |poke| {
                    changes.poke(k, poke)
                });
                self.computed[level][k] = true;
            }
        }
        let value = self.net.head(&self.stack[slot(self.top)..slot(self.top) + wide]);
        debug_assert!(
            (value - self.net.evaluate(&self.pos)).abs() < 1e-4,
            "the accumulators drifted from the position"
        );
        value
    }
    /// Count a node and notice the time or node limit.
    fn tick(&mut self) -> bool {
        self.nodes += 1;
        if self.limited
            && (self.nodes >= self.node_limit
                || (self.nodes % 1024 == 0 && (self.clock)() >= self.deadline))
        {
            self.stopped = true;
        }
        self.stopped
    }
    /// Leaf search: the model's value, extended along the moves that the model's
    /// own linear values say gain more than `margin`, until the position is
    /// quiet by its own account. Replies to check are always searched.
    fn quiesce(&mut self, mut alpha: f32, beta: f32, ply: usize, left: i32) -> f32 {
        if self.tick() {
            return 0.;
        }
        if self.pos.halfmove >= 100 || self.pos.material_draw() {
            return 0.;
        }
        let moves = self.pos.moves();
        let checked = self.pos.in_check();
        if moves.is_empty() {
            return if checked { -MATE + ply as f32 * 0.001 } else { 0. };
        }
        let stand = self.value();
        if left == 0 || ply >= 120 {
            return stand;
        }
        let mut best = -INF;
        if !checked {
            if stand >= beta {
                return stand;
            }
            best = stand;
            alpha = alpha.max(stand);
        }
        let mut candidates: Vec<(f32, u32)> = moves
            .iter()
            .map(|&m| (self.net.gain(&self.pos, m), m))
            .filter(|(gain, _)| checked || *gain > self.margin)
            .collect();
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        for (_, m) in candidates {
            let undo = self.make(m);
            let score = -self.quiesce(-beta, -alpha, ply + 1, left - 1);
            self.unmake(m, &undo);
            if self.stopped {
                return 0.;
            }
            best = best.max(score);
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        best
    }
    fn negamax(&mut self, depth: i32, mut alpha: f32, beta: f32, ply: usize) -> f32 {
        if depth <= 0 && self.margin.is_finite() {
            return self.quiesce(alpha, beta, ply, self.quiesce_plies);
        }
        if self.tick() {
            return 0.;
        }
        let key = self.pos.key();
        if ply > 0
            && (self.pos.halfmove >= 100
                || self.pos.material_draw()
                || self.history_keys.contains(&key)
                || self.path.contains(&key))
        {
            return 0.;
        }
        if depth <= 0 || ply >= 120 {
            // A leaf needs only to know whether the rules end the game here.
            if !self.pos.has_move() {
                return if self.pos.in_check() { -MATE + ply as f32 * 0.001 } else { 0. };
            }
            return self.value();
        }
        let mut moves = self.pos.moves();
        if moves.is_empty() {
            return if self.pos.in_check() { -MATE + ply as f32 * 0.001 } else { 0. };
        }
        let mut tt = 0;
        let e = self.table[key as usize & (TABLE - 1)];
        if e.key == key && e.movement != 0 {
            tt = e.movement;
            if ply > 0
                && e.depth >= depth
                && (e.bound == EXACT || (e.bound == LOWER && e.value >= beta) || (e.bound == UPPER && e.value <= alpha))
            {
                return e.value;
            }
        }
        self.order(&mut moves, ply, tt);
        let checked = self.pos.in_check();
        // Check extension: a king in check is searched a ply deeper.
        let depth = if self.check_extension && checked { depth + 1 } else { depth };
        // Null move: if passing still fails high, so would the best move. Skipped
        // in check, right after another null move, and with little mobility
        // (where passing could be forced to be worse).
        if self.null_move && depth >= 3 && ply > 0 && !checked && !self.last_null && moves.len() >= 6 && beta < 1.5 {
            let ep = self.pos.ep;
            let before = self.clocks();
            self.pos.side = -self.pos.side;
            self.pos.ep = -1;
            self.pos.ply += 1;
            self.advance(Default::default(), before);
            self.last_null = true;
            let score = -self.negamax(depth - 3, -beta, -beta + 1e-4, ply + 1);
            self.last_null = false;
            self.top -= 1;
            self.pos.ply -= 1;
            self.pos.ep = ep;
            self.pos.side = -self.pos.side;
            if self.stopped {
                return 0.;
            }
            if score >= beta {
                return beta;
            }
        }
        let alpha0 = alpha;
        let (mut best, mut best_move) = (-INF, moves[0]);
        self.path.push(key);
        // Futility at the frontier: the model's static value plus what it says
        // the move gains, plus a margin, must reach alpha to be worth a search.
        let futile = self.futility && depth == 1 && !checked && alpha.abs() < 1.5;
        let stand = if futile { self.value() } else { 0. };
        for (i, &m) in moves.iter().enumerate() {
            if futile && i > 0 && stand + self.net.gain(&self.pos, m) + FUTILITY_MARGIN <= alpha {
                continue;
            }
            let undo = self.make(m);
            let mut score;
            if i == 0 {
                score = -self.negamax(depth - 1, -beta, -alpha, ply + 1);
            } else {
                // Late moves are searched a ply shallower first; a surprise
                // (beating alpha) earns the full depth.
                let reduced = self.lmr && depth >= 3 && i >= 3 && !checked;
                let d = if reduced { depth - 2 } else { depth - 1 };
                score = -self.negamax(d, -alpha - 1e-4, -alpha, ply + 1);
                if reduced && score > alpha && !self.stopped {
                    score = -self.negamax(depth - 1, -alpha - 1e-4, -alpha, ply + 1);
                }
                if score > alpha && score < beta && !self.stopped {
                    score = -self.negamax(depth - 1, -beta, -alpha, ply + 1);
                }
            }
            self.unmake(m, &undo);
            if self.stopped {
                self.path.pop();
                return 0.;
            }
            if score > best {
                best = score;
                best_move = m;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                if self.killers[ply][0] != m {
                    self.killers[ply] = [m, self.killers[ply][0]];
                }
                let h = &mut self.history[Self::history_slot(m)];
                *h = (*h + (depth * depth) as u32).min(1 << 20);
                break;
            }
        }
        self.path.pop();
        if best.abs() < 1.5 {
            let bound = if best <= alpha0 { UPPER } else if best >= beta { LOWER } else { EXACT };
            self.table[key as usize & (TABLE - 1)] = Entry { key, depth, value: best, bound, movement: best_move };
        }
        best
    }
    /// Search for at most `millis`; `None` when the position has no legal move.
    /// Diagnostic: search to exactly `depth` (or an earlier early stop), untimed.
    pub fn think_depth(&mut self, depth: u32) -> Option<Found> {
        self.max_depth = depth as i32;
        self.think(f64::INFINITY)
    }
    /// Take the transposition table back, for the next search.
    pub fn into_table(self) -> Table {
        Table { entries: self.table }
    }
    /// Diagnostic: search until about `nodes` positions have been visited.
    pub fn think_nodes(&mut self, nodes: u64) -> Option<Found> {
        self.node_limit = nodes;
        self.think(f64::INFINITY)
    }
    /// Extend leaves along moves the model's linear values rate above `margin`
    /// (infinite: leaves are scored as they stand).
    pub fn quiescence(mut self, margin: f32) -> Self {
        self.margin = margin;
        self
    }
    /// Diagnostic: stop early once the best move has settled, and open each
    /// depth with a narrow window around the previous score.
    pub fn root_options(mut self, early_stop: bool, aspiration: bool) -> Self {
        self.early_stop = early_stop;
        self.aspiration = aspiration;
        self
    }
    /// Diagnostic: check extension and frontier futility pruning.
    pub fn pruning_extras(mut self, check_extension: bool, futility: bool) -> Self {
        self.check_extension = check_extension;
        self.futility = futility;
        self
    }
    /// Diagnostic: order quiet moves by the model's linear values.
    pub fn gain_ordering(mut self, on: bool) -> Self {
        self.gain_order = on;
        self
    }
    /// Diagnostic: cap the leaf extension depth.
    pub fn quiescence_depth(mut self, plies: i32) -> Self {
        self.quiesce_plies = plies;
        self
    }
    /// Diagnostic: switch late-move reductions and null-move pruning.
    pub fn pruning(mut self, lmr: bool, null_move: bool) -> Self {
        self.lmr = lmr;
        self.null_move = null_move;
        self
    }
    pub fn think(&mut self, millis: f64) -> Option<Found> {
        self.deadline = (self.clock)() + millis;
        let mut root = self.pos.moves();
        if root.is_empty() {
            return None;
        }
        self.order(&mut root, 0, 0);
        let mut scores = vec![0f32; root.len()];
        let (mut depth, mut completed) = (1, 0);
        let (mut held, mut previous) = (0, 0u32);
        loop {
            // Depth 1 always completes, so every root move has a score.
            self.limited = depth > 1;
            let mut running = vec![-INF; root.len()];
            // Aspiration: the first move opens with a window around the last
            // depth's score; a score outside it is searched again in full.
            let aspire = self.aspiration && depth >= 3;
            let (low, high) = if aspire { (scores[0] - 0.1, scores[0] + 0.1) } else { (-INF, INF) };
            let mut alpha = low;
            let key = self.pos.key();
            self.path.push(key);
            for (i, &m) in root.iter().enumerate() {
                let undo = self.make(m);
                let mut score = if i == 0 {
                    -self.negamax(depth - 1, -high, -alpha, 1)
                } else {
                    -self.negamax(depth - 1, -alpha - 1e-4, -alpha, 1)
                };
                if i == 0 && aspire && (score <= low || score >= high) && !self.stopped {
                    alpha = -INF;
                    score = -self.negamax(depth - 1, -INF, INF, 1);
                }
                if i > 0 && score > alpha && !self.stopped {
                    score = -self.negamax(depth - 1, -INF, -alpha, 1);
                }
                self.unmake(m, &undo);
                if self.stopped {
                    break;
                }
                running[i] = score;
                alpha = alpha.max(score);
            }
            self.path.pop();
            if self.stopped {
                break;
            }
            scores = running;
            completed = depth;
            // Best-first for the next depth.
            let mut order: Vec<usize> = (0..root.len()).collect();
            order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
            root = order.iter().map(|&i| root[i]).collect();
            scores = order.iter().map(|&i| scores[i]).collect();
            let best = scores[0];
            let second = scores.get(1).copied().unwrap_or(-INF);
            held = if root[0] == previous { held + 1 } else { 1 };
            previous = root[0];
            // Early stop on the model's own scores, as the GPU search does.
            let settled = self.early_stop && depth >= 3 && held >= 3 && best - second >= 0.1;
            if root.len() == 1 || (self.early_stop && best.abs() >= 0.95) || settled || depth >= self.max_depth {
                break;
            }
            depth += 1;
        }
        let best = scores.iter().cloned().fold(-INF, f32::max);
        let total: f32 = scores.iter().map(|s| ((s - best) / 0.05).exp()).sum();
        let mut branches: Vec<Branch> = root
            .iter()
            .zip(&scores)
            .map(|(&m, &s)| Branch { movement: m, score: s, share: ((s - best) / 0.05).exp() / total })
            .collect();
        branches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Some(Found { movement: branches[0].movement, value: best, depth: completed.max(1) as u32, nodes: self.nodes, branches })
    }
}

/// The start position as a game state.
pub fn start_state() -> Vec<u32> {
    let mut start = vec![0u32; WORDS];
    let back = [4i32, 2, 3, 5, 6, 3, 2, 4];
    for f in 0..8 {
        start[f] = back[f] as u32;
        start[16 + f] = 1;
        start[96 + f] = -1i32 as u32;
        start[112 + f] = (-back[f]) as u32;
    }
    start[128] = 1;
    start[129] = 15;
    start[130] = -1i32 as u32;
    start[133] = 4;
    start[134] = 116;
    start[135] = 1;
    start
}
/// One self-play game with the CPU search: `random_plies` random opening moves,
/// then `nodes` per move, sampling early moves by the softmax share of root
/// scores. Returns (features, value target for the side to move) per searched
/// position, the target blending `result_weight` of the game result into the
/// search value, plus the last position and the result for White.
pub struct SelfPlayGame {
    pub rows: Vec<(Vec<f32>, f32)>,
    pub last: Vec<u32>,
    pub result: i32,
    pub plies: u32,
}
/// With `lambda` > 0 the target is a λ-return instead: from the game's end,
/// R_t = (1 - λ) v_t + λ R_{t+1} with R after the last position the result, so
/// each label carries the lookahead of the searches that followed it.
pub fn self_play_game(
    net: &Net,
    nodes: u64,
    result_weight: f32,
    lambda: f32,
    random_plies: u32,
    random: &mut dyn FnMut() -> u32,
    clock: &dyn Fn() -> f64,
    table: &mut Table,
) -> SelfPlayGame {
    let mut pos = Position::from_state(&start_state());
    let mut keys = vec![pos.key()];
    let mut rows: Vec<(Vec<f32>, f32, i8)> = Vec::new();
    let result = loop {
        let legal = pos.moves();
        if legal.is_empty() {
            break if pos.in_check() { -(pos.side as i32) } else { 0 };
        }
        let key = pos.key();
        if pos.halfmove >= 100 || pos.material_draw() || keys.iter().filter(|&&k| k == key).count() >= 3 || pos.ply >= 240 {
            break 0;
        }
        let m = if pos.ply < random_plies {
            legal[random() as usize % legal.len()]
        } else {
            let state = state_words(&pos, &keys);
            let mut search = Search::with_table(net, &state, clock, std::mem::take(table));
            let found = search.think_nodes(nodes).unwrap();
            *table = search.into_table();
            // Mate scores (±2) lie outside the value range; a target the model
            // cannot reach only grows its weights until the output saturates.
            rows.push((features(&pos), found.value.clamp(-1., 1.), pos.side));
            if pos.ply < 20 {
                let mut r = random() as f32 / 4294967296.;
                let mut pick = found.movement;
                for b in &found.branches {
                    r -= b.share;
                    if r <= 0. {
                        pick = b.movement;
                        break;
                    }
                }
                pick
            } else {
                found.movement
            }
        };
        let _ = pos.make(m);
        keys.push(pos.key());
    };
    let plies = pos.ply;
    let last = state_words(&pos, &keys);
    let labeled = if lambda > 0. {
        // λ-return, in White's view, from the end of the game.
        let mut targets = vec![0f32; rows.len()];
        let mut ahead = result as f32;
        for (i, (_, v, side)) in rows.iter().enumerate().rev() {
            ahead = (1. - lambda) * v * *side as f32 + lambda * ahead;
            targets[i] = ahead;
        }
        rows.into_iter().zip(targets).map(|((x, _, side), t)| (x, t * side as f32)).collect()
    } else {
        rows.into_iter()
            .map(|(x, v, side)| (x, (1. - result_weight) * v + result_weight * result as f32 * side as f32))
            .collect()
    };
    SelfPlayGame { rows: labeled, last, result, plies }
}
/// A game state as chess.wgsl stores it, from a position and its key history
/// (the search reads repetitions from words 224..).
pub fn state_words(pos: &Position, keys: &[u64]) -> Vec<u32> {
    let mut state = vec![0u32; WORDS];
    for (i, b) in pos.board.iter().enumerate() {
        state[i] = *b as i32 as u32;
    }
    state[128] = pos.side as i32 as u32;
    state[129] = pos.rights;
    state[130] = pos.ep as u32;
    state[131] = pos.halfmove;
    state[132] = pos.ply;
    state[133] = pos.kings[0] as u32;
    state[134] = pos.kings[1] as u32;
    state[135] = 1;
    for (i, k) in keys.iter().enumerate().take(241) {
        state[224 + i * 2] = *k as u32;
        state[225 + i * 2] = (*k >> 32) as u32;
    }
    state
}
/// The UI's answer layout (as `commit` fills a game): the input state with the
/// chosen move (word 136), the value for White (152), total pseudo-visits (153),
/// depth (154), nodes (155) and the top five moves at 736 (move, visits,
/// score for the mover, share).
pub fn answer(state: &[u32], found: &Found) -> Vec<u32> {
    let mut out = state[..WORDS].to_vec();
    let side = state[128] as i32 as f32;
    out[136] = found.movement;
    out[152] = (found.value * side).to_bits();
    out[153] = 1000;
    out[154] = found.depth;
    out[155] = found.nodes.min(u32::MAX as u64) as u32;
    for (rank, b) in found.branches.iter().take(5).enumerate() {
        let at = 736 + rank * 4;
        out[at] = b.movement;
        out[at + 1] = (b.share * 1000.).round() as u32;
        out[at + 2] = b.score.to_bits();
        out[at + 3] = b.share.to_bits();
    }
    out
}
