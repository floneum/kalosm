import assert from 'node:assert/strict';
import { Position, moveKey, result, notation } from '../../src/engine/chess';
import { TinyModel, MATERIAL } from './model';
import type { Sample } from './model';
import { search } from './search';

function random(seed: number) {
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
    return (seed >>> 0) / 4294967296;
  };
}
function score(p: Position) {
  let value = 0;
  for (let s = 0; s < 128; s++) {
    if (s & 0x88) {
      s += 7;
      continue;
    }
    const piece = p.board[s];
    if (piece) value += Math.sign(piece) * MATERIAL[Math.abs(piece)];
  }
  return value * p.turn;
}
function reference(p: Position, depth: number, alpha = -10000, beta = 10000): number {
  const moves = p.legalMoves();
  if (!moves.length) return p.inCheck() ? -1000 - depth : 0;
  if (p.insufficient() || p.half >= 100) return 0;
  const checked = p.inCheck();
  if (depth <= 0 && !checked) {
    const stand = score(p);
    if (stand >= beta || depth <= -4) return stand;
    alpha = Math.max(alpha, stand);
  }
  const candidates =
    depth > 0 || checked ? moves : moves.filter((m) => p.board[m.to] || m.flags & 1 || m.promotion);
  candidates.sort(
    (a, b) =>
      MATERIAL[Math.abs(p.board[b.to])] * 10 -
      MATERIAL[Math.abs(p.board[b.from])] -
      (MATERIAL[Math.abs(p.board[a.to])] * 10 - MATERIAL[Math.abs(p.board[a.from])]),
  );
  if (depth <= -6) return score(p);
  for (const move of candidates) {
    const undo = p.make(move),
      value = -reference(p, depth - 1, -beta, -alpha);
    p.unmake(move, undo);
    if (value >= beta) return value;
    alpha = Math.max(alpha, value);
  }
  return alpha;
}
const fixtures = [
  'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4',
  '6k1/8/2n5/4p3/3Q4/8/8/6K1 w - - 0 1',
  '6k1/5ppp/8/8/8/3R4/5PPP/3r2K1 w - - 0 1',
  'r1bqk2r/pppp1ppp/2n2n2/2b1p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4',
  'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
  '3r2k1/5ppp/8/8/8/3Q4/5PPP/6K1 w - - 0 1',
  '4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1',
  'rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 2',
];
// Diverse, held-out middlegames. None are used by the training loop below.
const fixtureRandom = random(907);
for (let game = 0; game < 4; game++) {
  const p = new Position();
  for (let ply = 0; ply < 64; ply++) {
    const moves = p.legalMoves();
    if (result(p, [], moves)) break;
    const captures = moves.filter((m) => p.board[m.to]);
    const candidates = captures.length && fixtureRandom() < 0.55 ? captures : moves;
    p.push(candidates[Math.floor(fixtureRandom() * candidates.length)]);
    if (ply >= 16 && ply % 8 === 0 && !result(p, [])) fixtures.push(p.fen());
  }
}
function grade(model: TinyModel) {
  let blunders = 0,
    regret = 0;
  const choices = [];
  for (const fen of fixtures) {
    const p = new Position(fen),
      answer = search(p, model, [p.key()], { simulations: 768 });
    let best = -Infinity,
      chosen = -Infinity;
    for (const m of p.legalMoves()) {
      const u = p.make(m),
        value = -reference(p, 1);
      p.unmake(m, u);
      best = Math.max(best, value);
      if (answer.move && moveKey(m) === moveKey(answer.move)) chosen = value;
    }
    const loss = Math.min(10, Math.max(0, best - chosen));
    if (loss >= 1) blunders++;
    regret += loss;
    if (loss > 0.9) choices.push({ fen, move: answer.move && notation(p, answer.move), loss });
  }
  return { blunders, regret, choices };
}
const model = new TinyModel(),
  before = grade(model),
  rng = random(31);
const arenas = Array.from({ length: 12 }, () => {
  const p = new Position();
  return { p, keys: [p.key()], samples: [] as Sample[] };
});
const replay: Sample[] = [];
let batch: Sample[] = [],
  index = 0;
for (let i = 0; i < 16000; i++) {
  const a = arenas[i % arenas.length];
  const thought = search(a.p, model, a.keys, { simulations: 48, exploration: true, random: rng });
  if (!thought.move) throw new Error('no move');
  const sample = {
    position: a.p.clone(),
    moves: thought.moves,
    policy: thought.policy,
    value: thought.value * a.p.turn,
  };
  batch.push(sample);
  a.samples.push(sample);
  if (replay.length < 4096) replay.push(sample);
  else replay[index++ % 4096] = sample;
  a.p.push(thought.move);
  a.keys.push(a.p.key());
  const outcome = result(a.p, a.keys);
  if (outcome || a.p.ply >= 200) {
    if (outcome) for (const s of a.samples) s.value = s.value * 0.5 + outcome.winner * 0.5;
    a.p = new Position();
    a.keys = [a.p.key()];
    a.samples = [];
  }
  if (batch.length === 32) {
    model.train([
      ...batch,
      ...Array.from({ length: 32 }, () => replay[Math.floor(rng() * replay.length)]),
    ]);
    batch = [];
  }
}
const after = grade(model);
console.log(JSON.stringify({ before, after }, null, 2));
assert.ok(
  after.regret < before.regret && after.blunders <= before.blunders,
  'Self-play must measurably reduce tactical mistakes, not merely change weights.',
);
