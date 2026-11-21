import { writeFileSync } from 'node:fs';
import { Position, result } from '../src/engine/chess';
// Test oracle only: a material reference to grade the engine's moves against.
const MATERIAL = [0, 1, 3.2, 3.35, 5, 9.5, 0];
import { encodeHuman } from '../src/engine/gpu-state';

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

const rows = fixtures.map((fen) => {
  const p = new Position(fen),
    scores: Record<number, number> = {};
  let best = -Infinity;
  for (const m of p.legalMoves()) {
    const u = p.make(m),
      v = -reference(p, 1);
    p.unmake(m, u);
    const word = m.from | (m.to << 7) | ((m.promotion || 0) << 14) | (m.flags << 17);
    scores[word] = v;
    best = Math.max(best, v);
  }
  const regrets = Object.fromEntries(
    Object.entries(scores).map(([move, v]) => [move, Math.min(10, Math.max(0, best - v))]),
  );
  return { fen, state: Array.from(encodeHuman(fen, [fen], 1)), regrets };
});
writeFileSync('/private/tmp/rookie-strength-fixtures.json', JSON.stringify(rows));
console.log(`${rows.length} held-out tactical positions written`);
