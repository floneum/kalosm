import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Chess } from 'chess.js';
import { Position, moveKey, notation, result, parseSquare } from '../src/engine/chess';
import { PARAM_COUNT, TinyModel, validateWeights } from './legacy/model';
import { search } from './legacy/search';

function perft(p: Position, depth: number): number {
  if (!depth) return 1;
  let count = 0;
  for (const move of p.legalMoves()) {
    const undo = p.make(move);
    count += perft(p, depth - 1);
    p.unmake(move, undo);
  }
  return count;
}
function rng(seed = 17) {
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
    return (seed >>> 0) / 4294967296;
  };
}

test('starting position perft through depth four and reversible make/unmake', () => {
  const p = new Position(),
    fen = p.fen();
  assert.deepEqual(
    [1, 2, 3, 4].map((depth) => perft(p, depth)),
    [20, 400, 8902, 197281],
  );
  assert.equal(p.fen(), fen);
});

test('castling, pins, en passant and promotion perft positions', () => {
  const positions: [string, number[]][] = [
    ['r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1', [48, 2039, 97862]],
    ['8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1', [14, 191, 2812]],
    ['r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1', [6, 264, 9467]],
  ];
  for (const [fen, expected] of positions) {
    const p = new Position(fen);
    assert.deepEqual(
      expected.map((_, i) => perft(p, i + 1)),
      expected,
      fen,
    );
    assert.equal(p.fen(), fen);
  }
});

test('legal moves, FEN, SAN and outcomes match chess.js across seeded games', () => {
  const random = rng();
  for (let game = 0; game < 24; game++) {
    const reference = new Chess(),
      p = new Position(),
      keys = [p.key()];
    for (let ply = 0; ply < 150; ply++) {
      const moves = p.legalMoves();
      const refMoves = reference.moves({ verbose: true });
      assert.deepEqual(
        moves.map(moveKey).sort(),
        refMoves.map((m) => m.from + m.to + (m.promotion || '')).sort(),
        p.fen(),
      );
      const canonicalFen = p.fen().split(' ');
      if (!moves.some((m) => m.flags & 1)) canonicalFen[3] = '-';
      assert.equal(canonicalFen.join(' '), reference.fen());
      assert.equal(!!result(p, keys), reference.isGameOver(), p.fen());
      if (reference.isGameOver()) break;
      const move = moves[Math.floor(random() * moves.length)];
      const san = notation(p, move);
      const refMove = reference.move({
        from: moveKey(move).slice(0, 2),
        to: moveKey(move).slice(2, 4),
        promotion: moveKey(move).slice(4) || undefined,
      });
      assert.equal(san, refMove.san);
      p.make(move);
      keys.push(p.key());
    }
  }
});

test('repetition normalizes uncapturable and pinned en passant squares', () => {
  const pinned = new Position('k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1');
  const none = new Position('k3r3/8/8/3pP3/8/8/8/4K3 w - - 0 1');
  assert.equal(pinned.key(), none.key());
  const p = new Position(),
    keys = [p.key()];
  for (const uci of ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8']) {
    p.make(p.legalMoves().find((m) => moveKey(m) === uci)!);
    keys.push(p.key());
  }
  assert.equal(result(p, keys)?.reason, 'Threefold repetition');
});

test('all four promotion choices and mate/stalemate outcomes', () => {
  const p = new Position('7k/P7/8/8/8/8/8/7K w - - 0 1');
  assert.equal(p.legalMoves().filter((m) => m.from === parseSquare('a7')).length, 4);
  assert.deepEqual(result(new Position('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1'), []), {
    winner: 1,
    reason: 'Checkmate',
  });
  assert.deepEqual(result(new Position('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'), []), {
    winner: 0,
    reason: 'Stalemate',
  });
});

test('search finds forced mate and leaves root intact', () => {
  const p = new Position('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'),
    before = p.fen();
  const answer = search(p, new TinyModel(), [p.key()], { simulations: 800 });
  assert.ok(answer.move);
  assert.equal(p.fen(), before);
  p.make(answer.move!);
  assert.equal(result(p, [])?.reason, 'Checkmate');
  assert.ok(answer.depth > 1);
  assert.ok(Math.abs(answer.policy.reduce((a, b) => a + b, 0) - 1) < 1e-6);
});

test('both model heads learn from search targets and serialize exactly', () => {
  const p = new Position('rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2');
  const model = new TinyModel(),
    moves = p.legalMoves();
  const policy = moves.map((_, i) => (i === 0 ? 1 : 0));
  const sample = { position: p, moves, policy, value: 0.35 };
  const beforeValue = model.value(p),
    beforePolicy = model.policy(p, moves)[0];
  for (let i = 0; i < 150; i++) model.train([sample]);
  assert.ok(Math.abs(model.value(p) - 0.35) < Math.abs(beforeValue - 0.35));
  assert.ok(model.policy(p, moves)[0] > beforePolicy * 2);
  assert.equal(model.weights.length, PARAM_COUNT);
  assert.deepEqual(new TinyModel(Array.from(model.weights)).weights, model.weights);
  assert.equal(validateWeights(Array.from(model.weights)), true);
  assert.equal(validateWeights([NaN]), false);
});

test('self-play collects genuine positions, learns, and remains finite', () => {
  const random = rng(44),
    model = new TinyModel(),
    original = model.weights.slice();
  const p = new Position(),
    keys = [p.key()];
  let nodes = 0;
  for (let i = 0; i < 80 && !result(p, keys); i++) {
    const answer = search(p, model, keys, { simulations: 48, exploration: true, random });
    assert.ok(answer.move);
    model.train([
      {
        position: p.clone(),
        moves: answer.moves,
        policy: answer.policy,
        value: answer.value * p.turn,
      },
    ]);
    p.make(answer.move!);
    keys.push(p.key());
    nodes += answer.nodes;
  }
  assert.ok(nodes > 100);
  assert.ok(model.weights.some((w, i) => w !== original[i]));
  assert.ok(model.weights.every(Number.isFinite));
});
