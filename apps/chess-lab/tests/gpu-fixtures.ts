import { writeFileSync } from 'node:fs';
import { Position, result } from '../src/engine/chess';
const special = [
  undefined,
  'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
  'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1',
  '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1',
  '4k3/P7/8/8/8/8/7p/4K3 w - - 0 1',
  '4k3/P7/8/8/8/8/7p/4K3 b - - 0 1',
  '4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1',
  'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
  'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1',
  '7k/5Q2/7K/8/8/8/8/8 b - - 0 1',
];
const positions = special.map((fen) => new Position(fen));
let seed = 117;
const random = () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 4294967296;
};
let p = new Position();
while (positions.length < 128) {
  const moves = p.legalMoves();
  if (result(p, [], moves) || p.ply > 160) {
    p = new Position();
    continue;
  }
  p.push(moves[Math.floor(random() * moves.length)]);
  if (p.ply % 3 === 0) positions.push(p.clone());
}
writeFileSync(
  '/private/tmp/rookie-gpu-fixtures.json',
  JSON.stringify(
    positions.map((p) => {
      const state = new Array<number>(768).fill(0);
      state.splice(0, 128, ...p.board);
      state[128] = p.turn;
      state[129] = p.castle;
      state[130] = p.ep;
      state[131] = p.half;
      state[132] = p.ply;
      state[133] = p.kings[0];
      state[134] = p.kings[1];
      state[135] = 1;
      const after: Record<number, number[]> = {};
      const moves = p.legalMoves().map((m) => {
        const word = m.from | (m.to << 7) | ((m.promotion || 0) << 14) | (m.flags << 17);
        const child = p.clone();
        child.push(m);
        after[word] = [
          ...child.board,
          child.turn,
          child.castle,
          child.ep,
          child.half,
          child.ply,
          ...child.kings,
        ];
        return word;
      });
      return { fen: p.fen(), state, moves, after };
    }),
  ),
);
console.log('128 independent GPU rules fixtures written');
