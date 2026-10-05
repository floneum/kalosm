import type { Position, Move } from './chess';
export const INPUT = 832;
// Must match `count_scale` in fusor-chess/src/lib.rs.
const COUNT_SCALE = 0.25;
const square = (s: number, turn: number) => ((turn === 1 ? s : s ^ 112) >> 4) * 8 + (s & 7);
export const action = (p: Position, m: Move) => square(m.from, p.turn) * 64 + square(m.to, p.turn);
export function encode(p: Position, into: Float32Array, offset = 0) {
  into.fill(0, offset, offset + INPUT);
  for (let s = 0; s < 128; s++) {
    if (s & 0x88) {
      s += 7;
      continue;
    }
    const piece = p.board[s] * p.turn;
    if (piece) {
      const plane = piece > 0 ? piece - 1 : 6 - piece - 1;
      into[offset + plane * 64 + square(s, p.turn)] = 1;
      // Piece-count inputs: one learned weight per piece type, shared by its squares.
      into[offset + 782 + plane] += COUNT_SCALE;
    }
  }
  const rights = p.turn === 1 ? p.castle : ((p.castle & 3) << 2) | ((p.castle & 12) >> 2);
  for (let i = 0; i < 4; i++) into[offset + 768 + i] = (rights >> i) & 1;
  if (p.ep >= 0) into[offset + 772 + (p.ep & 7)] = 1;
  into[offset + 780] = Math.min(1, p.half / 100);
  into[offset + 781] = Math.min(1, p.ply / 200);
  into[offset + 831] = 1;
}
export interface TrainingSample {
  position: Position;
  value: number;
}
// Host batches for ChessGpu.predict/train; `batch` is ChessGpu.batch().
export class BatchBuffers {
  x: Float32Array;
  values: Float32Array;
  constructor(readonly batch: number) {
    this.x = new Float32Array(batch * INPUT);
    this.values = new Float32Array(batch);
  }
  pack(rows: { position: Position }[]) {
    if (!rows.length || rows.length > this.batch) throw new Error('Invalid GPU batch size');
    for (let i = 0; i < this.batch; i++) encode(rows[i % rows.length].position, this.x, i * INPUT);
  }
  training(rows: TrainingSample[]) {
    this.pack(rows);
    for (let i = 0; i < this.batch; i++) this.values[i] = rows[i % rows.length].value;
  }
}
