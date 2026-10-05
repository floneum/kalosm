import { CASTLE, EP, KING, PAWN, Position } from '../../src/engine/chess';
import type { Move } from '../../src/engine/chess';

export const VALUE_SIZE = 385;
export const POLICY_SIZE = 776;
export const PARAM_COUNT = VALUE_SIZE + POLICY_SIZE;
export const MODEL_BYTES = PARAM_COUNT * 4;
export const MATERIAL = [0, 1, 3.2, 3.35, 5, 9.5, 0];
export interface Sample {
  position: Position;
  moves: Move[];
  policy: number[];
  value: number;
}
const index = (piece: number, square: number) =>
  (Math.abs(piece) - 1) * 64 + ((piece > 0 ? square : square ^ 112) >> 4) * 8 + (square & 7);

// Two sparse linear heads: a color-symmetric piece-square value and move policy.
// A material/centralization initialization gives immediate, useful play.
// All 1,161 parameters remain trainable; there are no external model downloads.
export class TinyModel {
  weights: Float32Array;
  constructor(weights?: ArrayLike<number>) {
    this.weights = weights ? Float32Array.from(weights) : new Float32Array(PARAM_COUNT);
    if (weights) return;
    for (let piece = 1; piece <= 6; piece++) {
      for (let rank = 0; rank < 8; rank++) {
        for (let file = 0; file < 8; file++) {
          const center = 3.5 - (Math.abs(file - 3.5) + Math.abs(rank - 3.5)) / 2;
          const positional =
            piece === PAWN
              ? rank * 0.055 + center * 0.045
              : piece === KING
                ? -center * 0.035
                : center * 0.11;
          const i = (piece - 1) * 64 + rank * 8 + file;
          this.weights[i] = (MATERIAL[piece] + positional) / 8;
          this.weights[VALUE_SIZE + 384 + i] = piece === KING ? -center * 0.1 : center * 0.1;
        }
      }
    }
    this.weights[384] = 0.012;
    for (let i = 0; i < 6; i++) this.weights[VALUE_SIZE + 768 + i] = MATERIAL[i + 1] * 0.18;
    this.weights[VALUE_SIZE + 774] = 1.3;
    this.weights[VALUE_SIZE + 775] = 0.35;
  }

  value(p: Position): number {
    let sum = this.weights[384] * p.turn;
    for (let s = 0; s < 128; s++) {
      if (s & 0x88) {
        s += 7;
        continue;
      }
      const piece = p.board[s];
      if (piece) sum += this.weights[index(piece, s)] * Math.sign(piece);
    }
    return Math.tanh(sum);
  }

  policyFeatures(p: Position, move: Move): number[] {
    const piece = p.board[move.from];
    const indices = [
      VALUE_SIZE + index(piece, move.from),
      VALUE_SIZE + 384 + index(piece, move.to),
    ];
    const victim = move.flags & EP ? PAWN : Math.abs(p.board[move.to]);
    if (victim) indices.push(VALUE_SIZE + 767 + victim);
    if (move.promotion) indices.push(VALUE_SIZE + 774);
    if (move.flags & CASTLE) indices.push(VALUE_SIZE + 775);
    return indices;
  }

  policy(p: Position, moves: Move[]): number[] {
    const out = new Array<number>(moves.length);
    this.policyInto(p, moves, out);
    return out;
  }

  policyInto(p: Position, moves: Move[], out: Float64Array | number[]): void {
    let max = -Infinity;
    const weights = this.weights;
    for (let i = 0; i < moves.length; i++) {
      const move = moves[i],
        piece = p.board[move.from];
      let logit =
        weights[VALUE_SIZE + index(piece, move.from)] +
        weights[VALUE_SIZE + 384 + index(piece, move.to)];
      const victim = move.flags & EP ? PAWN : Math.abs(p.board[move.to]);
      if (victim) logit += weights[VALUE_SIZE + 767 + victim];
      if (move.promotion) logit += weights[VALUE_SIZE + 774];
      if (move.flags & CASTLE) logit += weights[VALUE_SIZE + 775];
      out[i] = logit;
      if (logit > max) max = logit;
    }
    let sum = 0;
    for (let i = 0; i < moves.length; i++) {
      out[i] = Math.exp(out[i] - max);
      sum += out[i];
    }
    for (let i = 0; i < moves.length; i++) out[i] /= sum;
  }

  train(batch: Sample[], learningRate = 0.045): { valueLoss: number; policyLoss: number } {
    const gradient = new Float32Array(PARAM_COUNT);
    let valueLoss = 0,
      policyLoss = 0;
    for (const sample of batch) {
      const p = sample.position,
        prediction = this.value(p);
      const error = prediction - sample.value;
      valueLoss += error * error;
      const derivative = 2 * error * (1 - prediction * prediction);
      for (let s = 0; s < 128; s++) {
        if (s & 0x88) {
          s += 7;
          continue;
        }
        const piece = p.board[s];
        if (piece) gradient[index(piece, s)] += derivative * Math.sign(piece);
      }
      gradient[384] += derivative * p.turn;
      const probs = this.policy(p, sample.moves);
      for (let m = 0; m < sample.moves.length; m++) {
        const target = sample.policy[m];
        policyLoss -= target * Math.log(Math.max(probs[m], 1e-8));
        const d = probs[m] - target;
        for (const i of this.policyFeatures(p, sample.moves[m])) gradient[i] += d;
      }
    }
    const scale = learningRate / Math.max(batch.length, 1);
    for (let i = 0; i < PARAM_COUNT; i++) {
      // Gradient clipping prevents rare terminal targets from destabilizing material values.
      this.weights[i] -= Math.max(-0.025, Math.min(0.025, gradient[i] * scale));
    }
    return {
      valueLoss: valueLoss / Math.max(batch.length, 1),
      policyLoss: policyLoss / Math.max(batch.length, 1),
    };
  }
}

export function validateWeights(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === PARAM_COUNT &&
    value.every((w) => typeof w === 'number' && Number.isFinite(w) && Math.abs(w) < 100)
  );
}
