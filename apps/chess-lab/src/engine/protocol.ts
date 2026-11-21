import type { Move } from './chess';
import type { Branch } from './search';

export interface Settings {
  /// Positions the search visits per self-play move.
  nodes: number;
}
export const DEFAULT_SETTINGS: Settings = {
  nodes: 400,
};
export interface ArenaSnapshot {
  id: number;
  fen: string;
  ply: number;
  episode: number;
  lastMove: Move | null;
  lastSan: string;
  score: number;
  branches: Branch[];
  depth: number;
  nodes: number;
  status: string;
}
export interface Metrics {
  positions: number;
  nodes: number;
  batches: number;
  games: number;
  truncated: number;
  whiteWins: number;
  blackWins: number;
  draws: number;
  valueLoss: number;
}
export const emptyMetrics = (): Metrics => ({
  positions: 0,
  nodes: 0,
  batches: 0,
  games: 0,
  truncated: 0,
  whiteWins: 0,
  blackWins: 0,
  draws: 0,
  valueLoss: 0,
});
export type TrainerRequest =
  | { type: 'init'; ids: number[]; settings: Settings; weights: number[]; running: boolean }
  | { type: 'running'; running: boolean }
  | { type: 'settings'; settings: Settings }
  | { type: 'sync'; round: number }
  | { type: 'weights'; weights: number[] };
export type TrainerResponse =
  | { type: 'snapshot'; arenas: ArenaSnapshot[]; metrics: Metrics }
  | { type: 'sync'; round: number; weights: number[] };
