import type { Move } from './chess';

export interface Branch {
  move: Move;
  uci: string;
  visits: number;
  score: number;
  prior: number;
}
export interface SearchResult {
  move: Move | null;
  moves: Move[];
  policy: number[];
  value: number;
  nodes: number;
  depth: number;
  branches: Branch[];
}
