import { Position, moveKey, result } from '../../src/engine/chess';
import type { Move } from '../../src/engine/chess';
import { TinyModel } from './model';

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
export interface SearchOptions {
  simulations: number;
  exploration?: boolean;
  random?: () => number;
  deadline?: number;
}

// Each worker owns one reusable, contiguous tree arena. No node objects, no
// tree-wide garbage collection, and no buffer allocations after warm-up.
class Tree {
  capacity = 4096;
  used = 0;
  prior = new Float64Array(this.capacity);
  total = new Float64Array(this.capacity);
  visits = new Uint32Array(this.capacity);
  first = new Uint32Array(this.capacity);
  count = new Uint16Array(this.capacity);
  expanded = new Uint8Array(this.capacity);
  terminal = new Int8Array(this.capacity);
  key = new Float64Array(this.capacity);
  moves: (Move | null)[] = [];

  ensure(size: number) {
    if (size <= this.capacity) return;
    while (this.capacity < size) this.capacity *= 2;
    const grow = <T extends Float64Array | Uint32Array | Uint16Array | Uint8Array | Int8Array>(
      old: T,
    ): T => {
      const Constructor = old.constructor as new (size: number) => T;
      const next = new Constructor(this.capacity);
      next.set(old);
      return next;
    };
    this.prior = grow(this.prior);
    this.total = grow(this.total);
    this.visits = grow(this.visits);
    this.first = grow(this.first);
    this.count = grow(this.count);
    this.expanded = grow(this.expanded);
    this.terminal = grow(this.terminal);
    this.key = grow(this.key);
  }
  add(move: Move | null, prior: number) {
    const id = this.used++;
    this.moves[id] = move;
    this.prior[id] = prior;
    this.total[id] = 0;
    this.visits[id] = 0;
    this.first[id] = 0;
    this.count[id] = 0;
    this.expanded[id] = 0;
    this.terminal[id] = 0;
    this.key[id] = 0;
    return id;
  }
}
const tree = new Tree(),
  priors = new Float64Array(256),
  noise = new Float64Array(256);
const path: number[] = [];

export function search(
  rootPosition: Position,
  model: TinyModel,
  history: number[],
  options: SearchOptions,
): SearchResult {
  const random = options.random || Math.random;
  tree.used = 0;
  tree.add(null, 1);
  tree.key[0] = history[history.length - 1] || rootPosition.key();
  const p = rootPosition.clone(),
    keys = history.slice();
  let nodes = 0,
    depth = 0;
  for (let simulation = 0; simulation < options.simulations; simulation++) {
    if (simulation > 0 && options.deadline && performance.now() >= options.deadline) break;
    p.copyFrom(rootPosition);
    keys.length = history.length;
    path.length = 1;
    path[0] = 0;
    let current = 0;
    while (tree.count[current]) {
      let best = tree.first[current],
        bestScore = -Infinity;
      const end = best + tree.count[current],
        factor = 1.65 * Math.sqrt(tree.visits[current] + 1);
      for (let child = best; child < end; child++) {
        const visits = tree.visits[child];
        const q = visits ? -tree.total[child] / visits : 0;
        const score = q + (factor * tree.prior[child]) / (1 + visits);
        if (score > bestScore) {
          bestScore = score;
          best = child;
        }
      }
      p.push(tree.moves[best]!);
      // Every tree node describes one immutable position. Hash it only once.
      if (!tree.key[best]) tree.key[best] = p.key();
      keys.push(tree.key[best]);
      current = best;
      path.push(current);
    }
    depth = Math.max(depth, path.length - 1);
    let value: number;
    if (tree.expanded[current]) value = tree.terminal[current];
    else {
      const moves = p.legalMoves();
      const outcome = result(p, keys, moves);
      nodes++;
      tree.expanded[current] = 1;
      if (outcome) {
        value = outcome.winner * p.turn;
        tree.terminal[current] = value;
      } else {
        value = model.value(p) * p.turn;
        model.policyInto(p, moves, priors);
        if (current === 0 && options.exploration) {
          let total = 0;
          for (let i = 0; i < moves.length; i++) {
            noise[i] = -Math.log(Math.max(random(), 1e-10));
            total += noise[i];
          }
          for (let i = 0; i < moves.length; i++)
            priors[i] = priors[i] * 0.75 + (0.25 * noise[i]) / total;
        }
        tree.ensure(tree.used + moves.length);
        tree.first[current] = tree.used;
        tree.count[current] = moves.length;
        for (let i = 0; i < moves.length; i++) tree.add(moves[i], priors[i]);
      }
    }
    for (let i = path.length - 1; i >= 0; i--) {
      const id = path[i];
      tree.visits[id]++;
      tree.total[id] += value;
      value = -value;
    }
  }
  const first = tree.first[0],
    count = tree.count[0],
    end = first + count;
  let totalVisits = 0,
    selected = first,
    maxVisits = -1;
  for (let id = first; id < end; id++) {
    totalVisits += tree.visits[id];
    if (tree.visits[id] > maxVisits) {
      maxVisits = tree.visits[id];
      selected = id;
    }
  }
  const moves: Move[] = [],
    policy: number[] = [],
    branches: Branch[] = [];
  for (let id = first; id < end; id++) {
    const move = tree.moves[id]!;
    moves.push(move);
    policy.push(totalVisits ? tree.visits[id] / totalVisits : tree.prior[id]);
    branches.push({
      move,
      uci: moveKey(move),
      visits: tree.visits[id],
      score: tree.visits[id] ? -tree.total[id] / tree.visits[id] : 0,
      prior: tree.prior[id],
    });
  }
  if (options.exploration && rootPosition.ply < 24 && count) {
    let choice = random();
    for (let i = 0; i < count; i++) {
      choice -= policy[i];
      if (choice <= 0) {
        selected = first + i;
        break;
      }
    }
  }
  return {
    move: count ? tree.moves[selected] : null,
    moves,
    policy,
    value: tree.visits[0] ? tree.total[0] / tree.visits[0] : 0,
    nodes,
    depth,
    branches: branches.sort((a, b) => b.visits - a.visits).slice(0, 5),
  };
}
