// Compact 0x88 rules engine. Search uses make/unmake; only tree roots are cloned.
export const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
export const PAWN = 1,
  KNIGHT = 2,
  BISHOP = 3,
  ROOK = 4,
  QUEEN = 5,
  KING = 6;
export const EP = 1,
  CASTLE = 2,
  DOUBLE = 4;
export type Color = 1 | -1;
export interface Move {
  from: number;
  to: number;
  promotion?: number;
  flags: number;
}
interface Undo {
  piece: number;
  captured: number;
  ep: number;
  castle: number;
  half: number;
  kings: [number, number];
}
const KNIGHT_STEPS = [-33, -31, -18, -14, 14, 18, 31, 33];
const BISHOP_STEPS = [-17, -15, 15, 17];
const ROOK_STEPS = [-16, -1, 1, 16];
const KING_STEPS = [...BISHOP_STEPS, ...ROOK_STEPS];
const SYMBOLS = ' pnbrqk';
// A 53-bit Zobrist key fits exactly in a JS number. Two independent 32-bit
// lanes avoid BigInt arithmetic and board-string allocation in the search loop.
const HASH_LO = new Uint32Array(13 * 128 + 32),
  HASH_HI = new Uint32Array(HASH_LO.length);
let hashSeed = 0x9e3779b9;
for (let i = 0; i < HASH_LO.length; i++) {
  hashSeed ^= hashSeed << 13;
  hashSeed ^= hashSeed >>> 17;
  hashSeed ^= hashSeed << 5;
  HASH_LO[i] = hashSeed >>> 0;
  hashSeed ^= hashSeed << 13;
  hashSeed ^= hashSeed >>> 17;
  hashSeed ^= hashSeed << 5;
  HASH_HI[i] = hashSeed & 0x1fffff;
}
export const squareName = (s: number) => 'abcdefgh'[s & 7] + ((s >> 4) + 1);
export const parseSquare = (s: string) => (Number(s[1]) - 1) * 16 + 'abcdefgh'.indexOf(s[0]);
export const moveKey = (m: Move) =>
  squareName(m.from) + squareName(m.to) + (m.promotion ? SYMBOLS[m.promotion] : '');
export const sameMove = (a: Move, b: Move) =>
  a.from === b.from && a.to === b.to && a.promotion === b.promotion;

export class Position {
  board = new Int8Array(128);
  turn: Color = 1;
  castle = 15;
  ep = -1;
  half = 0;
  ply = 0;
  kings: [number, number] = [4, 116];

  constructor(fen = START_FEN) {
    const [placement, side, rights, ep, half, full] = fen.split(' ');
    let rank = 7,
      file = 0;
    for (const char of placement) {
      if (char === '/') {
        rank--;
        file = 0;
      } else if (/\d/.test(char)) file += Number(char);
      else {
        const type = SYMBOLS.indexOf(char.toLowerCase());
        const color = char === char.toUpperCase() ? 1 : -1;
        const s = rank * 16 + file++;
        this.board[s] = type * color;
        if (type === KING) this.kings[color === 1 ? 0 : 1] = s;
      }
    }
    this.turn = side === 'w' ? 1 : -1;
    this.castle =
      (rights.includes('K') ? 1 : 0) |
      (rights.includes('Q') ? 2 : 0) |
      (rights.includes('k') ? 4 : 0) |
      (rights.includes('q') ? 8 : 0);
    this.ep = ep === '-' ? -1 : parseSquare(ep);
    this.half = Number(half) || 0;
    this.ply = ((Number(full) || 1) - 1) * 2 + (this.turn === -1 ? 1 : 0);
  }

  clone(): Position {
    const p = Object.create(Position.prototype) as Position;
    p.board = this.board.slice();
    p.turn = this.turn;
    p.castle = this.castle;
    p.ep = this.ep;
    p.half = this.half;
    p.ply = this.ply;
    p.kings = [...this.kings];
    return p;
  }

  copyFrom(p: Position) {
    this.board.set(p.board);
    this.turn = p.turn;
    this.castle = p.castle;
    this.ep = p.ep;
    this.half = p.half;
    this.ply = p.ply;
    this.kings[0] = p.kings[0];
    this.kings[1] = p.kings[1];
  }

  attacked(square: number, by: Color): boolean {
    for (const step of [-1, 1]) {
      const s = square - by * 16 + step;
      if (!(s & 0x88) && this.board[s] === by * PAWN) return true;
    }
    for (const step of KNIGHT_STEPS) {
      const s = square + step;
      if (!(s & 0x88) && this.board[s] === by * KNIGHT) return true;
    }
    for (const step of KING_STEPS) {
      let s = square + step,
        distance = 1;
      const diagonal = Math.abs(step) === 15 || Math.abs(step) === 17;
      while (!(s & 0x88)) {
        const piece = this.board[s];
        if (piece) {
          if (piece * by > 0) {
            const type = Math.abs(piece);
            if (
              type === QUEEN ||
              type === (diagonal ? BISHOP : ROOK) ||
              (type === KING && distance === 1)
            )
              return true;
          }
          break;
        }
        s += step;
        distance++;
      }
    }
    return false;
  }

  inCheck(): boolean {
    return this.attacked(this.kings[this.turn === 1 ? 0 : 1], -this.turn as Color);
  }

  make(m: Move): Undo {
    const piece = this.board[m.from];
    const u: Undo = {
      piece,
      captured: this.board[m.to],
      ep: this.ep,
      castle: this.castle,
      half: this.half,
      kings: [...this.kings],
    };
    this.push(m);
    return u;
  }

  // Search discards traversed boards and does not need an allocated undo record.
  push(m: Move): void {
    const piece = this.board[m.from],
      captured = this.board[m.to];
    this.board[m.to] = m.promotion ? m.promotion * this.turn : piece;
    this.board[m.from] = 0;
    if (m.flags & EP) this.board[m.to - this.turn * 16] = 0;
    if (m.flags & CASTLE) {
      const rookFrom = m.to > m.from ? m.from + 3 : m.from - 4;
      const rookTo = m.to > m.from ? m.from + 1 : m.from - 1;
      this.board[rookTo] = this.board[rookFrom];
      this.board[rookFrom] = 0;
    }
    if (Math.abs(piece) === KING) {
      this.kings[this.turn === 1 ? 0 : 1] = m.to;
      this.castle &= this.turn === 1 ? 12 : 3;
    }
    for (const s of [m.from, m.to]) {
      if (s === 0) this.castle &= ~2;
      if (s === 7) this.castle &= ~1;
      if (s === 112) this.castle &= ~8;
      if (s === 119) this.castle &= ~4;
    }
    this.ep = m.flags & DOUBLE ? m.from + this.turn * 16 : -1;
    this.half = Math.abs(piece) === PAWN || captured || m.flags & EP ? 0 : this.half + 1;
    this.ply++;
    this.turn = -this.turn as Color;
  }

  unmake(m: Move, u: Undo) {
    this.turn = -this.turn as Color;
    this.ply--;
    this.board[m.from] = u.piece;
    this.board[m.to] = u.captured;
    if (m.flags & EP) this.board[m.to - this.turn * 16] = -this.turn * PAWN;
    if (m.flags & CASTLE) {
      const rookFrom = m.to > m.from ? m.from + 3 : m.from - 4;
      const rookTo = m.to > m.from ? m.from + 1 : m.from - 1;
      this.board[rookFrom] = this.turn * ROOK;
      this.board[rookTo] = 0;
    }
    this.ep = u.ep;
    this.castle = u.castle;
    this.half = u.half;
    this.kings = u.kings;
  }

  legalMoves(): Move[] {
    const moves: Move[] = [];
    const king = this.kings[this.turn === 1 ? 0 : 1],
      enemy = -this.turn as Color;
    const checked = this.attacked(king, enemy);
    const pinned: number[] = [];
    // Only pinned pieces, king moves, and en passant can expose a safe king.
    // Compute pins once instead of doing make/unmake + eight attack rays for
    // every pseudo-legal move (the dominant cost at interior search nodes).
    if (!checked) {
      for (const step of KING_STEPS) {
        let candidate = -1;
        for (let s = king + step; !(s & 0x88); s += step) {
          const piece = this.board[s];
          if (!piece) continue;
          if (piece * this.turn > 0) {
            if (candidate >= 0) break;
            candidate = s;
          } else {
            const type = Math.abs(piece),
              diagonal = Math.abs(step) === 15 || Math.abs(step) === 17;
            if (candidate >= 0 && (type === QUEEN || type === (diagonal ? BISHOP : ROOK)))
              pinned.push(candidate);
            break;
          }
        }
      }
    }
    const add = (from: number, to: number, flags = 0) => {
      if (Math.abs(this.board[from]) === PAWN && (to >> 4 === 0 || to >> 4 === 7)) {
        for (const promotion of [QUEEN, ROOK, BISHOP, KNIGHT])
          moves.push({ from, to, promotion, flags });
      } else moves.push({ from, to, flags });
    };
    for (let from = 0; from < 128; from++) {
      if (from & 0x88) {
        from += 7;
        continue;
      }
      const piece = this.board[from];
      if (piece * this.turn <= 0) continue;
      const type = Math.abs(piece);
      if (type === PAWN) {
        const step = this.turn * 16,
          one = from + step;
        if (!(one & 0x88) && !this.board[one]) {
          add(from, one);
          if (from >> 4 === (this.turn === 1 ? 1 : 6) && !this.board[one + step])
            add(from, one + step, DOUBLE);
        }
        for (const to of [one - 1, one + 1]) {
          if (to & 0x88) continue;
          if (this.board[to] * this.turn < 0) add(from, to);
          else if (to === this.ep && this.board[to - step] === -this.turn * PAWN) add(from, to, EP);
        }
      } else {
        const steps =
          type === KNIGHT
            ? KNIGHT_STEPS
            : type === BISHOP
              ? BISHOP_STEPS
              : type === ROOK
                ? ROOK_STEPS
                : KING_STEPS;
        for (const step of steps) {
          for (let to = from + step; !(to & 0x88); to += step) {
            if (this.board[to] * this.turn > 0) break;
            add(from, to);
            if (this.board[to] || type === KNIGHT || type === KING) break;
          }
        }
        if (type === KING) {
          const home = this.turn === 1 ? 4 : 116;
          const enemy = -this.turn as Color;
          const rights = this.turn === 1 ? this.castle : this.castle >> 2;
          if (from === home && !checked) {
            if (
              rights & 1 &&
              this.board[home + 3] === this.turn * ROOK &&
              !this.board[home + 1] &&
              !this.board[home + 2] &&
              !this.attacked(home + 1, enemy) &&
              !this.attacked(home + 2, enemy)
            )
              add(from, home + 2, CASTLE);
            if (
              rights & 2 &&
              this.board[home - 4] === this.turn * ROOK &&
              !this.board[home - 1] &&
              !this.board[home - 2] &&
              !this.board[home - 3] &&
              !this.attacked(home - 1, enemy) &&
              !this.attacked(home - 2, enemy)
            )
              add(from, home - 2, CASTLE);
          }
        }
      }
    }
    let count = 0;
    for (const move of moves) {
      const piece = this.board[move.from];
      if (
        !checked &&
        Math.abs(piece) !== KING &&
        !(move.flags & EP) &&
        !pinned.includes(move.from)
      ) {
        moves[count++] = move;
        continue;
      }
      // Legality only needs occupancy; clocks, rights, and history never change.
      const captured = this.board[move.to];
      this.board[move.from] = 0;
      this.board[move.to] = piece;
      if (move.flags & EP) this.board[move.to - this.turn * 16] = 0;
      const legal = !this.attacked(Math.abs(piece) === KING ? move.to : king, enemy);
      this.board[move.from] = piece;
      this.board[move.to] = captured;
      if (move.flags & EP) this.board[move.to - this.turn * 16] = -this.turn * PAWN;
      if (legal) moves[count++] = move;
    }
    moves.length = count;
    return moves;
  }

  insufficient(): boolean {
    const minors: number[] = [];
    let bishopsOnly = true;
    for (let s = 0; s < 128; s++) {
      if (s & 0x88) {
        s += 7;
        continue;
      }
      const type = Math.abs(this.board[s]);
      if (!type || type === KING) continue;
      if (type === PAWN || type === ROOK || type === QUEEN) return false;
      minors.push(((s >> 4) + (s & 7)) % 2);
      if (type === KNIGHT) bishopsOnly = false;
    }
    return minors.length <= 1 || (bishopsOnly && minors.every((c) => c === minors[0]));
  }

  fen(): string {
    const rows: string[] = [];
    for (let rank = 7; rank >= 0; rank--) {
      let row = '',
        empty = 0;
      for (let file = 0; file < 8; file++) {
        const piece = this.board[rank * 16 + file];
        if (!piece) {
          empty++;
          continue;
        }
        if (empty) {
          row += empty;
          empty = 0;
        }
        const char = SYMBOLS[Math.abs(piece)];
        row += piece > 0 ? char.toUpperCase() : char;
      }
      if (empty) row += empty;
      rows.push(row);
    }
    const rights = ['K', 'Q', 'k', 'q'].filter((_, i) => this.castle & (1 << i)).join('') || '-';
    return `${rows.join('/')} ${this.turn === 1 ? 'w' : 'b'} ${rights} ${this.ep < 0 ? '-' : squareName(this.ep)} ${this.half} ${Math.floor(this.ply / 2) + 1}`;
  }

  key(): number {
    // FIDE repetition treats an uncapturable (including pinned) EP square as absent.
    let ep = -1;
    if (this.ep >= 0) {
      const captured = this.ep - this.turn * 16;
      for (const from of [captured - 1, captured + 1]) {
        if (from & 0x88 || this.board[from] !== this.turn * PAWN) continue;
        const move = { from, to: this.ep, flags: EP };
        const u = this.make(move);
        const legal = !this.attacked(this.kings[this.turn === -1 ? 0 : 1], this.turn);
        this.unmake(move, u);
        if (legal) {
          ep = this.ep & 7;
          break;
        }
      }
    }
    const state = 13 * 128;
    let lo = HASH_LO[state + this.castle],
      hi = HASH_HI[state + this.castle];
    if (this.turn === -1) {
      lo ^= HASH_LO[state + 16];
      hi ^= HASH_HI[state + 16];
    }
    if (ep >= 0) {
      lo ^= HASH_LO[state + 17 + ep];
      hi ^= HASH_HI[state + 17 + ep];
    }
    for (let s = 0; s < 128; s++) {
      if (s & 0x88) {
        s += 7;
        continue;
      }
      const piece = this.board[s];
      if (piece) {
        const i = (piece + 6) * 128 + s;
        lo ^= HASH_LO[i];
        hi ^= HASH_HI[i];
      }
    }
    return (hi >>> 0) * 4294967296 + (lo >>> 0);
  }
}

export interface GameResult {
  winner: Color | 0;
  reason: string;
}
export function result(p: Position, keys: number[], moves = p.legalMoves()): GameResult | null {
  if (!moves.length)
    return p.inCheck()
      ? { winner: -p.turn as Color, reason: 'Checkmate' }
      : { winner: 0, reason: 'Stalemate' };
  if (p.half >= 100) return { winner: 0, reason: 'Fifty-move rule' };
  if (p.insufficient()) return { winner: 0, reason: 'Insufficient material' };
  const key = keys[keys.length - 1];
  let repetitions = 0;
  for (let i = keys.length - 1; i >= Math.max(0, keys.length - p.half - 1); i -= 2) {
    if (keys[i] === key && ++repetitions >= 3) return { winner: 0, reason: 'Threefold repetition' };
  }
  return null;
}

export function notation(p: Position, move: Move): string {
  let text = '';
  const type = Math.abs(p.board[move.from]);
  if (move.flags & CASTLE) text = move.to > move.from ? 'O-O' : 'O-O-O';
  else {
    const capture = !!p.board[move.to] || !!(move.flags & EP);
    if (type !== PAWN) {
      text += SYMBOLS[type].toUpperCase();
      const others = p
        .legalMoves()
        .filter(
          (m) => m.to === move.to && m.from !== move.from && Math.abs(p.board[m.from]) === type,
        );
      if (others.length) {
        if (others.every((m) => (m.from & 7) !== (move.from & 7))) text += squareName(move.from)[0];
        else if (others.every((m) => m.from >> 4 !== move.from >> 4))
          text += squareName(move.from)[1];
        else text += squareName(move.from);
      }
    } else if (capture) text += squareName(move.from)[0];
    if (capture) text += 'x';
    text += squareName(move.to);
    if (move.promotion) text += '=' + SYMBOLS[move.promotion].toUpperCase();
  }
  const u = p.make(move);
  if (p.inCheck()) text += p.legalMoves().length ? '+' : '#';
  p.unmake(move, u);
  return text;
}
