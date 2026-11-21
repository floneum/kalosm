import { memo } from 'react';
import { Position, squareName } from '../engine/chess';
import type { Move } from '../engine/chess';

const pieceNames = ['', 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'];
export function Piece({ piece }: { piece: number }) {
  const white = piece > 0;
  return (
    <svg
      viewBox="0 0 48 48"
      className={`piece ${white ? 'white-piece' : 'black-piece'}`}
      aria-hidden="true"
    >
      <g
        fill={white ? '#fffaf0' : '#303b35'}
        stroke={white ? '#424b40' : '#202d25'}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {Math.abs(piece) === 1 && (
          <>
            <circle cx="24" cy="12" r="5.5" />
            <path d="M20 18h8l-1 5c0 5 3 8 5 10H16c2-2 5-5 5-10Z" />
            <path d="M16 33h16l3 6H13Z" />
            <path d="M12 39h24v3H12Z" />
          </>
        )}
        {Math.abs(piece) === 2 && (
          <>
            <path d="M13 35c0-8 8-11 12-16l-6 2-7 5-5-5 6-10 9-3 4-4 2 7c9 4 10 13 8 24Z" />
            <path d="m13 13 6 1M24 9c7 7 7 14 2 20" fill="none" />
            <circle cx="18" cy="15" r="1" fill={white ? '#303b35' : '#fffaf0'} stroke="none" />
            <path d="M13 35h23l2 7H10Z" />
          </>
        )}
        {Math.abs(piece) === 3 && (
          <>
            <circle cx="24" cy="6" r="2.5" />
            <path d="M24 8c-4 4-9 8-9 13 0 4 4 6 9 6s9-2 9-6c0-5-5-9-9-13Z" />
            <path d="m27 14-6 8" fill="none" />
            <path d="M20 27h8l3 8H17Z" />
            <path d="M15 35h18l3 7H12Z" />
            <path d="M18 30h12" />
          </>
        )}
        {Math.abs(piece) === 4 && (
          <>
            <path d="M12 7h6v6h4V7h4v6h4V7h6v12l-5 4v12H17V23l-5-4Z" />
            <path d="M12 19h24M17 24h14M17 32h14" fill="none" />
            <path d="M14 35h20l3 7H11Z" />
          </>
        )}
        {Math.abs(piece) === 5 && (
          <>
            <path d="m10 13 7 9 7-12 7 12 7-9-6 19H16Z" />
            <circle cx="9" cy="10" r="3" />
            <circle cx="24" cy="7" r="3" />
            <circle cx="39" cy="10" r="3" />
            <path d="M16 32h16v4H16Zm-2 4h20l3 6H11Z" />
            <path d="M17 28h14" />
          </>
        )}
        {Math.abs(piece) === 6 && (
          <>
            <path d="M24 3v10m-4-6h8" fill="none" strokeWidth="2.3" />
            <path d="M24 15c-4-6-12-4-12 3 0 5 6 9 7 14h10c1-5 7-9 7-14 0-7-8-9-12-3Z" />
            <path d="M24 16v12" fill="none" />
            <path d="M17 32h14v4H17Zm-3 4h20l3 6H11Z" />
          </>
        )}
      </g>
    </svg>
  );
}

interface BoardProps {
  fen: string;
  lastMove?: Move | null;
  selected?: number | null;
  legal?: Move[];
  onSquare?: (square: number) => void;
  flipped?: boolean;
  coordinates?: boolean;
  label?: string;
}
export const Board = memo(function Board({
  fen,
  lastMove,
  selected,
  legal = [],
  onSquare,
  flipped = false,
  coordinates = false,
  label = 'Chess position',
}: BoardProps) {
  const p = new Position(fen);
  const check = p.inCheck() ? p.kings[p.turn === 1 ? 0 : 1] : -1;
  return (
    <div
      className={`board ${coordinates ? 'has-coordinates' : ''} ${onSquare ? 'interactive-board' : ''}`}
      role={onSquare ? 'group' : 'img'}
      aria-label={label}
    >
      {Array.from({ length: 64 }, (_, i) => {
        const rank = flipped ? Math.floor(i / 8) : 7 - Math.floor(i / 8);
        const file = flipped ? 7 - (i % 8) : i % 8;
        const s = rank * 16 + file,
          piece = p.board[s];
        const target = legal.some((m) => m.to === s);
        const className = `square ${(rank + file) % 2 ? 'light' : 'dark'} ${lastMove && (lastMove.from === s || lastMove.to === s) ? 'last-move' : ''} ${selected === s ? 'selected' : ''} ${check === s ? 'in-check' : ''}`;
        const content = (
          <>
            {coordinates && i % 8 === 0 && <span className="rank-label">{rank + 1}</span>}
            {coordinates && i >= 56 && <span className="file-label">{'abcdefgh'[file]}</span>}
            {piece !== 0 && <Piece piece={piece} />}
            {target && <span className={`legal-dot ${piece ? 'capture-ring' : ''}`} />}
          </>
        );
        return onSquare ? (
          <button
            key={s}
            className={className}
            aria-label={`${squareName(s)}${piece ? ` ${piece > 0 ? 'white' : 'black'} ${pieceNames[Math.abs(piece)]}` : ''}${target ? ', legal move' : ''}`}
            aria-pressed={selected === s}
            onClick={() => onSquare(s)}
          >
            {content}
          </button>
        ) : (
          <div key={s} className={className}>
            {content}
          </div>
        );
      })}
    </div>
  );
});

export function EvaluationBar({ score }: { score: number }) {
  const percent = Math.round((score + 1) * 50);
  return (
    <div className="evaluation-bar" title={`White evaluation ${percent}%`}>
      <span style={{ width: `${percent}%` }} />
    </div>
  );
}
