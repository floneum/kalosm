import { useEffect, useMemo, useRef, useState } from 'react';
import { Position, START_FEN, moveKey, notation, result } from '../engine/chess';
import type { Color, Move } from '../engine/chess';
import type { SearchResult } from '../engine/search';
import type { Training } from '../useTraining';
import { Board, EvaluationBar, Piece } from './Board';
import { Icon } from './Icon';

interface Frame {
  fen: string;
  lastMove: Move | null;
  san: string;
}
const start = (): Frame[] => [{ fen: START_FEN, lastMove: null, san: '' }];
export function Play({ training, active }: { training: Training; active: boolean }) {
  const [frames, setFrames] = useState<Frame[]>(start);
  const [player, setPlayer] = useState<Color>(1);
  const [flipped, setFlipped] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [promotion, setPromotion] = useState<Move[]>([]);
  const [thinking, setThinking] = useState(false);
  const [analysis, setAnalysis] = useState<SearchResult | null>(null);
  const [error, setError] = useState('');
  const [whiteScore, setWhiteScore] = useState<number | null>(null);
  // Milliseconds the opponent may think per move.
  const [strength, setStrength] = useState(500);
  const [snapshot, setSnapshot] = useState(() => ({
    generation: training.generation,
  }));
  const frame = frames[frames.length - 1];
  const position = useMemo(() => new Position(frame.fen), [frame.fen]);
  const moves = useMemo(() => position.legalMoves(), [position]);
  const keys = useMemo(() => frames.map((f) => new Position(f.fen).key()), [frames]);
  const outcome = useMemo(() => result(position, keys, moves), [position, keys, moves]);
  // The model's own score from its last search; even until it has searched.
  const score = whiteScore ?? 0;
  const myTurn = position.turn === player && !outcome;

  useEffect(() => {
    if (active) setSnapshot({ generation: training.generation });
    // The generation this game started against; training continues meanwhile.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  // Every finished game becomes training data, once.
  const learned = useRef('');
  useEffect(() => {
    if (!outcome || frames.length < 2) return;
    const game = frames.map((f) => f.fen).join('|');
    if (learned.current === game) return;
    learned.current = game;
    training.learn(
      frames.slice(0, -1).map((f) => f.fen),
      frames.slice(1).map((f) => moveKey(f.lastMove!)),
      outcome.winner,
    );
  }, [outcome, frames, training]);

  function commit(move: Move) {
    const p = new Position(frame.fen),
      san = notation(p, move);
    p.make(move);
    setFrames((f) => [...f, { fen: p.fen(), lastMove: move, san }]);
    setSelected(null);
    setPromotion([]);
  }

  // While the human decides, the engine ponders their position.
  useEffect(() => {
    if (position.turn === player && !outcome && active && training.ready) {
      training.ponder(
        frame.fen,
        frames.map((f) => f.fen),
      );
      return () => training.stopPonder();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame, player, active, outcome, training.ready]);
  useEffect(() => {
    if (position.turn === player || outcome || !active || !training.ready) {
      setThinking(false);
      return;
    }
    setThinking(true);
    setError('');
    let live = true;
    void training
      .play(
        frame.fen,
        frames.map((f) => f.fen),
        strength,
      )
      .then((result) => {
        if (!live) return;
        setAnalysis(result);
        setWhiteScore(result.value * position.turn);
        if (result.move) commit(result.move);
        setThinking(false);
      })
      .catch((error) => {
        if (live) {
          setError(String(error));
          setThinking(false);
        }
      });
    return () => {
      live = false;
    };
    // Ignore stale results after undo/new game; the GPU worker serializes requests.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame, player, active, outcome, strength, training.ready]);

  function clickSquare(square: number) {
    if (!myTurn || thinking) return;
    const candidates = moves.filter((m) => m.from === selected && m.to === square);
    if (candidates.length > 1) {
      setPromotion(candidates);
      return;
    }
    if (candidates.length) {
      commit(candidates[0]);
      return;
    }
    setSelected(position.board[square] * player > 0 && square !== selected ? square : null);
  }
  function newGame(color = player) {
    setFrames(start());
    setSelected(null);
    setPromotion([]);
    setPlayer(color);
    setFlipped(color === -1);
    setSnapshot({ generation: training.generation });
    setAnalysis(null);
    setWhiteScore(null);
    setError('');
  }
  function undo() {
    setFrames((f) => f.slice(0, Math.max(1, f.length - (position.turn === player ? 2 : 1))));
    setSelected(null);
    setPromotion([]);
    setAnalysis(null);
    setWhiteScore(null);
  }
  const status = outcome
    ? outcome.winner === 0
      ? `Draw · ${outcome.reason}`
      : `${outcome.winner === player ? 'You win' : 'Rookie wins'} · ${outcome.reason}`
    : thinking
      ? 'Rookie is thinking…'
      : myTurn
        ? position.inCheck()
          ? 'You’re in check'
          : 'Your move'
        : 'Ready when you are';

  return (
    <section className="play-layout">
      <div className="play-board-panel">
        <div className="player-label">
          <div className="avatar engine-avatar">
            <Icon name="rook" size={23} />
          </div>
          <div>
            <strong>Rookie</strong>
            <span>
              Generation {snapshot.generation} · {player === 1 ? 'Black' : 'White'}
            </span>
          </div>
          <span className={`turn-pill ${thinking ? 'is-thinking' : ''}`}>
            {thinking ? (
              <>
                <span className="pulse-dot" /> Thinking
              </>
            ) : (
              'Rookie'
            )}
          </span>
        </div>
        <div className="play-board-wrap">
          <Board
            fen={frame.fen}
            lastMove={frame.lastMove}
            selected={selected}
            legal={moves.filter((m) => m.from === selected && myTurn)}
            onSquare={clickSquare}
            flipped={flipped}
            coordinates
            label="Play chess against Rookie. Select a piece, then a highlighted destination."
          />
          {promotion.length > 0 && (
            <div className="promotion-overlay">
              <div role="dialog" aria-label="Choose promotion piece">
                <h3>A well-earned promotion.</h3>
                <div>
                  {promotion.map((move) => (
                    <button
                      key={move.promotion}
                      aria-label={`Promote to ${['', 'pawn', 'knight', 'bishop', 'rook', 'queen'][move.promotion!]}`}
                      onClick={() => commit(move)}
                    >
                      <Piece piece={player * move.promotion!} />
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
        <EvaluationBar score={score} />
        <div className="player-label">
          <div className="avatar human-avatar">you</div>
          <div>
            <strong>You</strong>
            <span>{player === 1 ? 'White pieces' : 'Black pieces'} · Curiosity included</span>
          </div>
          <div className="board-tools">
            <button
              className="icon-button"
              aria-label="Undo last turn"
              title="Undo last turn"
              disabled={frames.length < 2}
              onClick={undo}
            >
              <Icon name="undo" />
            </button>
            <button
              className="icon-button"
              aria-label="Flip board"
              title="Flip board"
              onClick={() => setFlipped((f) => !f)}
            >
              <Icon name="flip" />
            </button>
          </div>
        </div>
      </div>
      <aside className="play-sidebar">
        <div className="panel game-status">
          <span className="eyebrow">HUMAN × LITTLE MACHINE</span>
          <h2>{status}</h2>
          <p aria-live="polite">
            {outcome
              ? 'Every game is a new conversation. Give the latest generation a try.'
              : 'Select a piece to see its legal moves.'}
          </p>
          {error && (
            <p className="error-message" role="alert">
              {error}
            </p>
          )}
          <div className="field">
            <label htmlFor="play-strength">Thinking budget</label>
            <select
              id="play-strength"
              value={strength}
              onChange={(e) => setStrength(Number(e.target.value))}
            >
              <option value="150">Quick · 0.15 s</option>
              <option value="500">Normal · 0.5 s</option>
              <option value="1500">Deep · 1.5 s</option>
            </select>
          </div>
          <div className="new-game-actions">
            <button className="button primary" onClick={() => newGame()}>
              <Icon name="reset" size={16} /> New game
            </button>
            <button className="button secondary" onClick={() => newGame(-player as Color)}>
              Play as {player === 1 ? 'Black' : 'White'}
            </button>
          </div>
          <p className="small-note">
            Play uses the same Fusor model as self-play. It keeps training while you play and
            learns from every game you finish.
          </p>
        </div>
        <div className="panel move-panel">
          <div className="panel-heading">
            <h3>The story so far</h3>
            <span className="micro">{frames.length - 1} plies</span>
          </div>
          <div className="move-list" aria-label="Move history">
            {frames.length === 1 ? (
              <div className="empty-moves">
                <Icon name="rook" size={28} />
                <span>The board is full of possibilities.</span>
                <small>{player === 1 ? 'Make the first move.' : 'Rookie opens with White.'}</small>
              </div>
            ) : (
              Array.from({ length: Math.ceil((frames.length - 1) / 2) }, (_, i) => (
                <div className="move-row" key={i}>
                  <span>{i + 1}.</span>
                  <strong>{frames[i * 2 + 1]?.san}</strong>
                  <strong>{frames[i * 2 + 2]?.san || '…'}</strong>
                </div>
              ))
            )}
          </div>
        </div>
        {analysis && (
          <div className="panel">
            <div className="panel-heading">
              <h3>Inside that last move</h3>
              <Icon name="tree" />
            </div>
            <div className="analysis-meta">
              <span>{analysis.nodes.toLocaleString()} positions</span>
              <span>{analysis.depth} ply explored</span>
            </div>
            {analysis.branches.slice(0, 4).map((branch) => (
              <div className="candidate" key={branch.uci}>
                <code>{branch.uci}</code>
                <div>
                  <span
                    style={{
                      width: `${Math.max(2, (branch.visits / Math.max(1, analysis.branches[0].visits)) * 100)}%`,
                    }}
                  />
                </div>
                <span>{branch.visits}</span>
              </div>
            ))}
            <p className="small-note">Visits per candidate in Rookie’s most recent search.</p>
          </div>
        )}
      </aside>
    </section>
  );
}
