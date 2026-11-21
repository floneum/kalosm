import type { HistoryPoint } from '../useTraining';

export function LossChart({ history }: { history: HistoryPoint[] }) {
  const max = Math.max(0.012, ...history.map((p) => p.loss)) * 1.15;
  const points = history.map(
    (p, i) => `${38 + (i / Math.max(history.length - 1, 1)) * 270},${118 - (p.loss / max) * 96}`,
  );
  const path = points.length ? `M${points.join(' L')}` : '';
  return (
    <div className="loss-chart">
      <svg
        viewBox="0 0 328 156"
        role="img"
        aria-label={
          history.length
            ? `Training value loss, latest ${history[history.length - 1].loss.toFixed(4)}. Lower values mean closer agreement with training targets.`
            : 'Training loss chart is waiting for its first batch.'
        }
      >
        <defs>
          <linearGradient id="loss-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#df6b43" stopOpacity=".15" />
            <stop offset="100%" stopColor="#df6b43" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[22, 54, 86, 118].map((y, i) => (
          <g key={y}>
            <line x1="38" x2="310" y1={y} y2={y} stroke="#e8e7e0" strokeDasharray="3 4" />
            <text x="28" y={y + 3} textAnchor="end">
              {(max * (1 - i / 3)).toFixed(3)}
            </text>
          </g>
        ))}
        {history.length > 1 && (
          <>
            <path d={`${path} L308,118 L38,118 Z`} fill="url(#loss-fill)" />
            <path d={path} fill="none" stroke="#d65e37" strokeWidth="2" strokeLinejoin="round" />
            <circle
              cx="308"
              cy={118 - (history[history.length - 1].loss / max) * 96}
              r="3.5"
              fill="#d65e37"
            />
          </>
        )}
        <text x="38" y="143">
          {history.length ? `${Math.floor(history[0].time)}s` : '0s'}
        </text>
        <text x="310" y="143" textAnchor="end">
          {history.length ? `${Math.floor(history[history.length - 1].time)}s` : 'Time'}
        </text>
      </svg>
      {history.length < 2 && <span className="chart-empty">Collecting the first batches…</span>}
    </div>
  );
}
