export type IconName =
  | 'rook'
  | 'grid'
  | 'play'
  | 'pause'
  | 'arrow'
  | 'download'
  | 'upload'
  | 'bolt'
  | 'cpu'
  | 'chevron'
  | 'close'
  | 'reset'
  | 'flip'
  | 'undo'
  | 'check'
  | 'settings'
  | 'expand'
  | 'tree'
  | 'sun'
  | 'info';
const paths: Record<IconName, React.ReactNode> = {
  rook: (
    <>
      <path d="M5 3h3v4h3V3h2v4h3V3h3v7l-3 3v6H8v-6l-3-3Z" />
      <path d="M6 21h12M8 16h8" />
    </>
  ),
  grid: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </>
  ),
  play: <path d="m8 4 12 8-12 8Z" />,
  pause: (
    <>
      <path d="M8 5v14M16 5v14" strokeWidth="3" />
    </>
  ),
  arrow: (
    <>
      <path d="M4 12h15m-6-6 6 6-6 6" />
    </>
  ),
  download: (
    <>
      <path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" />
    </>
  ),
  upload: (
    <>
      <path d="M12 16V4m-5 5 5-5 5 5M4 17v4h16v-4" />
    </>
  ),
  bolt: <path d="m13 2-9 12h7l-1 8 10-13h-7Z" />,
  cpu: (
    <>
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4" />
      <rect x="9" y="9" width="6" height="6" rx="1" />
    </>
  ),
  chevron: <path d="m8 4 8 8-8 8" />,
  close: <path d="m6 6 12 12M6 18 18 6" />,
  reset: (
    <>
      <path d="M4 10a8 8 0 1 1 1 7M4 4v6h6" />
    </>
  ),
  flip: (
    <>
      <path d="M4 7h15m-4-4 4 4-4 4M20 17H5m4-4-4 4 4 4" />
    </>
  ),
  undo: (
    <>
      <path d="m9 5-6 6 6 6M3 11h11a6 6 0 0 1 6 6" />
    </>
  ),
  check: <path d="m5 12 4 4L19 6" />,
  settings: (
    <>
      <path d="M4 6h16M4 12h16M4 18h16" />
      <circle cx="9" cy="6" r="2" />
      <circle cx="16" cy="12" r="2" />
      <circle cx="8" cy="18" r="2" />
    </>
  ),
  expand: (
    <>
      <path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5" />
    </>
  ),
  tree: (
    <>
      <rect x="9" y="2" width="6" height="5" rx="1" />
      <path d="M12 7v5M5 17v-5h14v5" />
      <rect x="2" y="17" width="6" height="5" rx="1" />
      <rect x="16" y="17" width="6" height="5" rx="1" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v6m0-10v.1" />
    </>
  ),
};
export function Icon({
  name,
  size = 18,
  className = '',
}: {
  name: IconName;
  size?: number;
  className?: string;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      {paths[name]}
    </svg>
  );
}
