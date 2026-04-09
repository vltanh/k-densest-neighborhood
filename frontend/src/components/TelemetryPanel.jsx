import { memo, useEffect, useRef, useState } from 'react';
import { Terminal, ChevronDown } from 'lucide-react';
import { classifyLine } from '../utils/telemetryParser';

// ── Line styling ──────────────────────────────────────────────────────────────
// Each classification maps to Tailwind utility classes applied to the log row.
// Keep these stable identifiers in sync with classifyLine() in telemetryParser.
const LINE_STYLES = {
  header:    'text-cyan-400 font-semibold',
  separator: 'text-gray-600',
  iteration: 'text-indigo-300 font-semibold',
  incumbent: 'text-yellow-300 font-semibold',
  prune:     'text-amber-300',
  warn:      'text-orange-400',
  success:   'text-emerald-300 font-semibold',
  error:     'text-red-400 font-semibold',
  info:      'text-green-400',
};

// Memoised so appending a new log line doesn't re-render every previous row.
const LogLine = memo(function LogLine({ text }) {
  const cls = LINE_STYLES[classifyLine(text)] ?? LINE_STYLES.info;
  return <div className={`${cls} whitespace-pre-wrap`}>{text}</div>;
});

// ── Formatting helpers ────────────────────────────────────────────────────────
const fmtInt = (x) => (x == null ? null : Number(x).toLocaleString());

const fmtFloat = (x, digits = 6) => {
  if (x == null || Number.isNaN(x)) return null;
  // Use fixed notation for values that round to something readable, exponential
  // when the magnitude is tiny (e.g. λ = 3.2e-5 during fractional phases).
  const abs = Math.abs(x);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return x.toExponential(3);
  return x.toFixed(digits);
};

const fmtElapsed = (startedAt, finishedAt) => {
  if (!startedAt) return null;
  const end = finishedAt ?? Date.now();
  const s = Math.max(0, (end - startedAt) / 1000);
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.floor(s % 60);
  return `${m}m ${rem.toString().padStart(2, '0')}s`;
};

// ── Small presentational pieces ───────────────────────────────────────────────
const Stat = ({ label, value, accent = false }) => (
  <div className="flex flex-col min-w-0">
    <span className="text-[9px] uppercase tracking-wider text-gray-500 font-semibold">{label}</span>
    <span
      className={`text-sm font-mono truncate ${accent ? 'text-indigo-300' : 'text-gray-100'}`}
      title={value ?? ''}
    >
      {value ?? <span className="text-gray-600">—</span>}
    </span>
  </div>
);

const STATUS_STYLES = {
  idle:      { label: 'Idle',      cls: 'bg-gray-700/60 text-gray-300 border-gray-600',            dot: 'bg-gray-400' },
  running:   { label: 'Running',   cls: 'bg-indigo-500/15 text-indigo-200 border-indigo-500/40',   dot: 'bg-indigo-400 animate-pulse' },
  converged: { label: 'Converged', cls: 'bg-emerald-500/15 text-emerald-200 border-emerald-500/40', dot: 'bg-emerald-400' },
  stopped:   { label: 'Stopped',   cls: 'bg-amber-500/15 text-amber-200 border-amber-500/40',       dot: 'bg-amber-400' },
  error:     { label: 'Error',     cls: 'bg-red-500/15 text-red-200 border-red-500/40',             dot: 'bg-red-400' },
};

const StatusBadge = ({ status }) => {
  const s = STATUS_STYLES[status] ?? STATUS_STYLES.idle;
  return (
    <span className={`inline-flex items-center gap-1.5 text-[10px] uppercase font-semibold tracking-wider px-2 py-0.5 rounded-full border ${s.cls}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
      {s.label}
    </span>
  );
};

// ── Main component ────────────────────────────────────────────────────────────
export default function TelemetryPanel({ telemetry, logs, loading }) {
  const containerRef = useRef(null);

  // Smart auto-scroll: only stick to the bottom while the user is already at
  // the bottom. If they scroll up to read earlier lines, we stop forcing them
  // back down and expose a "Jump" button instead.
  const [stickToBottom, setStickToBottom] = useState(true);

  // Tick the clock while a run is in flight so Elapsed keeps updating even
  // if no new log lines arrive.
  const [, forceTick] = useState(0);
  useEffect(() => {
    if (!loading || !telemetry.startedAt) return undefined;
    const id = setInterval(() => forceTick(t => (t + 1) % 1_000_000), 500);
    return () => clearInterval(id);
  }, [loading, telemetry.startedAt]);

  // On new logs, if the user was at the bottom we snap down using direct
  // scrollTop assignment (no smooth-scroll — it fights streaming updates).
  useEffect(() => {
    if (!stickToBottom) return;
    const el = containerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs, stickToBottom]);

  const onScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    if (nearBottom !== stickToBottom) setStickToBottom(nearBottom);
  };

  const jumpToBottom = () => {
    const el = containerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setStickToBottom(true);
  };

  const elapsed = fmtElapsed(telemetry.startedAt, telemetry.finishedAt);
  const incumbentDisplay = telemetry.incumbent
    ? `${fmtFloat(telemetry.incumbent.obj, 4)} · k=${telemetry.incumbent.size}`
    : null;
  const showFinalBlock =
    telemetry.density != null || telemetry.solverTime != null || telemetry.status === 'converged';

  return (
    <div className="flex flex-col flex-grow min-h-0 gap-3">
      {/* KPI dashboard ─────────────────────────────────────────────────────── */}
      <div className="rounded-lg border border-gray-700 bg-gray-800/60 shadow-inner p-3 shrink-0">
        <div className="flex items-center justify-between mb-2.5">
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-gray-400 font-semibold">
            <Terminal size={12} className="text-indigo-400" />
            Live Telemetry
          </div>
          <StatusBadge status={telemetry.status} />
        </div>

        <div className="grid grid-cols-3 gap-y-2.5 gap-x-3">
          <Stat label="Iter"       value={fmtInt(telemetry.iteration)} accent />
          <Stat label="λ"          value={fmtFloat(telemetry.lambda, 6)} accent />
          <Stat label="Elapsed"    value={elapsed} />
          <Stat label="B&B Nodes"  value={fmtInt(telemetry.bbNodes || null)} />
          <Stat label="LP Solves"  value={fmtInt(telemetry.lpSolves || null)} />
          <Stat label="Incumbent"  value={incumbentDisplay} />
        </div>

        {showFinalBlock && (
          <div className="mt-3 pt-2.5 border-t border-gray-700/70 grid grid-cols-3 gap-y-2 gap-x-3">
            <Stat label="Density"     value={fmtFloat(telemetry.density, 6)} accent />
            <Stat label="Size"        value={fmtInt(telemetry.size)} />
            <Stat label="Solver Time" value={telemetry.solverTime != null ? `${telemetry.solverTime.toFixed(3)}s` : null} />
            <Stat label="Cols Gen"    value={fmtInt(telemetry.columnsGenerated)} />
            <Stat label="BQP Cuts"    value={fmtInt(telemetry.cutsAdded)} />
          </div>
        )}
      </div>

      {/* Log stream ────────────────────────────────────────────────────────── */}
      <div className="flex flex-col flex-grow min-h-0 border border-gray-700 rounded bg-black shadow-inner overflow-hidden relative">
        <div className="bg-gray-800 px-3 py-1.5 text-[10px] text-gray-400 uppercase tracking-wider flex items-center justify-between border-b border-gray-700 shrink-0">
          <div className="flex items-center gap-2">
            <span>Solver Log</span>
            <span className="text-gray-600">· {logs.length}</span>
          </div>
          {!stickToBottom && logs.length > 0 && (
            <button
              type="button"
              onClick={jumpToBottom}
              className="flex items-center gap-1 text-[10px] text-indigo-300 hover:text-indigo-200 transition-colors"
              title="Resume auto-scroll"
            >
              <ChevronDown size={12} /> Jump to latest
            </button>
          )}
        </div>
        <div
          ref={containerRef}
          onScroll={onScroll}
          className="p-3 overflow-y-auto custom-scrollbar font-mono text-[10px] leading-relaxed flex-grow min-h-0"
        >
          {logs.length === 0 ? (
            <div className="text-gray-600 italic">Waiting for solver output…</div>
          ) : (
            logs.map((log, idx) => <LogLine key={idx} text={log} />)
          )}
          {loading && <span className="animate-pulse text-green-500">_</span>}
        </div>
      </div>
    </div>
  );
}
