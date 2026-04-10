import { memo, useEffect, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { classifyLine } from '../utils/telemetryParser';

// ── Line styling ──────────────────────────────────────────────────────────────
const LINE_STYLES = {
  header:    'text-[var(--gold)] font-semibold',
  separator: 'text-[var(--on-night-faint)]',
  iteration: 'text-[var(--on-night)] font-semibold',
  incumbent: 'text-[var(--gold)] font-semibold',
  prune:     'text-[#E8B04E]',
  warn:      'text-[#E08A3C]',
  success:   'text-[#94C466] font-semibold',
  error:     'text-[var(--vermillion)] font-semibold',
  info:      'text-[#8FB070]',
};

const LogLine = memo(function LogLine({ text }) {
  const cls = LINE_STYLES[classifyLine(text)] ?? LINE_STYLES.info;
  return <div className={`${cls} whitespace-pre-wrap`}>{text}</div>;
});

// ── Formatting helpers ────────────────────────────────────────────────────────
const fmtInt = (x) => (x == null ? null : Number(x).toLocaleString());

const fmtFloat = (x, digits = 6) => {
  if (x == null || Number.isNaN(x)) return null;
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

// ── Stat cell — editorial, no boxes ───────────────────────────────────────────
const Stat = ({ label, value, accent = false, big = false }) => (
  <div className="flex flex-col min-w-0 py-1">
    <span className="text-[10px] uppercase tracking-[0.18em] text-[var(--on-night-faint)] font-semibold">
      {label}
    </span>
    <span
      className={`font-mono tnum truncate mt-1 ${big ? 'text-[20px]' : 'text-[15px]'} ${
        accent ? 'text-[var(--gold)]' : 'text-[var(--on-night)]'
      }`}
      title={value ?? ''}
    >
      {value ?? <span className="text-[var(--on-night-faint)]">—</span>}
    </span>
  </div>
);

const STATUS_STYLES = {
  idle:      { label: 'Standby',   cls: 'text-[var(--on-night-faint)]',  dot: 'bg-[var(--on-night-faint)]' },
  running:   { label: 'On-Press',  cls: 'text-[var(--gold)]',             dot: 'bg-[var(--gold)] pulse-dot' },
  converged: { label: 'Filed',     cls: 'text-[#94C466]',                 dot: 'bg-[#94C466]' },
  stopped:   { label: 'Halted',    cls: 'text-[#E08A3C]',                 dot: 'bg-[#E08A3C]' },
  error:     { label: 'Retracted', cls: 'text-[var(--vermillion)]',       dot: 'bg-[var(--vermillion)]' },
};

export const StatusBadge = ({ status }) => {
  const s = STATUS_STYLES[status] ?? STATUS_STYLES.idle;
  return (
    <span className={`inline-flex items-center gap-2 eyebrow ${s.cls}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
      {s.label}
    </span>
  );
};

// ═══════════════════════════════════════════════════════════════════════════
// Dispatch view — KPI cards, scrolls internally if needed.
// ═══════════════════════════════════════════════════════════════════════════
export function DispatchView({ telemetry, loading }) {
  // Re-tick elapsed while running.
  const [, forceTick] = useState(0);
  useEffect(() => {
    if (!loading || !telemetry.startedAt) return undefined;
    const id = setInterval(() => forceTick(t => (t + 1) % 1_000_000), 500);
    return () => clearInterval(id);
  }, [loading, telemetry.startedAt]);

  const elapsed = fmtElapsed(telemetry.startedAt, telemetry.finishedAt);
  const bestObjDisplay  = telemetry.incumbent ? fmtFloat(telemetry.incumbent.obj, 4) : null;
  const bestSizeDisplay = telemetry.incumbent ? fmtInt(telemetry.incumbent.size) : null;
  const showFinalBlock =
    telemetry.density != null || telemetry.solverTime != null || telemetry.status === 'converged';

  return (
    <div className="flex flex-col flex-grow min-h-0 overflow-x-hidden fade-in">
      {/* Fixed-height KPI grids at the top */}
      <div className="shrink-0">
        <div className="grid grid-cols-4 gap-x-4 border-t border-b border-[var(--rule-night)] py-3">
          <Stat label="Iter"      value={fmtInt(telemetry.iteration)} accent />
          <Stat label="λ"         value={fmtFloat(telemetry.lambda, 6)} accent />
          <Stat label="Best Obj"  value={bestObjDisplay} accent />
          <Stat label="Best Size" value={bestSizeDisplay} accent />
          <Stat label="Elapsed"   value={elapsed} />
          <Stat label="Started"   value={telemetry.startedAt ? new Date(telemetry.startedAt).toLocaleTimeString() : null} />
          <Stat label="B&B"       value={fmtInt(telemetry.bbNodes || null)} />
          <Stat label="LP"        value={fmtInt(telemetry.lpSolves || null)} />
        </div>

        {showFinalBlock && (
          <div className="grid grid-cols-4 gap-x-4 border-b border-[var(--rule-night)] py-3 fade-in">
            <Stat label="Density"  value={fmtFloat(telemetry.density, 6)} accent />
            <Stat label="Size"     value={fmtInt(telemetry.size)} accent />
            <Stat label="Solve"    value={telemetry.solverTime != null ? `${telemetry.solverTime.toFixed(3)}s` : null} />
            <Stat label="Iters"    value={fmtInt(telemetry.iteration)} />
            <Stat label="B&B"      value={fmtInt(telemetry.bbNodes || null)} />
            <Stat label="LP"       value={fmtInt(telemetry.lpSolves || null)} />
            <Stat label="Cols Gen" value={fmtInt(telemetry.columnsGenerated)} />
            <Stat label="BQP Cuts" value={fmtInt(telemetry.cutsAdded)} />
          </div>
        )}
      </div>

      {/* Empty-state hint */}
      {!showFinalBlock && telemetry.iteration == null && telemetry.blacklisted.length === 0 && (
        <div className="pt-6 text-[14px] text-[var(--on-night-faint)] italic leading-snug shrink-0">
          No run in progress. Press <span className="text-[var(--gold)] not-italic">Extract Community</span> to
          begin — live metrics will appear here.
        </div>
      )}

      {/* Blacklist — grows to fill remaining space in the tab */}
      {telemetry.blacklisted.length > 0 && (
        <div className="mt-4 flex flex-col flex-grow min-h-0">
          <div className="flex items-baseline justify-between mb-2 shrink-0">
            <span className="eyebrow text-[var(--on-night-faint)]">Blacklisted</span>
            <span className="font-mono tnum text-[11px] text-[var(--on-night-faint)]">
              {telemetry.blacklisted.length}
            </span>
          </div>
          <ul className="space-y-1 flex-grow min-h-0 overflow-y-auto overflow-x-hidden custom-scrollbar">
            {telemetry.blacklisted.map((b) => (
              <li
                key={b.id}
                className="flex items-start gap-2 text-[12px] leading-snug"
                title={b.reason}
              >
                <span className="text-[var(--vermillion)] mt-[2px]">✕</span>
                <div className="min-w-0 flex-grow">
                  <div className="font-mono tnum text-[var(--on-night)] truncate">{b.id}</div>
                  {(b.phase || b.reason) && (
                    <div className="text-[11px] text-[var(--on-night-faint)] italic truncate">
                      {b.phase && <span>{b.phase}</span>}
                      {b.phase && b.reason && <span> · </span>}
                      {b.reason}
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Log view — raw streaming output.
// ═══════════════════════════════════════════════════════════════════════════
export function LogView({ logs, loading }) {
  const containerRef = useRef(null);
  const [stickToBottom, setStickToBottom] = useState(true);

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

  return (
    <div className="flex flex-col flex-grow min-h-0 border border-[var(--rule-night)] bg-[var(--night)] overflow-hidden relative scan-line">
      <div className="px-3 py-2.5 eyebrow text-[var(--on-night-faint)] flex items-center justify-between border-b border-[var(--rule-night)] shrink-0">
        <div className="flex items-center gap-2">
          <span>solver.log</span>
          <span className="text-[var(--on-night-faint)] normal-case tnum font-mono text-[12px]">{logs.length}</span>
        </div>
        {!stickToBottom && logs.length > 0 && (
          <button
            type="button"
            onClick={jumpToBottom}
            className="flex items-center gap-1 text-[11px] text-[var(--gold)] hover:text-[var(--on-night)] transition-colors normal-case tracking-normal"
            title="Resume auto-scroll"
          >
            <ChevronDown size={12} /> latest
          </button>
        )}
      </div>
      <div
        ref={containerRef}
        onScroll={onScroll}
        className="p-3 overflow-y-auto custom-scrollbar font-mono text-[12px] leading-[1.65] flex-grow min-h-0 relative z-10"
      >
        {logs.length === 0 ? (
          <div className="text-[var(--on-night-faint)] text-[14px] italic">
            Awaiting dispatch from the compositor…
          </div>
        ) : (
          logs.map((log, idx) => <LogLine key={idx} text={log} />)
        )}
        {loading && <span className="pulse-dot inline-block w-2 h-3.5 bg-[var(--gold)] ml-0.5" />}
      </div>
    </div>
  );
}
