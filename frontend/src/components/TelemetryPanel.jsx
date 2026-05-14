import { memo, useEffect, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { classifyLine } from '../utils/telemetryParser';
import { fmtFloat, fmtInt } from '../utils/format';

// ── Line styling ──────────────────────────────────────────────────────────────
const LINE_STYLES = {
  header:    'text-[var(--gold)] font-semibold',
  separator: 'text-[var(--on-night-faint)]',
  iteration: 'text-[var(--on-night)] font-semibold',
  incumbent: 'text-[var(--gold)] font-semibold',
  prune:     'text-[#E8B04E]',
  warn:      'text-[#E08A3C]',
  success:   'text-[#94C466] font-semibold',
  error:     'text-[var(--danger)] font-semibold',
  info:      'text-[#8FB070]',
};

const LogLine = memo(function LogLine({ text }) {
  const cls = LINE_STYLES[classifyLine(text)] ?? LINE_STYLES.info;
  return <div className={`${cls} whitespace-pre-wrap`}>{text}</div>;
});

// ── Elapsed time ──────────────────────────────────────────────────────────────
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
    <span className="text-[length:var(--text-xs)] uppercase tracking-[0.18em] text-[var(--on-night-faint)] font-semibold">
      {label}
    </span>
    <span
      className={`font-mono tnum truncate mt-1 ${big ? 'text-[length:var(--text-lg)]' : 'text-[length:var(--text-base)]'} ${
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
  error:     { label: 'Retracted', cls: 'text-[var(--danger)]',           dot: 'bg-[var(--danger)]' },
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
  const bestDensDisplay = telemetry.incumbent ? fmtFloat(telemetry.incumbent.density, 4) : null;
  const bestSizeDisplay = telemetry.incumbent ? fmtInt(telemetry.incumbent.size) : null;
  const showFinalBlock =
    telemetry.density != null || telemetry.solverTime != null || telemetry.status === 'converged' ||
    telemetry.qualities != null;
  const showLiveBlock =
    telemetry.iteration != null || telemetry.lambda != null || telemetry.incumbent != null ||
    telemetry.startedAt != null || telemetry.bbNodes != null || telemetry.lpSolves != null;

  return (
    <div className="flex flex-col flex-grow min-h-0 overflow-x-hidden fade-in telemetry-container">
      {/* KPI grids — reflow 4→2 cols under 380px via container query */}
      <div className="shrink-0">
        {showLiveBlock && (
        <div className="telemetry-grid pb-3">
          <Stat label="Iter"      value={fmtInt(telemetry.iteration)} accent />
          <Stat label="λ"         value={fmtFloat(telemetry.lambda, 6)} accent />
          <Stat label="Best Obj"  value={bestObjDisplay} accent />
          <Stat label="Best Dens" value={bestDensDisplay} accent />
          <Stat label="Best Size" value={bestSizeDisplay} accent />
          <Stat label="Elapsed"   value={elapsed} />
          <Stat label="Started"   value={telemetry.startedAt ? new Date(telemetry.startedAt).toLocaleTimeString() : null} />
          <Stat label="B&B"       value={fmtInt(telemetry.bbNodes || null)} />
          <Stat label="LP"        value={fmtInt(telemetry.lpSolves || null)} />
        </div>
        )}

        {showFinalBlock && (
          <div className="telemetry-grid border-t border-[var(--rule-night)] pt-3 mt-3 pb-1 fade-in">
            <Stat label="Avg Int Degree"  value={fmtFloat(telemetry.qualities?.avg_internal_degree, 4)} accent />
            <Stat label="Edge Density"    value={fmtFloat(telemetry.qualities?.edge_density, 4)} accent />
            <Stat label="Ext Conductance" value={fmtFloat(telemetry.qualities?.ext_conductance, 4)} accent />
            <Stat label="Int NCut"        value={fmtFloat(telemetry.qualities?.int_ncut, 4)} accent />
            <Stat label="Size"            value={fmtInt(telemetry.size)} />
            <Stat label="Solve"           value={telemetry.solverTime != null ? `${telemetry.solverTime.toFixed(3)}s` : null} />
            {!!telemetry.bbNodes && (
              <Stat label="B&B" value={fmtInt(telemetry.bbNodes)} />
            )}
            {!!telemetry.lpSolves && (
              <Stat label="LP" value={fmtInt(telemetry.lpSolves)} />
            )}
            {telemetry.columnsGenerated != null && (
              <Stat label="Cols Gen" value={fmtInt(telemetry.columnsGenerated)} />
            )}
            {telemetry.cutsAdded != null && (
              <Stat label="BQP Cuts" value={fmtInt(telemetry.cutsAdded)} />
            )}
          </div>
        )}
      </div>

      {/* Empty-state hint */}
      {!showFinalBlock && telemetry.iteration == null && telemetry.blacklisted.length === 0 && (
        <div className="pt-6 text-[length:var(--text-sm)] text-[var(--on-night-faint)] italic leading-snug shrink-0">
          No run in progress. Press <span className="text-[var(--gold)] not-italic">Extract Community</span> to
          begin — live metrics will appear here.
        </div>
      )}

      {/* Blacklist — grows to fill remaining space in the tab */}
      {telemetry.blacklisted.length > 0 && (
        <div className="mt-4 flex flex-col flex-grow min-h-0">
          <div className="flex items-baseline justify-between mb-2 shrink-0">
            <span className="eyebrow text-[var(--on-night-faint)]">Blacklisted</span>
            <span className="font-mono tnum text-[length:var(--text-xs)] text-[var(--on-night-faint)]">
              {telemetry.blacklisted.length}
            </span>
          </div>
          <ul className="space-y-1 flex-grow min-h-0 overflow-y-auto overflow-x-hidden custom-scrollbar">
            {telemetry.blacklisted.map((b) => (
              <li
                key={b.id}
                className="flex items-start gap-2 text-[length:var(--text-sm)] leading-snug"
                title={b.reason}
              >
                <span className="text-[var(--danger)] mt-[2px]">✕</span>
                <div className="min-w-0 flex-grow">
                  <div className="font-mono tnum text-[var(--on-night)] truncate">{b.id}</div>
                  {(b.phase || b.reason) && (
                    <div className="text-[length:var(--text-xs)] text-[var(--on-night-faint)] italic truncate">
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
    <div className="flex flex-col flex-grow min-h-0 overflow-hidden relative scan-line">
      {!stickToBottom && logs.length > 0 && (
        <button
          type="button"
          onClick={jumpToBottom}
          className="absolute top-2 right-3 z-20 flex items-center gap-1 text-[length:var(--text-xs)] text-[var(--gold)] hover:text-[var(--on-night)] bg-[var(--night-2)]/90 backdrop-blur border border-[var(--rule-night)] px-2 py-1 transition-colors normal-case tracking-normal"
          title="Resume auto-scroll"
        >
          <ChevronDown size={12} /> latest
        </button>
      )}
      <div
        ref={containerRef}
        onScroll={onScroll}
        className="p-3 overflow-y-auto custom-scrollbar font-mono text-[length:var(--text-sm)] leading-[1.65] flex-grow min-h-0 relative z-10"
      >
        {logs.length === 0 ? (
          <div className="text-[var(--on-night-faint)] text-[length:var(--text-sm)] italic">
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
