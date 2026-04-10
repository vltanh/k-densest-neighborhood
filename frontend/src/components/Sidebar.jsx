import { useState } from 'react';
import { Square, Play } from 'lucide-react';
import { DispatchView, LogView, StatusBadge } from './TelemetryPanel';

// ─────────────────────────────────────────────────────────────────────────────
// Tab — a minimal underline tab for the night-side chrome.
// Active: gold baseline + high-contrast label. Inactive: faint label.
// ─────────────────────────────────────────────────────────────────────────────
function Tab({ label, active, onClick, badge = null }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`relative pb-2 pt-1 eyebrow transition-colors flex items-center gap-2 ${
        active ? 'text-[var(--on-night)]' : 'text-[var(--on-night-faint)] hover:text-[var(--on-night-dim)]'
      }`}
    >
      <span>{label}</span>
      {badge}
      <span
        className={`absolute left-0 right-0 -bottom-px h-[2px] transition-colors ${
          active ? 'bg-[var(--gold)]' : 'bg-transparent'
        }`}
      />
    </button>
  );
}

function TabBar({ children }) {
  return (
    <div className="flex items-center gap-7 border-b border-[var(--rule-night)]">
      {children}
    </div>
  );
}

export default function Sidebar({ width, params, setParams, logs, telemetry, loading, onExtract, onStop }) {
  const set = (key) => (e) => setParams(prev => ({ ...prev, [key]: e.target.value }));
  const [configTab, setConfigTab] = useState('query');   // 'query' | 'advanced'
  const [feedTab, setFeedTab] = useState('dispatch');    // 'dispatch' | 'log'

  return (
    <div
      style={{ width: `${width}px` }}
      className="texture-night text-[var(--on-night)] flex flex-col h-full overflow-hidden shrink-0 z-20 relative"
    >
      {/* Hairline outer frame */}
      <div className="absolute inset-0 pointer-events-none border-r border-[var(--rule-night)]" />

      {/* ═══ HEADER ══════════════════════════════════════════════════ */}
      <header className="px-7 pt-8 pb-6 shrink-0 relative">
        <h1 className="font-display text-[26px] leading-tight text-[var(--on-night)] lowercase">
          k-densest subgraph explorer
        </h1>
      </header>

      {/* Running status marquee — height reserved even when idle */}
      <div className="h-[3px] shrink-0 relative">
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-px bg-[var(--rule-night)]" />
        {loading && <div className="absolute inset-0 marquee-bar" />}
      </div>

      {/* ═══ CONFIG — Query / Advanced tabs ══════════════════════════ */}
      <section className="px-7 pt-5 shrink-0">
        <TabBar>
          <Tab label="Query" active={configTab === 'query'} onClick={() => setConfigTab('query')} />
          <Tab label="Advanced" active={configTab === 'advanced'} onClick={() => setConfigTab('advanced')} />
        </TabBar>

        <div className="pt-5 pb-6">
          {configTab === 'query' && (
            <div className="space-y-5 fade-in">
              <div>
                <label className="field-label flex justify-between items-baseline">
                  <span>Seed Paper ID</span>
                  <span className="text-[var(--gold)] normal-case tracking-normal text-[11px] italic">entry point</span>
                </label>
                <input type="text" value={params.queryNode} onChange={set('queryNode')} className="field-input" />
              </div>
              <div>
                <label className="field-label">Min Community Size · k</label>
                <input type="number" min="2" step="1" value={params.k} onChange={set('k')} className="field-input" />
              </div>
            </div>
          )}

          {configTab === 'advanced' && (
            <div className="max-h-[46vh] overflow-y-auto custom-scrollbar pr-2 -mr-2 fade-in">
              <div className="space-y-5">
              <div className="grid grid-cols-2 gap-x-5 gap-y-5">
                <div>
                  <label className="field-label">Max In-Edges</label>
                  <input type="number" min="0" step="500" value={params.maxInEdges} onChange={set('maxInEdges')} className="field-input" />
                </div>
                <div>
                  <label className="field-label">Dinkelbach Iter</label>
                  <input type="number" min="1" max="200" step="1" value={params.dinkelbachIter} onChange={set('dinkelbachIter')} className="field-input" />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-x-5 gap-y-5">
                <div>
                  <label className="field-label">Time Limit (s)</label>
                  <input type="number" min="0" step="10" value={params.timeLimit} onChange={set('timeLimit')} className="field-input" />
                </div>
                <div>
                  <label className="field-label">Node Limit</label>
                  <input type="number" min="1" step="1000" value={params.nodeLimit} onChange={set('nodeLimit')} className="field-input" />
                </div>
                <div>
                  <label className="field-label">Gap Tol</label>
                  <input type="number" min="0" step="0.0001" value={params.gapTol} onChange={set('gapTol')} className="field-input" />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-x-5 gap-y-5">
                <div>
                  <label className="field-label">CG Batch Frac</label>
                  <input type="number" min="0.01" max="1.0" step="0.01" value={params.cgBatchFrac} onChange={set('cgBatchFrac')} className="field-input" />
                </div>
                <div>
                  <label className="field-label">Min Batch</label>
                  <input type="number" min="1" step="1" value={params.cgMinBatch} onChange={set('cgMinBatch')} className="field-input" />
                </div>
                <div>
                  <label className="field-label">Max Batch</label>
                  <input type="number" min="1" step="1" value={params.cgMaxBatch} onChange={set('cgMaxBatch')} className="field-input" />
                </div>
              </div>

              <div>
                <label className="field-label">Num Tol</label>
                <input type="number" min="0" step="0.000001" value={params.tol} onChange={set('tol')} className="field-input" />
              </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* ═══ FEED — Dispatch / Log tabs ══════════════════════════════ */}
      <section className="px-7 pb-6 flex flex-col flex-grow overflow-hidden min-h-0">
        <TabBar>
          <Tab label="Dispatch" active={feedTab === 'dispatch'} onClick={() => setFeedTab('dispatch')} />
          <Tab
            label="Log"
            active={feedTab === 'log'}
            onClick={() => setFeedTab('log')}
            badge={
              logs.length > 0 && (
                <span className="text-[10px] font-mono tnum text-[var(--on-night-faint)] normal-case tracking-normal">
                  {logs.length}
                </span>
              )
            }
          />
          <div className="flex-grow" />
          <div className="pb-2">
            <StatusBadge status={telemetry.status} />
          </div>
        </TabBar>

        <div className="pt-5 flex flex-col flex-grow min-h-0">
          {feedTab === 'dispatch' && <DispatchView telemetry={telemetry} loading={loading} />}
          {feedTab === 'log' && <LogView logs={logs} loading={loading} />}
        </div>
      </section>

      {/* ═══ FOOTER / ACTION ═════════════════════════════════════════ */}
      <footer className="px-7 pt-5 pb-7 shrink-0 border-t border-[var(--rule-night)] bg-[var(--night-2)] relative">
        <div className="flex items-center gap-3">
          {!loading ? (
            <button onClick={onExtract} className="btn-press">
              <Play size={13} fill="currentColor" />
              <span>Extract Community</span>
            </button>
          ) : (
            <>
              <button disabled className="btn-press">
                <span className="pulse-dot w-1.5 h-1.5 rounded-full bg-[var(--gold)] inline-block" />
                <span>Computing</span>
              </button>
              <button onClick={onStop} className="btn-stop" title="Stop">
                <Square size={14} fill="currentColor" />
              </button>
            </>
          )}
        </div>
        <div className="mt-4 eyebrow text-[var(--on-night-faint)]">
          Session · Active
        </div>
      </footer>
    </div>
  );
}
