import { useState } from 'react';
import { Square, Play, ChevronDown } from 'lucide-react';
import { DispatchView, LogView, StatusBadge } from './TelemetryPanel';
import { ORACLE_MODES, ORACLE_SIM, SIM_DATASETS } from '../constants';

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

function OracleDropdown({ value, onChange, disabled }) {
  const current = ORACLE_MODES.find(m => m.value === value) || ORACLE_MODES[0];
  return (
    <div className="relative inline-block">
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="appearance-none bg-transparent border border-[var(--rule-night)] hover:border-[var(--gold)] text-[var(--on-night)] text-[11px] uppercase tracking-[0.16em] pl-3 pr-7 py-1.5 font-mono cursor-pointer focus:outline-none focus:border-[var(--gold)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Switch oracle backend"
      >
        {ORACLE_MODES.map(m => (
          <option key={m.value} value={m.value} className="bg-[var(--night)] text-[var(--on-night)]">
            {m.label}
          </option>
        ))}
      </select>
      <ChevronDown size={12} className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-[var(--on-night-faint)]" />
      <span className="sr-only">{current.label}</span>
    </div>
  );
}

export default function Sidebar({ width, fluid = false, hideFeed = false, hideFooter = false, hideHeader = false, oracleMode, onOracleChange, params, setParams, logs, telemetry, loading, onExtract, onStop }) {
  const set = (key) => (e) => setParams(prev => ({ ...prev, [key]: e.target.value }));
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [feedTab, setFeedTab] = useState('dispatch');
  const isSim = oracleMode === ORACLE_SIM;

  return (
    <div
      style={fluid ? { width: '100%' } : { width: `${width}px` }}
      className={`texture-night text-[var(--on-night)] flex flex-col h-full ${hideFeed ? 'overflow-y-auto custom-scrollbar' : 'overflow-hidden'} shrink-0 z-20 relative`}
    >
      <div className="absolute inset-0 pointer-events-none border-r border-[var(--rule-night)]" />

      {/* HEADER */}
      {!hideHeader && (
      <header className="px-7 pt-8 pb-6 shrink-0 relative">
        <div className="flex items-start justify-between gap-4">
          <h1 className="font-display text-[26px] leading-tight text-[var(--on-night)] lowercase">
            k-densest subgraph explorer
          </h1>
          <OracleDropdown value={oracleMode} onChange={onOracleChange} disabled={loading} />
        </div>
        {isSim && (
          <div className="mt-2 text-[11px] tracking-[0.14em] uppercase text-[var(--gold)] font-mono">
            Simulation · annotated citation graphs
          </div>
        )}
      </header>
      )}

      <div className="h-[3px] shrink-0 relative">
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-px bg-[var(--rule-night)]" />
        {loading && <div className="absolute inset-0 marquee-bar" />}
      </div>

      {/* CONFIG */}
      <section className="px-7 pt-5 shrink-0">
        <div className="pb-5">
          {(
            <div className="space-y-5 fade-in">
              {hideHeader && (
                <div>
                  <label className="field-label">Dataset</label>
                  <select
                    value={isSim ? `sim:${params.dataset}` : 'openalex'}
                    disabled={loading}
                    onChange={(e) => {
                      const v = e.target.value;
                      if (v === 'openalex') {
                        onOracleChange(ORACLE_MODES[0].value);
                      } else if (v.startsWith('sim:')) {
                        const ds = v.slice(4);
                        onOracleChange(ORACLE_SIM);
                        setParams(prev => ({ ...prev, dataset: ds }));
                      }
                    }}
                    className="field-input"
                  >
                    <option value="openalex" className="bg-[var(--night)] text-[var(--on-night)]">OpenAlex</option>
                    {SIM_DATASETS.map(d => (
                      <option key={d} value={`sim:${d}`} className="bg-[var(--night)] text-[var(--on-night)]">[S] {d}</option>
                    ))}
                  </select>
                  {isSim && (
                    <div className="mt-1.5 text-[11px] text-[var(--gold)] italic">
                      [S] = simulated · annotated citation graph
                    </div>
                  )}
                </div>
              )}
              {isSim ? (
                <div>
                  <label htmlFor="seed-input" className="field-label flex justify-between items-baseline">
                    <span>Query Node ID</span>
                    <span className="inline-flex items-center gap-1.5 normal-case tracking-normal text-[11px] text-[var(--on-night-faint)]">
                      press <kbd className="kbd-hint" style={{ background: 'var(--night-3)', color: 'var(--on-night-dim)', borderColor: 'var(--rule-night-2)' }}>/</kbd>
                    </span>
                  </label>
                  <input
                    id="seed-input"
                    type="number"
                    min={0}
                    step={1}
                    value={params.queryNode}
                    onChange={set('queryNode')}
                    className="field-input"
                  />
                </div>
              ) : (
                <div>
                  <label htmlFor="seed-input" className="field-label flex justify-between items-baseline">
                    <span>Seed Paper ID</span>
                    <span className="inline-flex items-center gap-1.5 normal-case tracking-normal text-[11px] text-[var(--on-night-faint)]">
                      press <kbd className="kbd-hint" style={{ background: 'var(--night-3)', color: 'var(--on-night-dim)', borderColor: 'var(--rule-night-2)' }}>/</kbd>
                    </span>
                  </label>
                  <input
                    id="seed-input"
                    type="text"
                    value={params.queryNode}
                    onChange={set('queryNode')}
                    placeholder="W2741809807"
                    pattern="W\d+"
                    className="field-input"
                  />
                  {params.queryNode && !/^W\d+$/.test(params.queryNode) && (
                    <div className="mt-1 text-[11px] text-[var(--ember)] italic">
                      expected OpenAlex work ID (e.g. W2741809807)
                    </div>
                  )}
                </div>
              )}
              <div>
                <label className="field-label">Min Community Size · k</label>
                <input type="number" min="2" step="1" value={params.k} onChange={set('k')} className="field-input" />
              </div>
            </div>
          )}

        </div>

        <button
          type="button"
          onClick={() => setAdvancedOpen(v => !v)}
          className="w-full flex items-center justify-between py-3 border-t border-[var(--rule-night)] eyebrow text-[var(--on-night-dim)] hover:text-[var(--on-night)] transition-colors"
          aria-expanded={advancedOpen}
        >
          <span>Advanced</span>
          <ChevronDown size={13} className={`transition-transform ${advancedOpen ? 'rotate-180' : ''}`} />
        </button>

        <div className="pb-6">
          {advancedOpen && (
            <div className={`pt-4 ${hideFeed ? '' : 'max-h-[46vh] overflow-y-auto custom-scrollbar pr-2 -mr-2'} fade-in`}>
              <div className="space-y-5">
              <div className="grid grid-cols-2 gap-x-5 gap-y-5">
                <div>
                  <label className="field-label">Max In-Edges</label>
                  <input type="number" min="0" step="500" value={params.maxInEdges ?? 0} onChange={set('maxInEdges')} className="field-input" />
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
                  <input type="number" min="0" step="1" value={params.cgMinBatch} onChange={set('cgMinBatch')} className="field-input" />
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

      {/* FEED */}
      {!hideFeed && (
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
      )}

      {/* FOOTER */}
      {!hideFooter && (
      <footer className="px-7 pt-5 pb-7 shrink-0 border-t border-[var(--rule-night)] bg-[var(--night-2)] relative">
        <div className="flex items-center gap-3">
          {!loading ? (
            <button onClick={onExtract} className="btn-press" title="Extract Community (⌘/Ctrl + Enter)">
              <Play size={13} fill="currentColor" />
              <span>Extract</span>
              <span className="kbd">⌘↵</span>
            </button>
          ) : (
            <>
              <button disabled className="btn-press">
                <span className="pulse-dot w-1.5 h-1.5 rounded-full bg-[var(--ice)] inline-block" />
                <span>Computing</span>
              </button>
              <button onClick={onStop} className="btn-stop" title="Stop">
                <Square size={14} fill="currentColor" />
              </button>
            </>
          )}
        </div>
        <div className="mt-4 eyebrow text-[var(--on-night-faint)]">
          Session · Active · {isSim ? `${params.dataset}` : 'OpenAlex'}
        </div>
      </footer>
      )}

      {hideFeed && <div className="flex-grow" />}
    </div>
  );
}
