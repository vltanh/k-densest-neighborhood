import { useState, useCallback } from 'react';
import { Square, Play, ChevronDown } from 'lucide-react';
import {
  ORACLE_MODES,
  ORACLE_SIM,
  SIM_DATASETS,
  SOLVER_VARIANTS,
  VARIANT_BP,
  variantSpec,
} from '../constants';

function OracleDropdown({ value, onChange, disabled }) {
  const current = ORACLE_MODES.find(m => m.value === value) || ORACLE_MODES[0];
  return (
    <div className="relative inline-block">
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="appearance-none bg-transparent border border-[var(--rule-night)] hover:border-[var(--gold)] text-[var(--on-night)] text-[length:var(--text-xs)] uppercase tracking-[0.16em] pl-3 pr-7 py-1.5 font-mono cursor-pointer focus:outline-none focus:border-[var(--gold)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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

export default function Sidebar({ width, fluid = false, hideFooter = false, hideHeader = false, oracleMode, onOracleChange, params, setParams, loading, onExtract, onStop }) {
  const set = useCallback(
    (key) => (e) => setParams(prev => ({ ...prev, [key]: e.target.value })),
    [setParams],
  );
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [advancedTab, setAdvancedTab] = useState('solver');
  const isSim = oracleMode === ORACLE_SIM;

  const variant = params.variant || VARIANT_BP;
  const spec = variantSpec(variant);
  const usesK             = !!spec.uses?.k;
  const kOptional         = !!spec.uses?.kOptional;
  const usesKappa         = !!spec.uses?.kappa;
  const usesBfsDepth      = !!spec.uses?.bfsDepth;
  const usesBpInternals   = !!spec.uses?.bpInternals;
  const usesTimeBudget    = spec.uses?.timeBudget !== false;

  return (
    <div
      style={fluid ? { width: '100%' } : { width: `${width}px` }}
      className="texture-night text-[var(--on-night)] flex flex-col h-full overflow-y-auto custom-scrollbar shrink-0 z-20 relative"
    >
      <div className="absolute inset-0 pointer-events-none border-r border-[var(--rule-night)]" />

      {/* HEADER */}
      {!hideHeader && (
      <header className="px-7 pt-8 pb-6 shrink-0 relative">
        <div className="flex items-start justify-between gap-4">
          <h1 className="font-display text-[length:var(--text-xl)] leading-tight text-[var(--on-night)] lowercase">
            k-densest subgraph explorer
          </h1>
          <OracleDropdown value={oracleMode} onChange={onOracleChange} disabled={loading} />
        </div>
        {isSim && (
          <div className="mt-2 text-[length:var(--text-xs)] tracking-[0.14em] uppercase text-[var(--gold)] font-mono">
            Simulation · annotated citation graphs
          </div>
        )}
      </header>
      )}

      <div className="h-[4px] shrink-0 relative">
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
                    <div className="mt-1.5 text-[length:var(--text-xs)] text-[var(--gold)] italic">
                      [S] = simulated · annotated citation graph
                    </div>
                  )}
                </div>
              )}
              {!hideHeader && isSim && (
                <div>
                  <label className="field-label">Dataset</label>
                  <div className="relative">
                    <select
                      value={params.dataset}
                      disabled={loading}
                      onChange={(e) => setParams(prev => ({ ...prev, dataset: e.target.value }))}
                      className="field-input appearance-none pr-8"
                    >
                      {SIM_DATASETS.map(d => (
                        <option key={d} value={d} className="bg-[var(--night)] text-[var(--on-night)]">{d}</option>
                      ))}
                    </select>
                    <ChevronDown size={12} className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none text-[var(--on-night-faint)]" />
                  </div>
                </div>
              )}
              {isSim ? (
                <div>
                  <label htmlFor="seed-input" className="field-label flex justify-between items-baseline">
                    <span>Query Node ID</span>
                    <span className="inline-flex items-center gap-1.5 normal-case tracking-normal text-[length:var(--text-xs)] text-[var(--on-night-faint)]">
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
                    <span className="inline-flex items-center gap-1.5 normal-case tracking-normal text-[length:var(--text-xs)] text-[var(--on-night-faint)]">
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
                    <div className="mt-1 text-[length:var(--text-xs)] text-[var(--danger)] italic">
                      expected OpenAlex work ID (e.g. W2741809807)
                    </div>
                  )}
                </div>
              )}
              <div>
                <label className="field-label">Solver</label>
                <div className="relative">
                  <select
                    value={variant}
                    disabled={loading}
                    onChange={(e) => setParams(prev => ({ ...prev, variant: e.target.value }))}
                    className="field-input appearance-none pr-8"
                  >
                    {SOLVER_VARIANTS.map(v => (
                      <option key={v.value} value={v.value} className="bg-[var(--night)] text-[var(--on-night)]">
                        {v.label}
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={12} className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none text-[var(--on-night-faint)]" />
                </div>
                {spec.blurb && (
                  <div className="mt-1.5 text-[length:var(--text-xs)] leading-snug text-[var(--on-night-faint)] italic">
                    {spec.blurb}
                  </div>
                )}
              </div>

              {(usesK || usesKappa) && (
                <div className={`grid gap-x-5 gap-y-5 ${usesK && usesKappa ? 'grid-cols-2' : 'grid-cols-1'}`}>
                  {usesK && (
                    <div>
                      <label className="field-label">Min Community Size · k</label>
                      <input
                        type="number"
                        min={kOptional ? 0 : 2}
                        step="1"
                        value={params.k}
                        onChange={set('k')}
                        className="field-input"
                      />
                      {kOptional && (
                        <div className="mt-1 text-[length:var(--text-xs)] text-[var(--on-night-faint)] italic">
                          0 skips grow-to-k
                        </div>
                      )}
                    </div>
                  )}
                  {usesKappa && (
                    <div>
                      <label className="field-label">κ · Edge-Connectivity</label>
                      <input type="number" min="0" step="1" value={params.kappa} onChange={set('kappa')} className="field-input" />
                      <div className="mt-1 text-[length:var(--text-xs)] text-[var(--on-night-faint)] italic">0 disables</div>
                    </div>
                  )}
                </div>
              )}

              {usesBfsDepth && (
                <div>
                  <label className="field-label">BFS Depth</label>
                  <input type="number" min="0" step="1" value={params.bfsDepth} onChange={set('bfsDepth')} className="field-input" />
                  <div className="mt-1 text-[length:var(--text-xs)] text-[var(--on-night-faint)] italic">hops out from the seed</div>
                </div>
              )}
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
            <div className="pt-4 fade-in">
              <div role="tablist" aria-label="Advanced settings" className="flex gap-6 border-b border-[var(--rule-night)] mb-4">
                {[
                  { id: 'oracle', label: 'Oracle' },
                  { id: 'solver', label: 'Solver' },
                ].map(t => {
                  const active = advancedTab === t.id;
                  return (
                    <button
                      key={t.id}
                      type="button"
                      role="tab"
                      aria-selected={active}
                      onClick={() => setAdvancedTab(t.id)}
                      className={`relative pb-2 pt-1 eyebrow transition-colors duration-200 ease-out ${
                        active ? 'text-[var(--on-night)]' : 'text-[var(--on-night-faint)] hover:text-[var(--on-night-dim)]'
                      }`}
                    >
                      <span>{t.label}</span>
                      <span
                        className="absolute left-0 right-0 -bottom-px h-[2px] transition-[background-color,transform] duration-200 ease-out origin-left"
                        style={{
                          backgroundColor: active ? 'var(--gold)' : 'transparent',
                          transform: active ? 'scaleX(1)' : 'scaleX(0.4)',
                        }}
                      />
                    </button>
                  );
                })}
              </div>

              {advancedTab === 'oracle' && (
                <div className="space-y-5 fade-in">
                  <div>
                    <label className="field-label">Max In-Edges</label>
                    <input type="number" min="0" step="500" value={params.maxInEdges ?? 0} onChange={set('maxInEdges')} className="field-input" />
                    <div className="mt-1 text-[length:var(--text-xs)] text-[var(--on-night-faint)] italic">incoming edges to fetch · 0 disables</div>
                  </div>
                </div>
              )}

              {advancedTab === 'solver' && (
                <div className="space-y-5 fade-in">
                  {usesBpInternals ? (
                    <>
                      {usesTimeBudget && (
                        <>
                          <div className="grid grid-cols-2 gap-x-5 gap-y-5">
                            <div>
                              <label className="field-label">Soft Time (s)</label>
                              <input type="number" min="-1" step="10" value={params.timeLimit} onChange={set('timeLimit')} className="field-input" />
                              <div className="mt-1 text-[length:var(--text-xs)] text-[var(--on-night-faint)] italic">no-improvement cap · -1 disables</div>
                            </div>
                            <div>
                              <label className="field-label">Hard Time (s)</label>
                              <input type="number" min="-1" step="10" value={params.hardTimeLimit} onChange={set('hardTimeLimit')} className="field-input" />
                              <div className="mt-1 text-[length:var(--text-xs)] text-[var(--on-night-faint)] italic">-1 disables</div>
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-x-5 gap-y-5">
                            <div>
                              <label className="field-label">Node Limit</label>
                              <input type="number" min="-1" step="1000" value={params.nodeLimit} onChange={set('nodeLimit')} className="field-input" />
                              <div className="mt-1 text-[length:var(--text-xs)] text-[var(--on-night-faint)] italic">-1 disables</div>
                            </div>
                            <div>
                              <label className="field-label">Gap Tol</label>
                              <input type="number" min="-1" step="0.0001" value={params.gapTol} onChange={set('gapTol')} className="field-input" />
                              <div className="mt-1 text-[length:var(--text-xs)] text-[var(--on-night-faint)] italic">-1 disables</div>
                            </div>
                          </div>
                        </>
                      )}

                      <div>
                        <label className="field-label">Dinkelbach Iter</label>
                        <input type="number" min="-1" step="1" value={params.dinkelbachIter} onChange={set('dinkelbachIter')} className="field-input" />
                        <div className="mt-1 text-[length:var(--text-xs)] text-[var(--on-night-faint)] italic">-1 disables</div>
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
                    </>
                  ) : (
                    <div className="text-[length:var(--text-sm)] text-[var(--on-night-faint)] italic">
                      {spec.label} exposes no inner-loop tuning. The variant runs to completion using its own stopping rule.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </section>

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
        <div className="mt-4 eyebrow text-[var(--on-night-faint)] flex items-center gap-2 flex-wrap">
          <span>{isSim ? params.dataset : 'OpenAlex'}</span>
          <span className="rule-dot" />
          <span className="text-[var(--gold)]">{spec.label}</span>
        </div>
      </footer>
      )}

      <div className="flex-grow" />
    </div>
  );
}
