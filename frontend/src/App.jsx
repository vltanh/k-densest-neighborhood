import { useState, useEffect } from 'react';
import { Menu, Play, Square, Network, SlidersHorizontal, Activity, ChevronDown, ChevronUp } from 'lucide-react';
import { API_BASE_URL, ORACLE_OPENALEX, ORACLE_SIM, SIM_DATASETS, VARIANT_BP } from './constants';
import { useDragResize } from './hooks/useDragResize';
import { useMediaQuery } from './hooks/useMediaQuery';
import { useSubgraphExtractor } from './hooks/useSubgraphExtractor';
import Sidebar from './components/Sidebar';
import GraphView from './components/GraphView';
import PaperLedger from './components/PaperLedger';
import PaperModal from './components/PaperModal';
import { DispatchView, LogView } from './components/TelemetryPanel';

const SHARED_DEFAULTS = {
  variant: VARIANT_BP,
  k: 5,
  kappa: 0,
  bfsDepth: 1,
  timeLimit: -1,
  nodeLimit: -1,
  maxInEdges: 0,
  gapTol: -1,
  dinkelbachIter: -1,
  cgBatchFrac: 1.0,
  cgMinBatch: 0,
  cgMaxBatch: 50,
  tol: 0.000001,
};

const DEFAULT_OPENALEX_PARAMS = {
  ...SHARED_DEFAULTS,
  queryNode: 'W2741809807',
};

const DEFAULT_SIM_PARAMS = {
  ...SHARED_DEFAULTS,
  dataset: SIM_DATASETS[0],
  queryNode: '0',
};

export default function App() {
  const [sessionId] = useState(() => {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
    // Fallback for non-secure contexts (e.g. LAN HTTP from mobile)
    return 'sid-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
  });
  const [oracleMode, setOracleMode] = useState(ORACLE_OPENALEX);
  const [openalexParams, setOpenalexParams] = useState(DEFAULT_OPENALEX_PARAMS);
  const [simParams, setSimParams] = useState(DEFAULT_SIM_PARAMS);

  const params = oracleMode === ORACLE_SIM ? simParams : openalexParams;
  const setParams = oracleMode === ORACLE_SIM ? setSimParams : setOpenalexParams;

  const [extractedSeed, setExtractedSeed] = useState(DEFAULT_OPENALEX_PARAMS.queryNode);
  const [extractedMode, setExtractedMode] = useState(ORACLE_OPENALEX);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [clickedNode, setClickedNode] = useState(null);
  const [modalContent, setModalContent] = useState(null);

  const isMobile = useMediaQuery('(max-width: 900px)');
  const [mobileTab, setMobileTab] = useState('atlas');

  const { sidebarWidth, setSidebarWidth, setIsDraggingSidebar,
          ledgerHeightPct, setLedgerHeightPct, setIsDraggingLedger } = useDragResize();

  const { graphData, logs, telemetry, meta, loading, error, extractSubgraph, stopExtraction } = useSubgraphExtractor(sessionId);

  const fetchBibtex = async (doi) => {
    setModalContent({ type: 'loading_bib' });
    try {
      const res = await fetch(`${API_BASE_URL}/api/bibtex?doi=${encodeURIComponent(doi)}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setModalContent({ type: 'bibtex', content: data.bibtex });
    } catch (e) {
      setModalContent({ type: 'error', content: e.message });
    }
  };

  const handleExtract = () => {
    const seed = oracleMode === ORACLE_SIM ? String(params.queryNode) : params.queryNode;
    setExtractedSeed(seed);
    setExtractedMode(oracleMode);
    extractSubgraph(oracleMode, params);
    if (isMobile) setMobileTab('atlas');
  };

  const handleOracleChange = (next) => {
    if (next === oracleMode || loading) return;
    setOracleMode(next);
  };

  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target?.tagName;
      const typing = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';

      if (e.key === 'Escape') {
        if (modalContent) { setModalContent(null); return; }
      }
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        if (!loading) { e.preventDefault(); handleExtract(); }
        return;
      }
      if (e.key === '/' && !typing) {
        e.preventDefault();
        const el = document.getElementById('seed-input');
        el?.focus(); el?.select?.();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  if (isMobile) {
    return (
      <MobileShell
        tab={mobileTab}
        setTab={setMobileTab}
        loading={loading}
        onExtract={handleExtract}
        onStop={stopExtraction}
        telemetry={telemetry}
        graphData={graphData}
        sidebarProps={{
          fluid: true, hideFeed: true, hideFooter: true, hideHeader: true,
          oracleMode, onOracleChange: handleOracleChange,
          params, setParams, logs, telemetry, meta, loading,
          onExtract: handleExtract, onStop: stopExtraction,
        }}
        logs={logs}
        graphProps={{
          graphData, queryNode: extractedSeed, oracleMode: extractedMode, meta, error,
          hoveredNode, setHoveredNode, setClickedNode, heightPct: 100,
        }}
        ledgerProps={{
          nodes: graphData.nodes, queryNode: extractedSeed, oracleMode: extractedMode, meta,
          loading, error, hoveredNode, setHoveredNode, clickedNode,
          onDetails: (node) => setModalContent({ type: 'abstract', paper: node }),
          onBib: fetchBibtex, heightPct: 100,
        }}
        modal={<PaperModal content={modalContent} onClose={() => setModalContent(null)} />}
      />
    );
  }

  return (
    <div className="flex h-[100dvh] w-full overflow-hidden relative texture-paper">
      <PaperModal content={modalContent} onClose={() => setModalContent(null)} />

      <Sidebar
        width={sidebarWidth}
        oracleMode={oracleMode}
        onOracleChange={handleOracleChange}
        params={params}
        setParams={setParams}
        logs={logs}
        telemetry={telemetry}
        meta={meta}
        loading={loading}
        onExtract={handleExtract}
        onStop={stopExtraction}
      />

      <div
        className="divider-v"
        onMouseDown={(e) => { e.preventDefault(); setIsDraggingSidebar(true); }}
        onTouchStart={(e) => { e.preventDefault(); setIsDraggingSidebar(true); }}
        onDoubleClick={() => setSidebarWidth(sidebarWidth > 0 ? 0 : 460)}
        title="Drag to resize — double-click to collapse"
      />

      {sidebarWidth === 0 && (
        <button
          type="button"
          onClick={() => setSidebarWidth(Math.min(460, Math.floor((window.innerWidth || 1024) * 0.85)))}
          className="absolute top-3 left-3 z-30 btn-chrome"
          title="Open panel"
          aria-label="Open panel"
        >
          <Menu size={16} />
        </button>
      )}

      <div className="flex-grow flex flex-col relative min-w-0">
        <GraphView
          graphData={graphData}
          queryNode={extractedSeed}
          oracleMode={extractedMode}
          meta={meta}
          error={error}
          hoveredNode={hoveredNode}
          setHoveredNode={setHoveredNode}
          setClickedNode={setClickedNode}
          heightPct={100 - ledgerHeightPct}
        />

        <div
          className="divider-h"
          onMouseDown={(e) => { e.preventDefault(); setIsDraggingLedger(true); }}
          onTouchStart={(e) => { e.preventDefault(); setIsDraggingLedger(true); }}
          onDoubleClick={() => setLedgerHeightPct(ledgerHeightPct > 0 ? 0 : 42)}
          title="Drag to resize — double-click to collapse"
        />

        <PaperLedger
          nodes={graphData.nodes}
          queryNode={extractedSeed}
          oracleMode={extractedMode}
          meta={meta}
          loading={loading}
          error={error}
          hoveredNode={hoveredNode}
          setHoveredNode={setHoveredNode}
          clickedNode={clickedNode}
          onDetails={(node) => setModalContent({ type: 'abstract', paper: node })}
          onBib={fetchBibtex}
          heightPct={ledgerHeightPct}
        />
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// Mobile shell — tabs: Graph | Register | Config (config pane owns Log/Dispatch).
// ═════════════════════════════════════════════════════════════════════════════
function MobileShell({
  tab, setTab,
  loading, onExtract, onStop, telemetry, graphData, logs,
  sidebarProps, graphProps, ledgerProps, modal,
}) {
  const status = telemetry?.status ?? 'idle';
  const statusLabel = {
    idle: 'standby', running: 'on-press', converged: 'filed', stopped: 'halted', error: 'retracted',
  }[status] ?? 'standby';
  const coreCount = graphData?.nodes?.filter(n => n.type === 'core').length ?? 0;

  return (
    <div className="h-[100dvh] w-full flex flex-col overflow-hidden relative texture-paper">
      {modal}

      {/* ═══ TOP BAR ══════════════════════════════════════════════════════ */}
      <header
        className="shrink-0 border-b border-[var(--rule-paper-2)] bg-[var(--paper)]/95 backdrop-blur relative z-20"
        style={{ paddingTop: 'env(safe-area-inset-top, 0px)' }}
      >
        <div className="px-4 pt-3 pb-2.5 flex items-center gap-3">
          <div className="flex-1 min-w-0">
            <div className="eyebrow text-[var(--ink-dim)] flex items-center gap-1.5 flex-wrap">
              <span className={`w-1.5 h-1.5 rounded-full ${
                status === 'running' ? 'bg-[var(--accent)] pulse-dot' :
                status === 'error' ? 'bg-[var(--danger)]' :
                status === 'converged' ? 'bg-[#94C466]' :
                'bg-[var(--ink-faint)]'
              }`} />
              <span>{statusLabel}</span>
              {coreCount > 0 && (
                <>
                  <span className="rule-dot" />
                  <span className="font-mono tnum normal-case tracking-normal">{coreCount} core</span>
                </>
              )}
            </div>
            <h1 className="font-display text-[length:var(--text-lg)] leading-tight lowercase text-[var(--ink)] truncate">
              k-densest explorer
            </h1>
          </div>

          {!loading ? (
            <button
              type="button"
              onClick={onExtract}
              className="shrink-0 inline-flex items-center gap-2 bg-[var(--accent)] text-[var(--paper)] px-4 py-2.5 font-semibold text-[length:var(--text-sm)] tracking-[0.14em] uppercase border border-[var(--accent)] shadow-[2px_2px_0_0_var(--night-3)] active:translate-x-[2px] active:translate-y-[2px] active:shadow-none transition-transform"
              title="Extract"
            >
              <Play size={12} fill="currentColor" />
              <span>Run</span>
            </button>
          ) : (
            <button
              type="button"
              onClick={onStop}
              className="shrink-0 inline-flex items-center gap-2 bg-[var(--night)] text-[var(--danger)] px-4 py-2.5 border border-[var(--danger)] transition-colors active:bg-[var(--danger)] active:text-[var(--paper)]"
              title="Stop"
            >
              <Square size={12} fill="currentColor" />
              <span className="font-semibold text-[length:var(--text-sm)] tracking-[0.14em] uppercase">Stop</span>
            </button>
          )}
        </div>
        {loading && <div className="h-[3px] marquee-bar" />}
      </header>

      {/* ═══ TAB PANES — all mounted, toggled via display ═════════════════ */}
      <main className="relative flex-grow min-h-0">
        <TabPane active={tab === 'atlas'}>
          <SplitStack
            top={<GraphView {...graphProps} heightPct={100} />}
            bottom={<PaperLedger {...ledgerProps} heightPct={100} />}
            topLabel="Atlas"
            bottomLabel="Register"
          />
        </TabPane>
        <TabPane active={tab === 'feed'} dark>
          <SplitStack
            dark
            top={
              <div className="h-full flex flex-col px-5 pt-4 pb-4 text-[var(--on-night)] overflow-y-auto overflow-x-hidden custom-scrollbar">
                <DispatchView telemetry={telemetry} loading={loading} />
              </div>
            }
            bottom={
              <div className="h-full flex flex-col text-[var(--on-night)] overflow-hidden">
                <LogView logs={logs} loading={loading} />
              </div>
            }
            topLabel="Dispatch"
            bottomLabel="Log"
          />
        </TabPane>
        <TabPane active={tab === 'config'} dark>
          <Sidebar {...sidebarProps} />
        </TabPane>
      </main>

      {/* ═══ BOTTOM TAB BAR ═══════════════════════════════════════════════ */}
      <nav
        className="shrink-0 border-t border-[var(--rule-paper-2)] bg-[var(--paper)]/95 backdrop-blur grid grid-cols-3 relative z-20"
        style={{ paddingBottom: 'env(safe-area-inset-bottom, 0px)' }}
      >
        <TabButton active={tab === 'atlas'}  onClick={() => setTab('atlas')}  icon={<Network size={18} />}           label="Atlas" />
        <TabButton active={tab === 'feed'}   onClick={() => setTab('feed')}   icon={<Activity size={18} />}          label="Feed" badge={logs?.length} />
        <TabButton active={tab === 'config'} onClick={() => setTab('config')} icon={<SlidersHorizontal size={18} />} label="Config" />
      </nav>
    </div>
  );
}

function SplitStack({ top, bottom, topLabel, bottomLabel, dark = false }) {
  const [open, setOpen] = useState('both'); // 'top' | 'bottom' | 'both'
  const topOpen    = open === 'top' || open === 'both';
  const bottomOpen = open === 'bottom' || open === 'both';
  const topFlex    = topOpen && bottomOpen ? 58 : topOpen ? 100 : 0;
  const bottomFlex = topOpen && bottomOpen ? 42 : bottomOpen ? 100 : 0;

  const toggleTop    = () => setOpen(o => o === 'bottom' ? 'both' : o === 'both' ? 'bottom' : 'bottom');
  const toggleBottom = () => setOpen(o => o === 'top' ? 'both' : o === 'both' ? 'top' : 'top');

  return (
    <div className="h-full w-full flex flex-col">
      <PaneHeader label={topLabel} open={topOpen} onToggle={toggleTop} dark={dark} />
      <div
        style={{ flex: `${topFlex} 1 0`, minHeight: 0 }}
        className="relative transition-[flex] duration-300 ease-out overflow-hidden"
        aria-hidden={!topOpen}
      >
        {top}
      </div>
      <PaneHeader label={bottomLabel} open={bottomOpen} onToggle={toggleBottom} dark={dark} />
      <div
        style={{ flex: `${bottomFlex} 1 0`, minHeight: 0 }}
        className="relative transition-[flex] duration-300 ease-out overflow-hidden"
        aria-hidden={!bottomOpen}
      >
        {bottom}
      </div>
    </div>
  );
}

function PaneHeader({ label, open, onToggle, dark }) {
  const bg    = dark ? 'var(--night-2)' : 'var(--paper-2)';
  const rule  = dark ? 'var(--rule-night)' : 'var(--rule-paper-2)';
  const ink   = dark ? 'var(--on-night-dim)' : 'var(--ink-dim)';
  const faint = dark ? 'var(--on-night-faint)' : 'var(--ink-faint)';
  return (
    <button
      type="button"
      onClick={onToggle}
      className="shrink-0 h-8 flex items-center justify-between px-4 w-full active:brightness-110 transition"
      style={{ background: bg, borderTop: `1px solid ${rule}`, borderBottom: `1px solid ${rule}` }}
      aria-expanded={open}
    >
      <span className="eyebrow" style={{ color: ink, letterSpacing: '0.22em' }}>{label}</span>
      <span style={{ color: faint }} className="inline-flex items-center">
        {open ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
      </span>
    </button>
  );
}

function TabPane({ active, dark = false, children }) {
  return (
    <div
      className={`absolute inset-0 ${dark ? 'texture-night' : ''}`}
      style={{ display: active ? 'block' : 'none' }}
      aria-hidden={!active}
    >
      {children}
    </div>
  );
}

function TabButton({ active, onClick, icon, label, badge }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`relative py-2 flex flex-col items-center gap-0.5 transition-colors ${
        active ? 'text-[var(--accent)]' : 'text-[var(--ink-dim)] active:text-[var(--ink)]'
      }`}
      aria-current={active ? 'page' : undefined}
    >
      <span className="relative">
        {icon}
        {badge > 0 && (
          <span className="absolute -top-1 -right-2 min-w-[14px] h-[14px] px-1 rounded-full bg-[var(--accent)] text-[var(--paper)] text-[length:var(--text-xs)] font-mono tnum flex items-center justify-center">
            {badge > 99 ? '99+' : badge}
          </span>
        )}
      </span>
      <span className="text-[length:var(--text-xs)] tracking-[0.14em] uppercase font-mono">{label}</span>
      {active && <span className="absolute top-0 left-3 right-3 h-[2px] bg-[var(--accent)]" />}
    </button>
  );
}
