import { useState } from 'react';
import { API_BASE_URL } from './constants';
import { useDragResize } from './hooks/useDragResize';
import { useSubgraphExtractor } from './hooks/useSubgraphExtractor';
import Sidebar from './components/Sidebar';
import GraphView from './components/GraphView';
import PaperLedger from './components/PaperLedger';
import PaperModal from './components/PaperModal';

const DEFAULT_PARAMS = {
  queryNode: 'W2741809807',
  k: 5,
  timeLimit: 60.0,
  nodeLimit: 100000,
  maxInEdges: 0,
  gapTol: 0.0001,
  dinkelbachIter: 50,
  cgBatchFrac: 0.1,
  cgMinBatch: 5,
  cgMaxBatch: 50,
  tol: 0.000001,
};

export default function App() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [params, setParams] = useState(DEFAULT_PARAMS);
  // The seed ID that the currently-displayed graph was extracted with. This
  // is decoupled from params.queryNode so that typing a new ID in the input
  // field does not strip the seed highlight from the graph on screen.
  const [extractedSeed, setExtractedSeed] = useState(DEFAULT_PARAMS.queryNode);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [clickedNode, setClickedNode] = useState(null);
  const [modalContent, setModalContent] = useState(null);

  const { sidebarWidth, setSidebarWidth, setIsDraggingSidebar,
          ledgerHeightPct, setLedgerHeightPct, setIsDraggingLedger } = useDragResize();

  const { graphData, logs, telemetry, loading, error, extractSubgraph, stopExtraction } = useSubgraphExtractor(sessionId);

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

  return (
    <div className="flex h-screen w-full overflow-hidden relative texture-paper">
      <PaperModal content={modalContent} onClose={() => setModalContent(null)} />

      <Sidebar
        width={sidebarWidth}
        params={params}
        setParams={setParams}
        logs={logs}
        telemetry={telemetry}
        loading={loading}
        onExtract={() => { setExtractedSeed(params.queryNode); extractSubgraph(params); }}
        onStop={stopExtraction}
      />

      <div
        className="divider-v"
        onMouseDown={(e) => { e.preventDefault(); setIsDraggingSidebar(true); }}
        onDoubleClick={() => setSidebarWidth(sidebarWidth > 0 ? 0 : 460)}
        title="Drag to resize — double-click to collapse"
      />

      <div className="flex-grow flex flex-col relative min-w-0">
        <GraphView
          graphData={graphData}
          queryNode={extractedSeed}
          error={error}
          hoveredNode={hoveredNode}
          setHoveredNode={setHoveredNode}
          setClickedNode={setClickedNode}
          heightPct={100 - ledgerHeightPct}
        />

        <div
          className="divider-h"
          onMouseDown={(e) => { e.preventDefault(); setIsDraggingLedger(true); }}
          onDoubleClick={() => setLedgerHeightPct(ledgerHeightPct > 0 ? 0 : 42)}
          title="Drag to resize — double-click to collapse"
        />

        <PaperLedger
          nodes={graphData.nodes}
          queryNode={extractedSeed}
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
