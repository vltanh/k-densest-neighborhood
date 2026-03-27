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
  maxInEdges: 1500,
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
  const [hoveredNode, setHoveredNode] = useState(null);
  const [clickedNode, setClickedNode] = useState(null);
  const [modalContent, setModalContent] = useState(null);

  const { sidebarWidth, setSidebarWidth, setIsDraggingSidebar,
          ledgerHeightPct, setLedgerHeightPct, setIsDraggingLedger } = useDragResize();

  const { graphData, logs, loading, error, extractSubgraph, stopExtraction } = useSubgraphExtractor(sessionId);

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
    <div className="flex h-screen w-full bg-gray-50 overflow-hidden font-sans text-sm relative">
      <PaperModal content={modalContent} onClose={() => setModalContent(null)} />

      <Sidebar
        width={sidebarWidth}
        params={params}
        setParams={setParams}
        logs={logs}
        loading={loading}
        onExtract={() => extractSubgraph(params)}
        onStop={stopExtraction}
      />

      <div
        className="w-2 bg-gray-200 hover:bg-indigo-300 cursor-col-resize z-10 flex flex-col items-center justify-center transition-colors border-r border-gray-300"
        onMouseDown={(e) => { e.preventDefault(); setIsDraggingSidebar(true); }}
        onDoubleClick={() => setSidebarWidth(sidebarWidth > 0 ? 0 : 380)}
      >
        <div className="h-8 w-1 bg-gray-400 rounded-full" />
      </div>

      <div className="flex-grow flex flex-col relative min-w-0 bg-white">
        <GraphView
          graphData={graphData}
          queryNode={params.queryNode}
          error={error}
          hoveredNode={hoveredNode}
          setHoveredNode={setHoveredNode}
          setClickedNode={setClickedNode}
          heightPct={100 - ledgerHeightPct}
        />

        <div
          className="h-2 bg-gray-200 hover:bg-indigo-300 cursor-row-resize z-10 flex items-center justify-center border-y border-gray-300 shadow-sm transition-colors"
          onMouseDown={(e) => { e.preventDefault(); setIsDraggingLedger(true); }}
          onDoubleClick={() => setLedgerHeightPct(ledgerHeightPct > 0 ? 0 : 40)}
        >
          <div className="w-8 h-1 bg-gray-400 rounded-full" />
        </div>

        <PaperLedger
          nodes={graphData.nodes}
          queryNode={params.queryNode}
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