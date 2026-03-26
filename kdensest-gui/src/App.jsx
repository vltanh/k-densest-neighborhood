import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { ChevronLeft, ChevronRight, ChevronUp, ChevronDown, Activity, Terminal, Settings2, Square, ZoomIn, ZoomOut, Maximize, Download, X, BookOpen, Quote, BarChart2 } from 'lucide-react';

export default function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isLedgerOpen, setIsLedgerOpen] = useState(true);
  
  const [queryNode, setQueryNode] = useState('W2741809807');
  const [k, setK] = useState(5);
  const [timeLimit, setTimeLimit] = useState(600.0);
  const [nodeLimit, setNodeLimit] = useState(100000);
  const [gapTol, setGapTol] = useState(0.0001);
  const [dinkelbachIter, setDinkelbachIter] = useState(50);
  const [cgBatchFrac, setCgBatchFrac] = useState(0.1);
  const [cgMinBatch, setCgMinBatch] = useState(5);
  const [cgMaxBatch, setCgMaxBatch] = useState(50);
  const [tol, setTol] = useState(0.000001);
  
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [modalContent, setModalContent] = useState(null); 

  const svgRef = useRef();
  const logsEndRef = useRef(null);
  const abortControllerRef = useRef(null);
  const zoomBehaviorRef = useRef(null);

  useEffect(() => {
    if (logsEndRef.current) logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Derived Statistics
  const stats = useMemo(() => {
    const v = graphData.nodes.filter(n => n.type === 'core').length;
    const e = graphData.edges.filter(l => l.type === 'core').length;
    const density = v > 1 ? (e / (v * (v - 1))).toFixed(4) : "0.0000";
    return { v, e, density, ghosts: graphData.nodes.length - v };
  }, [graphData]);

  const extractSubgraph = async () => {
    setLoading(true); setError(null); setLogs([]); setGraphData({ nodes: [], edges: [] });
    abortControllerRef.current = new AbortController();
    try {
      const response = await fetch('http://127.0.0.1:8000/api/extract', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: abortControllerRef.current.signal,
        body: JSON.stringify({ query_node: queryNode, k: parseInt(k), time_limit: parseFloat(timeLimit), node_limit: parseInt(nodeLimit), gap_tol: parseFloat(gapTol), dinkelbach_iter: parseInt(dinkelbachIter), cg_batch_frac: parseFloat(cgBatchFrac), cg_min_batch: parseInt(cgMinBatch), cg_max_batch: parseInt(cgMaxBatch), tol: parseFloat(tol) })
      });
      if (!response.body) throw new Error("ReadableStream not supported by browser.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); 
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const packet = JSON.parse(line);
            if (packet.type === "log") setLogs(prev => [...prev, packet.content]);
            else if (packet.type === "result") setGraphData(packet.content);
            else if (packet.type === "error") setError(packet.content);
          } catch (e) { }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') setLogs(prev => [...prev, "\n[!] Connection closed by user."]);
      else setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const stopExtraction = async () => {
    try { await fetch('http://127.0.0.1:8000/api/stop', { method: 'POST' }); } catch (e) { }
    if (abortControllerRef.current) abortControllerRef.current.abort();
    setLoading(false);
  };

  const fetchBibtex = async (doi) => {
    setModalContent({ type: 'loading_bib' });
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/bibtex?doi=${encodeURIComponent(doi)}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setModalContent({ type: 'bibtex', content: data.bibtex });
    } catch (e) {
      setModalContent({ type: 'error', content: e.message });
    }
  };

  const exportSVG = () => {
    if (!svgRef.current) return;
    const svgData = new XMLSerializer().serializeToString(svgRef.current);
    const blob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a"); link.href = url; link.download = `kdensest_${queryNode}.svg`;
    document.body.appendChild(link); link.click(); document.body.removeChild(link);
  };

  // --- D3 GRAPH ---
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    
    // 1. Wipe canvas immediately
    svg.selectAll("*").remove(); 
    if (!graphData.nodes.length) return;

    const width = svg.node().getBoundingClientRect().width;
    const height = svg.node().getBoundingClientRect().height;

    const defs = svg.append("defs");
    defs.append("marker").attr("id", "arrow-core").attr("viewBox", "0 -5 10 10").attr("refX", 10).attr("refY", 0).attr("orient", "auto").attr("markerWidth", 6).attr("markerHeight", 6).append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#6b7280");
    defs.append("marker").attr("id", "arrow-ghost").attr("viewBox", "0 -5 10 10").attr("refX", 10).attr("refY", 0).attr("orient", "auto").attr("markerWidth", 5).attr("markerHeight", 5).append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#d1d5db");

    const g = svg.append("g");
    const zoom = d3.zoom().scaleExtent([0.1, 4]).on("zoom", (e) => g.attr("transform", e.transform));
    svg.call(zoom);
    zoomBehaviorRef.current = zoom;

    const nodes = graphData.nodes.map(d => Object.create(d));
    const links = graphData.edges.filter(d => d.source !== d.target).map(d => Object.create(d));

    // Tweaked physics for large ghost clouds
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(d => d.type === 'core' ? 140 : 50))
      .force("charge", d3.forceManyBody().strength(d => d.type === 'core' ? -800 : -20))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius(d => d.type === 'core' ? 30 : 6));

    const link = g.append("g").selectAll("line").data(links).join("line")
      .attr("stroke", d => d.type === 'core' ? "#9ca3af" : "#d1d5db")
      .attr("stroke-opacity", d => d.type === 'core' ? 0.8 : 0.3)
      .attr("stroke-width", d => d.type === 'core' ? 2 : 1)
      .attr("stroke-dasharray", d => d.type === 'core' ? "none" : "3,3")
      .attr("marker-end", d => d.type === 'core' ? "url(#arrow-core)" : "url(#arrow-ghost)");

    const node = g.append("g").selectAll("circle").data(nodes).join("circle")
      .attr("r", d => d.type === 'core' ? 24 : 4) // Tiny ghost nodes
      .attr("fill", d => d.type === 'core' ? (d.id === queryNode ? "#ef4444" : "#4f46e5") : "#e5e7eb")
      .attr("stroke", d => d.type === 'core' ? "#fff" : "none")
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .call(drag(simulation));

    const labels = g.append("g").selectAll("text").data(nodes.filter(n => n.type === 'core')).join("text")
      .text(d => d.displayNum)
      .attr("text-anchor", "middle").attr("dominant-baseline", "central")
      .attr("fill", "#ffffff").attr("font-size", "14px").attr("font-weight", "bold")
      .style("pointer-events", "none");

    node.on("mouseover", (event, d) => setHoveredNode(d.id)).on("mouseout", () => setHoveredNode(null));

    simulation.on("tick", () => {
      link.each(function(d) {
        const rSource = d.source.type === 'core' ? 24 : 4;
        const rTarget = d.target.type === 'core' ? 24 : 4;
        const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > 0) {
          const ux = dx / dist, uy = dy / dist;
          d3.select(this)
            .attr("x1", d.source.x + ux * rSource).attr("y1", d.source.y + uy * rSource)
            .attr("x2", d.target.x - ux * rTarget).attr("y2", d.target.y - uy * rTarget);
        }
      });
      node.attr("cx", d => d.x).attr("cy", d => d.y);
      labels.attr("x", d => d.x).attr("y", d => d.y);
    });

    function drag(sim) {
      return d3.drag().on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; });
    }
    return () => simulation.stop();
  }, [graphData, queryNode]);

  const handleZoom = (factor) => { if (svgRef.current && zoomBehaviorRef.current) d3.select(svgRef.current).transition().duration(300).call(zoomBehaviorRef.current.scaleBy, factor); };
  const resetZoom = () => { if (svgRef.current && zoomBehaviorRef.current) d3.select(svgRef.current).transition().duration(500).call(zoomBehaviorRef.current.transform, d3.zoomIdentity); };

  return (
    <div className="flex h-screen w-full bg-gray-50 overflow-hidden font-sans text-sm relative">
      
      {/* Modal Overlay */}
      {modalContent && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col overflow-hidden">
            <div className="px-6 py-4 border-b flex justify-between items-center bg-gray-50">
              <h3 className="font-bold text-gray-800 flex items-center gap-2">
                {modalContent.type === 'abstract' && <><BookOpen size={18}/> Paper Details</>}
                {modalContent.type === 'bibtex' && <><Quote size={18}/> BibTeX Export</>}
                {modalContent.type === 'loading_bib' && 'Fetching from DOI registry...'}
                {modalContent.type === 'error' && 'Error'}
              </h3>
              <button onClick={() => setModalContent(null)} className="text-gray-400 hover:text-gray-700 transition-colors"><X size={20}/></button>
            </div>
            <div className="p-6 overflow-y-auto">
              {modalContent.type === 'abstract' && (
                <div className="space-y-4">
                  <div><span className="font-semibold text-gray-900 block mb-1">Title</span><p className="text-gray-700">{modalContent.paper.title}</p></div>
                  <div className="grid grid-cols-2 gap-4">
                    <div><span className="font-semibold text-gray-900 block mb-1">Authors</span><p className="text-gray-700 text-sm">{modalContent.paper.author}</p></div>
                    <div><span className="font-semibold text-gray-900 block mb-1">Venue</span><p className="text-gray-700 text-sm">{modalContent.paper.journal} ({modalContent.paper.year})</p></div>
                  </div>
                  <div><span className="font-semibold text-gray-900 block mb-1">Abstract</span><p className="text-gray-700 text-sm leading-relaxed text-justify">{modalContent.paper.abstract}</p></div>
                </div>
              )}
              {modalContent.type === 'bibtex' && <pre className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-xs overflow-x-auto whitespace-pre-wrap">{modalContent.content}</pre>}
              {modalContent.type === 'loading_bib' && <div className="flex justify-center py-12"><div className="animate-spin w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full"></div></div>}
              {modalContent.type === 'error' && <div className="bg-red-50 text-red-700 p-4 rounded-lg border border-red-200">{modalContent.content}</div>}
            </div>
          </div>
        </div>
      )}

      {/* Sidebar */}
      <div className={`${isSidebarOpen ? 'w-[380px]' : 'w-0'} bg-gray-900 text-white flex flex-col transition-all duration-300 ease-in-out overflow-hidden shrink-0 shadow-xl z-20`}>
        <div className="p-5 overflow-y-auto flex-grow flex flex-col custom-scrollbar">
          <h1 className="text-xl font-bold mb-5 flex items-center gap-2"><Activity size={20} className="text-indigo-400"/> KDensest Explorer</h1>
          
          <div className="space-y-4 shrink-0 bg-gray-800 p-4 rounded-lg border border-gray-700">
            <div><label className="block text-gray-400 text-xs font-semibold uppercase mb-1">Query Node ID (OpenAlex)</label><input type="text" value={queryNode} onChange={e => setQueryNode(e.target.value)} className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white font-mono text-sm" /></div>
            <div><label className="block text-gray-400 text-xs font-semibold uppercase mb-1">Target Subgraph Size (k)</label><input type="number" min="2" value={k} onChange={e => setK(e.target.value)} className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white font-mono text-sm" /></div>
          </div>

          <details className="mt-4 border border-gray-700 rounded-lg bg-gray-800/50 group">
            <summary className="text-gray-300 font-semibold p-3 select-none hover:text-white cursor-pointer flex items-center gap-2"><Settings2 size={16}/> Advanced Solver Options</summary>
            <div className="p-4 pt-0 space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div><label className="block text-gray-400 text-[10px] uppercase mb-1">Time Limit (s)</label><input type="number" step="10" value={timeLimit} onChange={e => setTimeLimit(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
                <div><label className="block text-gray-400 text-[10px] uppercase mb-1">Node Limit</label><input type="number" step="1000" value={nodeLimit} onChange={e => setNodeLimit(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div><label className="block text-gray-400 text-[10px] uppercase mb-1">Gap Tol</label><input type="number" step="0.0001" value={gapTol} onChange={e => setGapTol(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
                <div><label className="block text-gray-400 text-[10px] uppercase mb-1">Num Tol</label><input type="number" step="0.000001" value={tol} onChange={e => setTol(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
              </div>
              <div><label className="flex justify-between text-gray-400 text-[10px] uppercase mb-1"><span>Dinkelbach Iters</span><span className="text-indigo-400">{dinkelbachIter}</span></label><input type="range" min="1" max="200" value={dinkelbachIter} onChange={e => setDinkelbachIter(e.target.value)} className="w-full accent-indigo-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer" /></div>
              <div><label className="flex justify-between text-gray-400 text-[10px] uppercase mb-1"><span>CG Batch Fraction</span><span className="text-indigo-400">{cgBatchFrac}</span></label><input type="range" min="0.01" max="1.0" step="0.01" value={cgBatchFrac} onChange={e => setCgBatchFrac(e.target.value)} className="w-full accent-indigo-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer" /></div>
              <div className="grid grid-cols-2 gap-3">
                <div><label className="block text-gray-400 text-[10px] uppercase mb-1">CG Min Batch</label><input type="number" value={cgMinBatch} onChange={e => setCgMinBatch(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
                <div><label className="block text-gray-400 text-[10px] uppercase mb-1">CG Max Batch</label><input type="number" value={cgMaxBatch} onChange={e => setCgMaxBatch(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
              </div>
            </div>
          </details>

          <div className="mt-4 flex flex-col flex-grow min-h-[180px] border border-gray-700 rounded bg-black shadow-inner overflow-hidden">
            <div className="bg-gray-800 px-3 py-1.5 text-[10px] text-gray-400 uppercase tracking-wider flex items-center gap-2 border-b border-gray-700 shrink-0"><Terminal size={12} /> Live Telemetry</div>
            <div className="p-3 overflow-y-auto font-mono text-[10px] text-green-400 leading-relaxed flex-grow">
              {logs.map((log, idx) => (<div key={idx} className="whitespace-pre-wrap">{log}</div>))}
              {loading && <span className="animate-pulse text-green-500">_</span>}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>
        
        <div className="p-4 border-t border-gray-800 shrink-0 bg-gray-900 flex gap-2">
          {!loading ? (
            <button onClick={extractSubgraph} className="w-full py-3 rounded font-bold shadow-lg bg-indigo-600 hover:bg-indigo-500 text-white hover:shadow-indigo-500/20">Extract Subgraph</button>
          ) : (
            <>
              <button disabled className="flex-1 py-3 rounded font-bold bg-indigo-900 text-indigo-300 cursor-not-allowed flex items-center justify-center gap-2"><span className="animate-pulse">Computing...</span></button>
              <button onClick={stopExtraction} className="px-5 py-3 rounded font-bold bg-red-600 hover:bg-red-500 text-white shadow-lg flex items-center justify-center"><Square size={16} fill="currentColor" /></button>
            </>
          )}
        </div>
      </div>

      <div className="w-4 bg-gray-200 hover:bg-indigo-200 cursor-col-resize flex items-center justify-center border-r border-gray-300 z-10" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
        {isSidebarOpen ? <ChevronLeft size={16} className="text-gray-500"/> : <ChevronRight size={16} className="text-gray-500"/>}
      </div>

      {/* Main Area */}
      <div className="flex-grow flex flex-col relative min-w-0 bg-white">
        
        {/* Viewport */}
        <div className={`w-full relative transition-all duration-300 ${isLedgerOpen ? 'h-[55%]' : 'h-full'}`}>
           
           {/* STATS OVERLAY */}
           {graphData.nodes.length > 0 && (
             <div className="absolute top-4 left-4 z-10 flex gap-4 pointer-events-none">
               <div className="bg-white/90 backdrop-blur px-4 py-2 rounded shadow-sm border border-gray-200 pointer-events-auto flex items-center gap-3">
                 <div className="bg-indigo-100 p-2 rounded text-indigo-600"><BarChart2 size={20}/></div>
                 <div>
                   <div className="text-[10px] uppercase font-bold text-gray-400 tracking-wider">Topology</div>
                   <div className="font-mono text-gray-900 font-semibold">{stats.v} <span className="text-gray-400 text-xs font-sans">Core</span> <span className="text-gray-300 mx-1">|</span> {stats.ghosts} <span className="text-gray-400 text-xs font-sans">Frontier</span></div>
                 </div>
               </div>
               <div className="bg-white/90 backdrop-blur px-4 py-2 rounded shadow-sm border border-gray-200 pointer-events-auto">
                 <div className="text-[10px] uppercase font-bold text-gray-400 tracking-wider mb-0.5">Core Density</div>
                 <div className="text-lg font-mono text-indigo-600 font-bold leading-none">{stats.density}</div>
               </div>
             </div>
           )}

           {/* Floating Tools */}
           <div className="absolute top-4 right-4 flex flex-col gap-2 z-10">
              <button onClick={() => handleZoom(1.3)} className="bg-white p-2 rounded shadow border border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-indigo-600 transition-colors" title="Zoom In"><ZoomIn size={18}/></button>
              <button onClick={() => handleZoom(0.7)} className="bg-white p-2 rounded shadow border border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-indigo-600 transition-colors" title="Zoom Out"><ZoomOut size={18}/></button>
              <button onClick={resetZoom} className="bg-white p-2 rounded shadow border border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-indigo-600 transition-colors" title="Reset Layout"><Maximize size={18}/></button>
              <div className="h-px bg-gray-300 my-1 w-full"></div>
              <button onClick={exportSVG} className="bg-white p-2 rounded shadow border border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-indigo-600 transition-colors" title="Export as SVG"><Download size={18}/></button>
           </div>

           {error && <div className="absolute bottom-4 left-4 bg-red-50 text-red-600 px-4 py-3 rounded shadow-lg border border-red-200 z-10 max-w-md font-mono text-xs"><strong>Execution Error:</strong><br/>{error}</div>}
           
           <svg ref={svgRef} className="w-full h-full cursor-grab active:cursor-grabbing outline-none" />
        </div>

        <div className="h-4 bg-gray-200 hover:bg-indigo-200 cursor-row-resize flex items-center justify-center border-y border-gray-300 z-10" onClick={() => setIsLedgerOpen(!isLedgerOpen)}>
          {isLedgerOpen ? <ChevronDown size={16} className="text-gray-500"/> : <ChevronUp size={16} className="text-gray-500"/>}
        </div>

        {/* Ledger */}
        <div className={`${isLedgerOpen ? 'flex-grow' : 'h-0'} bg-white overflow-y-auto transition-all duration-300`}>
          <table className="w-full text-left text-sm text-gray-600 border-collapse">
            <thead className="bg-gray-50 border-b border-gray-200 sticky top-0 z-10 text-xs uppercase tracking-wider text-gray-500 shadow-sm">
              <tr>
                <th className="px-4 py-3 font-semibold w-12 text-center">#</th>
                <th className="px-4 py-3 font-semibold w-32">ID</th>
                <th className="px-4 py-3 font-semibold w-20">Year</th>
                <th className="px-4 py-3 font-semibold">Title & Authors</th>
                <th className="px-4 py-3 font-semibold w-40">Journal</th>
                <th className="px-4 py-3 font-semibold w-24 text-right">Cites</th>
                <th className="px-4 py-3 font-semibold w-32 text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {graphData.nodes.filter(n => n.type === 'core').map(node => (
                <tr key={node.id} onMouseEnter={() => setHoveredNode(node.id)} onMouseLeave={() => setHoveredNode(null)} className={`transition-colors ${hoveredNode === node.id ? 'bg-indigo-50/50' : 'hover:bg-gray-50'}`}>
                  <td className="px-4 py-4"><div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white mx-auto ${node.id === queryNode ? 'bg-red-500' : 'bg-indigo-600'}`}>{node.displayNum}</div></td>
                  <td className="px-4 py-4 font-mono text-xs">{node.id}</td>
                  <td className="px-4 py-4">{node.year}</td>
                  <td className="px-4 py-4"><div className="font-semibold text-gray-900 mb-1 leading-snug">{node.title}</div><div className="text-xs text-gray-500">{node.author}</div></td>
                  <td className="px-4 py-4 text-xs italic">{node.journal}</td>
                  <td className="px-4 py-4 text-right font-mono text-xs text-gray-500">{node.citations ? node.citations.toLocaleString() : '-'}</td>
                  <td className="px-4 py-4"><div className="flex gap-2 justify-center"><button onClick={() => setModalContent({ type: 'abstract', paper: node })} className="px-2 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-xs font-medium transition-colors">Details</button><button onClick={() => fetchBibtex(node.doi)} className="px-2 py-1 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 rounded text-xs font-medium transition-colors">Bib</button></div></td>
                </tr>
              ))}
            </tbody>
          </table>
          {graphData.nodes.length === 0 && !loading && !error && <div className="text-center py-16 text-gray-400 italic bg-gray-50/50 h-full">Configure parameters and Extract Subgraph.</div>}
        </div>

      </div>
    </div>
  );
}