import React, { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { ZoomIn, ZoomOut, Maximize, Download, BarChart2 } from 'lucide-react';

export default function GraphView({ graphData, queryNode, error, hoveredNode, setHoveredNode, setClickedNode, heightPct }) {
  const svgRef = useRef();
  const zoomBehaviorRef = useRef(null);

  const stats = useMemo(() => {
    const v = graphData.nodes.filter(n => n.type === 'core').length;
    const e = graphData.edges.filter(l => l.type === 'core').length;
    const density = v > 1 ? (e / (v * (v - 1))).toFixed(4) : '0.0000';
    return { v, e, density, ghosts: graphData.nodes.length - v };
  }, [graphData]);

  const handleZoom = (factor) => {
    if (svgRef.current && zoomBehaviorRef.current)
      d3.select(svgRef.current).transition().duration(300).call(zoomBehaviorRef.current.scaleBy, factor);
  };
  const resetZoom = () => {
    if (svgRef.current && zoomBehaviorRef.current)
      d3.select(svgRef.current).transition().duration(500).call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
  };
  const exportSVG = () => {
    if (!svgRef.current) return;
    const blob = new Blob([new XMLSerializer().serializeToString(svgRef.current)], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url; link.download = `kdensest_${queryNode}.svg`;
    document.body.appendChild(link); link.click(); document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 100);
  };

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    if (!graphData.nodes.length) return;

    const { width, height } = svg.node().getBoundingClientRect();

    // 1. Precise Date Parsing for Sub-Day sorting
    const getEpoch = (n) => {
      if (n.date && n.date !== 'N/A') return new Date(n.date).getTime();
      if (n.year && n.year !== 'N/A') return new Date(n.year, 0, 1).getTime();
      return new Date(2000, 0, 1).getTime();
    };

    // 2. Rank Spreading Algorithm (Prevents cramping & spilling)
    const coreNodes = graphData.nodes.filter(n => n.type === 'core');
    coreNodes.sort((a, b) => getEpoch(a) - getEpoch(b));
    
    const rankMap = new Map();
    coreNodes.forEach((n, i) => rankMap.set(n.id, i));

    // Guarantee enough space so collision radius (35) doesn't cause immediate overlap
    const xSpacing = Math.max(70, (width * 0.8) / Math.max(1, coreNodes.length - 1));
    const totalWidth = xSpacing * (coreNodes.length - 1);
    const startX = (width - totalWidth) / 2;

    const nodes = graphData.nodes.map(d => {
      const copy = Object.create(d);
      if (copy.type === 'core') {
        const rank = rankMap.get(copy.id);
        copy.targetX = startX + rank * xSpacing;
        copy.x = copy.targetX;
        // Stagger above/below to seed charge-driven vertical spread deterministically
        copy.y = height / 2 + (rank % 2 === 0 ? 1 : -1) * (100 + Math.random() * 60);
      } else {
        copy.y = height / 2;
        copy.targetX = width / 2;
      }
      return copy;
    });

    const links = graphData.edges.filter(d => d.source !== d.target).map(d => Object.create(d));

    const defs = svg.append('defs');
    defs.append('marker').attr('id', 'arrow-core').attr('viewBox', '0 -5 10 10').attr('refX', 10).attr('refY', 0).attr('orient', 'auto').attr('markerWidth', 6).attr('markerHeight', 6).append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#6b7280');
    defs.append('marker').attr('id', 'arrow-ghost').attr('viewBox', '0 -5 10 10').attr('refX', 10).attr('refY', 0).attr('orient', 'auto').attr('markerWidth', 5).attr('markerHeight', 5).append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#d1d5db');

    const g = svg.append('g');
    
    // Auto-frame the widened graph on load
    const initialScale = totalWidth > width ? width / (totalWidth + 200) : 1;
    const initialTransform = d3.zoomIdentity.translate(width/2 - (width/2 * initialScale), height/2 - (height/2 * initialScale)).scale(initialScale);
    
    const zoom = d3.zoom().scaleExtent([0.05, 4]).on('zoom', (e) => g.attr('transform', e.transform));
    svg.call(zoom);
    svg.call(zoom.transform, initialTransform);
    zoomBehaviorRef.current = zoom;

    // 3. Physics tuned for vertical blooming
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(d => d.type === 'core' ? 120 : 40))
      // Stronger charge forces them out vertically
      .force('charge', d3.forceManyBody().strength(d => d.type === 'core' ? -1200 : -20))
      // Ultra-weak Y gravity allows the charge to push them tall
      .force('y', d3.forceY(height / 2).strength(0.02))
      // Strong X locks chronological order, but 0.8 allows a tiny bit of horizontal sliding to relieve pressure
      .force('x', d3.forceX(d => d.type === 'core' ? d.targetX : width/2).strength(d => d.type === 'core' ? 0.8 : 0.01))
      .force('collide', d3.forceCollide().radius(d => d.type === 'core' ? 35 : 8));

    const link = g.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', d => d.type === 'core' ? '#9ca3af' : '#d1d5db')
      .attr('stroke-opacity', d => d.type === 'core' ? 0.8 : 0.3)
      .attr('stroke-width', d => d.type === 'core' ? 2 : 1)
      .attr('stroke-dasharray', d => d.type === 'core' ? 'none' : '3,3')
      .attr('marker-end', d => d.type === 'core' ? 'url(#arrow-core)' : 'url(#arrow-ghost)');

    const node = g.append('g').selectAll('circle').data(nodes).join('circle')
      .attr('class', 'graph-node')
      .attr('r', d => d.type === 'core' ? 24 : 4)
      .attr('fill', d => d.type === 'core' ? (d.id === queryNode ? '#ef4444' : '#4f46e5') : '#e5e7eb')
      .attr('stroke', d => d.type === 'core' ? '#fff' : 'none')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end',   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

    const labels = g.append('g').selectAll('text').data(nodes.filter(n => n.type === 'core')).join('text')
      .text(d => d.displayNum)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('fill', '#ffffff').attr('font-size', '14px').attr('font-weight', 'bold')
      .style('pointer-events', 'none');

    // Link hover to Ledger, and Click to scroll the Ledger
    node.on('mouseover', (_, d) => setHoveredNode(d.id))
        .on('mouseout', () => setHoveredNode(null))
        .on('click', (_, d) => { if (d.type === 'core') setClickedNode(d.id); });

    simulation.on('tick', () => {
      link.each(function(d) {
        const rS = d.source.type === 'core' ? 24 : 4;
        const rT = d.target.type === 'core' ? 24 : 4;
        const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > 0) {
          const ux = dx / dist, uy = dy / dist;
          d3.select(this)
            .attr('x1', d.source.x + ux * rS).attr('y1', d.source.y + uy * rS)
            .attr('x2', d.target.x - ux * rT).attr('y2', d.target.y - uy * rT);
        }
      });
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      labels.attr('x', d => d.x).attr('y', d => d.y);
    });

    return () => simulation.stop();
  }, [graphData, queryNode, setHoveredNode, setClickedNode]);

  // Isolate visual updates for hover
  useEffect(() => {
    if (!svgRef.current) return;
    d3.select(svgRef.current).selectAll('.graph-node')
      .transition().duration(200)
      .attr('stroke', d => d.id === hoveredNode ? '#f59e0b' : (d.type === 'core' ? '#fff' : 'none'))
      .attr('stroke-width', d => d.id === hoveredNode ? 4 : 2)
      .attr('r', d => d.type === 'core' ? (d.id === hoveredNode ? 28 : 24) : 4);
  }, [hoveredNode]);

  return (
    <div style={{ height: `${heightPct}%` }} className="w-full relative overflow-hidden">
      {graphData.nodes.length > 0 && (
        <div className="absolute top-4 left-4 z-10 flex gap-4 pointer-events-none">
          <div className="bg-white/90 backdrop-blur px-4 py-2 rounded shadow-sm border border-gray-200 pointer-events-auto flex items-center gap-3">
            <div className="bg-indigo-100 p-2 rounded text-indigo-600"><BarChart2 size={20}/></div>
            <div>
              <div className="text-[10px] uppercase font-bold text-gray-400 tracking-wider">Topology</div>
              <div className="font-mono text-gray-900 font-semibold">
                {stats.v} <span className="text-gray-400 text-xs font-sans">Core</span>
                <span className="text-gray-300 mx-1">|</span>
                {stats.ghosts} <span className="text-gray-400 text-xs font-sans">Frontier</span>
              </div>
            </div>
          </div>
          <div className="bg-white/90 backdrop-blur px-4 py-2 rounded shadow-sm border border-gray-200 pointer-events-auto">
            <div className="text-[10px] uppercase font-bold text-gray-400 tracking-wider mb-0.5">Core Density</div>
            <div className="text-lg font-mono text-indigo-600 font-bold leading-none">{stats.density}</div>
          </div>
        </div>
      )}

      <div className="absolute top-4 right-4 flex flex-col gap-2 z-10">
        <button onClick={() => handleZoom(1.3)} className="bg-white p-2 rounded shadow border border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-indigo-600 transition-colors" title="Zoom In"><ZoomIn size={18}/></button>
        <button onClick={() => handleZoom(0.7)} className="bg-white p-2 rounded shadow border border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-indigo-600 transition-colors" title="Zoom Out"><ZoomOut size={18}/></button>
        <button onClick={resetZoom}            className="bg-white p-2 rounded shadow border border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-indigo-600 transition-colors" title="Reset Layout"><Maximize size={18}/></button>
        <div className="h-px bg-gray-300 my-1 w-full"></div>
        <button onClick={exportSVG}            className="bg-white p-2 rounded shadow border border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-indigo-600 transition-colors" title="Export as SVG"><Download size={18}/></button>
      </div>

      {error && (
        <div className="absolute bottom-4 left-4 bg-red-50 text-red-600 px-4 py-3 rounded shadow-lg border border-red-200 z-10 max-w-md font-mono text-xs">
          <strong>Execution Error:</strong><br/>{error}
        </div>
      )}
      <svg ref={svgRef} className="w-full h-full cursor-grab active:cursor-grabbing outline-none" />
    </div>
  );
}