import React, { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { ZoomIn, ZoomOut, Maximize, Download, BarChart2 } from 'lucide-react';

export default function GraphView({ graphData, queryNode, error, setHoveredNode, heightPct }) {
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

    const defs = svg.append('defs');
    defs.append('marker').attr('id', 'arrow-core').attr('viewBox', '0 -5 10 10').attr('refX', 10).attr('refY', 0).attr('orient', 'auto').attr('markerWidth', 6).attr('markerHeight', 6).append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#6b7280');
    defs.append('marker').attr('id', 'arrow-ghost').attr('viewBox', '0 -5 10 10').attr('refX', 10).attr('refY', 0).attr('orient', 'auto').attr('markerWidth', 5).attr('markerHeight', 5).append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#d1d5db');

    const g = svg.append('g');
    const zoom = d3.zoom().scaleExtent([0.1, 4]).on('zoom', (e) => g.attr('transform', e.transform));
    svg.call(zoom);
    zoomBehaviorRef.current = zoom;

    const nodes = graphData.nodes.map(d => Object.create(d));
    const links = graphData.edges.filter(d => d.source !== d.target).map(d => Object.create(d));

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(d => d.type === 'core' ? 140 : 50))
      .force('charge', d3.forceManyBody().strength(d => d.type === 'core' ? -800 : -20))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide().radius(d => d.type === 'core' ? 30 : 6));

    const link = g.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', d => d.type === 'core' ? '#9ca3af' : '#d1d5db')
      .attr('stroke-opacity', d => d.type === 'core' ? 0.8 : 0.3)
      .attr('stroke-width', d => d.type === 'core' ? 2 : 1)
      .attr('stroke-dasharray', d => d.type === 'core' ? 'none' : '3,3')
      .attr('marker-end', d => d.type === 'core' ? 'url(#arrow-core)' : 'url(#arrow-ghost)');

    const node = g.append('g').selectAll('circle').data(nodes).join('circle')
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

    node.on('mouseover', (_, d) => setHoveredNode(d.id)).on('mouseout', () => setHoveredNode(null));

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
  }, [graphData, queryNode]);

  return (
    <div style={{ height: `${heightPct}%` }} className="w-full relative">
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
