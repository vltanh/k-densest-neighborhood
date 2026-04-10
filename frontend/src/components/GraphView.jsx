import React, { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { ZoomIn, ZoomOut, Maximize, Download } from 'lucide-react';

// Palette locked to the design system — blue edition.
const COL_INK      = '#0B1A2E';
const COL_VERMIL   = '#3A7CE3';  // seed — saturated azure
const COL_GOLD     = '#7EB9E8';  // core ring highlight — ice blue
const COL_INK_SOFT = '#4D6585';
const COL_GHOST    = '#AFC0D6';
const COL_GHOST_LN = '#8297B2';
const COL_CORE     = '#1B3A66';  // core fill — deep navy

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

    const getEpoch = (n) => {
      if (n.date && n.date !== 'N/A') return new Date(n.date).getTime();
      if (n.year && n.year !== 'N/A') return new Date(n.year, 0, 1).getTime();
      return new Date(2000, 0, 1).getTime();
    };

    const coreNodes = graphData.nodes.filter(n => n.type === 'core');
    coreNodes.sort((a, b) => getEpoch(a) - getEpoch(b));

    const rankMap = new Map();
    coreNodes.forEach((n, i) => rankMap.set(n.id, i));

    const xSpacing = Math.max(70, (width * 0.8) / Math.max(1, coreNodes.length - 1));
    const totalWidth = xSpacing * (coreNodes.length - 1);
    const startX = (width - totalWidth) / 2;

    const nodes = graphData.nodes.map(d => {
      const copy = Object.create(d);
      if (copy.type === 'core') {
        const rank = rankMap.get(copy.id);
        copy.targetX = startX + rank * xSpacing;
        copy.x = copy.targetX;
        copy.y = height / 2 + (rank % 2 === 0 ? 1 : -1) * (100 + Math.random() * 60);
      } else {
        copy.y = height / 2;
        copy.targetX = width / 2;
      }
      return copy;
    });

    const links = graphData.edges.filter(d => d.source !== d.target).map(d => Object.create(d));

    const defs = svg.append('defs');
    defs.append('marker').attr('id', 'arrow-core').attr('viewBox', '0 -5 10 10')
      .attr('refX', 10).attr('refY', 0).attr('orient', 'auto').attr('markerWidth', 6).attr('markerHeight', 6)
      .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', COL_INK_SOFT);
    defs.append('marker').attr('id', 'arrow-ghost').attr('viewBox', '0 -5 10 10')
      .attr('refX', 10).attr('refY', 0).attr('orient', 'auto').attr('markerWidth', 5).attr('markerHeight', 5)
      .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', COL_GHOST_LN);

    // Soft glow for core nodes
    const filter = defs.append('filter').attr('id', 'node-glow').attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    filter.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'b');
    const merge = filter.append('feMerge');
    merge.append('feMergeNode').attr('in', 'b');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');

    const g = svg.append('g');

    const initialScale = totalWidth > width ? width / (totalWidth + 200) : 1;
    const initialTransform = d3.zoomIdentity.translate(width/2 - (width/2 * initialScale), height/2 - (height/2 * initialScale)).scale(initialScale);

    const zoom = d3.zoom().scaleExtent([0.05, 4]).on('zoom', (e) => g.attr('transform', e.transform));
    svg.call(zoom);
    svg.call(zoom.transform, initialTransform);
    zoomBehaviorRef.current = zoom;

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(d => d.type === 'core' ? 120 : 40))
      .force('charge', d3.forceManyBody().strength(d => d.type === 'core' ? -1200 : -20))
      .force('y', d3.forceY(height / 2).strength(0.02))
      .force('x', d3.forceX(d => d.type === 'core' ? d.targetX : width/2).strength(d => d.type === 'core' ? 0.8 : 0.01))
      .force('collide', d3.forceCollide().radius(d => d.type === 'core' ? 35 : 8));

    const link = g.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', d => d.type === 'core' ? COL_INK_SOFT : COL_GHOST_LN)
      .attr('stroke-opacity', d => d.type === 'core' ? 0.75 : 0.35)
      .attr('stroke-width', d => d.type === 'core' ? 1.5 : 1)
      .attr('stroke-dasharray', d => d.type === 'core' ? 'none' : '2,4')
      .attr('marker-end', d => d.type === 'core' ? 'url(#arrow-core)' : 'url(#arrow-ghost)');

    const node = g.append('g').selectAll('circle').data(nodes).join('circle')
      .attr('class', 'graph-node')
      .attr('r', d => d.type === 'core' ? 24 : 4)
      .attr('fill', d => d.type === 'core' ? (d.id === queryNode ? COL_VERMIL : COL_CORE) : COL_GHOST)
      .attr('stroke', d => d.type === 'core' ? COL_GOLD : 'none')
      .attr('stroke-width', d => d.type === 'core' ? 1.5 : 0)
      .style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end',   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

    const labels = g.append('g').selectAll('text').data(nodes.filter(n => n.type === 'core')).join('text')
      .text(d => d.displayNum)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('fill', '#E7EDF4')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-size', '13px').attr('font-weight', '600')
      .style('pointer-events', 'none');

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
    // queryNode is intentionally NOT in deps — typing in the Seed Paper ID
    // field should not rebuild the simulation (which caused a jitter on every
    // keystroke). Seed highlighting is handled by the lightweight effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphData, setHoveredNode, setClickedNode]);

  // Re-paint the seed node fill without touching the simulation.
  useEffect(() => {
    if (!svgRef.current) return;
    d3.select(svgRef.current).selectAll('.graph-node')
      .attr('fill', d => d.type === 'core' ? (d.id === queryNode ? COL_VERMIL : COL_CORE) : COL_GHOST);
  }, [queryNode]);

  useEffect(() => {
    if (!svgRef.current) return;
    d3.select(svgRef.current).selectAll('.graph-node')
      .transition().duration(200)
      .attr('stroke', d => d.id === hoveredNode ? COL_VERMIL : (d.type === 'core' ? COL_GOLD : 'none'))
      .attr('stroke-width', d => d.id === hoveredNode ? 3 : (d.type === 'core' ? 1.5 : 0))
      .attr('r', d => d.type === 'core' ? (d.id === hoveredNode ? 28 : 24) : 4);
  }, [hoveredNode]);

  return (
    <div style={{ height: `${heightPct}%` }} className="w-full relative overflow-hidden texture-grid">
      {/* Decorative corner marks */}
      <div className="absolute top-0 left-0 w-4 h-4 border-l border-t border-[var(--ink)] pointer-events-none" />
      <div className="absolute top-0 right-0 w-4 h-4 border-r border-t border-[var(--ink)] pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-4 h-4 border-l border-b border-[var(--ink)] pointer-events-none" />
      <div className="absolute bottom-0 right-0 w-4 h-4 border-r border-b border-[var(--ink)] pointer-events-none" />

      {/* ═══ EDITORIAL HEADER STRIP ═══════════════════════════════════════ */}
      <div className="absolute top-5 left-6 right-6 flex items-start justify-between pointer-events-none z-10">
        <div className="pointer-events-auto">
          <div className="eyebrow text-[var(--ink-dim)] flex items-center">
            <span>Plate I</span><span className="rule-dot" /><span>Community Atlas</span>
          </div>
          <h2 className="font-display text-[36px] leading-none mt-1.5 text-[var(--ink)]">
            The Dense Neighbourhood
          </h2>
          <p className="text-[14px] text-[var(--ink-soft)] mt-1.5 italic">
            sorted chronologically, west to east
          </p>
        </div>

        {graphData.nodes.length > 0 && (
          <div className="pointer-events-auto flex items-stretch gap-5 fade-in">
            <StatCell label="Core" value={stats.v} />
            <Divider />
            <StatCell label="Frontier" value={stats.ghosts} muted />
            <Divider />
            <StatCell label="Density" value={stats.density} accent mono />
          </div>
        )}
      </div>

      {/* ═══ ZOOM / EXPORT CONTROLS ═══════════════════════════════════════ */}
      <div className="absolute bottom-5 right-6 flex gap-2 z-10">
        <button onClick={() => handleZoom(1.3)} className="btn-chrome" title="Zoom In"><ZoomIn size={15}/></button>
        <button onClick={() => handleZoom(0.7)} className="btn-chrome" title="Zoom Out"><ZoomOut size={15}/></button>
        <button onClick={resetZoom}             className="btn-chrome" title="Reset"><Maximize size={15}/></button>
        <button onClick={exportSVG}             className="btn-chrome" title="Export SVG"><Download size={15}/></button>
      </div>

      {/* ═══ LEGEND ═══════════════════════════════════════════════════════ */}
      {graphData.nodes.length > 0 && (
        <div className="absolute bottom-5 left-6 flex items-center gap-5 z-10 pointer-events-none">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-[var(--vermillion)]" />
            <span className="text-[var(--ink-soft)] text-[13px] italic">seed</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-[#1B3A66] ring-1 ring-[var(--gold)]" />
            <span className="text-[var(--ink-soft)] text-[13px] italic">core member</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#AFC0D6]" />
            <span className="text-[var(--ink-soft)] text-[13px] italic">frontier</span>
          </div>
        </div>
      )}

      {error && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-[var(--paper)] text-[var(--ink)] px-7 py-6 border border-[var(--vermillion)] z-10 max-w-md shadow-[6px_6px_0_0_var(--vermillion)]">
          <div className="eyebrow text-[var(--vermillion)] mb-2">Erratum</div>
          <div className="text-[16px] leading-snug">{error}</div>
        </div>
      )}

      {graphData.nodes.length === 0 && !error && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center max-w-lg px-8">
            <div className="eyebrow text-[var(--ink-dim)]">An invitation</div>
            <h3 className="font-display text-[48px] leading-[0.95] mt-3 text-[var(--ink)]">
              Pick a paper.<br />
              <span className="text-[var(--vermillion)] italic font-normal">Read its block.</span>
            </h3>
            <p className="text-[16px] text-[var(--ink-soft)] mt-5 leading-relaxed">
              Configure your parameters to the left — the explorer will find the densest
              community around your seed paper and typeset it below.
            </p>
          </div>
        </div>
      )}

      <svg ref={svgRef} className="w-full h-full cursor-grab active:cursor-grabbing outline-none" />
    </div>
  );
}

// ── Small display helpers ──────────────────────────────────────────────────────
const Divider = () => (
  <div className="w-px self-stretch bg-[var(--rule-paper-2)] my-1" />
);

const StatCell = ({ label, value, muted = false, accent = false, mono = false }) => (
  <div className="flex flex-col items-end">
    <span className="eyebrow text-[var(--ink-dim)]">{label}</span>
    <span
      className={`${mono ? 'font-mono tnum' : 'font-display'} leading-none mt-1 ${
        accent ? 'text-[var(--vermillion)] text-[26px]' : muted ? 'text-[var(--ink-soft)] text-[28px]' : 'text-[var(--ink)] text-[28px]'
      }`}
    >
      {value}
    </span>
  </div>
);
