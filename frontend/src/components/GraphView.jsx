import React, { useEffect, useRef, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { ZoomIn, ZoomOut, Maximize, Download, Layers, X } from 'lucide-react';
import { ORACLE_SIM, classColor } from '../constants';

// Palette — azure chrome + ember hot accent for seed.
const COL_INK      = '#0B1A2E';
const COL_EMBER    = '#D97757';  // seed — warm hot accent
const COL_ICE      = '#7EB9E8';  // core ring highlight — ice blue
const COL_INK_SOFT = '#4D6585';
const COL_GHOST    = '#AFC0D6';
const COL_GHOST_LN = '#8297B2';
const COL_CORE     = '#1B3A66';  // core fill — deep navy

export default function GraphView({ graphData, queryNode, oracleMode, meta, error, hoveredNode, setHoveredNode, setClickedNode, heightPct }) {
  const isSim = oracleMode === ORACLE_SIM;
  const numClasses = meta?.numClasses ?? null;
  const coreFill = (d) => {
    if (d.type !== 'core') return COL_GHOST;
    if (isSim) return classColor(d.label, numClasses);
    return COL_CORE;
  };
  const svgRef = useRef();
  const zoomBehaviorRef = useRef(null);
  const [legendOpen, setLegendOpen] = useState(true);

  const stats = useMemo(() => {
    const v = graphData.nodes.filter(n => n.type === 'core').length;
    const e = graphData.edges.filter(l => l.type === 'core').length;
    const density = v > 1 ? (e / (v * (v - 1))).toFixed(4) : '0.0000';
    return { v, e, density, ghosts: graphData.nodes.length - v };
  }, [graphData]);

  const presentClasses = useMemo(() => {
    if (!isSim) return [];
    const counts = new Map();
    graphData.nodes.forEach(n => {
      if (n.type !== 'core' || n.label == null || n.label < 0) return;
      counts.set(n.label, (counts.get(n.label) || 0) + 1);
    });
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1] || a[0] - b[0])
      .map(([label, count]) => ({ label, count }));
  }, [graphData, isSim]);

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
    const prefix = isSim ? `kdensest_${meta?.dataset || 'sim'}` : 'kdensest';
    link.href = url; link.download = `${prefix}_${queryNode}.svg`;
    document.body.appendChild(link); link.click(); document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 100);
  };

  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    if (!svgRef.current) return;
    const el = svgRef.current;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize(prev => (prev.w === r.width && prev.h === r.height) ? prev : { w: r.width, h: r.height });
    });
    ro.observe(el);
    const r = el.getBoundingClientRect();
    setSize({ w: r.width, h: r.height });
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    if (!graphData.nodes.length) return;
    const { w: width, h: height } = size;
    if (width === 0 || height === 0) return;

    const getEpoch = (n) => {
      if (n.date && n.date !== 'N/A') return new Date(n.date).getTime();
      if (n.year && n.year !== 'N/A') return new Date(n.year, 0, 1).getTime();
      return new Date(2000, 0, 1).getTime();
    };

    const coreNodes = graphData.nodes.filter(n => n.type === 'core');
    if (isSim) {
      // Sort by class label, then node id — groups same-class cores together
      coreNodes.sort((a, b) => (a.label - b.label) || (a.rawId - b.rawId));
    } else {
      coreNodes.sort((a, b) => getEpoch(a) - getEpoch(b));
    }

    const rankMap = new Map();
    coreNodes.forEach((n, i) => rankMap.set(n.id, i));

    const xSpacing = Math.max(70, (width * 0.8) / Math.max(1, coreNodes.length - 1));
    const totalWidth = xSpacing * (coreNodes.length - 1);
    const startX = (width - totalWidth) / 2;
    const yBand = Math.min(height * 0.32, 280);

    const nodes = graphData.nodes.map(d => {
      const copy = Object.create(d);
      if (copy.type === 'core') {
        const rank = rankMap.get(copy.id);
        copy.targetX = startX + rank * xSpacing;
        // 3-row zigzag: rank %3 maps to top / middle / bottom band so adjacent
        // ranks never share a row and a single row never holds all nodes.
        const row = rank % 3;
        const band = row === 0 ? -1 : row === 1 ? 0 : 1;
        copy.targetY = height / 2 + band * yBand + (Math.random() - 0.5) * 60;
        copy.x = copy.targetX;
        copy.y = copy.targetY;
      } else {
        copy.targetY = height / 2;
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

    const totalHeight = 2 * yBand + 80;
    const scaleX = (totalWidth + 200) > width ? width / (totalWidth + 200) : 1;
    const scaleY = totalHeight > height ? height / totalHeight : 1;
    const initialScale = Math.min(scaleX, scaleY, 1);
    const initialTransform = d3.zoomIdentity
      .translate(width/2 - (width/2 * initialScale), height/2 - (height/2 * initialScale))
      .scale(initialScale);

    const zoom = d3.zoom().scaleExtent([0.05, 4]).on('zoom', (e) => g.attr('transform', e.transform));
    svg.call(zoom);
    svg.call(zoom.transform, initialTransform);
    zoomBehaviorRef.current = zoom;

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(d => d.type === 'core' ? 140 : 40))
      .force('charge', d3.forceManyBody().strength(d => d.type === 'core' ? -1800 : -25))
      .force('y', d3.forceY(d => d.type === 'core' ? d.targetY : height / 2)
                    .strength(d => d.type === 'core' ? 0.12 : 0.01))
      .force('x', d3.forceX(d => d.type === 'core' ? d.targetX : width/2).strength(d => d.type === 'core' ? 0.35 : 0.01))
      .force('collide', d3.forceCollide().radius(d => d.type === 'core' ? 42 : 8));

    const link = g.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', d => d.type === 'core' ? COL_INK_SOFT : COL_GHOST_LN)
      .attr('stroke-opacity', d => d.type === 'core' ? 0.75 : 0.35)
      .attr('stroke-width', d => d.type === 'core' ? 1.5 : 1)
      .attr('stroke-dasharray', d => d.type === 'core' ? 'none' : '2,4')
      .attr('marker-end', d => d.type === 'core' ? 'url(#arrow-core)' : 'url(#arrow-ghost)');

    // Seed halo — single pulsing ring, behind node. One cue, not five.
    const seedHalo = g.append('g').selectAll('g')
      .data(nodes.filter(n => n.id === queryNode))
      .join('g')
      .attr('class', 'seed-halo');
    seedHalo.append('circle')
      .attr('class', 'seed-halo-pulse')
      .attr('r', 32)
      .attr('fill', 'none')
      .attr('stroke', COL_EMBER)
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.7);

    const node = g.append('g').selectAll('circle').data(nodes).join('circle')
      .attr('class', 'graph-node')
      .attr('r', d => d.type === 'core' ? 24 : 4)
      .attr('fill', coreFill)
      .attr('stroke', d => d.type === 'core' ? (d.id === queryNode ? COL_EMBER : COL_ICE) : 'none')
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
      seedHalo.attr('transform', d => `translate(${d.x}, ${d.y})`);
    });

    // Animate the pulse ring — pause when tab hidden to save battery.
    let pulseTimer = null;
    const startPulse = () => {
      if (pulseTimer) return;
      pulseTimer = d3.interval((elapsed) => {
        const t = (elapsed % 1800) / 1800;
        const r = 28 + t * 22;
        const op = 0.65 * (1 - t);
        seedHalo.select('.seed-halo-pulse').attr('r', r).attr('stroke-opacity', op);
      }, 30);
    };
    const stopPulse = () => { if (pulseTimer) { pulseTimer.stop(); pulseTimer = null; } };
    const onVis = () => document.hidden ? stopPulse() : startPulse();
    startPulse();
    document.addEventListener('visibilitychange', onVis);

    return () => {
      simulation.stop();
      stopPulse();
      document.removeEventListener('visibilitychange', onVis);
    };
    // queryNode is intentionally NOT in deps — typing in the Seed Paper ID
    // field should not rebuild the simulation (which caused a jitter on every
    // keystroke). Seed highlighting is handled by the lightweight effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphData, setHoveredNode, setClickedNode, isSim, size]);

  // Re-paint the seed node fill without touching the simulation.
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('.graph-node')
      .attr('fill', coreFill)
      .attr('stroke', d => d.type === 'core' ? (d.id === queryNode ? COL_EMBER : COL_ICE) : 'none')
      .attr('stroke-width', d => d.type === 'core' ? 1.5 : 0);
    // coreFill is a closure rebuilt every render but it only reads isSim,
    // meta.numClasses, and the per-datum d.label; listing it would re-fire
    // this effect on every parent re-render even when no visual input changed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryNode, isSim]);

  useEffect(() => {
    if (!svgRef.current) return;
    d3.select(svgRef.current).selectAll('.graph-node')
      .transition().duration(200)
      .attr('stroke', d => {
        if (d.id === hoveredNode) return COL_EMBER;
        if (d.type !== 'core') return 'none';
        return d.id === queryNode ? COL_EMBER : COL_ICE;
      })
      .attr('stroke-width', d => d.id === hoveredNode ? 3 : (d.type === 'core' ? 1.5 : 0))
      .attr('r', d => d.type === 'core' ? (d.id === hoveredNode ? 28 : 24) : 4);
  }, [hoveredNode, queryNode]);

  return (
    <div style={{ height: `${heightPct}%` }} className="w-full relative overflow-hidden texture-grid">
      {/* Corner marks — owned here only, not in modals */}
      <div className="absolute top-0 left-0 w-4 h-4 border-l border-t border-[var(--ink)] pointer-events-none" />
      <div className="absolute top-0 right-0 w-4 h-4 border-r border-t border-[var(--ink)] pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-4 h-4 border-l border-b border-[var(--ink)] pointer-events-none" />
      <div className="absolute bottom-0 right-0 w-4 h-4 border-r border-b border-[var(--ink)] pointer-events-none" />

      {/* ═══ EDITORIAL HEADER STRIP ═══════════════════════════════════════ */}
      <div className="absolute top-5 left-6 right-6 max-[900px]:top-3 max-[900px]:left-4 max-[900px]:right-4 flex items-start justify-between gap-4 pointer-events-none z-10">
        <div className="pointer-events-auto max-[900px]:hidden">
          <div className="eyebrow text-[var(--ink-dim)] flex items-center">
            <span>Plate I</span><span className="rule-dot" />
            <span>{isSim ? `Community Atlas · ${meta?.dataset || 'sim'}` : 'Community Atlas'}</span>
          </div>
          <h2 className="type-hero text-[var(--ink)] mt-1.5">
            The Dense Neighbourhood
          </h2>
          <p className="type-sub text-[var(--ink-soft)] mt-2">
            {isSim ? 'colored by class label, grouped by class' : 'sorted chronologically, west to east'}
          </p>
        </div>

        {graphData.nodes.length > 0 && (
          <div className="pointer-events-auto ml-auto flex items-stretch gap-3 md:gap-5 fade-in">
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
        <button type="button" onClick={() => handleZoom(1.3)} className="btn-chrome" title="Zoom In"  aria-label="Zoom in"><ZoomIn size={15}/></button>
        <button type="button" onClick={() => handleZoom(0.7)} className="btn-chrome" title="Zoom Out" aria-label="Zoom out"><ZoomOut size={15}/></button>
        <button type="button" onClick={resetZoom}             className="btn-chrome" title="Reset"    aria-label="Reset zoom"><Maximize size={15}/></button>
        <button type="button" onClick={exportSVG}             className="btn-chrome" title="Export SVG" aria-label="Export as SVG"><Download size={15}/></button>
      </div>

      {/* ═══ LEGEND DRAWER — collapsible, bottom-left ═════════════════════ */}
      {graphData.nodes.length > 0 && (
        <div className="absolute bottom-5 left-6 max-[900px]:bottom-auto max-[900px]:top-3 max-[900px]:left-3 z-10 pointer-events-auto">
          <button
            type="button"
            onClick={() => setLegendOpen(o => !o)}
            className="inline-flex items-center gap-2 eyebrow text-[var(--ink-dim)] hover:text-[var(--ink)] bg-[var(--paper)]/90 backdrop-blur border border-[var(--rule-paper-2)] px-3 py-2 transition-colors"
            title={legendOpen ? 'Hide legend' : 'Show legend'}
          >
            <Layers size={12} />
            <span>{isSim ? `Classes · ${presentClasses.length}${numClasses ? `/${numClasses}` : ''}` : 'Legend'}</span>
          </button>
          {legendOpen && (
            <div className="mt-2 bg-[var(--paper)]/95 backdrop-blur border border-[var(--rule-paper-2)] px-4 py-3 max-w-[640px] max-[900px]:max-w-[min(calc(100vw-2rem),240px)] fade-in">
              {!isSim ? (
                <div className="flex items-center gap-5">
                  <LegendDot color="#1B3A66" ring="var(--ember)" label="seed" />
                  <LegendDot color="#1B3A66" ring="var(--ice)" label="core member" />
                  <LegendDot color="#AFC0D6" size={8} label="frontier" />
                </div>
              ) : (
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2 pb-2 border-b border-[var(--rule-paper)]">
                    <span className="w-3 h-3 rounded-full border-2 border-[var(--ember)]" style={{ boxSizing: 'border-box' }} />
                    <span className="text-[12px] font-mono tnum text-[var(--ink-soft)]">seed · ember ring</span>
                  </div>
                  <div className="flex flex-wrap gap-x-4 gap-y-1.5 max-h-40 overflow-y-auto scrollbar-paper pr-2">
                    {presentClasses.map(({ label, count }) => (
                      <div key={label} className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: classColor(label, numClasses) }} />
                        <span className="text-[11px] font-mono tnum text-[var(--ink-soft)]">c{label}·{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
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
            <h3 className="font-display text-[48px] max-[900px]:text-[26px] leading-[0.95] mt-3 text-[var(--ink)]">
              Pick a paper.<br />
              <span className="text-[var(--vermillion)] italic font-normal">Read its block.</span>
            </h3>
            <p className="text-[16px] max-[900px]:text-[13px] text-[var(--ink-soft)] mt-5 leading-relaxed">
              <span className="max-[900px]:hidden">Configure your parameters to the left — the explorer will find the densest
              community around your seed paper and typeset it below.</span>
              <span className="hidden max-[900px]:inline">Tap the gear to configure, then Run.</span>
            </p>
          </div>
        </div>
      )}

      <svg ref={svgRef} className="w-full h-full cursor-grab active:cursor-grabbing outline-none" style={{ touchAction: 'none' }} />
    </div>
  );
}

// ── Small display helpers ──────────────────────────────────────────────────────
const Divider = () => (
  <div className="w-px self-stretch bg-[var(--rule-paper-2)] my-1" />
);

const LegendDot = ({ color, label, ring = null, size = 12 }) => (
  <div className="flex items-center gap-2">
    <span
      className="rounded-full inline-block"
      style={{
        width: size, height: size,
        backgroundColor: color,
        boxShadow: ring ? `0 0 0 1px ${ring}` : 'none',
      }}
    />
    <span className="text-[13px] text-[var(--ink-soft)] italic">{label}</span>
  </div>
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
