import React, { useState, useMemo, useEffect } from 'react';
import { ArrowUp, ArrowDown } from 'lucide-react';

export default function PaperLedger({ nodes, queryNode, loading, error, hoveredNode, setHoveredNode, clickedNode, onDetails, onBib, heightPct }) {
  const [sortConfig, setSortConfig] = useState({ key: 'displayNum', direction: 'asc' });

  useEffect(() => {
    if (clickedNode) {
      const row = document.getElementById(`ledger-row-${clickedNode}`);
      if (row) row.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [clickedNode]);

  const requestSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') direction = 'desc';
    setSortConfig({ key, direction });
  };

  const sortedNodes = useMemo(() => {
    const coreNodes = nodes.filter(n => n.type === 'core');
    return coreNodes.sort((a, b) => {
      let aVal = a[sortConfig.key];
      let bVal = b[sortConfig.key];

      if (sortConfig.key === 'year' || sortConfig.key === 'citations') {
        aVal = (aVal === 'N/A' || aVal == null) ? -1 : parseInt(aVal);
        bVal = (bVal === 'N/A' || bVal == null) ? -1 : parseInt(bVal);
      } else if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase();
        bVal = bVal.toLowerCase();
      }

      if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
  }, [nodes, sortConfig]);

  const SortIcon = ({ columnKey }) => {
    if (sortConfig.key !== columnKey) return <span className="w-3 inline-block" />;
    return sortConfig.direction === 'asc'
      ? <ArrowUp size={11} className="inline ml-1 text-[var(--vermillion)]" />
      : <ArrowDown size={11} className="inline ml-1 text-[var(--vermillion)]" />;
  };

  const Th = ({ columnKey, title, className = "" }) => (
    <th
      onClick={() => requestSort(columnKey)}
      className={`px-5 py-4 eyebrow text-left text-[var(--ink-dim)] hover:text-[var(--ink)] cursor-pointer select-none transition-colors ${className}`}
    >
      <div className="flex items-center">
        {title} <SortIcon columnKey={columnKey} />
      </div>
    </th>
  );

  return (
    <div
      style={{ height: `${heightPct}%` }}
      className="overflow-y-auto scrollbar-paper relative bg-[var(--paper)] border-t border-[var(--rule-paper-2)]"
    >
      {/* Ledger masthead */}
      {sortedNodes.length > 0 && (
        <div className="px-7 pt-7 pb-4 flex items-end justify-between border-b border-[var(--rule-paper)]">
          <div>
            <div className="eyebrow text-[var(--ink-dim)] flex items-center">
              <span>Plate II</span><span className="rule-dot" /><span>The Register</span>
            </div>
            <h2 className="font-display text-[36px] leading-none mt-1.5 text-[var(--ink)]">
              Table of Contents
            </h2>
            <p className="text-[14px] text-[var(--ink-soft)] mt-1.5 italic">
              {sortedNodes.length} {sortedNodes.length === 1 ? 'paper' : 'papers'} in the densest block
            </p>
          </div>
          <div className="eyebrow text-[var(--ink-faint)]">— click column to sort —</div>
        </div>
      )}

      <table className="w-full text-left border-collapse">
        {sortedNodes.length > 0 && (
          <thead className="sticky top-0 z-10 bg-[var(--paper)]/95 backdrop-blur border-b border-[var(--rule-paper-2)]">
            <tr>
              <Th columnKey="displayNum" title="№" className="w-16" />
              <Th columnKey="year" title="Year" className="w-20" />
              <Th columnKey="title" title="Title · Authors" />
              <Th columnKey="journal" title="Venue" className="w-52" />
              <Th columnKey="citations" title="Cites" className="w-20" />
              <th className="px-5 py-4 w-36" />
            </tr>
          </thead>
        )}
        <tbody>
          {sortedNodes.map((node, idx) => {
            const isTarget = hoveredNode === node.id || clickedNode === node.id;
            const isSeed = node.id === queryNode;
            return (
              <tr
                id={`ledger-row-${node.id}`}
                key={node.id}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                className={`group border-b border-[var(--rule-paper)] transition-colors ${
                  isTarget ? 'bg-[var(--paper-2)]' : 'hover:bg-[var(--paper-2)]/60'
                }`}
              >
                <td className="px-5 py-5 align-top">
                  <div className="flex items-center gap-2">
                    <span
                      className={`font-display text-[26px] leading-none tnum ${
                        isSeed ? 'text-[var(--vermillion)]' : 'text-[var(--ink)]'
                      }`}
                    >
                      {node.displayNum}
                    </span>
                    {isSeed && (
                      <span
                        className="eyebrow text-[var(--vermillion)] writing-mode-vertical"
                        style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
                      >
                        seed
                      </span>
                    )}
                  </div>
                </td>
                <td className="px-5 py-5 align-top">
                  <span className="font-mono tnum text-[14px] text-[var(--ink-soft)]">{node.year}</span>
                </td>
                <td className="px-5 py-5 align-top max-w-0">
                  <div className="text-[16px] font-semibold leading-[1.3] text-[var(--ink)] mb-1.5">
                    {node.title}
                  </div>
                  <div className="text-[14px] text-[var(--ink-soft)] leading-snug italic">
                    {node.author}
                  </div>
                </td>
                <td className="px-5 py-5 align-top">
                  <div className="text-[14px] text-[var(--ink-soft)] leading-snug italic">
                    {node.journal}
                  </div>
                </td>
                <td className="px-5 py-5 align-top">
                  <span className="font-mono tnum text-[14px] text-[var(--ink)]">
                    {node.citations ? node.citations.toLocaleString() : '—'}
                  </span>
                </td>
                <td className="px-5 py-5 align-top">
                  <div className="flex items-center gap-3 opacity-60 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => onDetails(node)}
                      className="eyebrow text-[var(--ink)] hover:text-[var(--vermillion)] transition-colors border-b border-[var(--ink)] hover:border-[var(--vermillion)] pb-0.5"
                    >
                      Read
                    </button>
                    <button
                      onClick={() => onBib(node.doi)}
                      className="eyebrow text-[var(--ink-dim)] hover:text-[var(--vermillion)] transition-colors border-b border-[var(--rule-paper-2)] hover:border-[var(--vermillion)] pb-0.5"
                    >
                      .bib
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {nodes.length === 0 && !loading && !error && (
        <div className="h-full flex items-center justify-center py-16 px-8">
          <div className="text-center max-w-xl">
            <div className="eyebrow text-[var(--ink-dim)]">Colophon</div>
            <p className="text-[18px] text-[var(--ink-soft)] leading-relaxed mt-4 italic">
              The register is empty. Once the solver finishes its round,
              the papers of the densest block will be typeset here — numbered,
              sorted, and ready to read.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
