import { useState, useMemo, useEffect } from 'react';
import { ArrowUp, ArrowDown } from 'lucide-react';
import { ORACLE_SIM, classColor } from '../constants';

export default function PaperLedger({ nodes, queryNode, oracleMode, meta, loading, error, hoveredNode, setHoveredNode, clickedNode, onDetails, onBib, heightPct }) {
  const isSim = oracleMode === ORACLE_SIM;
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

      if (['year', 'citations', 'label', 'degree', 'rawId', 'displayNum'].includes(sortConfig.key)) {
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

  const classStats = useMemo(() => {
    if (!isSim) return null;
    const counts = {};
    sortedNodes.forEach(n => {
      counts[n.label] = (counts[n.label] || 0) + 1;
    });
    return counts;
  }, [isSim, sortedNodes]);

  const renderSortIcon = (columnKey) => {
    if (sortConfig.key !== columnKey) return <span className="w-3 inline-block" />;
    return sortConfig.direction === 'asc'
      ? <ArrowUp size={11} className="inline ml-1 text-[var(--accent)]" />
      : <ArrowDown size={11} className="inline ml-1 text-[var(--accent)]" />;
  };

  const renderTh = (columnKey, title, className = '') => {
    const isActive = sortConfig.key === columnKey;
    const ariaSort = !isActive ? 'none' : (sortConfig.direction === 'asc' ? 'ascending' : 'descending');
    return (
      <th
        key={columnKey}
        aria-sort={ariaSort}
        scope="col"
        className={`px-5 py-4 eyebrow text-left text-[var(--ink-dim)] ${className}`}
      >
        <button
          type="button"
          onClick={() => requestSort(columnKey)}
          className="flex items-center hover:text-[var(--ink)] transition-colors cursor-pointer select-none"
        >
          {title} {renderSortIcon(columnKey)}
        </button>
      </th>
    );
  };

  const splitBadge = (split) => {
    const colors = {
      train: 'bg-[var(--ink)]/10 text-[var(--ink)] border-[var(--ink)]/30',
      val:   'bg-[var(--accent)]/10 text-[var(--accent)] border-[var(--accent)]/40',
      test:  'bg-[var(--gold)]/15 text-[var(--ink)] border-[var(--gold)]/50',
      unlabeled: 'bg-transparent text-[var(--ink-faint)] border-[var(--rule-paper-2)]',
    };
    return (
      <span className={`inline-block px-2 py-0.5 border text-[length:var(--text-xs)] font-mono uppercase tracking-[0.12em] ${colors[split] || colors.unlabeled}`}>
        {split}
      </span>
    );
  };

  return (
    <div
      style={{ height: `${heightPct}%` }}
      className="overflow-y-auto scrollbar-paper relative bg-[var(--paper)] border-t border-[var(--rule-paper-2)]"
    >
      {sortedNodes.length > 0 && (
        <div className="px-7 pt-7 pb-4 max-[900px]:px-4 max-[900px]:pt-3 max-[900px]:pb-2 flex items-end justify-between gap-3 border-b border-[var(--rule-paper)]">
          <div className="min-w-0">
            <div className="eyebrow text-[var(--ink-dim)] flex items-center max-[900px]:hidden">
              <span>Plate II</span><span className="rule-dot" />
              <span>{isSim ? `The Register · ${meta?.dataset || 'sim'}` : 'The Register'}</span>
            </div>
            <h2 className="font-display text-[length:var(--text-2xl)] max-[900px]:text-[length:var(--text-base)] max-[900px]:tracking-normal leading-none mt-1.5 max-[900px]:mt-0 text-[var(--ink)]">
              {isSim ? 'Core Nodes' : 'Table of Contents'}
            </h2>
            <p className="text-[length:var(--text-sm)] max-[900px]:text-[length:var(--text-xs)] text-[var(--ink-soft)] mt-1.5 max-[900px]:mt-0.5 italic truncate">
              {sortedNodes.length} {sortedNodes.length === 1 ? 'node' : 'nodes'} in the densest block
            </p>
          </div>
          {isSim && classStats ? (
            <div className="flex items-center gap-3 overflow-x-auto scrollbar-paper max-w-[55%]">
              {Object.entries(classStats)
                .sort((a, b) => b[1] - a[1] || parseInt(a[0]) - parseInt(b[0]))
                .map(([lbl, cnt]) => (
                  <div key={lbl} className="flex items-center gap-1.5 shrink-0">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: classColor(parseInt(lbl), meta?.numClasses) }} />
                    <span className="text-[length:var(--text-sm)] font-mono tnum text-[var(--ink-soft)]">c{lbl}·{cnt}</span>
                  </div>
                ))}
            </div>
          ) : (
            <div className="eyebrow text-[var(--ink-faint)] max-[900px]:hidden">— click column to sort —</div>
          )}
        </div>
      )}

      <table className="w-full text-left border-collapse">
        {sortedNodes.length > 0 && (
          <thead className="sticky top-0 z-10 bg-[var(--paper)]/95 backdrop-blur border-b border-[var(--rule-paper-2)]">
            <tr>
              {renderTh('displayNum', '№', 'w-16')}
              {isSim ? (
                <>
                  {renderTh('rawId', 'Node ID', 'w-28')}
                  {renderTh('label', 'Class', 'w-28')}
                  {renderTh('split', 'Split', 'w-24')}
                  {renderTh('degree', 'Degree', 'w-24')}
                  <th className="px-5 py-4" />
                </>
              ) : (
                <>
                  {renderTh('year', 'Year', 'w-20')}
                  {renderTh('title', 'Title · Authors')}
                  {renderTh('journal', 'Venue', 'w-52')}
                  {renderTh('citations', 'Cites', 'w-20')}
                  <th className="px-5 py-4 w-36" />
                </>
              )}
            </tr>
          </thead>
        )}
        <tbody>
          {sortedNodes.map((node) => {
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
                      className={`font-display text-[length:var(--text-xl)] leading-none tnum ${
                        isSeed ? 'text-[var(--seed-stroke)]' : 'text-[var(--ink)]'
                      }`}
                    >
                      {node.displayNum}
                    </span>
                    {isSeed && (
                      <span
                        className="eyebrow text-[var(--seed-stroke)] writing-mode-vertical"
                        style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
                      >
                        seed
                      </span>
                    )}
                  </div>
                </td>

                {isSim ? (
                  <>
                    <td className="px-5 py-5 align-top">
                      <span className="font-mono tnum text-[length:var(--text-base)] text-[var(--ink)]">{node.rawId}</span>
                    </td>
                    <td className="px-5 py-5 align-top">
                      <div className="flex items-center gap-2">
                        <span
                          className="w-4 h-4 rounded-full border border-[var(--ink)]/20"
                          style={{ backgroundColor: classColor(node.label, meta?.numClasses) }}
                        />
                        <span className="font-mono tnum text-[length:var(--text-sm)] text-[var(--ink)]">c{node.label}</span>
                      </div>
                    </td>
                    <td className="px-5 py-5 align-top">
                      {splitBadge(node.split)}
                    </td>
                    <td className="px-5 py-5 align-top">
                      <span className="font-mono tnum text-[length:var(--text-sm)] text-[var(--ink-soft)]">{node.degree ?? '—'}</span>
                    </td>
                    <td className="px-5 py-5 align-top" />
                  </>
                ) : (
                  <>
                    <td className="px-5 py-5 align-top">
                      <span className="font-mono tnum text-[length:var(--text-sm)] text-[var(--ink-soft)]">{node.year}</span>
                    </td>
                    <td className="px-5 py-5 align-top max-w-0">
                      <div className="text-[length:var(--text-base)] font-semibold leading-[1.3] text-[var(--ink)] mb-1.5">
                        {node.title}
                      </div>
                      <div className="text-[length:var(--text-sm)] text-[var(--ink-soft)] leading-snug italic">
                        {node.author}
                      </div>
                    </td>
                    <td className="px-5 py-5 align-top">
                      <div className="text-[length:var(--text-sm)] text-[var(--ink-soft)] leading-snug italic">
                        {node.journal}
                      </div>
                    </td>
                    <td className="px-5 py-5 align-top">
                      <span className="font-mono tnum text-[length:var(--text-sm)] text-[var(--ink)]">
                        {node.citations ? node.citations.toLocaleString() : '—'}
                      </span>
                    </td>
                    <td className="px-5 py-5 align-top">
                      <div className="flex items-center gap-3 opacity-60 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={() => onDetails(node)}
                          className="link-tap eyebrow text-[var(--ink)] hover:text-[var(--accent)] transition-colors border-b border-[var(--ink)] hover:border-[var(--accent)] pb-0.5"
                        >
                          Read
                        </button>
                        <button
                          onClick={() => onBib(node.doi)}
                          className="link-tap eyebrow text-[var(--ink-dim)] hover:text-[var(--accent)] transition-colors border-b border-[var(--rule-paper-2)] hover:border-[var(--accent)] pb-0.5"
                        >
                          .bib
                        </button>
                      </div>
                    </td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>

      {nodes.length === 0 && !loading && !error && (
        <div className="h-full flex items-center justify-center py-16 px-8">
          <div className="text-center max-w-xl">
            <div className="eyebrow text-[var(--ink-dim)]">Colophon</div>
            <p className="text-[length:var(--text-base)] text-[var(--ink)] leading-snug mt-4">
              {isSim
                ? 'No run yet. Choose a dataset and a query node in the sidebar, then press Extract.'
                : 'No run yet. Enter a seed paper ID in the sidebar, then press Extract.'}
            </p>
            <p className="text-[length:var(--text-lg)] text-[var(--ink-dim)] leading-relaxed mt-3 italic">
              {isSim
                ? 'the densest block will be typeset here — colored by class label.'
                : 'the papers of the densest block will be typeset here — numbered, sorted, and ready to read.'}
            </p>
          </div>
        </div>
      )}

      {nodes.length === 0 && error && (
        <div className="h-full flex items-center justify-center py-16 px-8">
          <div className="max-w-xl border-l-4 border-[var(--danger)] bg-[var(--paper-2)] px-5 py-4">
            <div className="eyebrow text-[var(--danger)]">Erratum</div>
            <p className="text-[length:var(--text-base)] text-[var(--ink)] leading-snug mt-2">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
