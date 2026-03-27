import React, { useState, useMemo, useEffect } from 'react';
import { ArrowUp, ArrowDown } from 'lucide-react';

export default function PaperLedger({ nodes, queryNode, loading, error, hoveredNode, setHoveredNode, clickedNode, onDetails, onBib, heightPct }) {
  const [sortConfig, setSortConfig] = useState({ key: 'displayNum', direction: 'asc' });

  // Auto-scroll when a node is clicked in the graph
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
    if (sortConfig.key !== columnKey) return <span className="w-4 inline-block" />;
    return sortConfig.direction === 'asc' 
      ? <ArrowUp size={12} className="inline ml-1 text-indigo-500" /> 
      : <ArrowDown size={12} className="inline ml-1 text-indigo-500" />;
  };

  const Th = ({ columnKey, title, className = "" }) => (
    <th 
      onClick={() => requestSort(columnKey)} 
      className={`px-4 py-3 font-semibold cursor-pointer hover:bg-gray-100 hover:text-gray-700 transition-colors select-none ${className}`}
    >
      <div className="flex items-center">
        {title} <SortIcon columnKey={columnKey} />
      </div>
    </th>
  );

  return (
    <div style={{ height: `${heightPct}%` }} className="bg-white overflow-y-auto">
      <table className="w-full text-left text-sm text-gray-600 border-collapse">
        <thead className="bg-gray-50 border-b border-gray-200 sticky top-0 z-10 text-xs uppercase tracking-wider text-gray-500 shadow-sm">
          <tr>
            <Th columnKey="displayNum" title="#" className="w-16" />
            <Th columnKey="id" title="ID" className="w-32" />
            <Th columnKey="year" title="Year" className="w-24" />
            <Th columnKey="title" title="Title & Authors" />
            <Th columnKey="journal" title="Journal" className="w-48" />
            <Th columnKey="citations" title="Cites" className="w-28 text-right" />
            <th className="px-4 py-3 font-semibold w-32 text-center">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {sortedNodes.map(node => {
            const isTarget = hoveredNode === node.id || clickedNode === node.id;
            return (
              <tr
                id={`ledger-row-${node.id}`}
                key={node.id}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                className={`transition-colors ${isTarget ? 'bg-indigo-50/80 ring-1 ring-inset ring-indigo-200' : 'hover:bg-gray-50'}`}
              >
                <td className="px-4 py-4">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white mx-auto ${node.id === queryNode ? 'bg-red-500' : 'bg-indigo-600'}`}>
                    {node.displayNum}
                  </div>
                </td>
                <td className="px-4 py-4 font-mono text-xs">{node.id}</td>
                <td className="px-4 py-4">{node.year}</td>
                <td className="px-4 py-4">
                  <div className="font-semibold text-gray-900 mb-1 leading-snug">{node.title}</div>
                  <div className="text-xs text-gray-500">{node.author}</div>
                </td>
                <td className="px-4 py-4 text-xs italic">{node.journal}</td>
                <td className="px-4 py-4 text-right font-mono text-xs text-gray-500">{node.citations ? node.citations.toLocaleString() : '-'}</td>
                <td className="px-4 py-4">
                  <div className="flex gap-2 justify-center">
                    <button onClick={() => onDetails(node)} className="px-2 py-1 bg-white border border-gray-200 hover:bg-gray-50 text-gray-700 rounded text-xs font-medium transition-colors shadow-sm">Details</button>
                    <button onClick={() => onBib(node.doi)}  className="px-2 py-1 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 rounded text-xs font-medium transition-colors shadow-sm">Bib</button>
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {nodes.length === 0 && !loading && !error && (
        <div className="text-center py-16 text-gray-400 italic bg-gray-50/50 h-full">Configure parameters and Extract Subgraph.</div>
      )}
    </div>
  );
}