import React from 'react';

export default function PaperLedger({ nodes, queryNode, loading, error, hoveredNode, setHoveredNode, onDetails, onBib, heightPct }) {
  const coreNodes = nodes.filter(n => n.type === 'core');
  return (
    <div style={{ height: `${heightPct}%` }} className="bg-white overflow-y-auto">
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
          {coreNodes.map(node => (
            <tr
              key={node.id}
              onMouseEnter={() => setHoveredNode(node.id)}
              onMouseLeave={() => setHoveredNode(null)}
              className={`transition-colors ${hoveredNode === node.id ? 'bg-indigo-50/50' : 'hover:bg-gray-50'}`}
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
                  <button onClick={() => onDetails(node)} className="px-2 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-xs font-medium transition-colors">Details</button>
                  <button onClick={() => onBib(node.doi)}  className="px-2 py-1 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 rounded text-xs font-medium transition-colors">Bib</button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {nodes.length === 0 && !loading && !error && (
        <div className="text-center py-16 text-gray-400 italic bg-gray-50/50 h-full">Configure parameters and Extract Subgraph.</div>
      )}
    </div>
  );
}
