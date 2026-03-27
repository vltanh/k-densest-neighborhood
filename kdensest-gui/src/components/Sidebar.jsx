import React, { useEffect, useRef } from 'react';
import { Activity, Terminal, Settings2, Square } from 'lucide-react';

export default function Sidebar({ width, params, setParams, logs, loading, onExtract, onStop }) {
  const logsEndRef = useRef(null);
  
  useEffect(() => {
    if (logsEndRef.current) logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const set = (key) => (e) => setParams(prev => ({ ...prev, [key]: e.target.value }));

  return (
    <div style={{ width: `${width}px` }} className="bg-gray-900 text-white flex flex-col h-full overflow-hidden shrink-0 shadow-xl z-20">
      
      {/* 1. FIXED HEADER */}
      <div className="px-5 pt-5 pb-4 shrink-0 border-b border-gray-800/80">
        <h1 className="text-xl font-bold flex items-center gap-2 truncate">
          <Activity size={20} className="text-indigo-400 shrink-0"/> KDensest Explorer
        </h1>
      </div>

      {/* 2. FULLY EXPANDING OPTIONS (No Scrollbar) */}
      <div className="px-5 py-4 shrink-0 flex flex-col gap-4 border-b border-gray-800/50">
        <div className="space-y-4 shrink-0 bg-gray-800 p-4 rounded-lg border border-gray-700">
          <div>
            <label className="block text-gray-400 text-xs font-semibold uppercase mb-1">Query Node ID</label>
            <input type="text" value={params.queryNode} onChange={set('queryNode')} className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white font-mono text-sm" />
          </div>
          <div>
            <label className="block text-gray-400 text-xs font-semibold uppercase mb-1">Subgraph Size (k)</label>
            <input type="number" min="2" value={params.k} onChange={set('k')} className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white font-mono text-sm" />
          </div>
        </div>

        <details className="border border-gray-700 rounded-lg bg-gray-800/50 group shrink-0">
          <summary className="text-gray-300 font-semibold p-3 select-none hover:text-white cursor-pointer flex items-center gap-2">
            <Settings2 size={16}/> Adv. Solver Options
          </summary>
          <div className="p-4 pt-0 space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div><label className="block text-gray-400 text-[10px] uppercase mb-1">Time Limit</label><input type="number" step="10" value={params.timeLimit} onChange={set('timeLimit')} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
              <div><label className="block text-gray-400 text-[10px] uppercase mb-1">Node Limit</label><input type="number" step="1000" value={params.nodeLimit} onChange={set('nodeLimit')} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
            </div>
            <div>
              <label className="block text-gray-400 text-[10px] uppercase mb-1">Max In-Edges</label>
              <input type="number" step="500" value={params.maxInEdges} onChange={set('maxInEdges')} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div><label className="block text-gray-400 text-[10px] uppercase mb-1">Gap Tol</label><input type="number" step="0.0001" value={params.gapTol} onChange={set('gapTol')} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
              <div><label className="block text-gray-400 text-[10px] uppercase mb-1">Num Tol</label><input type="number" step="0.000001" value={params.tol} onChange={set('tol')} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
            </div>
            <div>
              <label className="flex justify-between text-gray-400 text-[10px] uppercase mb-1">
                <span>Dinkelbach Iter</span><span className="text-indigo-400">{params.dinkelbachIter}</span>
              </label>
              <input type="range" min="1" max="200" value={params.dinkelbachIter} onChange={set('dinkelbachIter')} className="w-full accent-indigo-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer" />
            </div>
            <div>
              <label className="flex justify-between text-gray-400 text-[10px] uppercase mb-1">
                <span>CG Batch Frac</span><span className="text-indigo-400">{params.cgBatchFrac}</span>
              </label>
              <input type="range" min="0.01" max="1.0" step="0.01" value={params.cgBatchFrac} onChange={set('cgBatchFrac')} className="w-full accent-indigo-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div><label className="block text-gray-400 text-[10px] uppercase mb-1">Min Batch</label><input type="number" value={params.cgMinBatch} onChange={set('cgMinBatch')} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
              <div><label className="block text-gray-400 text-[10px] uppercase mb-1">Max Batch</label><input type="number" value={params.cgMaxBatch} onChange={set('cgMaxBatch')} className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white font-mono text-xs" /></div>
            </div>
          </div>
        </details>
      </div>

      {/* 3. HIGHLY COMPRESSIBLE TERMINAL */}
      {/* min-h-0 allows the container to compress infinitely when pushed by the options above */}
      <div className="px-5 py-4 flex flex-col flex-grow overflow-hidden min-h-0">
        <div className="flex flex-col flex-grow border border-gray-700 rounded bg-black shadow-inner overflow-hidden min-h-0">
          <div className="bg-gray-800 px-3 py-1.5 text-[10px] text-gray-400 uppercase tracking-wider flex items-center gap-2 border-b border-gray-700 shrink-0">
            <Terminal size={12}/> Live Telemetry
          </div>
          <div className="p-3 overflow-y-auto custom-scrollbar font-mono text-[10px] text-green-400 leading-relaxed flex-grow whitespace-nowrap min-h-0">
            {logs.map((log, idx) => <div key={idx} className="whitespace-pre-wrap">{log}</div>)}
            {loading && <span className="animate-pulse text-green-500">_</span>}
            <div ref={logsEndRef} />
          </div>
        </div>
      </div>

      {/* 4. FIXED FOOTER */}
      <div className="p-4 border-t border-gray-800 shrink-0 bg-gray-900 flex gap-2">
        {!loading ? (
          <button onClick={onExtract} className="w-full py-3 rounded font-bold shadow-lg bg-indigo-600 hover:bg-indigo-500 text-white hover:shadow-indigo-500/20 transition-all">Extract Subgraph</button>
        ) : (
          <>
            <button disabled className="flex-1 py-3 rounded font-bold bg-indigo-900 text-indigo-300 cursor-not-allowed flex items-center justify-center gap-2"><span className="animate-pulse">Computing...</span></button>
            <button onClick={onStop} className="px-5 py-3 rounded font-bold bg-red-600 hover:bg-red-500 text-white shadow-lg flex items-center justify-center transition-all"><Square size={16} fill="currentColor"/></button>
          </>
        )}
      </div>
    </div>
  );
}