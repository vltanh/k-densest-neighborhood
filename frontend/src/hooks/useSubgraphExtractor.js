import { useState, useRef } from 'react';
import { API_BASE_URL } from '../constants';

export function useSubgraphExtractor(sessionId) {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortControllerRef = useRef(null);

  const processPacket = (line) => {
    if (!line.trim()) return;
    try {
      const packet = JSON.parse(line);
      if (packet.type === 'log') setLogs(prev => [...prev, packet.content]);
      else if (packet.type === 'result') setGraphData(packet.content);
      else if (packet.type === 'error') setError(packet.content);
    } catch (e) { console.error('Parse error:', e); }
  };

  const extractSubgraph = async ({ queryNode, k, timeLimit, nodeLimit, maxInEdges, gapTol, dinkelbachIter, cgBatchFrac, cgMinBatch, cgMaxBatch, tol }) => {
    setLoading(true); setError(null); setLogs([]); setGraphData({ nodes: [], edges: [] });
    abortControllerRef.current = new AbortController();

    const pi = (v, def) => { const n = parseInt(v);   return isNaN(n) ? def : n; };
    const pf = (v, def) => { const n = parseFloat(v); return isNaN(n) ? def : n; };

    try {
      const response = await fetch(`${API_BASE_URL}/api/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortControllerRef.current.signal,
        body: JSON.stringify({
          session_id: sessionId,
          query_node: queryNode,
          k:               pi(k,              5),
          time_limit:      pf(timeLimit,       60.0),
          node_limit:      pi(nodeLimit,       100000),
          max_in_edges:    pi(maxInEdges,      1500),
          gap_tol:         pf(gapTol,          0.0001),
          dinkelbach_iter: pi(dinkelbachIter,  50),
          cg_batch_frac:   pf(cgBatchFrac,     0.1),
          cg_min_batch:    pi(cgMinBatch,      5),
          cg_max_batch:    pi(cgMaxBatch,      50),
          tol:             pf(tol,             0.000001),
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail?.[0]?.msg || `HTTP Error ${response.status}`);
      }
      if (!response.body) throw new Error('ReadableStream not supported by browser.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        lines.forEach(processPacket);
      }
      if (buffer.trim()) processPacket(buffer);

    } catch (err) {
      if (err.name === 'AbortError') setLogs(prev => [...prev, '\n[!] Connection closed by user.']);
      else setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const stopExtraction = async () => {
    try { await fetch(`${API_BASE_URL}/api/stop?session_id=${sessionId}`, { method: 'POST' }); }
    catch (e) { console.error('Failed to send stop signal', e); }
    if (abortControllerRef.current) abortControllerRef.current.abort();
    setLoading(false);
  };

  return { graphData, logs, loading, error, extractSubgraph, stopExtraction };
}
