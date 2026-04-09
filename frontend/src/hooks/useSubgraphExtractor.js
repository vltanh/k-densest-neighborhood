import { useState, useRef, useCallback } from 'react';
import { API_BASE_URL } from '../constants';
import { parseLogLine, TELEMETRY_INITIAL } from '../utils/telemetryParser';

// Cap the in-memory log buffer so a long Dinkelbach run doesn't balloon
// React state and re-render thousands of <div>s per streamed line. The last
// MAX_LOG_LINES are kept; older ones scroll out of history silently.
const MAX_LOG_LINES = 2000;

export function useSubgraphExtractor(sessionId) {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [logs, setLogs] = useState([]);
  const [telemetry, setTelemetry] = useState(TELEMETRY_INITIAL);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortControllerRef = useRef(null);

  const processPacket = useCallback((line) => {
    if (!line.trim()) return;
    try {
      const packet = JSON.parse(line);
      if (packet.type === 'log') {
        setLogs(prev => {
          const next = [...prev, packet.content];
          return next.length > MAX_LOG_LINES ? next.slice(-MAX_LOG_LINES) : next;
        });
        setTelemetry(prev => parseLogLine(prev, packet.content));
      } else if (packet.type === 'result') {
        setGraphData(packet.content);
      } else if (packet.type === 'error') {
        setError(packet.content);
        setTelemetry(prev => ({ ...prev, status: 'error', finishedAt: Date.now() }));
      }
    } catch (e) { console.error('Parse error:', e); }
  }, []);

  const extractSubgraph = async ({ queryNode, k, timeLimit, nodeLimit, maxInEdges, gapTol, dinkelbachIter, cgBatchFrac, cgMinBatch, cgMaxBatch, tol }) => {
    setLoading(true);
    setError(null);
    setLogs([]);
    setGraphData({ nodes: [], edges: [] });
    setTelemetry({ ...TELEMETRY_INITIAL, status: 'running', startedAt: Date.now() });
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

      // Normal end of stream. Leave telemetry.status intact if the parser
      // already set it to 'converged' via a "Status: Converged" log line;
      // otherwise mark it as finished-but-unknown by stamping finishedAt.
      setTelemetry(prev => ({
        ...prev,
        status: prev.status === 'running' ? 'converged' : prev.status,
        finishedAt: prev.finishedAt ?? Date.now(),
      }));
    } catch (err) {
      if (err.name === 'AbortError') {
        setLogs(prev => [...prev, '\n[!] Connection closed by user.']);
        setTelemetry(prev => ({ ...prev, status: 'stopped', finishedAt: Date.now() }));
      } else {
        setError(err.message);
        setTelemetry(prev => ({ ...prev, status: 'error', finishedAt: Date.now() }));
      }
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

  return { graphData, logs, telemetry, loading, error, extractSubgraph, stopExtraction };
}
