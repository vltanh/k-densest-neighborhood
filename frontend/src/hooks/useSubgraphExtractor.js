import { useState, useRef, useCallback, useEffect } from 'react';
import { API_BASE_URL, ORACLE_SIM, ORACLE_OPENALEX, VARIANT_BP } from '../constants';
import { parseLogLine, TELEMETRY_INITIAL } from '../utils/telemetryParser';

const MAX_LOG_LINES = 2000;

export function useSubgraphExtractor(sessionId) {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [logs, setLogs] = useState([]);
  const [telemetry, setTelemetry] = useState(TELEMETRY_INITIAL);
  const [meta, setMeta] = useState(null);
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
      } else if (packet.type === 'meta') {
        setMeta(packet.content);
      } else if (packet.type === 'qualities') {
        setTelemetry(prev => ({ ...prev, qualities: packet.content }));
      } else if (packet.type === 'error') {
        setError(packet.content);
        setTelemetry(prev => ({ ...prev, status: 'error', finishedAt: Date.now() }));
      }
    } catch (e) { console.error('Parse error:', e); }
  }, []);

  const extractSubgraph = async (mode, params) => {
    setLoading(true);
    setError(null);
    setLogs([]);
    setGraphData({ nodes: [], edges: [] });
    setMeta(null);
    setTelemetry({ ...TELEMETRY_INITIAL, status: 'running', startedAt: Date.now() });
    // Cancel any in-flight extraction before starting a new one. Without this,
    // back-to-back Extract clicks would leave the old fetch streaming into state.
    if (abortControllerRef.current) abortControllerRef.current.abort();
    abortControllerRef.current = new AbortController();

    const pi = (v, def) => { const n = parseInt(v);   return isNaN(n) ? def : n; };
    const pf = (v, def) => { const n = parseFloat(v); return isNaN(n) ? def : n; };

    const sharedSolver = {
      variant:            params.variant || VARIANT_BP,
      k:                  pi(params.k,               5),
      kappa:              pi(params.kappa,           0),
      bfs_depth:          pi(params.bfsDepth,        1),
      time_limit:         pf(params.timeLimit,       60.0),
      node_limit:         pi(params.nodeLimit,       100000),
      max_in_edges:       pi(params.maxInEdges,      0),
      gap_tol:            pf(params.gapTol,          0.0001),
      dinkelbach_iter:    pi(params.dinkelbachIter,  50),
      cg_batch_frac:      pf(params.cgBatchFrac,     1.0),
      cg_min_batch:       pi(params.cgMinBatch,      0),
      cg_max_batch:       pi(params.cgMaxBatch,      50),
      tol:                pf(params.tol,             0.000001),
    };

    let endpoint, body;
    if (mode === ORACLE_SIM) {
      endpoint = '/api/extract-sim';
      body = { session_id: sessionId, dataset: params.dataset, query_node: pi(params.queryNode, 0), ...sharedSolver };
    } else {
      endpoint = '/api/extract';
      body = { session_id: sessionId, query_node: params.queryNode, ...sharedSolver };
    }

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortControllerRef.current.signal,
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail?.[0]?.msg || errData.detail || `HTTP Error ${response.status}`);
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

  // Abort any pending fetch when the component using this hook unmounts so
  // the stream reader does not keep pushing into stale setState.
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) abortControllerRef.current.abort();
    };
  }, []);

  return { graphData, logs, telemetry, meta, loading, error, extractSubgraph, stopExtraction };
}

export { ORACLE_OPENALEX, ORACLE_SIM };
