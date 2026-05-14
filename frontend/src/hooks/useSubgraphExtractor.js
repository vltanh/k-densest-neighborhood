import { useState, useRef, useCallback, useEffect } from 'react';
import { API_BASE_URL, ORACLE_SIM, ORACLE_OPENALEX, VARIANT_BP } from '../constants';
import { parseLogLine, TELEMETRY_INITIAL } from '../utils/telemetryParser';

const MAX_LOG_LINES = 2000;
const INCUMBENT_GRAPH_DEBOUNCE_MS = 350;

const pi = (v, def) => {
  const n = parseInt(v, 10);
  return Number.isNaN(n) ? def : n;
};

const pf = (v, def) => {
  const n = parseFloat(v);
  return Number.isNaN(n) ? def : n;
};

const nodeSignature = (nodes) =>
  Array.isArray(nodes) ? nodes.map(n => String(n)).join('|') : '';

const graphSignature = (graph) => {
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes.map(n => n.id).join('|') : '';
  const edges = Array.isArray(graph?.edges)
    ? graph.edges.map(e => `${e.source}->${e.target}:${e.type || ''}`).join('|')
    : '';
  return `${nodes}::${edges}`;
};

export function useSubgraphExtractor(sessionId) {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [logs, setLogs] = useState([]);
  const [telemetry, setTelemetry] = useState(TELEMETRY_INITIAL);
  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortControllerRef = useRef(null);
  const incumbentGraphAbortRef = useRef(null);
  const incumbentTimerRef = useRef(null);
  const graphRequestSeqRef = useRef(0);
  const latestGraphSignatureRef = useRef('');
  const runContextRef = useRef(null);

  const clearIncumbentGraphWork = useCallback((abortRequest = true) => {
    if (incumbentTimerRef.current) {
      clearTimeout(incumbentTimerRef.current);
      incumbentTimerRef.current = null;
    }
    if (abortRequest && incumbentGraphAbortRef.current) {
      incumbentGraphAbortRef.current.abort();
      incumbentGraphAbortRef.current = null;
    }
  }, []);

  const applyGraphData = useCallback((graph) => {
    const sig = graphSignature(graph);
    if (sig && sig === latestGraphSignatureRef.current) return;
    latestGraphSignatureRef.current = sig;
    setGraphData(graph);
  }, []);

  const fetchGraphForNodes = useCallback(async (nodes, signature, runId) => {
    const ctx = runContextRef.current;
    if (!ctx || ctx.runId !== runId || ctx.finalGraphReceived) return;

    const endpoint = ctx.mode === ORACLE_SIM ? '/api/graph-sim' : '/api/graph-openalex';
    const body = ctx.mode === ORACLE_SIM
      ? {
          dataset: ctx.params.dataset,
          query_node: pi(ctx.params.queryNode, 0),
          nodes: nodes.map(n => pi(n, NaN)).filter(Number.isFinite),
        }
      : {
          query_node: ctx.params.queryNode,
          max_in_edges: pi(ctx.params.maxInEdges, 0),
          nodes: nodes.map(n => String(n)),
        };
    if (!body.nodes.length) return;

    const seq = ++graphRequestSeqRef.current;
    if (incumbentGraphAbortRef.current) incumbentGraphAbortRef.current.abort();
    const controller = new AbortController();
    incumbentGraphAbortRef.current = controller;

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify(body),
      });
      if (!response.ok) return;
      const data = await response.json();
      const current = runContextRef.current;
      if (
        controller.signal.aborted ||
        seq !== graphRequestSeqRef.current ||
        !current ||
        current.runId !== runId ||
        current.finalGraphReceived ||
        current.lastIncumbentSignature !== signature
      ) return;
      if (data?.meta) setMeta(data.meta);
      if (data?.graph) applyGraphData(data.graph);
    } catch (err) {
      if (err.name !== 'AbortError') console.warn('Failed to build incumbent graph', err);
    } finally {
      if (incumbentGraphAbortRef.current === controller) {
        incumbentGraphAbortRef.current = null;
      }
    }
  }, [applyGraphData]);

  const scheduleIncumbentGraph = useCallback((inc) => {
    const ctx = runContextRef.current;
    if (!ctx || ctx.finalGraphReceived) return;
    const nodes = Array.isArray(inc?.nodes) ? inc.nodes : [];
    if (!nodes.length) return;
    const signature = nodeSignature(nodes);
    if (!signature || signature === ctx.lastIncumbentSignature) return;
    ctx.lastIncumbentSignature = signature;

    if (incumbentTimerRef.current) clearTimeout(incumbentTimerRef.current);
    const runId = ctx.runId;
    const nodeList = nodes.map(n => String(n));
    incumbentTimerRef.current = setTimeout(() => {
      incumbentTimerRef.current = null;
      fetchGraphForNodes(nodeList, signature, runId);
    }, INCUMBENT_GRAPH_DEBOUNCE_MS);
  }, [fetchGraphForNodes]);

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
        const ctx = runContextRef.current;
        if (ctx) ctx.finalGraphReceived = true;
        clearIncumbentGraphWork(false);
        applyGraphData(packet.content);
      } else if (packet.type === 'meta') {
        setMeta(packet.content);
      } else if (packet.type === 'qualities') {
        setTelemetry(prev => ({ ...prev, qualities: packet.content }));
      } else if (packet.type === 'incumbent') {
        const inc = packet.content || {};
        scheduleIncumbentGraph(inc);
        setTelemetry(prev => ({
          ...prev,
          lambda: typeof inc.lambda === 'number' ? inc.lambda : prev.lambda,
          bbNodes: typeof inc.bb_node === 'number' ? Math.max(prev.bbNodes || 0, inc.bb_node) : prev.bbNodes,
          size: typeof inc.size === 'number' ? inc.size : prev.size,
          incumbent: {
            obj: typeof inc.param_obj === 'number' ? inc.param_obj : null,
            density: typeof inc.density === 'number' ? inc.density : null,
            size: typeof inc.size === 'number' ? inc.size : null,
            nodes: Array.isArray(inc.nodes) ? inc.nodes : [],
            bbNode: typeof inc.bb_node === 'number' ? inc.bb_node : null,
            lambda: typeof inc.lambda === 'number' ? inc.lambda : null,
          },
        }));
      } else if (packet.type === 'error') {
        setError(packet.content);
        setTelemetry(prev => ({ ...prev, status: 'error', finishedAt: Date.now() }));
      }
    } catch (e) { console.error('Parse error:', e); }
  }, [applyGraphData, clearIncumbentGraphWork, scheduleIncumbentGraph]);

  const extractSubgraph = async (mode, params) => {
    const runId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    clearIncumbentGraphWork(true);
    graphRequestSeqRef.current += 1;
    latestGraphSignatureRef.current = '';
    runContextRef.current = {
      runId,
      mode,
      params: { ...params },
      finalGraphReceived: false,
      lastIncumbentSignature: '',
    };

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

    const sharedSolver = {
      variant:            params.variant || VARIANT_BP,
      k:                  pi(params.k,               5),
      kappa:              pi(params.kappa,           0),
      bfs_depth:          pi(params.bfsDepth,        1),
      time_limit:         pf(params.timeLimit,       60.0),
      hard_time_limit:    pf(params.hardTimeLimit,   -1.0),
      node_limit:         pi(params.nodeLimit,       100000),
      max_in_edges:       pi(params.maxInEdges,      0),
      gap_tol:            pf(params.gapTol,          0.0001),
      dinkelbach_iter:    pi(params.dinkelbachIter,  50),
      cg_batch_frac:      pf(params.cgBatchFrac,     1.0),
      cg_min_batch:       pi(params.cgMinBatch,      0),
      cg_max_batch:       pi(params.cgMaxBatch,      50),
      tol:                pf(params.tol,             0.000001),
      no_materialize:     !!params.noMaterialize,
      stream_incumbents:  true,
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
    clearIncumbentGraphWork(true);
    setLoading(false);
  };

  // Abort any pending fetch when the component using this hook unmounts so
  // the stream reader does not keep pushing into stale setState.
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) abortControllerRef.current.abort();
      clearIncumbentGraphWork(true);
    };
  }, [clearIncumbentGraphWork]);

  return { graphData, logs, telemetry, meta, loading, error, extractSubgraph, stopExtraction };
}

export { ORACLE_OPENALEX, ORACLE_SIM };
