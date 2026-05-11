// Structured parser for the C++ solver's stdout stream.
//
// The solver emits human-readable log lines that encode a lot of useful state
// (Dinkelbach iteration, current λ, running B&B counts, incumbent updates,
// convergence status, and the final statistics block). This module converts
// that stream into two things:
//
//   1. A `telemetry` snapshot (updated per line) that the UI can render as a
//      KPI dashboard instead of forcing the user to scan the raw log.
//   2. A `classifyLine` helper that tags each line with a semantic kind
//      (header / iteration / incumbent / warn / error / info), so the log
//      pane can style them distinctly.
//
// The parsers are tolerant: any line that doesn't match a pattern simply
// leaves the snapshot unchanged, so unknown output never crashes the UI.

const PATTERNS = {
  initActive:    /Init Active Set\s*\|\s*Size:\s*(\d+)\s*\|\s*Density:\s*([\d.eE+-]+)/,
  iteration:     /=== DINKELBACH ITERATION (\d+) \| Lambda = ([\d.eE+-]+) ===/,
  nodesExplored: /->\s*Nodes Explored\s*:\s*\d+\s*\(Total:\s*(\d+)\)/,
  lpSolves:      /->\s*LP Solves\s*:\s*\d+\s*\(Total:\s*(\d+)\)/,
  incumbent:     /Incumbent updated at Node \d+\s*\|\s*Obj:\s*([\d.eE+-]+)\s*\|\s*Size:\s*(\d+)/,
  foundSolution: /Found Solution\s*:\s*Size:\s*(\d+)\s*\|\s*New Density:\s*([\d.eE+-]+)/,
  converged:     /Status\s*:\s*Converged/,
  blacklisted:   /Blacklisting node (\S+?)(?:\s+(?:during|due to)\s+([^:]+?))?\s*:\s*(.+)$/,

  // Final statistics block printed once after the solver exits.
  finalBbNodes:     /^B&B Nodes Explored\s+:\s*(\d+)/,
  finalLpSolves:    /^Total LP Solves\s+:\s*(\d+)/,
  finalColumns:     /^Columns Generated\s+:\s*(\d+)/,
  finalCuts:        /^BQP Cuts Added\s+:\s*(\d+)/,
  finalSolverTime:  /^Total Solver Time\s+:\s*([\d.]+)s/,
  finalDensity:     /^Density\s+:\s*([\d.eE+-]+)/,
  finalSize:        /^Size\s+:\s*(\d+)/,
};

export const TELEMETRY_INITIAL = Object.freeze({
  status: 'idle',         // 'idle' | 'running' | 'converged' | 'stopped' | 'error'
  iteration: null,        // current Dinkelbach iteration (live)
  lambda: null,           // current density estimate (live)
  bbNodes: 0,             // cumulative B&B nodes explored
  lpSolves: 0,            // cumulative LP solves
  incumbent: null,        // { obj, size } — best integer solution so far
  density: null,          // final density (post-solve)
  size: null,             // final / current solution size
  columnsGenerated: null, // final
  cutsAdded: null,        // final
  solverTime: null,       // final (seconds)
  qualities: null,        // { avg_internal_degree, edge_density, ext_conductance, int_ncut }
  startedAt: null,        // epoch ms
  finishedAt: null,       // epoch ms
  blacklisted: [],        // [{ id, phase, reason }] — nodes dropped by the solver
});

/** Remove the `[YYYY-MM-DD HH:MM:SS,mmm]` timestamp prefix emitted by get_timestamp(). */
export function stripTimestamp(line) {
  return line.replace(/^\[[\d\- :,]+\]\s*/, '');
}

/**
 * Derive a new telemetry snapshot from the previous one and a fresh log line.
 * Returns the same reference when nothing matched, so React bails out of
 * unnecessary state updates downstream.
 */
export function parseLogLine(prev, line) {
  const clean = stripTimestamp(line);
  let m;

  if ((m = clean.match(PATTERNS.iteration)))
    return { ...prev, iteration: parseInt(m[1], 10), lambda: parseFloat(m[2]) };

  if ((m = clean.match(PATTERNS.initActive)))
    return { ...prev, size: parseInt(m[1], 10), lambda: parseFloat(m[2]) };

  if ((m = clean.match(PATTERNS.nodesExplored)))
    return { ...prev, bbNodes: parseInt(m[1], 10) };

  if ((m = clean.match(PATTERNS.lpSolves)))
    return { ...prev, lpSolves: parseInt(m[1], 10) };

  if ((m = clean.match(PATTERNS.incumbent)))
    return { ...prev, incumbent: { obj: parseFloat(m[1]), size: parseInt(m[2], 10) } };

  if ((m = clean.match(PATTERNS.foundSolution)))
    return { ...prev, size: parseInt(m[1], 10), lambda: parseFloat(m[2]) };

  if (PATTERNS.converged.test(clean))
    return { ...prev, status: 'converged' };

  if ((m = clean.match(PATTERNS.blacklisted))) {
    const id = m[1];
    // De-duplicate in case the solver re-reports the same node.
    if (prev.blacklisted.some(e => e.id === id)) return prev;
    const entry = { id, phase: (m[2] || '').trim() || null, reason: (m[3] || '').trim() };
    return { ...prev, blacklisted: [...prev.blacklisted, entry] };
  }

  if ((m = clean.match(PATTERNS.finalBbNodes)))
    return { ...prev, bbNodes: parseInt(m[1], 10) };

  if ((m = clean.match(PATTERNS.finalLpSolves)))
    return { ...prev, lpSolves: parseInt(m[1], 10) };

  if ((m = clean.match(PATTERNS.finalColumns)))
    return { ...prev, columnsGenerated: parseInt(m[1], 10) };

  if ((m = clean.match(PATTERNS.finalCuts)))
    return { ...prev, cutsAdded: parseInt(m[1], 10) };

  if ((m = clean.match(PATTERNS.finalSolverTime)))
    return { ...prev, solverTime: parseFloat(m[1]) };

  if ((m = clean.match(PATTERNS.finalDensity)))
    return { ...prev, density: parseFloat(m[1]) };

  if ((m = clean.match(PATTERNS.finalSize)))
    return { ...prev, size: parseInt(m[1], 10) };

  return prev;
}

/**
 * Classify a log line for styling. Returns one of:
 *   'header' | 'separator' | 'iteration' | 'incumbent' | 'prune' |
 *   'warn' | 'success' | 'error' | 'info'
 */
export function classifyLine(line) {
  const clean = stripTimestamp(line);
  if (/^={10,}/.test(clean)) return 'header';
  if (/^-{10,}/.test(clean)) return 'separator';
  if (/\[!\]/.test(clean)) return 'warn';
  if (/Incumbent updated/.test(clean)) return 'incumbent';
  if (/DINKELBACH ITERATION/.test(clean)) return 'iteration';
  if (/Pruning|Pruned node/.test(clean)) return 'prune';
  if (/Converged|FINAL SOLUTION|Graph built|Solved\./.test(clean)) return 'success';
  if (/Error|Exception|Fatal|Blacklisting/i.test(clean)) return 'error';
  return 'info';
}
