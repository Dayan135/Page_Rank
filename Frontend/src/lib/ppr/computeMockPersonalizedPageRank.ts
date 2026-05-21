import type { Graph, NodeId, PPRParams, PPRResult } from "./types";
import { computeDegrees } from "./degrees";

/**
 * Personalized PageRank via power iteration on the column-stochastic transition matrix M:
 *   r_{t+1} = α · M · r_t + (1 − α) · s
 * where s is the personalization vector (uniform if seeds is empty).
 *
 * Dangling nodes (out-degree 0): their column-mass is redistributed uniformly across all
 * nodes — the standard textbook fix that keeps the chain ergodic.
 */
export function computeMockPersonalizedPageRank(
  graph: Graph,
  params: PPRParams,
): PPRResult {
  const N = graph.nodes.length;
  if (N === 0) {
    return {
      ranks: new Map(),
      iterations: 0,
      converged: true,
      convergenceHistory: [],
      degrees: { in: new Map(), out: new Map() },
    };
  }

  const ids: NodeId[] = graph.nodes.map((n) => n.id);
  const indexOf = new Map<NodeId, number>();
  ids.forEach((id, i) => indexOf.set(id, i));

  // Weighted out-degree per source, then column-stochastic transitions.
  // outNeighbors[i] = array of {targetIndex, prob} where probs sum to 1.
  const outWeightSum = new Array<number>(N).fill(0);
  for (const e of graph.edges) {
    const s = indexOf.get(e.source);
    if (s === undefined) continue;
    if (!Number.isFinite(e.weight) || e.weight <= 0) continue;
    outWeightSum[s] += e.weight;
  }

  const outNeighbors: Array<Array<{ t: number; p: number }>> = Array.from(
    { length: N },
    () => [],
  );
  for (const e of graph.edges) {
    const s = indexOf.get(e.source);
    const t = indexOf.get(e.target);
    if (s === undefined || t === undefined) continue;
    if (!Number.isFinite(e.weight) || e.weight <= 0) continue;
    outNeighbors[s].push({ t, p: e.weight / outWeightSum[s] });
  }

  // Personalization vector.
  const s = new Array<number>(N).fill(0);
  if (params.seeds.length === 0) {
    const u = 1 / N;
    for (let i = 0; i < N; i++) s[i] = u;
  } else {
    const validSeeds = params.seeds.map((id) => indexOf.get(id)).filter(
      (i): i is number => i !== undefined,
    );
    if (validSeeds.length === 0) {
      const u = 1 / N;
      for (let i = 0; i < N; i++) s[i] = u;
    } else {
      const mass = 1 / validSeeds.length;
      for (const i of validSeeds) s[i] = mass;
    }
  }

  // Initial rank vector = personalization vector.
  let r = s.slice();
  const alpha = params.alpha;
  const teleport = 1 - alpha;
  const convergenceHistory: number[] = [];
  let iterations = 0;
  let converged = false;

  for (let it = 0; it < params.maxIter; it++) {
    iterations = it + 1;
    const next = new Array<number>(N).fill(0);

    // Dangling mass: nodes with no outgoing edges contribute uniformly.
    let danglingMass = 0;
    for (let i = 0; i < N; i++) {
      if (outNeighbors[i].length === 0) danglingMass += r[i];
    }
    const danglingShare = (alpha * danglingMass) / N;

    for (let i = 0; i < N; i++) {
      next[i] += teleport * s[i] + danglingShare;
    }
    for (let i = 0; i < N; i++) {
      if (outNeighbors[i].length === 0) continue;
      const contrib = alpha * r[i];
      for (const { t, p } of outNeighbors[i]) {
        next[t] += contrib * p;
      }
    }

    let l1 = 0;
    for (let i = 0; i < N; i++) l1 += Math.abs(next[i] - r[i]);
    convergenceHistory.push(l1);

    r = next;

    if (l1 < params.tolerance) {
      converged = true;
      break;
    }
  }

  // Renormalize against any drift from floating-point error.
  let sum = 0;
  for (let i = 0; i < N; i++) sum += r[i];
  if (sum > 0 && Math.abs(sum - 1) > 1e-12) {
    for (let i = 0; i < N; i++) r[i] /= sum;
  }

  const ranks = new Map<NodeId, number>();
  for (let i = 0; i < N; i++) ranks.set(ids[i], r[i]);

  return {
    ranks,
    iterations,
    converged,
    convergenceHistory,
    degrees: computeDegrees(graph),
  };
}
