import type { Graph, PPRParams, PPRResult } from "@/lib/ppr/types";

export function resultToJSON(
  graph: Graph,
  params: PPRParams,
  result: PPRResult,
): string {
  const ids = Array.from(result.ranks.keys());
  ids.sort((a, b) => (result.ranks.get(b) ?? 0) - (result.ranks.get(a) ?? 0));

  const payload = {
    generatedAt: new Date().toISOString(),
    graph: {
      format: graph.format,
      nodeCount: graph.nodes.length,
      edgeCount: graph.edges.length,
    },
    params,
    run: {
      iterations: result.iterations,
      converged: result.converged,
      convergenceHistory: result.convergenceHistory,
    },
    ranks: ids.map((id, idx) => ({
      rank: idx + 1,
      nodeId: id,
      score: result.ranks.get(id) ?? 0,
      inDegree: result.degrees.in.get(id) ?? 0,
      outDegree: result.degrees.out.get(id) ?? 0,
    })),
  };

  return JSON.stringify(payload, null, 2);
}
