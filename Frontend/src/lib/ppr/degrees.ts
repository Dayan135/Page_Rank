import type { Graph, NodeId } from "./types";

export function computeDegrees(graph: Graph): {
  in: Map<NodeId, number>;
  out: Map<NodeId, number>;
} {
  const inDeg = new Map<NodeId, number>();
  const outDeg = new Map<NodeId, number>();

  for (const node of graph.nodes) {
    inDeg.set(node.id, 0);
    outDeg.set(node.id, 0);
  }

  for (const edge of graph.edges) {
    outDeg.set(edge.source, (outDeg.get(edge.source) ?? 0) + 1);
    inDeg.set(edge.target, (inDeg.get(edge.target) ?? 0) + 1);
  }

  return { in: inDeg, out: outDeg };
}
