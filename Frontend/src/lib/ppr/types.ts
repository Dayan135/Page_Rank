export type NodeId = string;

export type CSVFormat = "edge" | "coo" | "adjacency";

export interface GraphNode {
  id: NodeId;
}

export interface GraphEdge {
  source: NodeId;
  target: NodeId;
  weight: number;
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  format: CSVFormat;
  warnings: string[];
}

export interface PPRParams {
  alpha: number;            // damping factor, [0, 1]
  maxIter: number;
  tolerance: number;
  topX: number;
  seeds: NodeId[];          // empty → uniform personalization
}

export interface PPRResult {
  ranks: Map<NodeId, number>;
  iterations: number;
  converged: boolean;
  convergenceHistory: number[];
  degrees: {
    in: Map<NodeId, number>;
    out: Map<NodeId, number>;
  };
}

export interface PPRAlgorithm {
  run(graph: Graph, params: PPRParams): Promise<PPRResult>;
}

export const DEFAULT_PARAMS: PPRParams = {
  alpha: 0.85,
  maxIter: 100,
  tolerance: 1e-6,
  topX: 10,
  seeds: [],
};
