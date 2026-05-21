import { describe, it, expect } from "vitest";
import { computeDegrees } from "@/lib/ppr/degrees";
import type { Graph } from "@/lib/ppr/types";

function graph(edges: [string, string][], extraNodes: string[] = []): Graph {
  const nodeSet = new Set<string>(extraNodes);
  for (const [s, t] of edges) {
    nodeSet.add(s);
    nodeSet.add(t);
  }
  return {
    nodes: Array.from(nodeSet).map((id) => ({ id })),
    edges: edges.map(([source, target]) => ({ source, target, weight: 1 })),
    format: "edge",
    warnings: [],
  };
}

describe("computeDegrees", () => {
  it("counts in-degrees and out-degrees on a directed graph", () => {
    const g = graph([
      ["A", "B"],
      ["A", "C"],
      ["B", "C"],
    ]);
    const d = computeDegrees(g);
    expect(d.out.get("A")).toBe(2);
    expect(d.out.get("B")).toBe(1);
    expect(d.out.get("C")).toBe(0);
    expect(d.in.get("A")).toBe(0);
    expect(d.in.get("B")).toBe(1);
    expect(d.in.get("C")).toBe(2);
  });

  it("returns 0 for isolated nodes", () => {
    const g = graph([], ["X"]);
    const d = computeDegrees(g);
    expect(d.in.get("X")).toBe(0);
    expect(d.out.get("X")).toBe(0);
  });
});
