import { describe, it, expect } from "vitest";
import { computeMockPersonalizedPageRank } from "@/lib/ppr/computeMockPersonalizedPageRank";
import { DEFAULT_PARAMS, type Graph, type PPRParams } from "@/lib/ppr/types";

function graph(ids: string[], edges: [string, string][]): Graph {
  return {
    nodes: ids.map((id) => ({ id })),
    edges: edges.map(([s, t]) => ({ source: s, target: t, weight: 1 })),
    format: "edge",
    warnings: [],
  };
}

function sumRanks(ranks: Map<string, number>): number {
  let s = 0;
  for (const v of ranks.values()) s += v;
  return s;
}

function params(overrides: Partial<PPRParams> = {}): PPRParams {
  return { ...DEFAULT_PARAMS, ...overrides };
}

describe("computeMockPersonalizedPageRank", () => {
  it("returns scores that sum to 1 on K5 (complete graph)", () => {
    const ids = ["A", "B", "C", "D", "E"];
    const edges: [string, string][] = [];
    for (const a of ids) for (const b of ids) if (a !== b) edges.push([a, b]);
    const g = graph(ids, edges);
    const r = computeMockPersonalizedPageRank(g, params());
    expect(sumRanks(r.ranks)).toBeCloseTo(1, 9);
    // Symmetric graph → uniform ranks
    for (const v of r.ranks.values()) expect(v).toBeCloseTo(1 / 5, 6);
  });

  it("returns scores that sum to 1 on a star graph", () => {
    const g = graph(
      ["c", "a", "b", "d"],
      [
        ["c", "a"],
        ["c", "b"],
        ["c", "d"],
        ["a", "c"],
        ["b", "c"],
        ["d", "c"],
      ],
    );
    const r = computeMockPersonalizedPageRank(g, params());
    expect(sumRanks(r.ranks)).toBeCloseTo(1, 9);
    // Hub "c" must outrank leaves
    expect(r.ranks.get("c")!).toBeGreaterThan(r.ranks.get("a")!);
  });

  it("returns scores that sum to 1 on a path graph", () => {
    const g = graph(
      ["A", "B", "C", "D"],
      [
        ["A", "B"],
        ["B", "C"],
        ["C", "D"],
      ],
    );
    const r = computeMockPersonalizedPageRank(g, params());
    expect(sumRanks(r.ranks)).toBeCloseTo(1, 9);
  });

  it("with alpha=0 reduces to the personalization vector", () => {
    const g = graph(
      ["A", "B", "C"],
      [
        ["A", "B"],
        ["B", "C"],
      ],
    );
    const r = computeMockPersonalizedPageRank(g, params({ alpha: 0, seeds: ["B"] }));
    expect(r.ranks.get("B")).toBeCloseTo(1, 9);
    expect(r.ranks.get("A")).toBeCloseTo(0, 9);
    expect(r.ranks.get("C")).toBeCloseTo(0, 9);
  });

  it("respects maxIter when alpha=1 (may not converge)", () => {
    const g = graph(
      ["A", "B", "C"],
      [
        ["A", "B"],
        ["B", "C"],
        ["C", "A"],
      ],
    );
    const r = computeMockPersonalizedPageRank(g, params({ alpha: 1, maxIter: 5 }));
    expect(r.iterations).toBeLessThanOrEqual(5);
    expect(sumRanks(r.ranks)).toBeCloseTo(1, 9);
  });

  it("handles a single isolated node + seed on it", () => {
    const g = graph(["X"], []);
    const r = computeMockPersonalizedPageRank(g, params({ seeds: ["X"] }));
    expect(r.ranks.get("X")).toBeCloseTo(1, 9);
  });

  it("biases mass toward the seeded component in a disconnected graph", () => {
    const g = graph(
      ["A", "B", "C", "D"],
      [
        ["A", "B"],
        ["B", "A"],
        ["C", "D"],
        ["D", "C"],
      ],
    );
    const r = computeMockPersonalizedPageRank(g, params({ seeds: ["A"] }));
    const compAB = (r.ranks.get("A") ?? 0) + (r.ranks.get("B") ?? 0);
    const compCD = (r.ranks.get("C") ?? 0) + (r.ranks.get("D") ?? 0);
    expect(compAB).toBeGreaterThan(compCD);
    expect(sumRanks(r.ranks)).toBeCloseTo(1, 9);
  });

  it("records a convergence history", () => {
    const g = graph(
      ["A", "B"],
      [
        ["A", "B"],
        ["B", "A"],
      ],
    );
    const r = computeMockPersonalizedPageRank(g, params());
    expect(r.convergenceHistory.length).toBe(r.iterations);
    expect(r.converged).toBe(true);
  });

  it("returns empty result for an empty graph", () => {
    const r = computeMockPersonalizedPageRank(graph([], []), params());
    expect(r.ranks.size).toBe(0);
    expect(r.iterations).toBe(0);
  });
});
