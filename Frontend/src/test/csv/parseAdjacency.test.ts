import { describe, it, expect } from "vitest";
import { parseGraphCSV } from "@/lib/csv";

describe("parseAdjacency", () => {
  it("round-trips a known 4x4 matrix", async () => {
    const csv = ",A,B,C,D\nA,0,1,1,0\nB,1,0,0,0\nC,0,0,0,1\nD,1,0,0,0\n";
    const g = await parseGraphCSV(csv, "adjacency");
    expect(g.nodes.map((n) => n.id)).toEqual(["A", "B", "C", "D"]);
    expect(g.edges).toEqual([
      { source: "A", target: "B", weight: 1 },
      { source: "A", target: "C", weight: 1 },
      { source: "B", target: "A", weight: 1 },
      { source: "C", target: "D", weight: 1 },
      { source: "D", target: "A", weight: 1 },
    ]);
    expect(g.format).toBe("adjacency");
  });

  it("preserves label ordering from the header", async () => {
    const csv = ",Z,A\nZ,0,1\nA,1,0\n";
    const g = await parseGraphCSV(csv, "adjacency");
    expect(g.nodes.map((n) => n.id)).toEqual(["Z", "A"]);
  });

  it("rejects a non-square matrix", async () => {
    const csv = ",A,B,C\nA,0,1,0\nB,1,0,0\n";
    await expect(parseGraphCSV(csv, "adjacency")).rejects.toThrow(/square/);
  });
});
