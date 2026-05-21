import { describe, it, expect } from "vitest";
import { parseGraphCSV } from "@/lib/csv";

describe("parseEdgeList", () => {
  it("parses a simple edge list with default weights", async () => {
    const csv = "source,target\nA,B\nB,C\n";
    const g = await parseGraphCSV(csv, "edge");
    expect(g.nodes.map((n) => n.id).sort()).toEqual(["A", "B", "C"]);
    expect(g.edges).toEqual([
      { source: "A", target: "B", weight: 1 },
      { source: "B", target: "C", weight: 1 },
    ]);
    expect(g.format).toBe("edge");
  });

  it("respects an explicit weight column", async () => {
    const csv = "source,target,weight\nA,B,0.5\nB,C,2\n";
    const g = await parseGraphCSV(csv, "edge");
    expect(g.edges.map((e) => e.weight)).toEqual([0.5, 2]);
  });

  it("accepts header aliases (from/to/w)", async () => {
    const csv = "from,to,w\nA,B,1\n";
    const g = await parseGraphCSV(csv, "edge");
    expect(g.edges).toEqual([{ source: "A", target: "B", weight: 1 }]);
  });

  it("warns about self-loops but keeps them", async () => {
    const csv = "source,target\nA,A\nA,B\n";
    const g = await parseGraphCSV(csv, "edge");
    expect(g.edges).toHaveLength(2);
    expect(g.warnings.some((w) => w.includes("self-loop"))).toBe(true);
  });

  it("throws when source/target columns are missing", async () => {
    const csv = "a,b,c\n1,2,3\n";
    await expect(parseGraphCSV(csv, "edge")).rejects.toThrow(/source/);
  });

  it("rejects a non-numeric weight with the row number", async () => {
    const csv = "source,target,weight\nA,B,oops\n";
    await expect(parseGraphCSV(csv, "edge")).rejects.toThrow(/Row 2.*weight/);
  });
});
