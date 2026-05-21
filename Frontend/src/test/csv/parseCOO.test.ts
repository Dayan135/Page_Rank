import { describe, it, expect } from "vitest";
import { parseGraphCSV } from "@/lib/csv";

describe("parseCOO", () => {
  it("generates n0..nN node ids and parses values", async () => {
    const csv = "row_idx,col_idx,value\n0,1,1.0\n1,2,0.5\n";
    const g = await parseGraphCSV(csv, "coo");
    expect(g.nodes.map((n) => n.id)).toEqual(["n0", "n1", "n2"]);
    expect(g.edges).toEqual([
      { source: "n0", target: "n1", weight: 1.0 },
      { source: "n1", target: "n2", weight: 0.5 },
    ]);
  });

  it("skips explicit zeros", async () => {
    const csv = "row_idx,col_idx,value\n0,1,1\n0,2,0\n";
    const g = await parseGraphCSV(csv, "coo");
    expect(g.edges).toHaveLength(1);
  });

  it("throws when required columns are missing", async () => {
    const csv = "i,j\n0,1\n";
    await expect(parseGraphCSV(csv, "coo")).rejects.toThrow(/row_idx|col_idx|value/);
  });

  it("rejects non-integer indices", async () => {
    const csv = "row_idx,col_idx,value\n0.5,1,1\n";
    await expect(parseGraphCSV(csv, "coo")).rejects.toThrow(/integer/);
  });
});
