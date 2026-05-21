import { describe, it, expect } from "vitest";
import { parseGraphCSV } from "@/lib/csv";

describe("parseGraphCSV dispatch", () => {
  it("dispatches based on the format argument", async () => {
    const csv = "source,target\nA,B\n";
    const g = await parseGraphCSV(csv, "edge");
    expect(g.format).toBe("edge");
  });

  it("surfaces an actionable error when the file shape doesn't match the chosen format", async () => {
    const csv = "source,target\nA,B\n";
    await expect(parseGraphCSV(csv, "coo")).rejects.toThrow(/row_idx|col_idx|value/);
  });
});
