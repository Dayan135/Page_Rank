import { describe, it, expect } from "vitest";
import { parseCustom } from "@/lib/csv/parseCustom";

const makeRows = (csv: string, sep = ",") => {
  const [headerLine, ...dataLines] = csv.trim().split("\n");
  const headers = headerLine.split(sep).map((h) => h.trim());
  return {
    headers,
    rows: dataLines.map((line) => {
      const vals = line.split(sep);
      return Object.fromEntries(headers.map((h, i) => [h, vals[i] ?? ""]));
    }),
  };
};

describe("parseCustom", () => {
  it("parses a simple edge list with arbitrary column names", () => {
    const { rows, headers } = makeRows("from_id,to_id\n1,2\n2,3\n");
    const g = parseCustom(rows, headers, { fromCol: "from_id", toCol: "to_id" });
    expect(g.nodes.map((n) => n.id).sort()).toEqual(["1", "2", "3"]);
    expect(g.edges).toEqual([
      { source: "1", target: "2", weight: 1 },
      { source: "2", target: "3", weight: 1 },
    ]);
    expect(g.format).toBe("custom-edge-list");
    expect(g.labels).toBeUndefined();
  });

  it("collects labels from the label column", () => {
    const { rows, headers } = makeRows(
      "id_from,id_to,title\n10,20,Cat\n20,30,Animal\n",
    );
    const g = parseCustom(rows, headers, {
      fromCol: "id_from",
      toCol: "id_to",
      labelCol: "title",
    });
    expect(g.labels).toEqual({ "10": "Cat", "20": "Animal" });
  });

  it("parses a TSV (tab-delimited) edge list", () => {
    const tsv = makeRows("src\tdst\tname\nA\tB\tAlpha\nB\tC\tBeta\n", "\t");
    const g = parseCustom(tsv.rows, tsv.headers, {
      fromCol: "src",
      toCol: "dst",
      labelCol: "name",
    });
    expect(g.edges).toEqual([
      { source: "A", target: "B", weight: 1 },
      { source: "B", target: "C", weight: 1 },
    ]);
    expect(g.labels).toEqual({ A: "Alpha", B: "Beta" });
  });

  it("respects an explicit weight column", () => {
    const { rows, headers } = makeRows("u,v,w\nA,B,0.5\nB,C,2.0\n");
    const g = parseCustom(rows, headers, { fromCol: "u", toCol: "v", weightCol: "w" });
    expect(g.edges.map((e) => e.weight)).toEqual([0.5, 2.0]);
  });

  it("skips rows with empty from or to and adds a warning", () => {
    const { rows, headers } = makeRows("a,b\n,2\n1,2\n");
    const g = parseCustom(rows, headers, { fromCol: "a", toCol: "b" });
    expect(g.edges).toHaveLength(1);
    expect(g.warnings.length).toBeGreaterThan(0);
  });

  it("throws when fromCol is not in headers", () => {
    const { rows, headers } = makeRows("x,y\n1,2\n");
    expect(() => parseCustom(rows, headers, { fromCol: "missing", toCol: "y" })).toThrow(
      /missing/,
    );
  });

  it("throws when toCol is not in headers", () => {
    const { rows, headers } = makeRows("x,y\n1,2\n");
    expect(() => parseCustom(rows, headers, { fromCol: "x", toCol: "missing" })).toThrow(
      /missing/,
    );
  });

  it("throws on non-numeric weight", () => {
    const { rows, headers } = makeRows("a,b,w\n1,2,oops\n");
    expect(() =>
      parseCustom(rows, headers, { fromCol: "a", toCol: "b", weightCol: "w" }),
    ).toThrow(/non-numeric weight/);
  });
});
