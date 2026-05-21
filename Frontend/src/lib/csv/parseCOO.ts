import type { Graph } from "@/lib/ppr/types";

interface Row {
  [key: string]: string | undefined;
}

const ROW_ALIASES = ["row_idx", "row", "i"];
const COL_ALIASES = ["col_idx", "col", "j"];
const VAL_ALIASES = ["value", "val", "v", "weight"];

function findColumn(headers: string[], aliases: string[]): string | null {
  const lower = headers.map((h) => h.trim().toLowerCase());
  for (const alias of aliases) {
    const idx = lower.indexOf(alias);
    if (idx !== -1) return headers[idx];
  }
  return null;
}

export function parseCOO(rows: Row[], headers: string[]): Graph {
  const warnings: string[] = [];
  const rowCol = findColumn(headers, ROW_ALIASES);
  const colCol = findColumn(headers, COL_ALIASES);
  const valCol = findColumn(headers, VAL_ALIASES);

  if (!rowCol || !colCol || !valCol) {
    throw new Error(
      `COO expects columns "row_idx", "col_idx", "value" (or aliases). Found: ${headers.join(", ")}`,
    );
  }

  const maxIndex = { v: -1 };
  const edges: Graph["edges"] = [];

  rows.forEach((row, i) => {
    const rRaw = (row[rowCol] ?? "").trim();
    const cRaw = (row[colCol] ?? "").trim();
    const vRaw = (row[valCol] ?? "").trim();
    if (rRaw === "" || cRaw === "" || vRaw === "") {
      warnings.push(`Row ${i + 2}: empty cell — skipped.`);
      return;
    }

    const rIdx = Number(rRaw);
    const cIdx = Number(cRaw);
    const val = Number(vRaw);

    if (!Number.isInteger(rIdx) || rIdx < 0) {
      throw new Error(`Row ${i + 2}: row_idx must be a non-negative integer, got "${rRaw}".`);
    }
    if (!Number.isInteger(cIdx) || cIdx < 0) {
      throw new Error(`Row ${i + 2}: col_idx must be a non-negative integer, got "${cRaw}".`);
    }
    if (!Number.isFinite(val)) {
      throw new Error(`Row ${i + 2}: non-numeric value "${vRaw}".`);
    }
    if (val === 0) return; // skip explicit zeros

    if (rIdx > maxIndex.v) maxIndex.v = rIdx;
    if (cIdx > maxIndex.v) maxIndex.v = cIdx;

    edges.push({ source: `n${rIdx}`, target: `n${cIdx}`, weight: val });
  });

  const nodes: Graph["nodes"] = [];
  for (let i = 0; i <= maxIndex.v; i++) {
    nodes.push({ id: `n${i}` });
  }

  return { nodes, edges, format: "coo", warnings };
}
