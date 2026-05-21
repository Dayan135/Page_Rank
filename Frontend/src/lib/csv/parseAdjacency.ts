import type { Graph } from "@/lib/ppr/types";

export function parseAdjacency(rows: string[][]): Graph {
  const warnings: string[] = [];

  if (rows.length === 0) {
    throw new Error("Adjacency matrix is empty.");
  }

  const headerRow = rows[0];
  // First cell may be empty or contain a corner label — both treated as the corner.
  const nodeLabels = headerRow.slice(1).map((s) => s.trim()).filter((s) => s !== "");

  if (nodeLabels.length === 0) {
    throw new Error("Adjacency matrix has no node labels in the header row.");
  }

  const dataRows = rows.slice(1).filter((r) => r.some((c) => c.trim() !== ""));
  if (dataRows.length !== nodeLabels.length) {
    throw new Error(
      `Adjacency matrix is not square: header declares ${nodeLabels.length} columns but found ${dataRows.length} data rows.`,
    );
  }

  if (nodeLabels.length > 500) {
    warnings.push(
      `Large adjacency matrix (${nodeLabels.length}×${nodeLabels.length}) — consider using edge-list format for better performance.`,
    );
  }

  const nodes: Graph["nodes"] = nodeLabels.map((id) => ({ id }));
  const edges: Graph["edges"] = [];

  dataRows.forEach((row, rIdx) => {
    const rowLabel = (row[0] ?? "").trim();
    if (rowLabel !== nodeLabels[rIdx]) {
      warnings.push(
        `Row ${rIdx + 2}: row label "${rowLabel}" does not match column label "${nodeLabels[rIdx]}".`,
      );
    }
    const cells = row.slice(1);
    if (cells.length < nodeLabels.length) {
      throw new Error(
        `Row ${rIdx + 2}: expected ${nodeLabels.length} values, found ${cells.length}.`,
      );
    }
    for (let cIdx = 0; cIdx < nodeLabels.length; cIdx++) {
      const raw = (cells[cIdx] ?? "").trim();
      if (raw === "") continue;
      const value = Number(raw);
      if (!Number.isFinite(value)) {
        throw new Error(`Row ${rIdx + 2}, col ${cIdx + 2}: non-numeric value "${raw}".`);
      }
      if (value === 0) continue;
      edges.push({
        source: nodeLabels[rIdx],
        target: nodeLabels[cIdx],
        weight: value,
      });
    }
  });

  return { nodes, edges, format: "adjacency", warnings };
}
