import type { Graph } from "@/lib/ppr/types";

export interface CustomMapping {
  fromCol: string;
  toCol: string;
  labelCol?: string;
  weightCol?: string;
}

export function parseCustom(
  rows: Record<string, string>[],
  headers: string[],
  mapping: CustomMapping,
): Graph {
  const { fromCol, toCol, labelCol, weightCol } = mapping;

  if (!headers.includes(fromCol)) {
    throw new Error(`Column "${fromCol}" not found. Available: ${headers.join(", ")}`);
  }
  if (!headers.includes(toCol)) {
    throw new Error(`Column "${toCol}" not found. Available: ${headers.join(", ")}`);
  }

  const warnings: string[] = [];
  const nodeSet = new Set<string>();
  const labels: Record<string, string> = {};
  const edges: Graph["edges"] = [];

  rows.forEach((row, i) => {
    const from = (row[fromCol] ?? "").trim();
    const to = (row[toCol] ?? "").trim();

    if (!from || !to) {
      warnings.push(`Row ${i + 2}: empty source or target — skipped.`);
      return;
    }

    let weight = 1.0;
    if (weightCol) {
      const raw = (row[weightCol] ?? "").trim();
      if (raw !== "") {
        const parsed = Number(raw);
        if (!Number.isFinite(parsed)) {
          throw new Error(`Row ${i + 2}: non-numeric weight "${raw}".`);
        }
        weight = parsed;
      }
    }

    nodeSet.add(from);
    nodeSet.add(to);

    if (labelCol) {
      const lbl = (row[labelCol] ?? "").trim();
      if (lbl) labels[from] = lbl;
    }

    edges.push({ source: from, target: to, weight });
  });

  return {
    nodes: Array.from(nodeSet).map((id) => ({ id })),
    edges,
    format: "custom-edge-list",
    warnings,
    labels: Object.keys(labels).length > 0 ? labels : undefined,
  };
}
