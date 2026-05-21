import type { Graph } from "@/lib/ppr/types";

interface Row {
  [key: string]: string | undefined;
}

const SOURCE_ALIASES = ["source", "from", "src", "u"];
const TARGET_ALIASES = ["target", "to", "dst", "v"];
const WEIGHT_ALIASES = ["weight", "w", "value"];

function findColumn(headers: string[], aliases: string[]): string | null {
  const lower = headers.map((h) => h.trim().toLowerCase());
  for (const alias of aliases) {
    const idx = lower.indexOf(alias);
    if (idx !== -1) return headers[idx];
  }
  return null;
}

export function parseEdgeList(rows: Row[], headers: string[]): Graph {
  const warnings: string[] = [];
  const sourceCol = findColumn(headers, SOURCE_ALIASES);
  const targetCol = findColumn(headers, TARGET_ALIASES);

  if (!sourceCol || !targetCol) {
    throw new Error(
      `Edge list expects columns "source" and "target" (or aliases). Found: ${headers.join(", ")}`,
    );
  }

  const weightCol = findColumn(headers, WEIGHT_ALIASES);
  const nodeSet = new Set<string>();
  const edges: Graph["edges"] = [];

  rows.forEach((row, i) => {
    const source = (row[sourceCol] ?? "").trim();
    const target = (row[targetCol] ?? "").trim();
    if (!source || !target) {
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

    if (source === target) {
      warnings.push(`Row ${i + 2}: self-loop on "${source}".`);
    }

    nodeSet.add(source);
    nodeSet.add(target);
    edges.push({ source, target, weight });
  });

  return {
    nodes: Array.from(nodeSet).map((id) => ({ id })),
    edges,
    format: "edge",
    warnings,
  };
}
