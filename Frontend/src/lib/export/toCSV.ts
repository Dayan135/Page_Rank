import type { NodeId, PPRResult } from "@/lib/ppr/types";

function csvEscape(s: string): string {
  return s.includes(",") || s.includes('"') || s.includes("\n")
    ? `"${s.replace(/"/g, '""')}"`
    : s;
}

export function resultToCSV(result: PPRResult, labels?: Record<NodeId, string>): string {
  const ids = Array.from(result.ranks.keys());
  ids.sort((a, b) => (result.ranks.get(b) ?? 0) - (result.ranks.get(a) ?? 0));

  const hasLabels = labels && Object.keys(labels).length > 0;
  const header = hasLabels
    ? "rank,nodeId,name,score,in_degree,out_degree"
    : "rank,nodeId,score,in_degree,out_degree";

  const body = ids
    .map((id, rankIdx) => {
      const rank = rankIdx + 1;
      const score = result.ranks.get(id) ?? 0;
      const inD = result.degrees.in.get(id) ?? 0;
      const outD = result.degrees.out.get(id) ?? 0;
      const safeId = csvEscape(id);
      if (hasLabels) {
        const name = csvEscape(labels![id] ?? "");
        return `${rank},${safeId},${name},${score},${inD},${outD}`;
      }
      return `${rank},${safeId},${score},${inD},${outD}`;
    })
    .join("\n");

  return `${header}\n${body}\n`;
}
