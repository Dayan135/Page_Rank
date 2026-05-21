import type { PPRResult } from "@/lib/ppr/types";

export function resultToCSV(result: PPRResult): string {
  const rows: Array<[number, string, number, number, number]> = [];
  const ids = Array.from(result.ranks.keys());
  ids.sort((a, b) => (result.ranks.get(b) ?? 0) - (result.ranks.get(a) ?? 0));

  ids.forEach((id, rankIdx) => {
    rows.push([
      rankIdx + 1,
      id,
      result.ranks.get(id) ?? 0,
      result.degrees.in.get(id) ?? 0,
      result.degrees.out.get(id) ?? 0,
    ]);
  });

  const header = "rank,nodeId,score,in_degree,out_degree";
  const body = rows
    .map(([r, id, s, inD, outD]) => {
      const safeId = id.includes(",") || id.includes('"') ? `"${id.replace(/"/g, '""')}"` : id;
      return `${r},${safeId},${s},${inD},${outD}`;
    })
    .join("\n");

  return `${header}\n${body}\n`;
}
