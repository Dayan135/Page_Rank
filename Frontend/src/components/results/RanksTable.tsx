import { useMemo, useState } from "react";
import { ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { useAppStore } from "@/store/useAppStore";
import { cn, formatScore } from "@/lib/utils";

type SortKey = "rank" | "id" | "score" | "in" | "out";
type SortDir = "asc" | "desc";

interface Row {
  rank: number;
  id: string;
  score: number;
  in: number;
  out: number;
}

export function RanksTable() {
  const result = useAppStore((s) => s.result);
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const rows = useMemo<Row[]>(() => {
    if (!result) return [];
    const arr = Array.from(result.ranks.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([id, score], i) => ({
        rank: i + 1,
        id,
        score,
        in: result.degrees.in.get(id) ?? 0,
        out: result.degrees.out.get(id) ?? 0,
      }));
    return arr;
  }, [result]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const base = q ? rows.filter((r) => r.id.toLowerCase().includes(q)) : rows;
    const sorted = [...base].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "string" && typeof bv === "string") {
        return sortDir === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      return sortDir === "asc"
        ? (av as number) - (bv as number)
        : (bv as number) - (av as number);
    });
    return sorted;
  }, [rows, query, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "id" ? "asc" : "desc");
    }
  };

  const Icon = ({ k }: { k: SortKey }) =>
    sortKey !== k ? (
      <ArrowUpDown className="ml-1 inline h-3 w-3 opacity-40" />
    ) : sortDir === "asc" ? (
      <ArrowUp className="ml-1 inline h-3 w-3" />
    ) : (
      <ArrowDown className="ml-1 inline h-3 w-3" />
    );

  if (!result) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <Input
          placeholder="Search node ID…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="max-w-sm"
        />
        <div className="text-xs text-muted-foreground tabular-nums">
          {filtered.length.toLocaleString()} / {rows.length.toLocaleString()} rows
        </div>
      </div>
      <div className="rounded-md border">
        <div className="max-h-[60dvh] overflow-auto">
          <Table>
            <TableHeader className="sticky top-0 z-10 bg-background">
              <TableRow>
                <SortHeader k="rank" current={sortKey} dir={sortDir} onClick={toggleSort}>
                  Rank <Icon k="rank" />
                </SortHeader>
                <SortHeader k="id" current={sortKey} dir={sortDir} onClick={toggleSort}>
                  Node <Icon k="id" />
                </SortHeader>
                <SortHeader k="score" current={sortKey} dir={sortDir} onClick={toggleSort} numeric>
                  Score <Icon k="score" />
                </SortHeader>
                <SortHeader k="in" current={sortKey} dir={sortDir} onClick={toggleSort} numeric>
                  In-deg <Icon k="in" />
                </SortHeader>
                <SortHeader k="out" current={sortKey} dir={sortDir} onClick={toggleSort} numeric>
                  Out-deg <Icon k="out" />
                </SortHeader>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.slice(0, 1000).map((r) => (
                <TableRow key={r.id}>
                  <TableCell className="font-mono tabular-nums">{r.rank}</TableCell>
                  <TableCell className="font-mono">{r.id}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {formatScore(r.score)}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{r.in}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{r.out}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        {filtered.length > 1000 && (
          <div className="border-t bg-muted/30 p-2 text-center text-xs text-muted-foreground">
            Showing first 1000 rows — narrow with search to see more.
          </div>
        )}
      </div>
    </div>
  );
}

function SortHeader({
  k,
  current,
  dir,
  onClick,
  numeric,
  children,
}: {
  k: SortKey;
  current: SortKey;
  dir: SortDir;
  onClick: (k: SortKey) => void;
  numeric?: boolean;
  children: React.ReactNode;
}) {
  return (
    <TableHead className={cn(numeric && "text-right")}>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => onClick(k)}
        aria-sort={current === k ? (dir === "asc" ? "ascending" : "descending") : "none"}
        className={cn("h-7 px-2 font-medium", numeric && "ml-auto")}
      >
        {children}
      </Button>
    </TableHead>
  );
}
