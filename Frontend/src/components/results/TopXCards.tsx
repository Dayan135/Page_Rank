import { Card, CardContent } from "@/components/ui/card";
import { useAppStore } from "@/store/useAppStore";
import { formatScore } from "@/lib/utils";

export function TopXCards() {
  const result = useAppStore((s) => s.result);
  const topX = useAppStore((s) => s.params.topX);
  const labels = useAppStore((s) => s.graph?.labels);

  if (!result) return null;

  const sorted = Array.from(result.ranks.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, topX);

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {sorted.map(([id, score], i) => {
        const inD = result.degrees.in.get(id) ?? 0;
        const outD = result.degrees.out.get(id) ?? 0;
        const label = labels?.[id];
        return (
          <Card key={id} className={i === 0 ? "border-accent ring-1 ring-accent/30" : ""}>
            <CardContent className="space-y-2 p-4">
              <div className="flex items-baseline justify-between">
                <span className="font-mono text-xs uppercase text-muted-foreground">
                  rank #{i + 1}
                </span>
                <span className="font-mono text-xs text-muted-foreground tabular-nums">
                  in {inD} · out {outD}
                </span>
              </div>
              <div className="truncate font-mono text-lg font-semibold" title={label ?? id}>
                {label ?? id}
              </div>
              {label && (
                <div className="truncate font-mono text-xs text-muted-foreground" title={id}>
                  {id}
                </div>
              )}
              <div className="font-mono text-2xl tabular-nums text-primary">
                {formatScore(score)}
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
