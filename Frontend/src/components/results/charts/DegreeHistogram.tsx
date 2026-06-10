import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAppStore } from "@/store/useAppStore";
import { ChartInfo } from "./ChartInfo";

const PRIMARY = "hsl(226 71% 40%)";
const ACCENT = "hsl(31 90% 44%)";

export function DegreeHistogram() {
  const result = useAppStore((s) => s.result);

  const data = useMemo(() => {
    if (!result) return [];
    const buckets = new Map<number, { inCount: number; outCount: number }>();
    const inc = (k: number, side: "in" | "out") => {
      const e = buckets.get(k) ?? { inCount: 0, outCount: 0 };
      if (side === "in") e.inCount += 1;
      else e.outCount += 1;
      buckets.set(k, e);
    };
    for (const v of result.degrees.in.values()) inc(v, "in");
    for (const v of result.degrees.out.values()) inc(v, "out");
    return Array.from(buckets.entries())
      .sort(([a], [b]) => a - b)
      .slice(0, 30)
      .map(([degree, c]) => ({ degree, inCount: c.inCount, outCount: c.outCount }));
  }, [result]);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-1.5">
          <CardTitle className="text-base">Degree distribution</CardTitle>
          <ChartInfo label="What is the degree distribution chart?">
            How connectivity is spread across the graph. For each degree value on the x-axis,
            the bars count nodes with that many incoming links (in-degree) and outgoing links
            (out-degree). Real-world graphs are typically heavy-tailed: most nodes have few
            links and a small set of hubs has many. Nodes at out-degree 0 are dangling — their
            PageRank mass is redistributed uniformly. Only the first 30 degree values are shown.
          </ChartInfo>
        </div>
        <p className="text-xs text-muted-foreground">
          Number of nodes at each in- and out-degree.
        </p>
      </CardHeader>
      <CardContent className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ left: 16, right: 16 }}>
            <CartesianGrid stroke="hsl(var(--border))" />
            <XAxis
              dataKey="degree"
              stroke="hsl(var(--muted-foreground))"
              tick={{ fontSize: 11, fontFamily: "Fira Code" }}
            />
            <YAxis
              stroke="hsl(var(--muted-foreground))"
              tick={{ fontSize: 11, fontFamily: "Fira Code" }}
            />
            <Tooltip
              contentStyle={{
                background: "hsl(var(--popover))",
                border: "1px solid hsl(var(--border))",
                color: "hsl(var(--popover-foreground))",
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12, fontFamily: "Fira Sans" }} />
            <Bar dataKey="inCount" name="in-degree" fill={PRIMARY} />
            <Bar dataKey="outCount" name="out-degree" fill={ACCENT} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
