import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAppStore } from "@/store/useAppStore";

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
        <CardTitle className="text-base">Degree distribution</CardTitle>
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
