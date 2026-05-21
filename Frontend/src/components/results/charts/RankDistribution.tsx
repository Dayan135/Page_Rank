import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAppStore } from "@/store/useAppStore";

const ACCENT = "hsl(31 90% 44%)";
const PRIMARY = "hsl(226 71% 40%)";
const SECONDARY = "hsl(217 91% 60%)";

export function RankDistribution() {
  const result = useAppStore((s) => s.result);

  const data = useMemo(() => {
    if (!result) return [];
    return Array.from(result.ranks.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 30)
      .map(([id, score]) => ({ id, score }));
  }, [result]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Rank distribution (top 30)</CardTitle>
      </CardHeader>
      <CardContent className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 16, right: 16 }}>
            <CartesianGrid horizontal={false} stroke="hsl(var(--border))" />
            <XAxis type="number" stroke="hsl(var(--muted-foreground))" />
            <YAxis
              dataKey="id"
              type="category"
              width={100}
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
            <Bar dataKey="score">
              {data.map((_, i) => (
                <Cell key={i} fill={i === 0 ? ACCENT : i < 3 ? SECONDARY : PRIMARY} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
