import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAppStore } from "@/store/useAppStore";

const PRIMARY = "hsl(226 71% 40%)";

export function ConvergenceChart() {
  const result = useAppStore((s) => s.result);
  if (!result) return null;

  const data = result.convergenceHistory.map((residual, idx) => ({
    iter: idx + 1,
    residual: Math.max(residual, 1e-16),
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Convergence</CardTitle>
        <p className="text-xs text-muted-foreground">
          L₁ residual per iteration (log scale). {result.iterations} iter ·{" "}
          {result.converged ? "converged" : "max iter reached"}.
        </p>
      </CardHeader>
      <CardContent className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ left: 16, right: 16 }}>
            <CartesianGrid stroke="hsl(var(--border))" />
            <XAxis
              dataKey="iter"
              stroke="hsl(var(--muted-foreground))"
              tick={{ fontSize: 11, fontFamily: "Fira Code" }}
            />
            <YAxis
              scale="log"
              domain={["auto", "auto"]}
              stroke="hsl(var(--muted-foreground))"
              tick={{ fontSize: 11, fontFamily: "Fira Code" }}
              tickFormatter={(v) => v.toExponential(0)}
            />
            <Tooltip
              contentStyle={{
                background: "hsl(var(--popover))",
                border: "1px solid hsl(var(--border))",
                color: "hsl(var(--popover-foreground))",
              }}
              formatter={(v: number) => v.toExponential(3)}
            />
            <Line
              type="monotone"
              dataKey="residual"
              stroke={PRIMARY}
              strokeWidth={2}
              dot={{ r: 2 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
