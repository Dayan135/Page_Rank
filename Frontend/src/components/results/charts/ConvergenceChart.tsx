import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAppStore } from "@/store/useAppStore";
import { ChartInfo } from "./ChartInfo";

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
        <div className="flex items-center gap-1.5">
          <CardTitle className="text-base">Convergence</CardTitle>
          <ChartInfo label="What is the convergence chart?">
            How fast the power iteration settles. Each point is the L₁ residual — the total
            absolute change of the rank vector in that iteration (max over seed columns), on a
            log scale. Each step contracts the error by roughly α, so a healthy run is a
            straight descending line that stops when it drops below the tolerance. A flat or
            rising curve means the run did not converge — treat the scores with suspicion.
          </ChartInfo>
        </div>
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
