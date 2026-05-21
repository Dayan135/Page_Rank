import { useNavigate } from "react-router-dom";
import { useEffect } from "react";
import { ArrowLeft } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ConfigPanel } from "@/components/configure/ConfigPanel";
import { useAppStore } from "@/store/useAppStore";

export default function ConfigurePage() {
  const navigate = useNavigate();
  const graph = useAppStore((s) => s.graph);

  useEffect(() => {
    if (!graph) navigate("/upload", { replace: true });
  }, [graph, navigate]);

  if (!graph) return null;

  return (
    <div className="space-y-6">
      <Button variant="ghost" size="sm" onClick={() => navigate("/upload")} className="gap-2 -ml-2">
        <ArrowLeft className="h-4 w-4" />
        Back to Upload
      </Button>
    <div className="grid gap-6 lg:grid-cols-[1fr_minmax(0,1.2fr)]">
      <ConfigPanel />
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Graph summary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 font-mono text-sm">
          <Row label="Format" value={graph.format} />
          <Row label="Nodes" value={graph.nodes.length.toLocaleString()} />
          <Row label="Edges" value={graph.edges.length.toLocaleString()} />
          {graph.warnings.length > 0 && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Warnings ({graph.warnings.length})
              </div>
              <ul className="mt-2 max-h-40 overflow-auto rounded-md border bg-muted/30 p-3 text-xs">
                {graph.warnings.slice(0, 50).map((w, i) => (
                  <li key={i} className="text-muted-foreground">
                    {w}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between border-b border-border/40 py-1">
      <span className="text-muted-foreground">{label}</span>
      <span className="tabular-nums">{value}</span>
    </div>
  );
}
