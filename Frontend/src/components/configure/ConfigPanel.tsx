import { useNavigate } from "react-router-dom";
import { Play, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlphaSlider } from "./AlphaSlider";
import { IterationInput } from "./IterationInput";
import { TopXInput } from "./TopXInput";
import { SeedNodePicker } from "./SeedNodePicker";
import { useAppStore } from "@/store/useAppStore";
import { algorithm } from "@/lib/ppr/adapter";
import { toast } from "@/components/ui/use-toast";

export function ConfigPanel() {
  const navigate = useNavigate();
  const graph = useAppStore((s) => s.graph);
  const params = useAppStore((s) => s.params);
  const setParams = useAppStore((s) => s.setParams);
  const setSeeds = useAppStore((s) => s.setSeeds);
  const runStatus = useAppStore((s) => s.runStatus);
  const setRunStatus = useAppStore((s) => s.setRunStatus);
  const setResult = useAppStore((s) => s.setResult);

  if (!graph) return null;

  const compute = async () => {
    setRunStatus("computing");
    try {
      const result = await algorithm.run(graph, params);
      setResult(result);
      navigate("/results");
    } catch (err) {
      setRunStatus("error");
      toast({
        title: "Computation failed",
        description: err instanceof Error ? err.message : String(err),
        variant: "destructive",
      });
    }
  };

  const isRunning = runStatus === "computing";
  const noSeeds = params.seeds.length === 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Parameters</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <AlphaSlider value={params.alpha} onChange={(alpha) => setParams({ alpha })} />
        <IterationInput
          maxIter={params.maxIter}
          tolerance={params.tolerance}
          onMaxIterChange={(maxIter) => setParams({ maxIter })}
          onToleranceChange={(tolerance) => setParams({ tolerance })}
        />
        <TopXInput
          value={Math.min(params.topX, graph.nodes.length)}
          max={graph.nodes.length}
          onChange={(topX) => setParams({ topX })}
        />
        <SeedNodePicker nodes={graph.nodes} seeds={params.seeds} onChange={setSeeds} labels={graph.labels} />
        <Button
          variant="accent"
          size="lg"
          onClick={compute}
          disabled={isRunning || noSeeds}
          className="w-full"
        >
          {isRunning ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Computing…
            </>
          ) : (
            <>
              <Play className="h-4 w-4" /> Compute PPR
            </>
          )}
        </Button>
        {noSeeds && (
          <p className="text-center text-xs text-muted-foreground">
            Select at least one seed node to compute.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
