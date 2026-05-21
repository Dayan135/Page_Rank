import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TopXCards } from "@/components/results/TopXCards";
import { RanksTable } from "@/components/results/RanksTable";
import { RankDistribution } from "@/components/results/charts/RankDistribution";
import { ConvergenceChart } from "@/components/results/charts/ConvergenceChart";
import { DegreeHistogram } from "@/components/results/charts/DegreeHistogram";
import { NetworkGraph } from "@/components/results/NetworkGraph";
import { ExportMenu } from "@/components/results/ExportMenu";
import { useAppStore } from "@/store/useAppStore";

export default function ResultsPage() {
  const navigate = useNavigate();
  const result = useAppStore((s) => s.result);

  useEffect(() => {
    if (!result) navigate("/upload", { replace: true });
  }, [result, navigate]);

  if (!result) return null;

  return (
    <div className="space-y-6">
      <Button variant="ghost" size="sm" onClick={() => navigate("/configure")} className="gap-2 -ml-2">
        <ArrowLeft className="h-4 w-4" />
        Back to Configure
      </Button>
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="font-mono text-2xl font-semibold tracking-tight">Results</h1>
          <p className="mt-1 text-sm text-muted-foreground tabular-nums">
            {result.iterations} iter · {result.converged ? "converged" : "max iter reached"} ·{" "}
            {result.ranks.size.toLocaleString()} nodes
          </p>
        </div>
        <ExportMenu />
      </div>

      <Tabs defaultValue="top">
        <TabsList>
          <TabsTrigger value="top">Top X</TabsTrigger>
          <TabsTrigger value="table">Table</TabsTrigger>
          <TabsTrigger value="charts">Charts</TabsTrigger>
          <TabsTrigger value="network">Network</TabsTrigger>
        </TabsList>
        <TabsContent value="top" className="pt-4">
          <TopXCards />
        </TabsContent>
        <TabsContent value="table" className="pt-4">
          <RanksTable />
        </TabsContent>
        <TabsContent value="charts" className="pt-4">
          <div className="grid gap-6 lg:grid-cols-2">
            <RankDistribution />
            <ConvergenceChart />
            <div className="lg:col-span-2">
              <DegreeHistogram />
            </div>
          </div>
        </TabsContent>
        <TabsContent value="network" className="pt-4">
          <NetworkGraph />
        </TabsContent>
      </Tabs>
    </div>
  );
}
