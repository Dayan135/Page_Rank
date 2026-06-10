import { saveAs } from "file-saver";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAppStore } from "@/store/useAppStore";
import { resultToCSV } from "@/lib/export/toCSV";
import { resultToJSON } from "@/lib/export/toJSON";

export function ExportMenu() {
  const graph = useAppStore((s) => s.graph);
  const params = useAppStore((s) => s.params);
  const result = useAppStore((s) => s.result);

  if (!graph || !result) return null;

  const downloadCSV = () => {
    const csv = resultToCSV(result, graph.labels);
    saveAs(new Blob([csv], { type: "text/csv;charset=utf-8" }), "ppr-results.csv");
  };

  const downloadJSON = () => {
    const json = resultToJSON(graph, params, result);
    saveAs(new Blob([json], { type: "application/json" }), "ppr-results.json");
  };

  return (
    <div className="flex items-center gap-2">
      <Button variant="outline" size="sm" onClick={downloadCSV} data-testid="export-csv">
        <Download className="h-4 w-4" /> CSV
      </Button>
      <Button variant="outline" size="sm" onClick={downloadJSON} data-testid="export-json">
        <Download className="h-4 w-4" /> JSON
      </Button>
    </div>
  );
}
