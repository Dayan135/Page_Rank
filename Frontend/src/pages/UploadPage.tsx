import { useNavigate } from "react-router-dom";
import { CheckCircle2, ArrowRight, RefreshCw } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FormatPicker } from "@/components/upload/FormatPicker";
import { FileDropzone } from "@/components/upload/FileDropzone";
import { SampleGraphPicker } from "@/components/upload/SampleGraphPicker";
import { useAppStore } from "@/store/useAppStore";
import { parseGraphCSV } from "@/lib/csv";
import { toast } from "@/components/ui/use-toast";

const FORMAT_LABEL: Record<string, string> = {
  edge: "Edge list",
  coo: "COO triplets",
  adjacency: "Adjacency matrix",
};

export default function UploadPage() {
  const navigate = useNavigate();
  const format = useAppStore((s) => s.format);
  const graph = useAppStore((s) => s.graph);
  const fileName = useAppStore((s) => s.fileName);
  const setFormat = useAppStore((s) => s.setFormat);
  const setGraph = useAppStore((s) => s.setGraph);
  const setRunStatus = useAppStore((s) => s.setRunStatus);

  const handleText = async (text: string, fmt: NonNullable<typeof format>, name?: string) => {
    try {
      setRunStatus("parsing");
      const g = await parseGraphCSV(text, fmt);
      setGraph(g, name);
      setRunStatus("idle");
      if (g.warnings.length > 0) {
        toast({
          title: `Parsed with ${g.warnings.length} warning${g.warnings.length === 1 ? "" : "s"}`,
          description: g.warnings.slice(0, 3).join(" · "),
        });
      }
      navigate("/configure");
    } catch (err) {
      setRunStatus("idle");
      const message = err instanceof Error ? err.message : String(err);
      toast({ title: "Could not parse file", description: message, variant: "destructive" });
    }
  };

  const handleReupload = () => {
    setGraph({ nodes: [], edges: [], format: format ?? "edge", warnings: [] });
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="font-mono text-2xl font-semibold tracking-tight">Upload graph</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Pick the CSV layout that matches your file, then drop it below.
          </p>
        </div>
        {graph && graph.nodes.length > 0 && (
          <Button onClick={() => navigate("/configure")} className="shrink-0 gap-2">
            Next
            <ArrowRight className="h-4 w-4" />
          </Button>
        )}
      </div>

      {graph && graph.nodes.length > 0 && (
        <Card className="border-primary/40 bg-primary/5">
          <CardContent className="flex items-center justify-between gap-4 py-4">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="h-5 w-5 shrink-0 text-primary" />
              <div className="font-mono text-sm">
                <p className="font-semibold">
                  {fileName ?? "Graph loaded"}
                </p>
                <p className="text-muted-foreground tabular-nums">
                  {graph.nodes.length.toLocaleString()} nodes ·{" "}
                  {graph.edges.length.toLocaleString()} edges ·{" "}
                  {format ? FORMAT_LABEL[format] : ""}
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleReupload}
              className="shrink-0 gap-2"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Re-upload
            </Button>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">1. Choose CSV format</CardTitle>
          <CardDescription>The parser expects exactly the layout you select.</CardDescription>
        </CardHeader>
        <CardContent>
          <FormatPicker value={format} onChange={setFormat} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">2. Upload file</CardTitle>
          <CardDescription>
            {format ? "Drop your CSV, or click to browse." : "Pick a format above to enable upload."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone
            disabled={!format}
            onFile={async (file) => {
              if (!format) return;
              const text = await file.text();
              handleText(text, format, file.name);
            }}
            onError={(message) =>
              toast({ title: "File rejected", description: message, variant: "destructive" })
            }
          />
          <SampleGraphPicker
            onLoad={({ text, format: f, name }) => {
              setFormat(f);
              handleText(text, f, name);
            }}
          />
        </CardContent>
      </Card>

      {graph && graph.nodes.length > 0 && (
        <div className="flex justify-end">
          <Button onClick={() => navigate("/configure")} className="gap-2">
            Next: Configure
            <ArrowRight className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
