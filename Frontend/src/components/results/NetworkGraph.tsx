import { useMemo } from "react";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  type Edge as RFEdge,
  type Node as RFNode,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAppStore } from "@/store/useAppStore";

const ACCENT = "#D97706";
const PRIMARY = "#1E40AF";
const SECONDARY = "#93C5FD";

function circleLayout(ids: string[]): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const radius = Math.max(120, ids.length * 12);
  ids.forEach((id, i) => {
    const angle = (2 * Math.PI * i) / ids.length;
    positions.set(id, { x: radius * Math.cos(angle), y: radius * Math.sin(angle) });
  });
  return positions;
}

export function NetworkGraph() {
  const graph = useAppStore((s) => s.graph);
  const result = useAppStore((s) => s.result);

  const { nodes, edges } = useMemo<{ nodes: RFNode[]; edges: RFEdge[] }>(() => {
    if (!graph || !result) return { nodes: [], edges: [] };

    const cap = 200;
    const ranked = Array.from(result.ranks.entries()).sort((a, b) => b[1] - a[1]);
    const visible = new Set(ranked.slice(0, cap).map(([id]) => id));
    const visibleIds = ranked.filter(([id]) => visible.has(id)).map(([id]) => id);
    const positions = circleLayout(visibleIds);

    const maxScore = visibleIds.reduce((m, id) => Math.max(m, result.ranks.get(id) ?? 0), 0);

    const rfNodes: RFNode[] = visibleIds.map((id, idx) => {
      const score = result.ranks.get(id) ?? 0;
      const size = 12 + 60 * Math.sqrt(score / (maxScore || 1));
      const fill = idx < 5 ? ACCENT : idx < 20 ? PRIMARY : SECONDARY;
      return {
        id,
        position: positions.get(id) ?? { x: 0, y: 0 },
        data: { label: id },
        style: {
          width: size,
          height: size,
          background: fill,
          border: "2px solid white",
          borderRadius: "50%",
          color: "white",
          fontFamily: "Fira Code",
          fontSize: Math.max(9, size / 6),
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        },
      };
    });

    const rfEdges: RFEdge[] = [];
    for (const e of graph.edges) {
      if (!visible.has(e.source) || !visible.has(e.target)) continue;
      rfEdges.push({
        id: `${e.source}->${e.target}`,
        source: e.source,
        target: e.target,
        style: { stroke: "#94a3b8", strokeWidth: 1 },
      });
      if (rfEdges.length >= 1000) break;
    }

    return { nodes: rfNodes, edges: rfEdges };
  }, [graph, result]);

  if (!graph || !result) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Network</CardTitle>
        <p className="text-xs text-muted-foreground">
          Node size ∝ √(PPR score). Showing up to 200 top-ranked nodes and 1000 edges.
        </p>
      </CardHeader>
      <CardContent className="h-[560px] p-0">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          minZoom={0.1}
          maxZoom={3}
          nodesDraggable
          nodesConnectable={false}
          elementsSelectable
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={24} color="hsl(var(--border))" />
          <MiniMap pannable zoomable />
          <Controls showInteractive={false} />
        </ReactFlow>
      </CardContent>
    </Card>
  );
}
