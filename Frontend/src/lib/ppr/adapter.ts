import type { Graph, NodeId, PPRAlgorithm, PPRParams, PPRResult } from "./types";
import { computeMockPersonalizedPageRank } from "./computeMockPersonalizedPageRank";

// In-browser reference implementation. Kept for tests and offline use.
export const mockAdapter: PPRAlgorithm = {
  run: (g, p) =>
    new Promise((resolve) => {
      // Yield to the event loop so the UI can render the spinner before the heavy work.
      setTimeout(() => resolve(computeMockPersonalizedPageRank(g, p)), 0);
    }),
};

// Base URL for the FastAPI/CUDA backend. In dev, "/api" is proxied to the GPU
// host by Vite (see vite.config.ts). For a static build, set VITE_PPR_API to the
// backend's absolute URL.
const API_BASE = import.meta.env.VITE_PPR_API ?? "/api";

// Wire shape of POST /api/ppr — Map fields travel as plain JSON objects.
interface PPRResultWire {
  ranks: Record<NodeId, number>;
  iterations: number;
  converged: boolean;
  convergenceHistory: number[];
  degrees: { in: Record<NodeId, number>; out: Record<NodeId, number> };
}

function reviveResult(w: PPRResultWire): PPRResult {
  return {
    ranks: new Map(Object.entries(w.ranks)),
    iterations: w.iterations,
    converged: w.converged,
    convergenceHistory: w.convergenceHistory ?? [],
    degrees: {
      in: new Map(Object.entries(w.degrees.in)),
      out: new Map(Object.entries(w.degrees.out)),
    },
  };
}

// Real backend: POST the graph + params to the CUDA PPR service and rebuild Maps.
export const httpAdapter: PPRAlgorithm = {
  async run(graph: Graph, params: PPRParams): Promise<PPRResult> {
    const res = await fetch(`${API_BASE}/ppr`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ graph, params }),
    });
    if (!res.ok) {
      // FastAPI errors are { detail: string | object }.
      let message = `PPR backend error (${res.status})`;
      try {
        const body = await res.json();
        if (body?.detail) {
          message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
        }
      } catch {
        /* non-JSON error body — keep the status message */
      }
      throw new Error(message);
    }
    return reviveResult((await res.json()) as PPRResultWire);
  },
};

// The UI calls algorithm.run(graph, params); this is the single swap-point.
export const algorithm: PPRAlgorithm = httpAdapter;
