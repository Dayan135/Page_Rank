import { create } from "zustand";
import type { CSVFormat, Graph, NodeId, PPRParams, PPRResult } from "@/lib/ppr/types";
import { DEFAULT_PARAMS } from "@/lib/ppr/types";

export type RunStatus = "idle" | "parsing" | "computing" | "done" | "error";

interface AppState {
  format: CSVFormat | null;
  graph: Graph | null;
  fileName: string | null;
  params: PPRParams;
  result: PPRResult | null;
  runStatus: RunStatus;
  error: string | null;

  setFormat: (format: CSVFormat) => void;
  setGraph: (graph: Graph, fileName?: string) => void;
  setParams: (patch: Partial<PPRParams>) => void;
  setSeeds: (seeds: NodeId[]) => void;
  setResult: (result: PPRResult) => void;
  setRunStatus: (status: RunStatus) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  format: null,
  graph: null,
  fileName: null,
  params: { ...DEFAULT_PARAMS },
  result: null,
  runStatus: "idle",
  error: null,

  setFormat: (format) => set({ format }),
  setGraph: (graph, fileName) => set({ graph, fileName: fileName ?? null, result: null, error: null }),
  setParams: (patch) =>
    set((state) => ({ params: { ...state.params, ...patch } })),
  setSeeds: (seeds) =>
    set((state) => ({ params: { ...state.params, seeds } })),
  setResult: (result) => set({ result, runStatus: "done" }),
  setRunStatus: (runStatus) => set({ runStatus }),
  setError: (error) => set({ error, runStatus: error ? "error" : "idle" }),
  reset: () =>
    set({
      format: null,
      graph: null,
      fileName: null,
      params: { ...DEFAULT_PARAMS },
      result: null,
      runStatus: "idle",
      error: null,
    }),
}));
