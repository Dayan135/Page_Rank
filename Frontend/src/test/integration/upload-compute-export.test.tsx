import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/toaster";
import { Routes, Route, Navigate } from "react-router-dom";
import UploadPage from "@/pages/UploadPage";
import ConfigurePage from "@/pages/ConfigurePage";
import ResultsPage from "@/pages/ResultsPage";
import { useAppStore } from "@/store/useAppStore";

// Mock React Flow — the integration test only cares about data flow, not the canvas render.
vi.mock("@xyflow/react", () => ({
  ReactFlow: () => null,
  Background: () => null,
  Controls: () => null,
  MiniMap: () => null,
}));

// Mock file-saver so we can spy on saveAs without actually triggering a download.
const saveAsSpy = vi.fn();
vi.mock("file-saver", () => ({
  saveAs: (...args: unknown[]) => saveAsSpy(...args),
}));

// The app's adapter now POSTs to the CUDA backend over HTTP. For this UI
// data-flow test (no server in happy-dom), route algorithm.run through the
// in-browser reference implementation instead.
vi.mock("@/lib/ppr/adapter", async () => {
  const { computeMockPersonalizedPageRank } = await import(
    "@/lib/ppr/computeMockPersonalizedPageRank"
  );
  return {
    algorithm: {
      run: async (
        graph: import("@/lib/ppr/types").Graph,
        params: import("@/lib/ppr/types").PPRParams,
      ) => computeMockPersonalizedPageRank(graph, params),
    },
  };
});

function renderApp() {
  return render(
    <ThemeProvider attribute="class" defaultTheme="light">
      <TooltipProvider>
        <MemoryRouter initialEntries={["/upload"]}>
          <Routes>
            <Route path="/" element={<Navigate to="/upload" replace />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/configure" element={<ConfigurePage />} />
            <Route path="/results" element={<ResultsPage />} />
          </Routes>
        </MemoryRouter>
        <Toaster />
      </TooltipProvider>
    </ThemeProvider>,
  );
}

describe("upload → compute → export integration", () => {
  beforeEach(() => {
    useAppStore.getState().reset();
    saveAsSpy.mockClear();
  });

  it("walks the full happy path", async () => {
    renderApp();

    // 1. Select Edge list format.
    await userEvent.click(screen.getByText("Edge list"));
    expect(useAppStore.getState().format).toBe("edge");

    // 2. Upload a tiny CSV via the file input.
    const csv = "source,target\nA,B\nB,C\nC,A\n";
    const file = new File([csv], "test.csv", { type: "text/csv" });
    await userEvent.upload(screen.getByTestId("file-input"), file);

    // 3. Configure page renders with a graph in the store.
    await waitFor(() => expect(useAppStore.getState().graph).not.toBeNull());
    expect(useAppStore.getState().graph?.nodes).toHaveLength(3);

    // 4. PPR requires ≥1 seed. Select one via the store (the seed-picker popover
    //    is Radix/cmdk and not what this data-flow test exercises), then Compute.
    useAppStore.getState().setSeeds(["A"]);
    const computeBtn = await screen.findByRole("button", { name: /compute ppr/i });
    await waitFor(() => expect(computeBtn).toBeEnabled());
    await userEvent.click(computeBtn);

    // 5. Results land in the store and a top node appears.
    await waitFor(() => expect(useAppStore.getState().result).not.toBeNull());
    const result = useAppStore.getState().result!;
    let sum = 0;
    for (const v of result.ranks.values()) sum += v;
    expect(sum).toBeCloseTo(1, 6);

    // 6. Export to CSV.
    const csvBtn = await screen.findByTestId("export-csv");
    await userEvent.click(csvBtn);
    expect(saveAsSpy).toHaveBeenCalled();
    const blob = saveAsSpy.mock.calls[0][0] as Blob;
    const text = await blob.text();
    expect(text.startsWith("rank,nodeId,score,in_degree,out_degree")).toBe(true);
    expect(text.split("\n").filter((line) => line).length).toBe(1 + 3); // header + 3 nodes
  });
});
