import type { PPRAlgorithm } from "./types";
import { computeMockPersonalizedPageRank } from "./computeMockPersonalizedPageRank";

export const mockAdapter: PPRAlgorithm = {
  run: (g, p) =>
    new Promise((resolve) => {
      // Yield to the event loop so the UI can render the spinner before the heavy work.
      setTimeout(() => resolve(computeMockPersonalizedPageRank(g, p)), 0);
    }),
};

// Future drop-in for the real CUDA backend:
//
// export const httpAdapter: PPRAlgorithm = {
//   async run(graph, params) {
//     const res = await fetch("/api/ppr", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ graph, params }),
//     });
//     if (!res.ok) throw new Error(`PPR API error ${res.status}`);
//     return await res.json();
//   },
// };

export const algorithm: PPRAlgorithm = mockAdapter;
