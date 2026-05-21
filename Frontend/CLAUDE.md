# PPR Analyzer — agent guide

Self-contained React SPA for Personalized PageRank analysis. Client-side **mock** computation today; a single-file swap connects it to the GPU backend in [`../Backend/graph_link/`](../Backend/graph_link/).

## What this app does

User flow: `Upload` → `Configure` → `Results`. The user picks a CSV format (edge list / COO / adjacency), drops a file, tunes params (α, maxIter, tolerance, topX, seed nodes), and explores PPR results as cards, a table, charts, and a node-link graph.

State management is **Zustand**, kept in a single store at [`src/store/useAppStore.ts`](src/store/useAppStore.ts):

```ts
{ format, graph, params, result, runStatus, error }
```

## Adapter swap-point — wiring the real CUDA backend

The whole point of the abstraction. The UI calls `algorithm.run(graph, params)` and does not care whether the result comes from JS or a remote GPU.

**File:** [`src/lib/ppr/adapter.ts`](src/lib/ppr/adapter.ts)

To plug in the real backend later:

1. Add a FastAPI endpoint in `../Backend/` that wraps `graph_link.run_personalized_pagerank` and returns the same JSON shape as `PPRResult` ([`src/lib/ppr/types.ts`](src/lib/ppr/types.ts)).
2. In `adapter.ts`, uncomment the `httpAdapter` block and switch the exported `algorithm` to point at it.
3. Make sure the FastAPI response converts `Map<NodeId, number>` to plain objects (and the client back into Maps — adjust the JSON parsing accordingly).

The component layer never sees this change. Tests for the mock continue to live under `src/test/ppr/`; add separate integration tests against a real backend if needed.

## File map (where to look for what)

| Concern                          | Path                                                      |
|----------------------------------|-----------------------------------------------------------|
| CSV parsers (per format)         | [`src/lib/csv/parseEdgeList.ts`](src/lib/csv/parseEdgeList.ts), [`parseCOO.ts`](src/lib/csv/parseCOO.ts), [`parseAdjacency.ts`](src/lib/csv/parseAdjacency.ts) |
| CSV dispatch entry-point         | [`src/lib/csv/index.ts`](src/lib/csv/index.ts) — `parseGraphCSV(input, format)` |
| PPR algorithm (mock)             | [`src/lib/ppr/computeMockPersonalizedPageRank.ts`](src/lib/ppr/computeMockPersonalizedPageRank.ts) |
| Algorithm interface + defaults   | [`src/lib/ppr/types.ts`](src/lib/ppr/types.ts) |
| Adapter (swap-point)             | [`src/lib/ppr/adapter.ts`](src/lib/ppr/adapter.ts) |
| Export (CSV / JSON)              | [`src/lib/export/toCSV.ts`](src/lib/export/toCSV.ts), [`toJSON.ts`](src/lib/export/toJSON.ts) |
| Global store                     | [`src/store/useAppStore.ts`](src/store/useAppStore.ts) |
| Routing + providers              | [`src/App.tsx`](src/App.tsx) |
| Shell, theme toggle              | [`src/components/layout/`](src/components/layout/) |
| Upload UI                        | [`src/components/upload/`](src/components/upload/) |
| Config UI                        | [`src/components/configure/`](src/components/configure/) |
| Results UI                       | [`src/components/results/`](src/components/results/) |
| Pages                            | [`src/pages/`](src/pages/) |

## Design system conventions

- **Style:** Data-Dense Dashboard (analytics palette: primary blue `#1E40AF`, accent amber `#D97706`, light slate background).
- **Fonts:** `Fira Sans` for body, `Fira Code` for numerics, node IDs, code. Loaded via `<link>` in [`index.html`](index.html).
- **Numeric columns:** apply `font-mono tabular-nums` (or the `.tabular-nums` utility) so the column doesn't jitter when ranks update.
- **Icons:** Only `lucide-react` SVG icons. Never emoji.
- **Primary CTA:** Each page has one — the "Compute PPR" button uses `variant="accent"` (amber). Other actions are `default` (blue), `outline`, or `ghost`.
- **Tokens:** All colors live in [`src/styles/globals.css`](src/styles/globals.css) as HSL CSS variables; light + dark variants. Tailwind aliases them in [`tailwind.config.ts`](tailwind.config.ts).
- **Reduced motion:** Honored globally by a `@media (prefers-reduced-motion: reduce)` block at the bottom of `globals.css`.

## Testing

`Vitest` + `@testing-library/react` + `happy-dom`. Run `npm run test`.

| Suite                                     | Path                                                                                  |
|-------------------------------------------|---------------------------------------------------------------------------------------|
| CSV parsers                               | [`src/test/csv/`](src/test/csv/)                                                      |
| PPR algorithm + degrees                   | [`src/test/ppr/`](src/test/ppr/)                                                      |
| Component tests (form controls, dropzone) | [`src/test/components/`](src/test/components/)                                        |
| Full upload → compute → export            | [`src/test/integration/upload-compute-export.test.tsx`](src/test/integration/upload-compute-export.test.tsx) |

**What's intentionally not tested:** React Flow internals, Recharts internals, shadcn primitives — we trust the upstream libs. The integration test mocks `@xyflow/react` (canvas) and `file-saver` (download).

## Common changes — recipes

### Add a new chart

1. Create `src/components/results/charts/MyChart.tsx`. Pull data via `useAppStore((s) => s.result)`. Use the design-system colors (`hsl(var(--primary))` etc.).
2. Mount it in the Charts tab grid in [`src/pages/ResultsPage.tsx`](src/pages/ResultsPage.tsx).

### Add a new CSV format

1. Add a parser at `src/lib/csv/parseMyFormat.ts` returning a `Graph`.
2. Add a `CSVFormat` variant in [`src/lib/ppr/types.ts`](src/lib/ppr/types.ts).
3. Dispatch from [`src/lib/csv/index.ts`](src/lib/csv/index.ts).
4. Add a card in [`src/components/upload/FormatPicker.tsx`](src/components/upload/FormatPicker.tsx) with a 4-row example.

### Add a new param

1. Add to `PPRParams` in [`src/lib/ppr/types.ts`](src/lib/ppr/types.ts) (and `DEFAULT_PARAMS`).
2. Add a control to [`src/components/configure/ConfigPanel.tsx`](src/components/configure/ConfigPanel.tsx) wired to `useAppStore.setParams`.
3. Consume it in `computeMockPersonalizedPageRank.ts` and the future backend adapter.

## Pitfalls / gotchas

- **`computeMockPersonalizedPageRank` uses sparse maps** to avoid an N×N allocation. Don't switch to dense unless N is guaranteed small.
- **Dangling nodes** (out-degree 0) are handled by redistributing their mass uniformly. If you change this, update the disconnected-graph test.
- **The first cell of an adjacency CSV** may be empty (`,A,B,C`) — the parser allows both empty and a corner label; don't add a check that requires one or the other.
- **React Flow** needs a fixed-height container (otherwise it renders empty). Don't remove the `h-[560px]` on the card content in `NetworkGraph.tsx`.
- **The dropzone** is intentionally disabled until a format is picked — don't gate this on a "has any uploaded file" check; the format must come first so error messages are meaningful.

## Backend coexistence

`Frontend/test_pipeline.py` is a Python test of the parent project's pipeline. It is unrelated to this React app and should stay where it is — leave it alone.
