# PPR Analyzer

Client-side React SPA for analyzing graphs with Personalized PageRank (PPR). Upload a graph CSV, tune the parameters, and explore the result as ranked cards, sortable tables, distribution charts, and an interactive node-link diagram.

The PPR algorithm runs in the browser via a JS power-iteration mock. The mock lives behind an `Algorithm` adapter (see [`src/lib/ppr/adapter.ts`](src/lib/ppr/adapter.ts)) so the real CUDA backend in [`../Backend/graph_link/`](../Backend/graph_link/) can replace it without UI changes.

## Quick start

```bash
cd page_rank/Frontend
npm install
npm run dev          # http://localhost:5173
```

## Scripts

| Command              | What it does                                  |
|----------------------|-----------------------------------------------|
| `npm run dev`        | Vite dev server with HMR                      |
| `npm run build`      | Type-check (tsc) and bundle to `dist/`        |
| `npm run preview`    | Serve the production build                    |
| `npm run test`       | Vitest, run once                              |
| `npm run test:watch` | Vitest in watch mode                          |
| `npm run test:coverage` | Vitest with v8 coverage                    |
| `npm run lint`       | ESLint (zero-warning policy)                  |
| `npm run typecheck`  | `tsc --noEmit`                                |

## Supported CSV formats

You **explicitly pick** one of three formats on the upload page:

1. **Edge list** — `source,target[,weight]`. Default weight 1.0. Aliases: `from`/`to`/`src`/`dst`/`w`.
2. **COO triplets** — `row_idx,col_idx,value`. 0-based integer indices; nodes named `n0`, `n1`, ….
3. **Adjacency matrix** — first row and column are node labels; `,A,B,C` style.

There is no auto-detect — picking the format up front avoids ambiguity and gives clear error messages when a file does not match.

Sample graphs are available on the upload page (Zachary's karate club, a tiny COO, a tiny adjacency).

## Parameters

| Field       | Default | Range                | Notes                                       |
|-------------|--------:|----------------------|---------------------------------------------|
| α (damping) |    0.85 | [0, 1]               | Higher = more weight on graph structure     |
| Max iter    |     100 | [1, 10 000]          | Power-iteration cap                         |
| Tolerance   |    1e-6 | > 0                  | L₁ convergence threshold                    |
| Top X       |      10 | [1, N]               | Top-ranked nodes to highlight as cards      |
| Seeds       |    none | subset of node IDs   | Empty = uniform PageRank                    |

## Output

After computing, the **Results** view has four tabs:

- **Top X** — KPI cards for the top-ranked nodes.
- **Table** — sortable / filterable list of all nodes with rank, score, in-degree, out-degree.
- **Charts** — rank distribution (top 30), convergence (log L₁ residual per iter), degree histogram.
- **Network** — interactive React Flow node-link diagram, node size ∝ √(PPR score).

Export the result as CSV (`rank,nodeId,score,in_degree,out_degree`) or JSON (full payload with params, run metadata, and convergence history).

## Adding a new sample graph

1. Drop the CSV in `public/samples/`.
2. Add an entry to the `SAMPLES` array in [`src/components/upload/SampleGraphPicker.tsx`](src/components/upload/SampleGraphPicker.tsx).

That's it — sample buttons auto-select the matching format on click.

## Wiring the real CUDA backend later

See [`CLAUDE.md`](CLAUDE.md) → "Adapter swap-point".

## Project layout

```
src/
  lib/
    csv/                # CSV parsers (edge, COO, adjacency) + dispatcher
    ppr/                # Algorithm types, mock PPR, degrees, adapter
    export/             # CSV / JSON serializers
  store/                # Zustand global store
  components/
    ui/                 # shadcn/ui primitives
    layout/             # AppShell, ThemeToggle
    upload/             # FormatPicker, FileDropzone, SampleGraphPicker
    configure/          # ConfigPanel, AlphaSlider, SeedNodePicker, …
    results/            # TopXCards, RanksTable, charts/, NetworkGraph, ExportMenu
  pages/                # UploadPage, ConfigurePage, ResultsPage
  styles/globals.css    # Tailwind layers + design tokens
  test/                 # Unit, component, integration tests
public/samples/         # Bundled demo CSVs
```

## Tech stack

Vite · React 18 · TypeScript (strict) · Tailwind CSS · shadcn/ui (Radix) · Zustand · React Router 6 · PapaParse · Recharts · React Flow · file-saver · next-themes · Vitest · @testing-library/react · happy-dom.
