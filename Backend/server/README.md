# PPR backend service

FastAPI wrapper around `graph_link.run_personalized_pagerank`, consumed by the
React frontend (`Frontend/src/lib/ppr/adapter.ts`).

## Run (on a CUDA host)

`graph_link` must be importable (built + on `PYTHONPATH` — see
[`../graph_link/CLAUDE.md`](../graph_link/CLAUDE.md)). Then:

```bash
pip install -r requirements.txt
cd Backend/server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Check it: `curl localhost:8000/api/health` → `{"status":"ok","cuda":true,...}`.

## Endpoints

- `GET  /api/health` — liveness + CUDA device info.
- `POST /api/ppr` — body `{ graph, params }` (see `Frontend/src/lib/ppr/types.ts`);
  returns the `PPRResult` shape with `ranks` and `degrees.{in,out}` as plain JSON
  objects (the frontend rebuilds them into `Map`s).

**Personalized PageRank requires ≥1 seed** — a no-seed request returns `422`.
Personalization is uniform over the seed set (mean of the per-seed columns).

## Connecting the frontend

The Vite dev server proxies `/api` → `http://localhost:8000` by default
(override with `VITE_PPR_TARGET`). If the backend runs on a remote GPU box,
tunnel the port to your dev machine:

```bash
ssh -N -L 8000:localhost:8000 dayanb@<gpu-host>
```

For a production build, set `VITE_PPR_API` to the backend's absolute URL instead
of relying on the dev proxy.
