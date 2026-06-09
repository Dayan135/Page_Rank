"""Run GPU Personalized PageRank on a WikiLinkGraphs edge list.

Dataset: https://zenodo.org/records/2539424 — files named
`{lang}wiki.wikilink_graph.{YYYY-MM-01}.csv.gz`, TAB-separated with header
`page_id_from  page_title_from  page_id_to  page_title_to` (one directed
internal-link edge per row).

This bypasses the web UI (which serializes the whole graph to JSON) and drives
graph_link directly, so it scales to millions of nodes. Seeds default to the
highest out-degree articles.

    python Tests/wiki_ppr.py --graph itwiki.wikilink_graph.2018-03-01.csv.gz

Requires a CUDA GPU and the built graph_link extension.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import graph_link


def load_edges(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read the (gz, tab-separated) edge list → (from_idx, to_idx, node_ids, N).

    Only the two id columns are read for the graph; titles are resolved later
    for just the reported nodes, so memory stays bounded on large editions.
    """
    t0 = time.time()
    froms, tos = [], []
    rows = 0
    for chunk in pd.read_csv(
        path, sep="\t", compression="gzip",
        usecols=["page_id_from", "page_id_to"],
        dtype={"page_id_from": np.int64, "page_id_to": np.int64},
        chunksize=2_000_000,
    ):
        froms.append(chunk["page_id_from"].to_numpy())
        tos.append(chunk["page_id_to"].to_numpy())
        rows += len(chunk)
    from_ids = np.concatenate(froms)
    to_ids = np.concatenate(tos)
    del froms, tos

    # Compact the sparse page_ids to a dense 0..N-1 index space.
    node_ids = np.unique(np.concatenate([from_ids, to_ids]))
    from_idx = np.searchsorted(node_ids, from_ids)
    to_idx = np.searchsorted(node_ids, to_ids)
    print(f"[load] {rows:,} edges, {len(node_ids):,} nodes in {time.time()-t0:.1f}s")
    return from_idx, to_idx, node_ids, len(node_ids)


def resolve_titles(path: str, wanted_ids: set[int]) -> dict[int, str]:
    """Second targeted pass: map only the requested page_ids → titles."""
    found: dict[int, str] = {}
    for cols in (("page_id_from", "page_title_from"), ("page_id_to", "page_title_to")):
        for chunk in pd.read_csv(
            path, sep="\t", compression="gzip", usecols=list(cols),
            dtype={cols[0]: np.int64, cols[1]: "string"}, chunksize=2_000_000,
        ):
            ids = chunk[cols[0]].to_numpy()
            mask = np.isin(ids, list(wanted_ids))
            if mask.any():
                titles = chunk[cols[1]].to_numpy()
                for i, t in zip(ids[mask], titles[mask]):
                    found.setdefault(int(i), str(t))
            if len(found) >= len(wanted_ids):
                return found
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True, help="path to {lang}wiki.wikilink_graph.*.csv.gz")
    ap.add_argument("--num-seeds", type=int, default=4, help="top out-degree nodes used as PPR seeds")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--topk", type=int, default=20, help="how many top-ranked articles to print")
    ap.add_argument("--block", type=int, default=8, help="PBR block size")
    ap.add_argument("--min-nnz", type=int, default=1, help="PBR min nnz per block")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required.")
    dev = torch.cuda.get_device_name(0)

    from_idx, to_idx, node_ids, N = load_edges(args.graph)

    # Transition matrix: A[target, source] = 1, so column-normalizing by each
    # source's out-degree yields a column-stochastic A (mass flows source->target).
    t0 = time.time()
    data = np.ones(len(from_idx), dtype=np.float32)
    A = sp.csr_matrix((data, (to_idx, from_idx)), shape=(N, N))
    A.sum_duplicates()
    nnz_raw = A.nnz

    # Scale-safe column-stochastic normalization. We do NOT use
    # graph_link.normalize_transition_matrix here: its dangling-node handling
    # builds a DENSE N x n_dangling uniform-restart block (279 GiB on itwiki),
    # which only works for tiny graphs. Instead divide each column by its
    # out-degree and give dangling nodes (zero out-degree) a self-loop, so the
    # matrix stays column-stochastic and mass is conserved -- all O(nnz).
    Acsc = A.tocsc()
    col_sums = np.asarray(Acsc.sum(axis=0), dtype=np.float64).ravel()
    out_deg = col_sums  # pre-normalization out-degree, for seed selection
    inv = np.zeros_like(col_sums)
    np.divide(1.0, col_sums, out=inv, where=col_sums > 0)
    A_norm = Acsc.multiply(inv).tocsr().astype(np.float32)
    dangling = np.where(col_sums == 0)[0]
    if dangling.size:
        loops = sp.csr_matrix(
            (np.ones(dangling.size, np.float32), (dangling, dangling)), shape=(N, N)
        )
        A_norm = (A_norm + loops).tocsr()
    A_norm.sort_indices()
    print(f"[build] CSR + normalize in {time.time()-t0:.1f}s  "
          f"(nnz={nnz_raw:,}, dangling={dangling.size:,})")

    seed_idx = np.argsort(out_deg)[::-1][: args.num_seeds].astype(np.int64)

    t0 = time.time()
    pbr = graph_link.csr_to_pbr(
        A_norm, block_rows=args.block, block_cols=args.block, min_nnz_per_block=args.min_nnz
    ).to("cuda")
    torch.cuda.synchronize()
    print(f"[pbr] csr_to_pbr + upload in {time.time()-t0:.1f}s  "
          f"(blocks={pbr.accounted_blocks():,}, remainder_nnz={pbr.remainder_nnz():,})")

    sources = torch.tensor(seed_idx, dtype=torch.int32, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    X, iters, converged, history = graph_link.run_personalized_pagerank(
        pbr, sources, alpha=args.alpha, max_iterations=args.max_iter,
        tolerance=args.tol, return_history=True,
    )
    torch.cuda.synchronize()
    ppr_s = time.time() - t0

    # Uniform personalization over the seed set = mean of the per-seed columns.
    r = X.mean(dim=1)
    r = r / r.sum()
    top_scores, top_idx = torch.topk(r, min(args.topk, N))
    top_idx = top_idx.cpu().numpy()
    top_scores = top_scores.cpu().numpy()

    wanted = {int(node_ids[i]) for i in top_idx} | {int(node_ids[i]) for i in seed_idx}
    titles = resolve_titles(args.graph, wanted)

    def title(i: int) -> str:
        return titles.get(int(node_ids[i]), f"<id {int(node_ids[i])}>")

    print("\n=== PPR run ===")
    print(f"device         : {dev}")
    print(f"nodes / edges  : {N:,} / {A.nnz:,}")
    print(f"alpha          : {args.alpha}")
    print(f"iterations     : {iters}  (converged={converged})")
    print(f"ppr wall time  : {ppr_s*1000:.1f} ms  ({ppr_s/max(iters,1)*1000:.2f} ms/iter)")
    print(f"peak GPU mem   : {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"final L1 error : {history[-1]:.2e}")
    print(f"\nseeds (top out-degree): " + ", ".join(title(i) for i in seed_idx))
    print(f"\ntop {len(top_idx)} articles by personalized rank:")
    for rank, (i, s) in enumerate(zip(top_idx, top_scores), 1):
        print(f"  {rank:2d}. {s:.6e}  {title(i)}")


if __name__ == "__main__":
    main()
