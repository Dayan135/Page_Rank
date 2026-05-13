"""
Standalone SpMM benchmark for graph_link (PBR format).

Compares:
  - PyTorch cuSPARSE  (torch.sparse.mm)
  - PBR CUDA          (graph_link.pbr_matmul, custom C++ + CUDA kernel)

Usage:
  python Tests/benchmark_spdmm.py [--wandb]

Pass --wandb to log results to Weights & Biases (project "graph-link-spdmm").
"""

import gc
import sys
import inspect
import argparse
import numpy as np
import torch
import scipy.sparse as sp
from itertools import product
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

# ---------- local imports ----------
sys.path.insert(0, "Tests")
from matrices import (
    generate_scattered_block_matrix,
    generate_random_sparse,
    generate_diag_blocks_plus_noise,
)

# sys.path.insert(0, "Backend/graph_link")
import graph_link
from graph_link import (
    csr_to_pbr,
    pbr_matmul,
    pbr_batched_matmul_triton_gpu,
    pbr_batched_matmul_cuda,
    get_pbr_gpu_meta,
    clear_pbr_gpu_meta,
)


# =====================================================================
# 0. OPTIONAL W&B
# =====================================================================

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wandb", action="store_true", help="Log results to W&B")
    return p.parse_args()


# =====================================================================
# 1. DATA STRUCTURES
# =====================================================================

@dataclass
class TaskConfig:
    name: str
    sparse_mat_generator_func: Callable
    params: Dict[str, List[Any]]


@dataclass
class TaskMeta:
    profile_name: str
    size: int
    features: int
    dtype: torch.dtype


@dataclass
class TaskTensors:
    scipy_mat: sp.csr_matrix
    sparse_torch: torch.Tensor   # torch sparse CSR on CUDA
    dense_torch: torch.Tensor    # shape (1, size, features) on CUDA


@dataclass
class BenchmarkTask:
    meta: TaskMeta
    tensors: TaskTensors


# =====================================================================
# 2. TASK GENERATOR
# =====================================================================

def _scipy_to_torch_csr(mat: sp.csr_matrix, dtype: torch.dtype) -> torch.Tensor:
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    mat = mat.astype(np_dtype)
    return torch.sparse_csr_tensor(
        torch.tensor(mat.indptr,  dtype=torch.int32),
        torch.tensor(mat.indices, dtype=torch.int32),
        torch.tensor(mat.data,    dtype=dtype),
        size=mat.shape,
        device="cuda",
    )


def generic_task_generator(
    task_configs: List[TaskConfig],
    features: List[int],
    dtypes: List[torch.dtype],
):
    for config in task_configs:
        sig = inspect.signature(config.sparse_mat_generator_func)
        expected_args = set(sig.parameters.keys())

        keys   = list(config.params.keys())
        values = [[v] if not isinstance(v, list) else v for v in config.params.values()]

        for combo in product(*values):
            current_params = dict(zip(keys, combo))
            gen_kwargs     = {k: v for k, v in current_params.items() if k in expected_args}

            for feat, dtype in product(features, dtypes):
                size = current_params.get("system_size")
                if size is None:
                    raise ValueError(f"TaskConfig '{config.name}' must define 'system_size'.")

                print(f"\n[GENERATING] {config.name} | Params: {current_params} | Feat: {feat} | DType: {dtype}")

                scipy_mat    = config.sparse_mat_generator_func(**gen_kwargs)
                np_dtype     = np.float64 if dtype == torch.float64 else np.float32
                scipy_mat    = scipy_mat.astype(np_dtype)
                sparse_torch = _scipy_to_torch_csr(scipy_mat, dtype)
                dense_torch  = torch.randn((1, size, feat), dtype=dtype, device="cuda")

                yield BenchmarkTask(
                    meta=TaskMeta(
                        profile_name=config.name,
                        size=size,
                        features=feat,
                        dtype=dtype,
                    ),
                    tensors=TaskTensors(
                        scipy_mat=scipy_mat,
                        sparse_torch=sparse_torch,
                        dense_torch=dense_torch,
                    ),
                )


# =====================================================================
# 3. TIMING UTILITY
# =====================================================================

def measure_kernel_time(func, *args, warmup: int = 5, iters: int = 30) -> float:
    """Accurate GPU wall-clock measurement using paired CUDA Events."""
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        func(*args)
        ends[i].record()

    torch.cuda.synchronize()
    return float(np.mean([s.elapsed_time(e) for s, e in zip(starts, ends)]))


# =====================================================================
# 4. EVALUATOR
# =====================================================================

def evaluate_task(task: BenchmarkTask, pbr_configs: list, use_wandb: bool):
    meta    = task.meta
    tensors = task.tensors

    sparse_torch = tensors.sparse_torch
    dense_2d     = tensors.dense_torch[0]   # (size, features) — 2-D slice
    scipy_mat    = tensors.scipy_mat

    # ------------------------------------------------------------------
    def safe_benchmark_and_log(algo_name: str, family_name: str, func, *args):
        run_config = {
            "Algorithm":      algo_name,
            "Family":         family_name,
            "Matrix Profile": meta.profile_name,
            "System Size":    meta.size,
            "Features":       meta.features,
            "DType":          str(meta.dtype),
        }

        run = None
        if use_wandb:
            import wandb
            run = wandb.init(
                project="graph-link-spdmm",
                group=meta.profile_name,
                name=f"{algo_name}_{meta.size}_{meta.features}",
                config=run_config,
                reinit=True,
            )

        try:
            func(*args)                   # single warmup
            torch.cuda.synchronize()
            time_ms = measure_kernel_time(func, *args, warmup=0, iters=30)
            print(f"  -> {algo_name}: {time_ms:.3f} ms")
            if run:
                run.log({"Time (ms)": time_ms})

        except torch.OutOfMemoryError:
            print(f"  -> {algo_name}: OOM KILLED")
            if run:
                run.log({"Time (ms)": -1.0, "Error": "OOM"})
            torch.cuda.empty_cache()

        except Exception as e:
            msg = str(e).splitlines()[0]
            print(f"  -> {algo_name}: FAILED — {msg}")
            if run:
                run.log({"Time (ms)": -2.0, "Error": msg})

        finally:
            if run:
                run.finish()

    # ------------------------------------------------------------------
    # Baselines
    safe_benchmark_and_log(
        "PyTorch cuSPARSE", "Baseline",
        torch.sparse.mm, sparse_torch, dense_2d,
    )

    # ------------------------------------------------------------------
    # PBR variants
    for cfg in pbr_configs:
        bs      = cfg["block_size"]
        min_nnz = cfg["min_nnz"]

        pbr_obj = csr_to_pbr(scipy_mat, block_rows=bs, block_cols=bs, min_nnz_per_block=min_nnz)
        if pbr_obj.accounted_blocks() == 0 and pbr_obj.remainder_nnz() == 0:
            print(f"  -> PBR_CUDA_{bs}x{bs}_{min_nnz}: skipped (empty)")
            continue

        pbr_gpu = pbr_obj.to("cuda")

        # --- CUDA kernel ---
        safe_benchmark_and_log(
            f"PBR_CUDA_{bs}x{bs}_{min_nnz}", f"PBR CUDA {bs}x{bs}",
            pbr_matmul, pbr_gpu, dense_2d,
        )

        clear_pbr_gpu_meta(pbr_gpu)
        del pbr_gpu


# =====================================================================
# 5. MAIN
# =====================================================================

def main():
    args = _parse_args()

    system_sizes  = [65536, 262144]
    features_list = [128, 512]
    dtypes        = [torch.float32]

    pbr_configs = [
        {"block_size": 2, "min_nnz": 1},
        {"block_size": 2, "min_nnz": 4},
        {"block_size": 4, "min_nnz": 4},
        {"block_size": 4, "min_nnz": 8},
        {"block_size": 8, "min_nnz": 4},
        {"block_size": 8, "min_nnz": 8},
    ]

    task_configs = [
        TaskConfig(
            name="Scattered Blocks 1×8 (0.05% density)",
            sparse_mat_generator_func=generate_scattered_block_matrix,
            params={
                "system_size": system_sizes,
                "block_size":  [8],
                "profiles":    [[(1, 8, 0.0005)]],
            },
        ),
        TaskConfig(
            name="Scattered Blocks 3×3 (0.05% density)",
            sparse_mat_generator_func=generate_scattered_block_matrix,
            params={
                "system_size": system_sizes,
                "block_size":  [8],
                "profiles":    [[(3, 3, 0.0005)]],
            },
        ),
        TaskConfig(
            name="Diag blocks (50% density) + noise (0.001%)",
            sparse_mat_generator_func=generate_diag_blocks_plus_noise,
            params={
                "system_size":    system_sizes,
                "block_size":     [64],
                "diag_density":   [0.5],
                "off_diag_density": [0.00001],
            },
        ),
        TaskConfig(
            name="Diag blocks (20% density) + noise (0.001%)",
            sparse_mat_generator_func=generate_diag_blocks_plus_noise,
            params={
                "system_size":    system_sizes,
                "block_size":     [64],
                "diag_density":   [0.2],
                "off_diag_density": [0.00001],
            },
        ),
        TaskConfig(
            name="Random Uniform (65k 0.01% density)",
            sparse_mat_generator_func=generate_random_sparse,
            params={
                "system_size": [65536],
                "density":     [0.0001],
            },
        ),
        TaskConfig(
            name="Random Uniform (262k 0.01% density)",
            sparse_mat_generator_func=generate_random_sparse,
            params={
                "system_size": [262144],
                "density":     [0.0001],
            },
        ),
    ]

    task_gen = generic_task_generator(task_configs, features_list, dtypes)

    for task in task_gen:
        evaluate_task(task, pbr_configs, use_wandb=args.wandb)
        del task
        gc.collect()
        torch.cuda.empty_cache()

    print("\nBenchmark complete.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
