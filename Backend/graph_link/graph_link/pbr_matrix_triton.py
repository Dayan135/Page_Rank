import torch
import triton
import triton.language as tl
from .pbr_registry import get_pbr_gpu_meta

# Define the hardware configurations Triton will "race" against each other.
# We vary BLOCK_W (features processed at once), num_warps (threads per block), 
# and num_stages (SRAM prefetching depth - Ampere loves 4 or 5 stages).
def get_autotune_configs():
    return [
        triton.Config({'BLOCK_W': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_W': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_W': 128}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_W': 256}, num_warps=8, num_stages=3),
    ]

@triton.autotune(
    configs=get_autotune_configs(),
    key=['num_features'], # Retune if the user passes a different feature width
    reset_to_zero=['y_ptr'] # We reset the output pointer to zero for each run to ensure a fair comparison, especially since we're using atomic adds
)
@triton.jit
def pbr_matmul_kernel(
    codes_ptr, coords_ptr, offsets_ptr, data_ptr, 
    x_ptr, y_ptr,
    stride_xb, stride_xh, stride_xw,
    stride_yb, stride_yh, stride_yw,
    rows, cols,
    num_features,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
    BLOCK_W: tl.constexpr # Now managed by the autotuner!
):
    # IDs
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    feat_tile_idx = tl.program_id(2)

    # Load Metadata
    base_coord_ptr = coords_ptr + block_idx * 2
    row_origin = tl.load(base_coord_ptr)
    col_origin = tl.load(base_coord_ptr + 1)
    code = tl.load(codes_ptr + block_idx)
    data_start = tl.load(offsets_ptr + block_idx)

    # Setup Feature range
    offs_w = feat_tile_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < num_features

    dtype = x_ptr.dtype.element_ty

    nnz_counter = 0
    for r in range(BLOCK_ROWS):
        global_row = row_origin + r
        if global_row < rows:
            row_res = tl.zeros([BLOCK_W], dtype=dtype)
            for c in range(BLOCK_COLS):
                if (code >> (r * BLOCK_COLS + c)) & 1:
                    global_col = col_origin + c
                    if global_col < cols: 
                        # Load weight
                        val = tl.load(data_ptr + data_start + nnz_counter)
                        
                        # Load X
                        x_row_ptr = x_ptr + batch_idx * stride_xb + global_col * stride_xh + offs_w * stride_xw
                        x_val = tl.load(x_row_ptr, mask=mask_w, other=0.0)
                        
                        row_res += val * x_val
                        nnz_counter += 1
            
            # Write back
            y_row_ptr = y_ptr + batch_idx * stride_yb + global_row * stride_yh + offs_w * stride_yw
            # tl.add(y_row_ptr, row_res, mask=mask_w)  # Using atomic add to handle potential write conflicts across threads
            tl.atomic_add(y_row_ptr, row_res, mask=mask_w)

def pbr_batched_matmul_triton_gpu(pbr_mat, x: torch.Tensor, y: torch.Tensor, batch_size: int, width: int):
    meta = get_pbr_gpu_meta(pbr_mat)
    if meta is None:
        print(f"Warning: PBR metadata not available for {pbr_mat}")
        #throw exception (?)
        # pbr_mat.to(x.device)
        # meta = get_pbr_gpu_meta(pbr_mat)

    if x.dim() == 2:
        stride_xb, stride_xh, stride_xw = 0, x.stride(0), x.stride(1)
        stride_yb, stride_yh, stride_yw = 0, y.stride(0), y.stride(1)
        actual_batch = 1
    else:
        stride_xb, stride_xh, stride_xw = x.stride(0), x.stride(1), x.stride(2)
        stride_yb, stride_yh, stride_yw = y.stride(0), y.stride(1), y.stride(2)
        actual_batch = batch_size

    num_blocks = pbr_mat.accounted_blocks()
    
    # CRITICAL CHANGE: The grid is now a lambda function.
    # Because BLOCK_W is chosen dynamically by the autotuner, 
    # Triton passes the winning META configuration into this lambda so we can calculate the grid size.
    grid = lambda META: (actual_batch, num_blocks, triton.cdiv(width, META['BLOCK_W']))

    pbr_matmul_kernel[grid](
        meta['codes'], meta['coords'], meta['offsets'], meta['data'],
        x, y,
        stride_xb, stride_xh, stride_xw,
        stride_yb, stride_yh, stride_yw,
        pbr_mat.rows, pbr_mat.cols,
        width,
        BLOCK_ROWS=pbr_mat.block_rows,
        BLOCK_COLS=pbr_mat.block_cols
        # Notice we removed BLOCK_W here. The autotuner injects it automatically!
    )