import torch
import graph_link
import scipy.sparse as sp
import numpy as np

# 1. Identity Matrix
A = sp.eye(8, format='csr').astype(np.float32)
# 2. Input of all ones
X = torch.ones((8, 1), device='cuda', dtype=torch.float32)
# 3. Convert with standard 8x8 blocks
pbr = graph_link.csr_to_pbr(A, block_rows=8, block_cols=8).to('cuda')
# 4. Run
Y = graph_link.pbr_matmul(pbr, X)

print("Output Y (Should be all 1.0):")
print(Y)