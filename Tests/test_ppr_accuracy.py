import pytest
import torch
import numpy as np
import scipy.sparse as sp
import networkx as nx
import graph_link

def create_directed_graph_with_sinkholes(N=50, p=0.1):
    """
    Creates a random directed graph and forces at least one node to be a sinkhole 
    (no outgoing edges) to rigorously test the missing mass teleportation logic.
    """
    G = nx.erdos_renyi_graph(N, p, directed=True, seed=42)
    
    # Force Node 0 to be a sinkhole
    G.remove_edges_from(list(G.out_edges(0)))
    
    # Convert NetworkX to Transition Matrix (CSR)
    nodes = max(G.nodes()) + 1
    row_idx = []
    col_idx = []
    data = []
    
    for u in range(nodes):
        out_degree = G.out_degree(u)
        if out_degree > 0:
            val = 1.0 / out_degree
            for _, v in G.out_edges(u):
                row_idx.append(v)  # Target row
                col_idx.append(u)  # Source col
                data.append(val)
                
    csr = sp.csr_matrix((data, (row_idx, col_idx)), shape=(nodes, nodes), dtype=np.float32)
    return G, csr

def test_batched_ppr_accuracy():
    """
    Tests Batched Personalized PageRank on the GPU vs 4 separate NetworkX runs.
    """
    # 1. Setup the Graph
    N = 100
    G, csr = create_directed_graph_with_sinkholes(N, p=0.05)
    
    # 2. Define our Batch of "Users" (Source Nodes)
    sources = [0, 5, 25, 99]
    num_features = len(sources)
    
    # 3. Load Matrix to GPU
    pbr_host = graph_link.csr_to_pbr(csr, block_rows=2, block_cols=2, min_nnz_per_block=1)
    pbr_device = pbr_host.to("cuda")
    
    # 4. Prepare PyTorch Input Tensor
    source_tensor = torch.tensor(sources, dtype=torch.int32, device="cuda")
    
    # 5. FIRE THE CUSTOM ENGINE
    damping = 0.85
    # Relaxed to 1e-5 to account for Float32 precision noise floor in parallel reductions
    scores_tensor, iters, converged = graph_link.run_personalized_pagerank(
        pbr_device, source_tensor, damping_factor=damping, max_iterations=200, tolerance=1e-5
    )
    
    assert converged, f"CUDA Engine failed to converge within 200 iterations!"
    
    # Bring the 2D tensor back to CPU for assertion checking
    scores_cpu = scores_tensor.cpu().numpy()
    
    # 6. Verify each column against NetworkX
    for i, src_node in enumerate(sources):
        
        # Build the NetworkX personalization dictionary
        nx_pers = {n: 0.0 for n in range(N)}
        nx_pers[src_node] = 1.0
        
        # Run NetworkX (Golden Standard)
        # We use the same tolerance and start vector to ensure a fair comparison
        nx_scores_dict = nx.pagerank(
            G, 
            alpha=damping, 
            personalization=nx_pers, 
            nstart=nx_pers, 
            tol=1e-5
        )
        
        # Convert NetworkX dict to a NumPy array
        nx_array = np.zeros(N)
        for node_id, score in nx_scores_dict.items():
            nx_array[node_id] = score
            
        # Extract the specific column from our PyTorch tensor output
        cuda_array = scores_cpu[:, i]
        
        # Check if the Total Mass sums to 1.0 (Sanity check)
        assert np.isclose(cuda_array.sum(), 1.0, atol=1e-5), f"Mass leaked! Sum is {cuda_array.sum()}"
        
        # Check against NetworkX
        l1_diff = np.abs(nx_array - cuda_array).sum()
        assert np.allclose(cuda_array, nx_array, atol=1e-4), \
            f"Accuracy Mismatch for Source Node {src_node}! Total L1 Error: {l1_diff}"
            
    print(f"\n[SUCCESS] Engine calculated {num_features} Personalized PageRanks in {iters} iterations.")