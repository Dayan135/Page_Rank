import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_benchmark_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    records = []
    for b in data['benchmarks']:
        name = b['name']

        try:
            # Extract the parameter string inside brackets []
            if '[' not in name: continue
            params_str = name.split('[')[1].split(']')[0]
            parts = params_str.split('-')
            
            algo = None
            if "torch_ppr" in name:
                feat_val = int(parts[0])
                size_val = int(parts[1])
                algo = "Torch cuSPARSE (GPU)"
            elif "pbr_engine" in name:
                bs = parts[0]
                feat_val = int(parts[1])
                size_val = int(parts[2])
                algo = f"PBR Engine ({bs}x{bs})"

            if algo:
                # Extract every individual run for the seaborn error bars
                for val in b['stats']['data']:
                    records.append({
                        "Algorithm": algo,
                        "Features (Batch Size)": feat_val,
                        "Graph Size (Nodes)": size_val,
                        "Execution Time (ms)": val * 1e3, # Convert seconds to ms
                        "Workload": f"Batch: {feat_val}"
                    })

        except Exception as e:
            print(f"Skipping malformed test name: {name} | Error: {e}")

    df = pd.DataFrame(records)
    
    if not df.empty:
        df = df.sort_values(by=["Graph Size (Nodes)", "Features (Batch Size)"])
        
    return df

def generate_plots(df):
    sns.set_theme(style="whitegrid")
    
    graph_sizes = sorted(df["Graph Size (Nodes)"].unique())
    num_sizes = len(graph_sizes)
    
    fig, axes = plt.subplots(1, num_sizes, figsize=(9 * num_sizes, 7), sharey=True)
    
    if num_sizes == 1:
        axes = [axes]
    
    # --- THE FIX IS HERE ---
    # We now tell Seaborn to look for 2x2 and 4x4 block sizes!
    algo_order = [
        "Torch cuSPARSE (GPU)", 
        "PBR Engine (2x2)",
        "PBR Engine (4x4)",
        "PBR Engine (8x8)"
    ]
    
    for ax, size in zip(axes, graph_sizes):
        subset = df[df["Graph Size (Nodes)"] == size]
        
        if subset.empty:
            continue

        sns.barplot(
            data=subset, 
            x="Workload", 
            y="Execution Time (ms)", 
            hue="Algorithm", 
            hue_order=[a for a in algo_order if a in subset["Algorithm"].unique()],
            ax=ax,
            palette="Set1",
            errorbar="sd",
            capsize=0.05
        )
        
        ax.set_title(f"Graph Size: {size} Nodes", fontsize=14, fontweight="bold")
        ax.set_ylabel("Execution Time (ms)" if ax == axes[0] else "")
        ax.set_xlabel("Concurrent Users (Batch Size)")
        
        if ax == axes[0]:
            ax.legend(title="Implementation", loc='upper left')
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    plt.suptitle("Batched Personalized PageRank: cuSPARSE vs PBR Engine\n(Lower is Better)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    output_filename = "ppr_benchmark_results.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    json_file = "output.json"
    print(f"Parsing {json_file}...")
    try:
        df = parse_benchmark_json(json_file)
        if not df.empty:
            generate_plots(df)
        else:
            print("DataFrame is empty. Check if your JSON file has the expected format.")
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Run pytest with --benchmark-json=output.json first.")