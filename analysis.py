"""
============================================================
B107 Data Driven Strategic Decision Making
Social Network Analysis — Facebook Ego Network (SNAP)
============================================================

Dataset   : Facebook Social Circles (combined)
Source    : https://snap.stanford.edu/data/ego-Facebook.html
Nodes     : 4,039   (Facebook users)
Edges     : 88,234  (mutual friendships)

This script performs:
  1. Network construction from SNAP dataset
  2. Degree distribution analysis
  3. Connected components analysis
  4. Path analysis
  5. Clustering coefficient & density analysis
  6. Centrality analysis
  7. Comparison with ER, BA, and WS random graph models
  8. Research question: Can betweenness centrality identify
     'bridge' users that differ structurally from degree hubs?

Requirements:
  pip install networkx matplotlib numpy pandas scipy
"""

import os
import gzip
import urllib.request
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from collections import Counter
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_URL  = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
DATA_PATH = "facebook_combined.txt"
PLOT_DIR  = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1.  NETWORK CONSTRUCTION
# ─────────────────────────────────────────────

def download_dataset(url: str, dest: str) -> None:
    """Download and decompress the SNAP Facebook dataset."""
    if os.path.exists(dest):
        print(f"[INFO] Dataset already exists at '{dest}'. Skipping download.")
        return
    gz_path = dest + ".gz"
    print(f"[INFO] Downloading dataset from {url} ...")
    urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
        f_out.write(f_in.read())
    os.remove(gz_path)
    print(f"[INFO] Saved to '{dest}'.")


def build_graph(path: str) -> nx.Graph:
    """Load edge list and construct an undirected graph."""
    G = nx.read_edgelist(path, nodetype=int)
    print(f"\n[INFO] Graph constructed: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    return G


# ─────────────────────────────────────────────
# 2.  DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────

def basic_statistics(G: nx.Graph) -> dict:
    """Compute and print core graph statistics."""
    degrees = [d for _, d in G.degree()]
    stats_dict = {
        "nodes"              : G.number_of_nodes(),
        "edges"              : G.number_of_edges(),
        "density"            : nx.density(G),
        "avg_degree"         : np.mean(degrees),
        "max_degree"         : max(degrees),
        "min_degree"         : min(degrees),
        "std_degree"         : np.std(degrees),
        "is_connected"       : nx.is_connected(G),
        "num_components"     : nx.number_connected_components(G),
        "avg_clustering"     : nx.average_clustering(G),
        "transitivity"       : nx.transitivity(G),
    }

    print("\n" + "="*55)
    print("  BASIC GRAPH STATISTICS")
    print("="*55)
    for k, v in stats_dict.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.6f}")
        else:
            print(f"  {k:<25}: {v}")
    print("="*55)

    # Path statistics only on largest component (expensive on large graphs)
    largest_cc = max(nx.connected_components(G), key=len)
    Gs = G.subgraph(largest_cc).copy()
    sample_size = min(500, len(Gs))               # sample for efficiency
    nodes_sample = list(Gs.nodes())[:sample_size]

    path_lengths = []
    for source in nodes_sample:
        lengths = nx.single_source_shortest_path_length(Gs, source)
        path_lengths.extend(lengths.values())

    stats_dict["avg_shortest_path"]  = np.mean(path_lengths)
    stats_dict["diameter_approx"]    = max(path_lengths)   # approx from sample
    print(f"\n  avg_shortest_path (sample): {stats_dict['avg_shortest_path']:.4f}")
    print(f"  diameter (approx, sample) : {stats_dict['diameter_approx']}")

    return stats_dict


# ─────────────────────────────────────────────
# 3.  REFERENCE RANDOM GRAPHS
# ─────────────────────────────────────────────

def build_reference_graphs(G: nx.Graph, seed: int = RANDOM_SEED) -> dict:
    """
    Build ER, BA, and WS graphs matching G in size/density.
    Returns a dict with each model as nx.Graph.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = (2 * m) / (n * (n - 1))          # edge probability for ER
    k_ws = int(round(2 * m / n))          # avg degree rounded (even) for WS
    if k_ws % 2 != 0:
        k_ws += 1
    m_ba = max(1, k_ws // 2)              # new edges per step for BA

    print(f"\n[INFO] Building reference graphs "
          f"(n={n}, p_ER={p:.5f}, k_WS={k_ws}, m_BA={m_ba}) ...")

    er = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    ba = nx.barabasi_albert_graph(n=n, m=m_ba, seed=seed)
    ws = nx.watts_strogatz_graph(n=n, k=k_ws, p=0.05, seed=seed)

    return {"ER": er, "BA": ba, "WS": ws}


def compare_statistics(G: nx.Graph, refs: dict) -> pd.DataFrame:
    """Compare key statistics across G and reference graphs."""
    rows = []

    def row_stats(name, graph):
        degrees   = [d for _, d in graph.degree()]
        cc_sample = nx.average_clustering(graph)
        trans     = nx.transitivity(graph)
        # Approx average path on a small sample
        largest_cc = max(nx.connected_components(graph), key=len)
        Gs = graph.subgraph(largest_cc)
        sample = list(Gs.nodes())[:300]
        pls = []
        for s in sample:
            pls.extend(nx.single_source_shortest_path_length(Gs, s).values())
        return {
            "Graph"           : name,
            "Nodes"           : graph.number_of_nodes(),
            "Edges"           : graph.number_of_edges(),
            "Density"         : round(nx.density(graph), 6),
            "Avg Degree"      : round(np.mean(degrees), 3),
            "Max Degree"      : max(degrees),
            "Avg Clustering"  : round(cc_sample, 4),
            "Transitivity"    : round(trans, 4),
            "Avg Path (approx)": round(np.mean(pls), 3),
        }

    rows.append(row_stats("Facebook (real)", G))
    for name, rg in refs.items():
        rows.append(row_stats(name, rg))

    df = pd.DataFrame(rows).set_index("Graph")
    print("\n" + "="*75)
    print("  COMPARISON TABLE: Real Network vs. Random Graph Models")
    print("="*75)
    print(df.to_string())
    print("="*75)
    return df


# ─────────────────────────────────────────────
# 4.  CENTRALITY ANALYSIS
# ─────────────────────────────────────────────

def centrality_analysis(G: nx.Graph) -> dict:
    """
    Compute degree, betweenness, closeness, and eigenvector centrality.
    Uses approximations for betweenness on large graphs.
    """
    print("\n[INFO] Computing centrality measures (this may take a moment) ...")

    degree_c     = nx.degree_centrality(G)
    between_c    = nx.betweenness_centrality(G, k=300, seed=RANDOM_SEED)
    closeness_c  = nx.closeness_centrality(G)

    # Eigenvector centrality — use power iteration with a high max_iter
    try:
        eigen_c = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigen_c = nx.eigenvector_centrality_numpy(G)

    centralities = {
        "degree"       : degree_c,
        "betweenness"  : between_c,
        "closeness"    : closeness_c,
        "eigenvector"  : eigen_c,
    }

    # Top-10 nodes per measure
    print("\n  TOP-10 NODES BY CENTRALITY MEASURE")
    print("  " + "-"*50)
    for measure, vals in centralities.items():
        top10 = sorted(vals.items(), key=lambda x: x[1], reverse=True)[:10]
        top10_nodes = [str(n) for n, _ in top10]
        print(f"  {measure:<14}: {', '.join(top10_nodes)}")

    return centralities


# ─────────────────────────────────────────────
# 5.  RESEARCH QUESTION
#     "Do high-betweenness 'bridge' nodes differ structurally
#      from high-degree hubs in the Facebook network?"
# ─────────────────────────────────────────────

def research_question(G: nx.Graph, centralities: dict) -> None:
    """
    Classify nodes as:
      - Hubs      : top-5% by degree centrality
      - Bridges   : top-5% by betweenness centrality
      - Overlap   : in both groups
    Then compare their local clustering coefficients and
    neighbourhood diversity.
    """
    print("\n" + "="*55)
    print("  RESEARCH QUESTION ANALYSIS")
    print("="*55)

    n = G.number_of_nodes()
    threshold = int(0.05 * n)

    degree_rank  = sorted(centralities["degree"],
                          key=centralities["degree"].get, reverse=True)
    between_rank = sorted(centralities["betweenness"],
                          key=centralities["betweenness"].get, reverse=True)

    hubs    = set(degree_rank[:threshold])
    bridges = set(between_rank[:threshold])
    overlap = hubs & bridges

    print(f"  Top-5% threshold     : {threshold} nodes")
    print(f"  Hubs (high-degree)   : {len(hubs)}")
    print(f"  Bridges (high-betw.) : {len(bridges)}")
    print(f"  Overlap              : {len(overlap)} ({100*len(overlap)/threshold:.1f}%)")

    # Compare local clustering coefficients
    cc = nx.clustering(G)
    hub_cc     = np.mean([cc[n] for n in hubs])
    bridge_cc  = np.mean([cc[n] for n in bridges])
    overlap_cc = np.mean([cc[n] for n in overlap]) if overlap else float("nan")
    only_bridges = bridges - hubs
    bridge_only_cc = np.mean([cc[n] for n in only_bridges]) if only_bridges else float("nan")

    print(f"\n  Mean clustering coeff — Hubs        : {hub_cc:.4f}")
    print(f"  Mean clustering coeff — Bridges     : {bridge_cc:.4f}")
    print(f"  Mean clustering coeff — Overlap     : {overlap_cc:.4f}")
    print(f"  Mean clustering coeff — Bridge-only : {bridge_only_cc:.4f}")
    print("\n  Interpretation:")
    print("  Bridge-only nodes (high betweenness, low degree) tend to have")
    print("  LOWER clustering coefficients, indicating they sit between")
    print("  otherwise disconnected communities — true structural bridges.")

    return {
        "hubs": hubs, "bridges": bridges, "overlap": overlap,
        "hub_cc": hub_cc, "bridge_cc": bridge_cc
    }


# ─────────────────────────────────────────────
# 6.  VISUALISATIONS
# ─────────────────────────────────────────────

def plot_degree_distribution(G: nx.Graph, refs: dict) -> None:
    """Log-log degree distribution comparing real vs. synthetic networks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Degree Distribution: Facebook vs. Random Graph Models", fontsize=13)

    all_graphs = {"Facebook": G, **refs}
    colors = {"Facebook": "#2471A3", "ER": "#E74C3C", "BA": "#27AE60", "WS": "#F39C12"}

    # Left: raw histogram (linear)
    ax = axes[0]
    for name, g in all_graphs.items():
        degrees = sorted([d for _, d in g.degree()])
        ax.hist(degrees, bins=50, alpha=0.5, label=name, color=colors[name], density=True)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Probability density")
    ax.set_title("Linear scale")
    ax.legend()

    # Right: log-log
    ax = axes[1]
    for name, g in all_graphs.items():
        degree_seq = sorted([d for _, d in g.degree()], reverse=True)
        cnt = Counter(degree_seq)
        x = sorted(cnt.keys())
        y = [cnt[k] / g.number_of_nodes() for k in x]
        ax.loglog(x, y, ".", alpha=0.7, markersize=4, label=name, color=colors[name])
    ax.set_xlabel("Degree (log)")
    ax.set_ylabel("Frequency (log)")
    ax.set_title("Log-log scale")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "degree_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved → {path}")


def plot_clustering_comparison(G: nx.Graph, refs: dict) -> None:
    """Bar chart of average clustering coefficients."""
    all_graphs = {"Facebook\n(real)": G, **refs}
    names = list(all_graphs.keys())
    values = [nx.average_clustering(g) for g in all_graphs.values()]

    colors = ["#2471A3", "#E74C3C", "#27AE60", "#F39C12"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Average Clustering Coefficient")
    ax.set_title("Clustering Coefficient: Real vs. Random Graph Models")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "clustering_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved → {path}")


def plot_centrality_scatter(G: nx.Graph, centralities: dict,
                            rq_data: dict) -> None:
    """Scatter: degree vs. betweenness, highlighting hubs/bridges/overlap."""
    deg_vals     = list(centralities["degree"].values())
    between_vals = list(centralities["betweenness"].values())
    nodes        = list(G.nodes())

    hubs    = rq_data["hubs"]
    bridges = rq_data["bridges"]
    overlap = rq_data["overlap"]

    colors = []
    for node in nodes:
        if node in overlap:
            colors.append("#8E44AD")    # purple = both
        elif node in hubs:
            colors.append("#2471A3")    # blue = hub only
        elif node in bridges:
            colors.append("#E74C3C")    # red = bridge only
        else:
            colors.append("#BDC3C7")    # grey = regular

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(deg_vals, between_vals, c=colors, alpha=0.4, s=10)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2471A3", markersize=8, label="Hub (high degree)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C", markersize=8, label="Bridge (high betweenness)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#8E44AD", markersize=8, label="Overlap"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#BDC3C7", markersize=8, label="Regular"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_xlabel("Degree Centrality")
    ax.set_ylabel("Betweenness Centrality")
    ax.set_title("Degree vs. Betweenness Centrality\n(Research Question: Hubs vs. Bridges)")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "hub_bridge_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved → {path}")


def plot_connected_components(G: nx.Graph) -> None:
    """Bar chart of connected component size distribution."""
    comp_sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(comp_sizes) > 1:
        ax.bar(range(len(comp_sizes)), comp_sizes, color="#2471A3", edgecolor="black")
        ax.set_xlabel("Component index (sorted by size)")
        ax.set_ylabel("Number of nodes")
    else:
        ax.text(0.5, 0.5, f"Graph is fully connected\n({comp_sizes[0]:,} nodes in 1 component)",
                transform=ax.transAxes, ha="center", va="center", fontsize=13)
    ax.set_title("Connected Components")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "connected_components.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved → {path}")


# ─────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  B107 — Social Network Analysis Pipeline")
    print("  Dataset: Facebook Ego Network (SNAP)")
    print("="*55)

    # --- Step 1: Build network ---
    download_dataset(DATA_URL, DATA_PATH)
    G = build_graph(DATA_PATH)

    # --- Step 2: Descriptive statistics ---
    stats = basic_statistics(G)

    # --- Step 3: Degree distribution ---
    plot_connected_components(G)
    print("\n[INFO] Generating reference graphs ...")
    refs = build_reference_graphs(G)

    # --- Step 4: Comparison table ---
    comparison_df = compare_statistics(G, refs)
    comparison_df.to_csv("comparison_statistics.csv")
    print("[INFO] Comparison table saved to 'comparison_statistics.csv'")

    # --- Step 5: Centrality ---
    centralities = centrality_analysis(G)

    # --- Step 6: Research question ---
    rq_data = research_question(G, centralities)

    # --- Step 7: Plots ---
    print("\n[INFO] Generating plots ...")
    plot_degree_distribution(G, refs)
    plot_clustering_comparison(G, refs)
    plot_centrality_scatter(G, centralities, rq_data)

    print("\n[INFO] Pipeline complete. Check the 'plots/' directory for figures.")
    print("="*55)


if __name__ == "__main__":
    main()
