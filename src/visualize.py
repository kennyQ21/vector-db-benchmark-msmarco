"""
Visualization — Publication-quality charts for the vector search benchmark.

Reads benchmark_results.json, feature_results.json, and advanced_results.json
to generate the 5 Elite-Tier charts for the final blog post:
  1. Feature Matrix (Heatmap)
  2. Latency vs Scale (Line chart: 10k -> 100k)
  3. Qdrant Pareto Plot (Recall vs Latency tradeoff for m=16 vs m=32)
  4. FAISS-HNSW Recall@k Parity Curves
  5. Concurrency Saturation (QPS vs Workers)
  
All charts saved as high-res PNGs in results/charts/.
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

CHARTS_DIR = Path(config.RESULTS_DIR) / "charts"

# ─── Style ────────────────────────────────────────────────

def setup_style_dark():
    """Premium dark theme for standard benchmark plots."""
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "grid.alpha": 0.6,
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "figure.titlesize": 18,
        "figure.titleweight": "bold",
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300
    })

def setup_style_light():
    """Clean light theme for research/advanced plots."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300
    })

ENGINE_COLORS = {
    "FAISS-FLAT": "#58a6ff",
    "FAISS-IVF":  "#3fb950",
    "Qdrant":     "#d29922",
    "ES-Vector":  "#f85149",
    "ES-Hybrid":  "#bc8cff",
}
ENGINE_ORDER = ["FAISS-FLAT", "FAISS-IVF", "Qdrant", "ES-Vector", "ES-Hybrid"]

def get_color(engine_name: str) -> str:
    return ENGINE_COLORS.get(engine_name, "#8b949e")

def get_engines_sorted(results: list[dict]) -> list[str]:
    present = {r["engine"] for r in results}
    return [e for e in ENGINE_ORDER if e in present]

def save_chart(fig, name: str):
    path = CHARTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  💾 Saved: {path.name}")
    return path


# ─── Chart 1: Feature Matrix Heatmap ─────────────────────
def plot_feature_matrix():
    feature_path = Path(config.RESULTS_DIR) / "feature_results.json"
    if not feature_path.exists():
        return
    with open(feature_path) as f:
        data = json.load(f)["star_matrix"]

    features = list(data.keys())
    engines = ["FAISS", "Qdrant", "Elasticsearch"]
    scores = np.array([[data[f][e]["stars"] for e in engines] for f in features], dtype=float)

    setup_style_dark()
    fig, ax = plt.subplots(figsize=(10, 6))

    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ["#2d1117", "#da3633", "#d29922", "#3fb950", "#2ea043"]
    cmap = LinearSegmentedColormap.from_list("stars", colors_list, N=6)

    im = ax.imshow(scores, cmap=cmap, aspect="auto", vmin=0, vmax=5)

    ax.set_xticks(np.arange(len(engines)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(engines, fontsize=13, fontweight="bold")
    ax.set_yticklabels(features, fontsize=11)

    for i in range(len(features)):
        for j in range(len(engines)):
            val = int(scores[i, j])
            text = "★" * val if val > 0 else "—"
            color = "#0d1117" if val >= 4 else "#c9d1d9"
            ax.text(j, i, text, ha="center", va="center", fontsize=14, color=color)

    ax.set_title("Feature Comparison Matrix", pad=15)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_chart(fig, "feature_matrix")


# ─── Chart 2: Latency vs Scale Line Chart ────────────────
def plot_latency_vs_scale():
    bench_path = Path(config.RESULTS_DIR) / "benchmark_results.json"
    if not bench_path.exists():
        return
    with open(bench_path) as f:
        results = json.load(f)["results"]

    scales = sorted(set(r["scale"] for r in results))
    if len(scales) < 2:
        return

    setup_style_dark()
    engines = get_engines_sorted(results)
    fig, ax = plt.subplots(figsize=(10, 6))

    for engine in engines:
        engine_data = sorted([r for r in results if r["engine"] == engine], key=lambda r: r["scale"])
        x = [r["scale"] for r in engine_data]
        y = [r["latency_p50_ms"] for r in engine_data]
        ax.plot(x, y, marker="o", linewidth=2.5, markersize=8,
                color=get_color(engine), label=engine, alpha=0.9)

    ax.set_xlabel("Dataset Size (passages)")
    ax.set_ylabel("P50 Latency (ms)")
    ax.set_title("Query Latency vs Dataset Scale")
    ax.legend(loc="upper left", frameon=True, facecolor="#161b22", edgecolor="#30363d", framealpha=0.9)
    ax.grid(True, linestyle="--")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

    save_chart(fig, "latency_vs_scale")


# ─── Chart 3: Qdrant Pareto Plot ─────────────────────────
def plot_pareto_frontier(advanced_data: dict):
    if "qdrant_m_comparison_100k" not in advanced_data:
        return
        
    m_data = advanced_data["qdrant_m_comparison_100k"]
    setup_style_light()
    fig = plt.figure(figsize=(10, 6))
    
    colors = {"m16": "#FF9F43", "m32": "#00d2d3"}
    markers = {"m16": "o", "m32": "s"}
    labels = {"m16": "Qdrant HNSW (m=16)", "m32": "Qdrant HNSW (m=32)"}
    
    for m_key in ["m16", "m32"]:
        if m_key not in m_data: continue
        data = m_data[m_key]
        latencies = [d["latency_p50_ms"] for d in data]
        recalls = [d["recall_at_10"] for d in data]
        ef_values = [d["ef_search"] for d in data]
        
        plt.plot(latencies, recalls, '-', color=colors[m_key], alpha=0.5, zorder=1)
        plt.scatter(latencies, recalls, s=120, c=colors[m_key], label=labels[m_key], 
                    marker=markers[m_key], edgecolor='white', linewidth=1.5, zorder=2)
        
        for lat, rec, ef in zip(latencies, recalls, ef_values):
            plt.annotate(f"ef={ef}", (lat, rec), xytext=(8, -8), 
                         textcoords="offset points", fontsize=10, 
                         color="#555555", zorder=3)

    plt.title("HNSW Pareto Frontier at 100k Scale\nLatency vs Recall Tradeoff", pad=20, fontweight="bold")
    plt.xlabel("P50 Latency (ms)", fontweight="bold")
    plt.ylabel("Recall@10", fontweight="bold")
    plt.legend(loc="lower left", frameon=True, shadow=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    save_chart(fig, "pareto_frontier_qdrant")


# ─── Chart 4: Recall Parity Curves ───────────────────────
def plot_recall_at_k(advanced_data: dict):
    if "recall_at_k" not in advanced_data:
        return
        
    data = advanced_data["recall_at_k"]
    setup_style_light()
    fig = plt.figure(figsize=(10, 6))
    
    colors = {"FAISS-Flat": "#5f27cd", "Qdrant": "#ff6b6b", "ES-Vector": "#01a3a4", "FAISS-HNSW": "#ff9f43"}
    markers = {"FAISS-Flat": "o", "Qdrant": "s", "ES-Vector": "^", "FAISS-HNSW": "D"}
        
    for engine, metrics in data.items():
        if engine not in colors: continue
        
        k_values = sorted([int(k) for k in metrics.keys()])
        recalls = [metrics[str(k)] for k in k_values]
        
        offset = {"FAISS-Flat": -0.2, "Qdrant": -0.1, "ES-Vector": 0.1, "FAISS-HNSW": 0.2}.get(engine, 0)
        x_jittered = [k + offset for k in k_values]
        
        plt.plot(x_jittered, recalls, marker=markers.get(engine, 'o'), 
                 markersize=10, linewidth=2.5, alpha=0.8,
                 color=colors.get(engine, '#333'), label=engine)
                 
    plt.title("Recall Parity Across Engines (10k scale)", pad=20, fontweight="bold")
    plt.xlabel("k (Top-k results retrieved)", fontweight="bold")
    plt.ylabel("Recall@k", fontweight="bold")
    plt.xscale('log')
    plt.xticks(k_values, labels=[str(k) for k in k_values])
    plt.ylim(0.2, 1.05)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc="lower right", frameon=True, shadow=True, fancybox=True, framealpha=0.9)
    
    plt.tight_layout()
    save_chart(fig, "recall_at_k_curves")


# ─── Chart 5: Concurrency Scaling ────────────────────────
def plot_concurrency(advanced_data: dict):
    if "concurrency" not in advanced_data:
        return
        
    data = advanced_data["concurrency"]
    setup_style_light()
    fig = plt.figure(figsize=(10, 6))
    
    workers = [d["workers"] for d in data]
    qps = [d["throughput_qps"] for d in data]
    
    plt.plot(workers, qps, marker='o', markersize=12, linewidth=3, 
             color="#10ac84", markerfacecolor="white", markeredgewidth=3)
             
    for i, d in enumerate(data):
        w = d["workers"]
        q = d["throughput_qps"]
        p50 = d["latency_p50_ms"]
        p95 = d["latency_p95_ms"]
        
        plt.annotate(f"P50: {p50}ms\nP95: {p95}ms", (w, q), 
                     xytext=(0, -25 if i != 1 else 15), 
                     textcoords="offset points", ha="center", va="top" if i != 1 else "bottom",
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#10ac84", alpha=0.8))

    plt.title("Throughput Saturation (FAISS In-Memory)\nQPS scaling with worker threads", pad=20, fontweight="bold")
    plt.xlabel("Number of Concurrent Worker Threads", fontweight="bold")
    plt.ylabel("Throughput (Queries per Second)", fontweight="bold")
    plt.xticks(workers)
    plt.ylim(0, max(qps) * 1.2)
    
    plt.axhline(y=max(qps), color='r', linestyle='--', alpha=0.3, label=f"Saturation: ~{int(max(qps))} QPS")
    plt.legend(loc="lower right", frameon=True, framealpha=0.9, shadow=True)
    
    plt.tight_layout()
    save_chart(fig, "concurrency_scaling")


# ─── Main Orchestrator ──────────────────────────────────
def generate_all_charts():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  📊 Generating Top-5 Elite Tier Charts")
    print(f"{'='*60}\n")
    
    # 1. Dark Theme Plots (From benchmark/feature results)
    plot_feature_matrix()
    plot_latency_vs_scale()
    
    # 2. Light Theme Plots (From advanced results)
    adv_path = Path(config.RESULTS_DIR) / "advanced_results.json"
    if adv_path.exists():
        with open(adv_path) as f:
            adv_data = json.load(f)
        plot_pareto_frontier(adv_data)
        plot_recall_at_k(adv_data)
        plot_concurrency(adv_data)

    print(f"\n✅ Generated charts gracefully in {CHARTS_DIR.name}/")

if __name__ == "__main__":
    generate_all_charts()
