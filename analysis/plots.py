"""
Matplotlib visualizations for experiment results.

Generates publication-quality plots comparing local vs Kubernetes
inference benchmarks, saving all output as PNG files.

Usage::

    python -m analysis.plots \\
        --local  logs/local_results_20260311_143000.json \\
        --k8s    logs/k8s_results_20260311_143500.json \\
        --output analysis/output

Or programmatically::

    from analysis.analyzer import ExperimentAnalyzer
    from analysis.plots import plot_all
    analyzer = ExperimentAnalyzer()
    analyzer.load_local("logs/local_results.json")
    analyzer.load_k8s("logs/k8s_results.json")
    plot_all(analyzer, output_dir="analysis/output")
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from analysis.analyzer import ExperimentAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

_COLORS = {"local": "#2196F3", "kubernetes": "#FF5722"}
_FIG_SIZE = (10, 6)
_DPI = 150


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


# ---------------------------------------------------------------------------
# 1. Power over time
# ---------------------------------------------------------------------------

def plot_power_over_time(
    analyzer: ExperimentAnalyzer,
    output_dir: str,
) -> Optional[str]:
    """Line plot of GPU power draw over experiment duration (both modes)."""
    local_samples = analyzer.get_local_gpu_samples()
    k8s_nodes = analyzer.get_k8s_node_metrics()

    if not local_samples and not k8s_nodes:
        logger.warning("No GPU samples available for power plot")
        return None

    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    if local_samples:
        times = list(range(len(local_samples)))
        powers = [s.get("power_w", 0.0) for s in local_samples]
        ax.plot(times, powers, color=_COLORS["local"], label="Local", linewidth=1.5)

    # For K8s, if we have per-node gpu_samples, overlay them;
    # otherwise just show avg_power_w as a horizontal line
    k8s_plotted = False
    for node in k8s_nodes:
        node_samples = node.get("gpu_samples", [])
        if node_samples:
            times = list(range(len(node_samples)))
            powers = [s.get("power_w", 0.0) for s in node_samples]
            label = f"K8s ({node.get('node_name', 'node')})"
            ax.plot(times, powers, linestyle="--", label=label, linewidth=1.5)
            k8s_plotted = True

    if not k8s_plotted and k8s_nodes:
        avg_p = sum(n.get("avg_power_w", 0.0) for n in k8s_nodes if "error" not in n)
        if avg_p > 0:
            ax.axhline(y=avg_p, color=_COLORS["kubernetes"], linestyle="--",
                       label=f"K8s avg ({avg_p:.1f} W)", linewidth=1.5)

    ax.set_xlabel("Sample Index (1s intervals)")
    ax.set_ylabel("GPU Power (W)")
    ax.set_title("GPU Power Consumption Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = str(Path(output_dir) / "power_over_time.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 2. GPU utilization over time
# ---------------------------------------------------------------------------

def plot_gpu_utilization_over_time(
    analyzer: ExperimentAnalyzer,
    output_dir: str,
) -> Optional[str]:
    """Line plot of GPU utilization over time (both modes)."""
    local_samples = analyzer.get_local_gpu_samples()
    k8s_nodes = analyzer.get_k8s_node_metrics()

    if not local_samples and not k8s_nodes:
        logger.warning("No GPU samples available for utilization plot")
        return None

    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    if local_samples:
        times = list(range(len(local_samples)))
        utils = [s.get("gpu_utilization_pct", 0.0) for s in local_samples]
        ax.plot(times, utils, color=_COLORS["local"], label="Local", linewidth=1.5)

    for node in k8s_nodes:
        node_samples = node.get("gpu_samples", [])
        if node_samples:
            times = list(range(len(node_samples)))
            utils = [s.get("gpu_utilization_pct", 0.0) for s in node_samples]
            label = f"K8s ({node.get('node_name', 'node')})"
            ax.plot(times, utils, linestyle="--", label=label, linewidth=1.5)

    ax.set_xlabel("Sample Index (1s intervals)")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title("GPU Utilization Over Time")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = str(Path(output_dir) / "gpu_utilization_over_time.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 3. Latency comparison (bar chart)
# ---------------------------------------------------------------------------

def plot_latency_comparison(
    analyzer: ExperimentAnalyzer,
    output_dir: str,
) -> Optional[str]:
    """Bar chart comparing average latency: local vs K8s."""
    la = analyzer.analyse_local()
    ka = analyzer.analyse_k8s()

    fig, ax = plt.subplots(figsize=(8, 6))

    modes = ["Local", "Kubernetes"]
    avgs = [la["avg_latency_ms"], ka["avg_latency_ms"]]
    colors = [_COLORS["local"], _COLORS["kubernetes"]]

    bars = ax.bar(modes, avgs, color=colors, width=0.5, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Average Latency (ms)")
    ax.set_title("Inference Latency Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    path = str(Path(output_dir) / "latency_comparison.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 4. Energy comparison (bar chart)
# ---------------------------------------------------------------------------

def plot_energy_comparison(
    analyzer: ExperimentAnalyzer,
    output_dir: str,
) -> Optional[str]:
    """Bar chart comparing total energy consumed: local vs K8s."""
    la = analyzer.analyse_local()
    ka = analyzer.analyse_k8s()

    fig, ax = plt.subplots(figsize=(8, 6))

    modes = ["Local", "Kubernetes"]
    vals = [la["total_energy_wh"], ka["total_energy_wh"]]
    colors = [_COLORS["local"], _COLORS["kubernetes"]]

    bars = ax.bar(modes, vals, color=colors, width=0.5, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.6f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Total Energy (Wh)")
    ax.set_title("Total Energy Consumption Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    path = str(Path(output_dir) / "energy_comparison.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 5. Energy efficiency (bar chart)
# ---------------------------------------------------------------------------

def plot_energy_efficiency(
    analyzer: ExperimentAnalyzer,
    output_dir: str,
) -> Optional[str]:
    """Bar chart comparing energy per inference: local vs K8s."""
    la = analyzer.analyse_local()
    ka = analyzer.analyse_k8s()

    fig, ax = plt.subplots(figsize=(8, 6))

    modes = ["Local", "Kubernetes"]
    vals = [la["energy_per_inference_wh"], ka["energy_per_inference_wh"]]
    colors = [_COLORS["local"], _COLORS["kubernetes"]]

    bars = ax.bar(modes, vals, color=colors, width=0.5, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.8f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Energy per Inference (Wh)")
    ax.set_title("Energy Efficiency Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    path = str(Path(output_dir) / "energy_efficiency.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 6. Throughput comparison (bar chart)
# ---------------------------------------------------------------------------

def plot_throughput_comparison(
    analyzer: ExperimentAnalyzer,
    output_dir: str,
) -> Optional[str]:
    """Bar chart comparing throughput: local vs K8s."""
    la = analyzer.analyse_local()
    ka = analyzer.analyse_k8s()

    fig, ax = plt.subplots(figsize=(8, 6))

    modes = ["Local", "Kubernetes"]
    vals = [la["throughput_inf_per_sec"], ka["throughput_inf_per_sec"]]
    colors = [_COLORS["local"], _COLORS["kubernetes"]]

    bars = ax.bar(modes, vals, color=colors, width=0.5, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Throughput (inferences/sec)")
    ax.set_title("Throughput Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    path = str(Path(output_dir) / "throughput_comparison.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 7. Latency distribution (histogram / box plot)
# ---------------------------------------------------------------------------

def plot_latency_distribution(
    analyzer: ExperimentAnalyzer,
    output_dir: str,
) -> Optional[str]:
    """Box plot showing latency distributions for both modes."""
    local_lats = analyzer.get_local_latencies()
    k8s_lats = analyzer.get_k8s_latencies()

    if not local_lats and not k8s_lats:
        logger.warning("No latency data for distribution plot")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    data = []
    labels = []
    colors_list = []
    if local_lats:
        data.append(local_lats)
        labels.append("Local")
        colors_list.append(_COLORS["local"])
    if k8s_lats:
        data.append(k8s_lats)
        labels.append("Kubernetes")
        colors_list.append(_COLORS["kubernetes"])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.4)
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Distribution")
    ax.grid(True, axis="y", alpha=0.3)

    path = str(Path(output_dir) / "latency_distribution.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 8. Per-chunk latency breakdown (K8s only)
# ---------------------------------------------------------------------------

def plot_per_chunk_latency(
    analyzer: ExperimentAnalyzer,
    output_dir: str,
) -> Optional[str]:
    """Bar chart of average per-chunk latency (K8s only)."""
    ka = analyzer.analyse_k8s()
    chunk_lats = ka.get("per_chunk_avg_latency_ms", [])

    if not chunk_lats:
        logger.warning("No per-chunk latency data")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    chunk_labels = [f"Chunk {i}" for i in range(len(chunk_lats))]
    bars = ax.bar(chunk_labels, chunk_lats, color="#4CAF50", width=0.5,
                  edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, chunk_lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    # Add network overhead as annotation
    net_oh = ka.get("avg_network_overhead_ms", 0.0)
    ax.set_xlabel("Model Chunk")
    ax.set_ylabel("Average Latency (ms)")
    ax.set_title(f"Per-Chunk Inference Latency (network overhead: {net_oh:.2f} ms)")
    ax.grid(True, axis="y", alpha=0.3)

    path = str(Path(output_dir) / "per_chunk_latency.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------

def plot_all(
    analyzer: ExperimentAnalyzer,
    output_dir: str = "analysis/output",
) -> List[str]:
    """Generate all available plots and save to output_dir.

    Parameters
    ----------
    analyzer : ExperimentAnalyzer
        Analyzer with at least one result set loaded.
    output_dir : str
        Directory for PNG outputs.

    Returns
    -------
    list[str]
        Paths of generated plot files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    has_local = analyzer.local_data is not None
    has_k8s = analyzer.k8s_data is not None

    # Time-series plots (need raw samples)
    if has_local or has_k8s:
        p = plot_power_over_time(analyzer, output_dir)
        if p:
            generated.append(p)
        p = plot_gpu_utilization_over_time(analyzer, output_dir)
        if p:
            generated.append(p)

    # Comparison bar charts (need both)
    if has_local and has_k8s:
        for fn in (plot_latency_comparison, plot_energy_comparison,
                   plot_energy_efficiency, plot_throughput_comparison):
            p = fn(analyzer, output_dir)
            if p:
                generated.append(p)

    # Distribution plot
    p = plot_latency_distribution(analyzer, output_dir)
    if p:
        generated.append(p)

    # K8s-specific
    if has_k8s:
        p = plot_per_chunk_latency(analyzer, output_dir)
        if p:
            generated.append(p)

    logger.info("Generated %d plots in %s", len(generated), output_dir)
    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis plots from experiment results",
    )
    parser.add_argument(
        "--local", type=str, default=None,
        help="Path to local results JSON",
    )
    parser.add_argument(
        "--k8s", type=str, default=None,
        help="Path to K8s results JSON",
    )
    parser.add_argument(
        "--combined", type=str, default=None,
        help="Path to combined experiment JSON",
    )
    parser.add_argument(
        "--output", type=str, default="analysis/output",
        help="Output directory for plot PNGs (default: analysis/output)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    analyzer = ExperimentAnalyzer()

    if args.combined:
        analyzer.load_combined(args.combined)
    else:
        if args.local:
            analyzer.load_local(args.local)
        if args.k8s:
            analyzer.load_k8s(args.k8s)

    if analyzer.local_data is None and analyzer.k8s_data is None:
        logger.error("No result files provided. Use --local, --k8s, or --combined.")
        return

    plot_all(analyzer, output_dir=args.output)


if __name__ == "__main__":
    main()
