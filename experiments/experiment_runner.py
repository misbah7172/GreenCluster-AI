"""
Unified experiment runner.

Orchestrates local, Kubernetes, or both inference experiments from a
single CLI entry-point and produces comparable, side-by-side results.

Usage::

    # Run only local experiment
    python -m experiments.experiment_runner --mode local --model transformer

    # Run only Kubernetes experiment
    python -m experiments.experiment_runner --mode kubernetes --model cnn --num-chunks 4

    # Run both and produce a comparison
    python -m experiments.experiment_runner --mode both --model transformer

Or programmatically::

    from experiments.experiment_runner import run_experiment
    results = run_experiment(mode="both", model_type="transformer")
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import helpers — lazy so the module can always be imported even if
# only one mode's dependencies are available.
# ---------------------------------------------------------------------------

def _run_local(
    model_type: str,
    iterations: int,
    batch_size: int,
    device: str,
    output_dir: str,
    monitor_interval: float,
    warmup: int,
) -> Dict[str, Any]:
    from experiments.local_runner import run_local_experiment
    return run_local_experiment(
        model_type=model_type,
        iterations=iterations,
        batch_size=batch_size,
        device_str=device,
        output_dir=output_dir,
        monitor_interval=monitor_interval,
        warmup_iterations=warmup,
    )


def _run_k8s(
    model_type: str,
    num_chunks: int,
    iterations: int,
    batch_size: int,
    output_dir: str,
    gateway_url: Optional[str],
    wait_timeout: float,
    warmup: int,
    auto_teardown: bool,
) -> Dict[str, Any]:
    # k8s_runner uses importlib to load the controller; import it the same way
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    import importlib.util as _ilu
    _runner_path = os.path.join(_project_root, "experiments", "k8s_runner.py")
    _spec = _ilu.spec_from_file_location("k8s_runner", _runner_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    return _mod.run_k8s_experiment(
        model_type=model_type,
        num_chunks=num_chunks,
        iterations=iterations,
        batch_size=batch_size,
        output_dir=output_dir,
        gateway_url=gateway_url,
        wait_timeout=wait_timeout,
        warmup_iterations=warmup,
        auto_teardown=auto_teardown,
    )


def _run_local_hf(
    hf_model: str,
    iterations: int,
    device: str,
    output_dir: str,
    monitor_interval: float,
    warmup: int,
) -> Dict[str, Any]:
    """Run a local HuggingFace model benchmark with energy monitoring.

    Loads the model, runs repeated inference, and collects power/latency
    metrics just like the standard local runner but using a real HF model.
    """
    import time
    import torch
    from model.hf_loader import HFModelLoader
    from model.layer_chunker import LayerChunker
    from model.generation import DistributedGenerator

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    loader = HFModelLoader(hf_model, dtype="float16")
    chunker = LayerChunker(loader)
    chunks = chunker.create_chunks(1)  # single chunk for local mode

    # Load weights
    from transformers import AutoModelForCausalLM
    try:
        real_model = AutoModelForCausalLM.from_pretrained(
            hf_model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        )
        real_model.eval()
        from model.hf_loader import HFModelLoader as _HFL
        embed, blocks, norm, lm_head = _HFL._detect_structure(real_model)
        real_map = {"embed": embed}
        for i, b in enumerate(blocks):
            real_map[f"layer_{i}"] = b
        if norm is not None:
            real_map["norm"] = norm
        if lm_head is not None:
            real_map["lm_head"] = lm_head
        for chunk in chunks:
            for name in chunk.layer_names:
                if name in real_map:
                    chunk.layers[name] = real_map[name]
            chunk.to(device)
            chunk.eval()
    except Exception as e:
        logger.warning("Could not load HF model for benchmark: %s", e)

    tokenizer = loader.get_tokenizer()
    gen = DistributedGenerator(chunks, tokenizer, device=device)

    # Start monitoring
    try:
        from monitoring.metrics import MetricsCollector
        collector = MetricsCollector(interval=monitor_interval)
        collector.start()
        has_monitor = True
    except Exception:
        has_monitor = False

    # Warmup
    for _ in range(warmup):
        gen.generate(prompt="Hello", max_new_tokens=5, temperature=0.7)

    # Measured iterations
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        gen.generate(prompt="The quick brown fox", max_new_tokens=10, temperature=0.7)
        latencies.append((time.perf_counter() - t0) * 1000)

    if has_monitor:
        collector.stop()

    avg_latency = sum(latencies) / len(latencies)
    total_time_s = sum(latencies) / 1000
    throughput = iterations / total_time_s if total_time_s > 0 else 0

    result = {
        "hf_model": hf_model,
        "iterations": iterations,
        "avg_latency_ms": round(avg_latency, 4),
        "throughput_inferences_per_sec": round(throughput, 4),
        "latencies_ms": [round(l, 4) for l in latencies],
    }

    if has_monitor:
        metrics = collector.compute_summary()
        result["avg_power_w"] = metrics.get("avg_gpu_power_w", 0.0)
        result["total_energy_wh"] = metrics.get("total_energy_wh", 0.0)
        result["energy_per_inference_wh"] = round(
            result["total_energy_wh"] / iterations, 8
        ) if iterations > 0 else 0.0

    logger.info("HF benchmark: model=%s, avg_latency=%.2fms, throughput=%.2f inf/s",
                hf_model, avg_latency, throughput)
    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _build_comparison(
    local_result: Dict[str, Any],
    k8s_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a side-by-side comparison of local vs Kubernetes results."""

    def _safe_div(a: float, b: float) -> Optional[float]:
        if b == 0:
            return None
        return round(a / b, 4)

    local_latency = local_result.get("avg_latency_ms", 0.0)
    k8s_latency = k8s_result.get("avg_e2e_latency_ms", 0.0)

    local_power = local_result.get("avg_power_w", 0.0)
    k8s_power = k8s_result.get("total_avg_power_w", 0.0)

    local_energy = local_result.get("total_energy_wh", 0.0)
    k8s_energy = k8s_result.get("total_energy_wh", 0.0)

    local_throughput = local_result.get("throughput_inferences_per_sec", 0.0)
    k8s_throughput = k8s_result.get("throughput_inferences_per_sec", 0.0)

    local_epi = local_result.get("energy_per_inference_wh", 0.0)
    k8s_epi = k8s_result.get("energy_per_inference_wh", 0.0)

    comparison = {
        "latency_ms": {
            "local": round(local_latency, 4),
            "kubernetes": round(k8s_latency, 4),
            "ratio_k8s_over_local": _safe_div(k8s_latency, local_latency),
        },
        "throughput_inf_per_sec": {
            "local": round(local_throughput, 4),
            "kubernetes": round(k8s_throughput, 4),
            "ratio_k8s_over_local": _safe_div(k8s_throughput, local_throughput),
        },
        "avg_power_w": {
            "local": round(local_power, 2),
            "kubernetes": round(k8s_power, 2),
            "ratio_k8s_over_local": _safe_div(k8s_power, local_power),
        },
        "total_energy_wh": {
            "local": round(local_energy, 6),
            "kubernetes": round(k8s_energy, 6),
            "ratio_k8s_over_local": _safe_div(k8s_energy, local_energy),
        },
        "energy_per_inference_wh": {
            "local": round(local_epi, 8),
            "kubernetes": round(k8s_epi, 8),
            "ratio_k8s_over_local": _safe_div(k8s_epi, local_epi),
        },
        "network_overhead_ms": {
            "local": 0.0,
            "kubernetes": round(k8s_result.get("avg_network_overhead_ms", 0.0), 4),
        },
    }

    return comparison


def _print_comparison(comparison: Dict[str, Any]) -> None:
    """Log a human-readable comparison table."""
    logger.info("=" * 60)
    logger.info("  COMPARISON: Local vs Kubernetes")
    logger.info("=" * 60)
    header = f"  {'Metric':<30} {'Local':>12} {'K8s':>12} {'Ratio':>8}"
    logger.info(header)
    logger.info("  " + "-" * 62)

    rows = [
        ("Avg Latency (ms)", "latency_ms"),
        ("Throughput (inf/s)", "throughput_inf_per_sec"),
        ("Avg GPU Power (W)", "avg_power_w"),
        ("Total Energy (Wh)", "total_energy_wh"),
        ("Energy/Inference (Wh)", "energy_per_inference_wh"),
        ("Network Overhead (ms)", "network_overhead_ms"),
    ]

    for label, key in rows:
        entry = comparison[key]
        local_val = entry["local"]
        k8s_val = entry["kubernetes"]
        ratio = entry.get("ratio_k8s_over_local")
        ratio_str = f"{ratio:.2f}x" if ratio is not None else "N/A"
        logger.info(
            "  %-30s %12.4f %12.4f %8s",
            label, local_val, k8s_val, ratio_str,
        )

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_experiment(
    mode: str = "both",
    model_type: str = "transformer",
    num_chunks: int = 3,
    iterations: int = 50,
    batch_size: int = 8,
    output_dir: str = "logs",
    device: str = "auto",
    monitor_interval: float = 1.0,
    warmup: int = 5,
    gateway_url: Optional[str] = None,
    wait_timeout: float = 300.0,
    auto_teardown: bool = True,
    hf_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run local, kubernetes, or both experiments.

    Parameters
    ----------
    mode : str
        ``"local"``, ``"kubernetes"``, or ``"both"``.
    model_type : str
        ``"transformer"`` or ``"cnn"``.
    num_chunks : int
        Number of model chunks for Kubernetes mode.
    iterations : int
        Number of measured inference iterations.
    batch_size : int
        Input batch size.
    output_dir : str
        Directory for per-mode result JSON files.
    device : str
        Torch device for local mode (``"auto"`` picks CUDA if available).
    monitor_interval : float
        GPU/CPU sampling interval in seconds (local mode).
    warmup : int
        Warmup iterations (both modes).
    gateway_url : str, optional
        Override gateway URL for Kubernetes mode.
    wait_timeout : float
        Pod readiness timeout in seconds.
    auto_teardown : bool
        Teardown K8s resources after the experiment.
    hf_model : str, optional
        HuggingFace model name for benchmarking with real HF models.

    Returns
    -------
    dict
        Combined results with optional comparison.
    """
    model_label = hf_model or model_type
    logger.info(
        "Experiment: mode=%s, model=%s, iterations=%d, batch_size=%d",
        mode, model_label, iterations, batch_size,
    )

    combined: Dict[str, Any] = {
        "mode": mode,
        "model_type": model_type,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if hf_model:
        combined["hf_model"] = hf_model

    local_result = None
    k8s_result = None

    # --- Local ---
    if mode in ("local", "both"):
        logger.info("--- Starting LOCAL experiment ---")
        if hf_model:
            local_result = _run_local_hf(
                hf_model=hf_model,
                iterations=iterations,
                device=device,
                output_dir=output_dir,
                monitor_interval=monitor_interval,
                warmup=warmup,
            )
        else:
            local_result = _run_local(
                model_type=model_type,
                iterations=iterations,
                batch_size=batch_size,
                device=device,
                output_dir=output_dir,
                monitor_interval=monitor_interval,
                warmup=warmup,
            )
        combined["local"] = local_result
        logger.info("--- LOCAL experiment complete ---")

    # --- Kubernetes ---
    if mode in ("kubernetes", "both"):
        logger.info("--- Starting KUBERNETES experiment ---")
        k8s_result = _run_k8s(
            model_type=model_type,
            num_chunks=num_chunks,
            iterations=iterations,
            batch_size=batch_size,
            output_dir=output_dir,
            gateway_url=gateway_url,
            wait_timeout=wait_timeout,
            warmup=warmup,
            auto_teardown=auto_teardown,
        )
        combined["kubernetes"] = k8s_result
        logger.info("--- KUBERNETES experiment complete ---")

    # --- Comparison ---
    if local_result is not None and k8s_result is not None:
        comparison = _build_comparison(local_result, k8s_result)
        combined["comparison"] = comparison
        _print_comparison(comparison)

    # --- Save combined results ---
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_{ts}.json"
    result_file = out_path / filename

    with open(result_file, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info("Combined results saved to %s", result_file)

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified experiment runner — local, kubernetes, or both",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["local", "kubernetes", "both"],
        help="Experiment mode (default: both)",
    )
    parser.add_argument(
        "--model", type=str, default="transformer",
        choices=["transformer", "cnn"],
        help="Model architecture (default: transformer)",
    )
    parser.add_argument(
        "--num-chunks", type=int, default=3,
        help="Number of model chunks for K8s mode (default: 3)",
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Number of inference iterations (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Input batch size (default: 8)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="logs",
        help="Output directory for results (default: logs)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Torch device for local mode (default: auto)",
    )
    parser.add_argument(
        "--monitor-interval", type=float, default=1.0,
        help="GPU/CPU sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--gateway-url", type=str, default=None,
        help="Override gateway URL for K8s mode",
    )
    parser.add_argument(
        "--wait-timeout", type=float, default=300.0,
        help="Pod readiness timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--no-teardown", action="store_true",
        help="Keep K8s resources after the experiment",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_experiment(
        mode=args.mode,
        model_type=args.model,
        num_chunks=args.num_chunks,
        iterations=args.iterations,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        monitor_interval=args.monitor_interval,
        warmup=args.warmup,
        gateway_url=args.gateway_url,
        wait_timeout=args.wait_timeout,
        auto_teardown=not args.no_teardown,
    )


if __name__ == "__main__":
    main()
