"""
KAI -- Streamlit Dashboard

Interactive web dashboard for visualizing experiment results comparing
local vs Kubernetes AI inference energy benchmarking.

Run with::

    streamlit run dashboard/app.py
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="KAI - Energy Benchmarking Dashboard",
    page_icon="=",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOGS_DIR = os.environ.get("KAI_LOGS_DIR", "logs")


@st.cache_data
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _find_result_files(logs_dir: str) -> List[str]:
    """Discover all JSON result files in the logs directory."""
    p = Path(logs_dir)
    if not p.exists():
        return []
    files = sorted(p.glob("*.json"), reverse=True)
    return [str(f) for f in files]


def _detect_mode(data: Dict[str, Any]) -> str:
    """Detect whether a JSON file is local, k8s, or combined."""
    if "local" in data and "kubernetes" in data:
        return "combined"
    if data.get("execution_mode") == "kubernetes" or "avg_e2e_latency_ms" in data:
        return "kubernetes"
    if data.get("execution_mode") == "local" or "avg_latency_ms" in data:
        return "local"
    return "unknown"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("KAI Dashboard")
st.sidebar.markdown("---")

result_files = _find_result_files(LOGS_DIR)

if not result_files:
    st.sidebar.warning(f"No result files found in `{LOGS_DIR}/`")
    st.title("KAI - Energy Benchmarking Dashboard")
    st.info(
        "No experiment results found. Run an experiment first:\n\n"
        "```\npython -m experiments.experiment_runner --mode both\n```"
    )
    st.stop()

# File selection
file_labels = [os.path.basename(f) for f in result_files]
selected_idx = st.sidebar.selectbox(
    "Select result file",
    range(len(file_labels)),
    format_func=lambda i: file_labels[i],
)
selected_file = result_files[selected_idx]

data = _load_json(selected_file)
detected_mode = _detect_mode(data)

st.sidebar.markdown(f"**Mode:** {detected_mode}")
st.sidebar.markdown(f"**File:** `{os.path.basename(selected_file)}`")

# Split combined files into local/k8s parts
local_data: Optional[Dict[str, Any]] = None
k8s_data: Optional[Dict[str, Any]] = None

if detected_mode == "combined":
    local_data = data.get("local")
    k8s_data = data.get("kubernetes")
elif detected_mode == "local":
    local_data = data
elif detected_mode == "kubernetes":
    k8s_data = data

# Optional: load a second file for comparison
st.sidebar.markdown("---")
compare_enabled = st.sidebar.checkbox("Load second file for comparison")
if compare_enabled:
    other_idx = st.sidebar.selectbox(
        "Compare with",
        range(len(file_labels)),
        format_func=lambda i: file_labels[i],
        key="compare_file",
    )
    other_file = result_files[other_idx]
    other_data = _load_json(other_file)
    other_mode = _detect_mode(other_data)

    if other_mode == "local" and local_data is None:
        local_data = other_data
    elif other_mode == "kubernetes" and k8s_data is None:
        k8s_data = other_data
    elif other_mode == "combined":
        if local_data is None:
            local_data = other_data.get("local")
        if k8s_data is None:
            k8s_data = other_data.get("kubernetes")

has_local = local_data is not None
has_k8s = k8s_data is not None

# Mode filter
mode_filter = st.sidebar.radio(
    "Display mode",
    ["Both", "Local only", "K8s only"],
    index=0,
)
if mode_filter == "Local only":
    has_k8s = False
    k8s_data = None
elif mode_filter == "K8s only":
    has_local = False
    local_data = None

st.sidebar.markdown("---")
st.sidebar.markdown("**Auto-refresh**")
auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Interval (seconds)", 5, 120, 30)
    st.sidebar.info(f"Refreshing every {refresh_interval}s")
    import time as _time
    _time.sleep(0.1)  # ensure page renders before rerun timer
    st.rerun() if False else None  # placeholder; actual rerun via fragment below


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("KAI - Energy Benchmarking Dashboard")

# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------

st.header("Summary Metrics")

if has_local and has_k8s:
    # Build comparison table
    local_latency = local_data.get("avg_latency_ms", 0.0)
    k8s_latency = k8s_data.get("avg_e2e_latency_ms", 0.0)

    local_throughput = local_data.get("throughput_inferences_per_sec", 0.0)
    k8s_throughput = k8s_data.get("throughput_inferences_per_sec", 0.0)

    local_power = local_data.get("avg_power_w", 0.0)
    k8s_power = k8s_data.get("total_avg_power_w", 0.0)

    local_energy = local_data.get("total_energy_wh", 0.0)
    k8s_energy = k8s_data.get("total_energy_wh", 0.0)

    local_epi = local_data.get("energy_per_inference_wh", 0.0)
    k8s_epi = k8s_data.get("energy_per_inference_wh", 0.0)

    def _ratio(a, b):
        return round(a / b, 2) if b != 0 else None

    summary_df = pd.DataFrame({
        "Metric": [
            "Avg Latency (ms)", "Throughput (inf/s)", "Avg GPU Power (W)",
            "Total Energy (Wh)", "Energy/Inference (Wh)", "Network Overhead (ms)",
        ],
        "Local": [
            f"{local_latency:.2f}", f"{local_throughput:.2f}",
            f"{local_power:.2f}", f"{local_energy:.6f}",
            f"{local_epi:.8f}", "0.00",
        ],
        "Kubernetes": [
            f"{k8s_latency:.2f}", f"{k8s_throughput:.2f}",
            f"{k8s_power:.2f}", f"{k8s_energy:.6f}",
            f"{k8s_epi:.8f}",
            f"{k8s_data.get('avg_network_overhead_ms', 0.0):.2f}",
        ],
        "Ratio (K8s/Local)": [
            str(_ratio(k8s_latency, local_latency)),
            str(_ratio(k8s_throughput, local_throughput)),
            str(_ratio(k8s_power, local_power)),
            str(_ratio(k8s_energy, local_energy)),
            str(_ratio(k8s_epi, local_epi)),
            "N/A",
        ],
    })
    st.dataframe(summary_df, width="stretch", hide_index=True)

elif has_local:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Latency", f"{local_data.get('avg_latency_ms', 0):.2f} ms")
    col2.metric("Throughput", f"{local_data.get('throughput_inferences_per_sec', 0):.2f} inf/s")
    col3.metric("Avg Power", f"{local_data.get('avg_power_w', 0):.2f} W")
    col4.metric("Energy", f"{local_data.get('total_energy_wh', 0):.6f} Wh")

elif has_k8s:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg E2E Latency", f"{k8s_data.get('avg_e2e_latency_ms', 0):.2f} ms")
    col2.metric("Throughput", f"{k8s_data.get('throughput_inferences_per_sec', 0):.2f} inf/s")
    col3.metric("Total Power", f"{k8s_data.get('total_avg_power_w', 0):.2f} W")
    col4.metric("Energy", f"{k8s_data.get('total_energy_wh', 0):.6f} Wh")

else:
    st.warning("No data to display")
    st.stop()

# ---------------------------------------------------------------------------
# Power Consumption Panel
# ---------------------------------------------------------------------------

st.header("GPU Power Consumption Over Time")

power_chart_data = {}

if has_local:
    gpu_samples = local_data.get("gpu_samples", [])
    if gpu_samples:
        power_chart_data["Local"] = [s.get("power_w", 0.0) for s in gpu_samples]

if has_k8s:
    node_metrics = k8s_data.get("node_metrics", [])
    for node in node_metrics:
        node_samples = node.get("gpu_samples", [])
        if node_samples:
            label = f"K8s ({node.get('node_name', 'node')})"
            power_chart_data[label] = [s.get("power_w", 0.0) for s in node_samples]
    # If no per-sample data, show avg as message
    if not any(n.get("gpu_samples") for n in node_metrics) and node_metrics:
        avg_powers = [n.get("avg_power_w", 0) for n in node_metrics if "error" not in n]
        if avg_powers:
            st.info(f"K8s average GPU power across nodes: {sum(avg_powers):.2f} W (no per-sample data)")

if power_chart_data:
    max_len = max(len(v) for v in power_chart_data.values())
    chart_df = pd.DataFrame({
        k: v + [None] * (max_len - len(v))
        for k, v in power_chart_data.items()
    })
    st.line_chart(chart_df)
elif not power_chart_data:
    st.info("No GPU power sample data available in selected file(s)")

# ---------------------------------------------------------------------------
# GPU Utilization Panel
# ---------------------------------------------------------------------------

st.header("GPU Utilization Over Time")

util_chart_data = {}

if has_local:
    gpu_samples = local_data.get("gpu_samples", [])
    if gpu_samples:
        util_chart_data["Local"] = [s.get("gpu_utilization_pct", 0.0) for s in gpu_samples]

if has_k8s:
    node_metrics = k8s_data.get("node_metrics", [])
    for node in node_metrics:
        node_samples = node.get("gpu_samples", [])
        if node_samples:
            label = f"K8s ({node.get('node_name', 'node')})"
            util_chart_data[label] = [s.get("gpu_utilization_pct", 0.0) for s in node_samples]

if util_chart_data:
    max_len = max(len(v) for v in util_chart_data.values())
    chart_df = pd.DataFrame({
        k: v + [None] * (max_len - len(v))
        for k, v in util_chart_data.items()
    })
    st.line_chart(chart_df)
else:
    st.info("No GPU utilization sample data available in selected file(s)")

# ---------------------------------------------------------------------------
# Latency Comparison Panel
# ---------------------------------------------------------------------------

st.header("Latency Comparison")

if has_local and has_k8s:
    lat_col1, lat_col2 = st.columns(2)

    with lat_col1:
        st.subheader("Average Latency")
        lat_df = pd.DataFrame({
            "Mode": ["Local", "Kubernetes"],
            "Latency (ms)": [
                local_data.get("avg_latency_ms", 0.0),
                k8s_data.get("avg_e2e_latency_ms", 0.0),
            ],
        })
        st.bar_chart(lat_df.set_index("Mode"))

    with lat_col2:
        st.subheader("Latency Distribution")
        dist_data = {}
        local_lats = local_data.get("inference_latencies_ms", [])
        k8s_lats = k8s_data.get("e2e_latencies_ms", [])
        if local_lats or k8s_lats:
            max_len = max(len(local_lats), len(k8s_lats))
            if local_lats:
                dist_data["Local"] = local_lats + [None] * (max_len - len(local_lats))
            if k8s_lats:
                dist_data["Kubernetes"] = k8s_lats + [None] * (max_len - len(k8s_lats))
            st.line_chart(pd.DataFrame(dist_data))
        else:
            st.info("No per-iteration latency data available")

elif has_local:
    local_lats = local_data.get("inference_latencies_ms", [])
    if local_lats:
        st.line_chart(pd.DataFrame({"Local Latency (ms)": local_lats}))
    else:
        st.metric("Avg Latency", f"{local_data.get('avg_latency_ms', 0):.2f} ms")

elif has_k8s:
    k8s_lats = k8s_data.get("e2e_latencies_ms", [])
    if k8s_lats:
        st.line_chart(pd.DataFrame({"K8s E2E Latency (ms)": k8s_lats}))
    else:
        st.metric("Avg E2E Latency", f"{k8s_data.get('avg_e2e_latency_ms', 0):.2f} ms")

# ---------------------------------------------------------------------------
# Energy Comparison Panel
# ---------------------------------------------------------------------------

st.header("Energy Comparison")

if has_local and has_k8s:
    en_col1, en_col2 = st.columns(2)

    with en_col1:
        st.subheader("Total Energy Consumed")
        energy_df = pd.DataFrame({
            "Mode": ["Local", "Kubernetes"],
            "Energy (Wh)": [
                local_data.get("total_energy_wh", 0.0),
                k8s_data.get("total_energy_wh", 0.0),
            ],
        })
        st.bar_chart(energy_df.set_index("Mode"))

    with en_col2:
        st.subheader("Energy per Inference")
        epi_df = pd.DataFrame({
            "Mode": ["Local", "Kubernetes"],
            "Energy/Inference (Wh)": [
                local_data.get("energy_per_inference_wh", 0.0),
                k8s_data.get("energy_per_inference_wh", 0.0),
            ],
        })
        st.bar_chart(epi_df.set_index("Mode"))

elif has_local:
    col1, col2 = st.columns(2)
    col1.metric("Total Energy", f"{local_data.get('total_energy_wh', 0):.6f} Wh")
    col2.metric("Energy/Inference", f"{local_data.get('energy_per_inference_wh', 0):.8f} Wh")

elif has_k8s:
    col1, col2 = st.columns(2)
    col1.metric("Total Energy", f"{k8s_data.get('total_energy_wh', 0):.6f} Wh")
    col2.metric("Energy/Inference", f"{k8s_data.get('energy_per_inference_wh', 0):.8f} Wh")

# ---------------------------------------------------------------------------
# Per-chunk latency (K8s only)
# ---------------------------------------------------------------------------

if has_k8s:
    chunk_lats = k8s_data.get("per_chunk_avg_latency_ms", [])
    if chunk_lats:
        st.header("Per-Chunk Latency (Kubernetes)")
        chunk_df = pd.DataFrame({
            "Chunk": [f"Chunk {i}" for i in range(len(chunk_lats))],
            "Avg Latency (ms)": chunk_lats,
        })
        st.bar_chart(chunk_df.set_index("Chunk"))

        net_overhead = k8s_data.get("avg_network_overhead_ms", 0.0)
        st.caption(f"Average network overhead: {net_overhead:.2f} ms")

# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

with st.expander("Experiment Configuration", expanded=False):
    if has_local:
        st.subheader("Local")
        cfg = local_data.get("experiment_config", {})
        if cfg:
            st.json(cfg)
        else:
            st.info("No configuration metadata in local results")

    if has_k8s:
        st.subheader("Kubernetes")
        cfg = k8s_data.get("experiment_config", {})
        if cfg:
            st.json(cfg)
        else:
            st.info("No configuration metadata in K8s results")

# ---------------------------------------------------------------------------
# Raw JSON viewer
# ---------------------------------------------------------------------------

with st.expander("Raw JSON Data", expanded=False):
    st.json(data)

# ---------------------------------------------------------------------------
# Auto-refresh logic
# ---------------------------------------------------------------------------

if auto_refresh:
    import time as _time
    _time.sleep(refresh_interval)
    st.rerun()
