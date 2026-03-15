"""
KAI -- Unified Control Dashboard

Interactive web interface for the full KAI platform: system scanning,
model inference, cluster setup, Kubernetes deployment, benchmarking,
live GPU monitoring, and experiment analysis.

Run with::

    streamlit run dashboard/unified_app.py
    # or
    python kai_cli.py dashboard
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is on sys.path so local imports work
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Page config -- must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="KAI - Unified Control Dashboard",
    page_icon="=",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POPULAR_MODELS = [
    "sshleifer/tiny-gpt2",
    "openai-community/gpt2",
    "microsoft/phi-2",
    "google/gemma-2b",
    "tiiuae/falcon-7b",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen2-7B",
]

MODEL_SIZES_MB = {
    "GPT-2 (124M)": 250,
    "Phi-2 (2.7B)": 5400,
    "Gemma-2B": 4000,
    "Falcon-7B": 14000,
    "Llama-2-7B": 14000,
    "Mistral-7B": 14000,
    "Llama-2-13B": 26000,
    "Llama-2-70B": 140000,
}

LOGS_DIR = os.environ.get("KAI_LOGS_DIR", "logs")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _init(key: str, default: Any = None) -> Any:
    """Initialise session_state key if absent and return its value."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


@st.cache_data
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _find_result_files(logs_dir: str) -> List[str]:
    p = Path(logs_dir)
    if not p.exists():
        return []
    return [str(f) for f in sorted(p.glob("*.json"), reverse=True)]


def _detect_mode(data: Dict[str, Any]) -> str:
    if "local" in data and "kubernetes" in data:
        return "combined"
    if data.get("execution_mode") == "kubernetes" or "avg_e2e_latency_ms" in data:
        return "kubernetes"
    if data.get("execution_mode") == "local" or "avg_latency_ms" in data:
        return "local"
    return "unknown"


def _subprocess_reader(proc, state_key: str) -> None:
    """Background thread: reads subprocess stdout line-by-line into session_state."""
    try:
        for line in iter(proc.stdout.readline, ""):
            st.session_state[state_key] = st.session_state.get(state_key, "") + line
        proc.stdout.close()
    except Exception:
        pass


def _run_subprocess(cmd: List[str], log_key: str, proc_key: str) -> None:
    """Launch a subprocess and start a reader thread."""
    st.session_state[log_key] = ""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=_PROJECT_ROOT,
    )
    st.session_state[proc_key] = proc
    t = threading.Thread(target=_subprocess_reader, args=(proc, log_key), daemon=True)
    t.start()


def _proc_is_running(proc_key: str) -> bool:
    proc = st.session_state.get(proc_key)
    if proc is None:
        return False
    return proc.poll() is None


# ===================================================================
# PAGE FUNCTIONS
# ===================================================================


# -------------------------------------------------------------------
# Page 1: Home
# -------------------------------------------------------------------

def page_home():
    st.title("KAI - System Overview")
    st.markdown("Scan your system resources and check model compatibility.")

    # --- Scan ---
    col_btn, _ = st.columns([1, 5])
    with col_btn:
        rescan = st.button("Rescan System")

    if rescan or "home_scan" not in st.session_state:
        try:
            from model.resource_detector import ResourceDetector
            detector = ResourceDetector(mode="local")
            st.session_state["home_scan"] = detector.scan_summary()
        except Exception as e:
            st.error(f"Scan failed: {e}")
            return

    summary = st.session_state["home_scan"]
    nodes = summary.get("nodes", [])
    node = nodes[0] if nodes else {}

    # --- Metric cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GPU", node.get("gpu_type", "None detected"))
    c2.metric("GPU VRAM", f"{node.get('gpu_vram_mb', 0):.0f} MB")
    c3.metric("System RAM", f"{summary.get('total_ram_mb', 0):.0f} MB")
    c4.metric("CPU Cores", summary.get("total_cpu_cores", 0))

    st.markdown("---")

    # --- Multi-node table ---
    if len(nodes) > 1:
        st.subheader(f"Cluster Nodes ({len(nodes)})")
        df = pd.DataFrame(nodes)
        st.dataframe(df, hide_index=True)

    # --- Model compatibility ---
    st.subheader("Model Compatibility")
    total_usable = summary.get("total_usable_mb", 0)
    compat_rows = []
    for name, size_mb in MODEL_SIZES_MB.items():
        fits = "Yes" if size_mb <= total_usable else "No"
        compat_rows.append({"Model": name, "Size (MB)": size_mb, "Fits?": fits})
    st.dataframe(pd.DataFrame(compat_rows), hide_index=True)

    # --- K8s probe ---
    st.markdown("---")
    with st.expander("Kubernetes Cluster Check"):
        try:
            from model.resource_detector import ResourceDetector
            k8s_det = ResourceDetector(mode="kubernetes")
            k8s_sum = k8s_det.scan_summary()
            k8s_sum = _augment_k8s_with_local_gpu(k8s_sum)
            st.success(f"Connected! {k8s_sum['num_nodes']} node(s), {k8s_sum.get('gpu_nodes', 0)} GPU node(s)")
            st.dataframe(pd.DataFrame(k8s_sum.get("nodes", [])), hide_index=True)
        except Exception as e:
            st.warning(f"Kubernetes not reachable: {e}")


# -------------------------------------------------------------------
# Page 2: Run Inference
# -------------------------------------------------------------------

def page_run_inference():
    st.title("Run Inference")

    # --- Model selection ---
    col_model, col_custom = st.columns([2, 3])
    with col_model:
        model_options = POPULAR_MODELS + ["Custom..."]
        model_choice = st.selectbox("Select model", model_options, key="run_model_sel")
    with col_custom:
        if model_choice == "Custom...":
            model_name = st.text_input("HuggingFace model name", key="run_custom_model")
        else:
            model_name = model_choice
            st.text_input("Model", value=model_name, disabled=True, key="run_model_disp")

    if not model_name:
        st.info("Select or enter a model to continue.")
        return

    # --- Model info ---
    with st.expander("Model Info", expanded=False):
        if st.button("Load model info", key="run_load_info"):
            with st.spinner("Fetching model metadata..."):
                try:
                    from model.hf_loader import HFModelLoader
                    loader = HFModelLoader(model_name, dtype="float16")
                    info = loader.get_model_size_estimate()
                    st.session_state["run_model_info"] = info
                    loader.validate_architecture()
                    st.session_state["run_arch_valid"] = True
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state["run_arch_valid"] = False

        info = st.session_state.get("run_model_info")
        if info:
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Parameters", f"{info.get('params_millions', 0):.1f}M")
            ic2.metric("FP32 Size", f"{info.get('float32_mb', 0):.0f} MB")
            ic3.metric("FP16 Size", f"{info.get('float16_mb', 0):.0f} MB")
            ic4.metric("INT8 Size", f"{info.get('int8_mb', 0):.0f} MB")
            if st.session_state.get("run_arch_valid"):
                st.success("Architecture supported")

    st.markdown("---")

    # --- Generation config ---
    st.subheader("Generation Config")
    col_a, col_b = st.columns(2)
    with col_a:
        prompt = st.text_area("Prompt", value="The future of AI is", height=100, key="run_prompt")
        max_tokens = st.slider("Max tokens", 1, 512, 100, key="run_max_tokens")
        stream = st.checkbox("Stream output", value=True, key="run_stream")
    with col_b:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="run_temp")
        top_k = st.slider("Top-k", 0, 200, 50, key="run_top_k")
        top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05, key="run_top_p")
        rep_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.1, 0.05, key="run_rep")

    col_c, col_d = st.columns(2)
    with col_c:
        device = st.selectbox("Device", ["cpu", "cuda:0", "auto"], key="run_device")
        dtype = st.selectbox("Dtype", ["float16", "bfloat16", "float32"], key="run_dtype")
    with col_d:
        num_chunks = st.number_input("Num chunks", 1, 8, 1, key="run_chunks")

    # --- Offload options ---
    with st.expander("Offloading Options"):
        offload = st.checkbox("Enable FlexGen-style offloading", key="run_offload")
        if offload:
            gpu_budget = st.number_input("GPU budget (MB, 0=auto)", 0, 100000, 0, key="run_gpu_budget")
            disk_dir = st.text_input("Disk swap dir", "/tmp/kai_swap", key="run_disk_dir")

    # --- Partition preview ---
    with st.expander("Partition Preview"):
        if st.button("Preview Partition", key="run_preview"):
            with st.spinner("Computing partition plan..."):
                try:
                    from model.hf_loader import HFModelLoader
                    from model.resource_detector import ResourceDetector
                    from model.auto_partitioner import AutoPartitioner
                    loader = HFModelLoader(model_name, dtype=dtype)
                    nodes = ResourceDetector(mode="local").scan()
                    plan = AutoPartitioner().create_plan(loader, nodes)
                    st.code(plan.summary())
                    if plan.feasible:
                        st.success("Model fits in available resources!")
                    else:
                        st.error(f"Partition not feasible: {plan.error_message}")
                except Exception as e:
                    st.error(f"Partition error: {e}")

    st.markdown("---")

    # --- Generate ---
    _init("run_generating", False)
    _init("run_output", "")
    _init("run_error", "")

    if st.button("Generate", type="primary", disabled=st.session_state["run_generating"]):
        st.session_state["run_generating"] = True
        st.session_state["run_output"] = ""
        st.session_state["run_error"] = ""

        def _generate_thread():
            try:
                import torch
                from model.hf_loader import HFModelLoader
                from model.layer_chunker import LayerChunker
                from model.generation import DistributedGenerator

                dev = device
                if dev == "auto":
                    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

                loader = HFModelLoader(model_name, dtype=dtype)
                chunker = LayerChunker(loader)
                chunks = chunker.create_chunks(num_chunks)

                # Load weights
                try:
                    from transformers import AutoModelForCausalLM
                    real_model = AutoModelForCausalLM.from_pretrained(
                        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True,
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
                        chunk.to(dev)
                        chunk.eval()
                except Exception:
                    for chunk in chunks:
                        chunk.to(dev)
                        chunk.eval()

                tokenizer = loader.get_tokenizer()
                gen = DistributedGenerator(chunks, tokenizer, device=dev)

                if stream:
                    output_text = ""
                    for token in gen.generate_stream(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=rep_penalty,
                    ):
                        output_text += token
                        st.session_state["run_output"] = output_text
                else:
                    result = gen.generate(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=rep_penalty,
                    )
                    st.session_state["run_output"] = result

            except Exception as e:
                st.session_state["run_error"] = str(e)
            finally:
                st.session_state["run_generating"] = False

        t = threading.Thread(target=_generate_thread, daemon=True)
        t.start()
        st.rerun()

    # --- Display output ---
    if st.session_state["run_generating"]:
        st.info("Generating...")
        time.sleep(1)
        st.rerun()

    if st.session_state.get("run_output"):
        st.subheader("Output")
        st.markdown(f"```\n{st.session_state['run_output']}\n```")

    if st.session_state.get("run_error"):
        st.error(f"Generation error: {st.session_state['run_error']}")


# -------------------------------------------------------------------
# Page 3: Cluster Setup
# -------------------------------------------------------------------

def _augment_k8s_with_local_gpu(summary: Dict[str, Any]) -> Dict[str, Any]:
    """When K8s reports no GPUs (e.g. Docker Desktop), augment nodes with local GPU info.

    Docker Desktop Kubernetes doesn't expose nvidia.com/gpu resources unless
    the NVIDIA device plugin DaemonSet is installed. This helper detects the
    local GPU via NVML and patches the K8s node data so the dashboard shows
    accurate GPU information.
    """
    if summary.get("gpu_nodes", 0) > 0:
        return summary  # already has GPU info

    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        if gpu_count == 0:
            pynvml.nvmlShutdown()
            return summary

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_vram_mb = mem_info.total / (1024 ** 2)
        gpu_type = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_type, bytes):
            gpu_type = gpu_type.decode("utf-8")
        pynvml.nvmlShutdown()
    except Exception:
        return summary  # no local GPU either

    # Patch the first node (Docker Desktop = single node running locally)
    nodes = summary.get("nodes", [])
    if nodes:
        nodes[0]["gpu_type"] = gpu_type
        nodes[0]["gpu_vram_mb"] = gpu_vram_mb
        usable_vram = max(0, gpu_vram_mb - 500)
        nodes[0]["usable_mb"] = usable_vram

    summary["gpu_nodes"] = 1
    summary["cpu_only_nodes"] = max(0, summary.get("cpu_only_nodes", 1) - 1)
    summary["total_gpu_vram_mb"] = gpu_vram_mb
    summary["total_usable_mb"] = sum(n.get("usable_mb", 0) for n in nodes)
    return summary


def page_cluster_setup():
    st.title("Cluster Setup")

    col_mode, col_btn, _ = st.columns([1, 1, 4])
    with col_mode:
        scan_mode = st.radio("Scan mode", ["local", "kubernetes"], key="cluster_mode")
    with col_btn:
        st.markdown("")  # spacer
        do_scan = st.button("Scan Resources")

    if do_scan:
        with st.spinner(f"Scanning ({scan_mode})..."):
            try:
                from model.resource_detector import ResourceDetector
                det = ResourceDetector(mode=scan_mode)
                summary = det.scan_summary()
                # For K8s mode: augment with local GPU when K8s doesn't expose GPU resources
                if scan_mode == "kubernetes":
                    summary = _augment_k8s_with_local_gpu(summary)
                st.session_state["cluster_summary"] = summary
            except Exception as e:
                st.error(f"Scan failed: {e}")
                return

    summary = st.session_state.get("cluster_summary")
    if not summary:
        st.info("Press 'Scan Resources' to discover nodes.")
        return

    # --- Aggregate metrics ---
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Nodes", summary.get("num_nodes", 0))
    mc2.metric("GPU Nodes", summary.get("gpu_nodes", 0))
    mc3.metric("Total VRAM", f"{summary.get('total_gpu_vram_mb', 0):.0f} MB")
    mc4.metric("Total RAM", f"{summary.get('total_ram_mb', 0):.0f} MB")

    # --- Docker Desktop hint ---
    nodes = summary.get("nodes", [])
    is_docker_desktop = any(n.get("name") == "docker-desktop" for n in nodes)
    if is_docker_desktop:
        st.info(
            "**Docker Desktop detected.** GPU info was augmented from your local machine via NVML. "
            "For native K8s GPU scheduling, install the "
            "[NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin)."
        )

    # --- Node table ---
    st.subheader("Discovered Nodes")
    if nodes:
        st.dataframe(pd.DataFrame(nodes), hide_index=True)

    # --- Model compatibility ---
    st.subheader("Model Compatibility")
    total_usable = summary.get("total_usable_mb", 0)
    compat_rows = []
    for name, size_mb in MODEL_SIZES_MB.items():
        fits = "Yes" if size_mb <= total_usable else "No"
        compat_rows.append({"Model": name, "Size (MB)": size_mb, "Fits?": fits})
    st.dataframe(pd.DataFrame(compat_rows), hide_index=True)

    # --- Custom model check ---
    st.markdown("---")
    st.subheader("Check Custom Model")
    custom_model = st.text_input("Enter HuggingFace model name", key="cluster_custom_check")
    if st.button("Check Compatibility", key="cluster_check_btn") and custom_model:
        with st.spinner("Loading model metadata..."):
            try:
                from model.hf_loader import HFModelLoader
                loader = HFModelLoader(custom_model, dtype="float16")
                info = loader.get_model_size_estimate()
                fp16_mb = info.get("float16_mb", 0)
                if fp16_mb <= total_usable:
                    st.success(f"{custom_model}: {fp16_mb:.0f} MB (FP16) - FITS in cluster ({total_usable:.0f} MB usable)")
                else:
                    st.error(f"{custom_model}: {fp16_mb:.0f} MB (FP16) - DOES NOT FIT ({total_usable:.0f} MB usable)")
            except Exception as e:
                st.error(f"Error: {e}")


# -------------------------------------------------------------------
# Page 4: Kubernetes Deploy
# -------------------------------------------------------------------

def page_k8s_deploy():
    st.title("Kubernetes Deployment")

    # ---- Step 1: Prepare Weights ----
    st.header("Step 1: Prepare Weights")
    with st.form("prepare_form"):
        pc1, pc2 = st.columns(2)
        with pc1:
            prep_model = st.text_input("Model", "sshleifer/tiny-gpt2", key="k8s_prep_model")
            prep_chunks = st.number_input("Num chunks", 1, 8, 3, key="k8s_prep_chunks")
        with pc2:
            prep_dtype = st.selectbox("Dtype", ["float16", "bfloat16", "float32"], key="k8s_prep_dtype")
            prep_outdir = st.text_input("Output dir", "data/chunks", key="k8s_prep_outdir")
        prep_submit = st.form_submit_button("Prepare Weights")

    if prep_submit:
        cmd = [
            sys.executable, "kai_cli.py", "prepare",
            "--model", prep_model,
            "--num-chunks", str(prep_chunks),
            "--dtype", prep_dtype,
            "--output-dir", prep_outdir,
        ]
        _run_subprocess(cmd, "k8s_prep_log", "k8s_prep_proc")
        st.rerun()

    if _proc_is_running("k8s_prep_proc"):
        st.info("Preparing weights...")
        st.code(st.session_state.get("k8s_prep_log", ""), language="text")
        time.sleep(2)
        st.rerun()
    elif st.session_state.get("k8s_prep_log"):
        proc = st.session_state.get("k8s_prep_proc")
        if proc and proc.returncode == 0:
            st.success("Weight preparation complete!")
        elif proc:
            st.error(f"Preparation failed (exit code {proc.returncode})")
        st.code(st.session_state.get("k8s_prep_log", ""), language="text")

    st.markdown("---")

    # ---- Step 2: Build Docker Images ----
    st.header("Step 2: Build Docker Images")
    with st.form("build_form"):
        bc1, bc2 = st.columns(2)
        with bc1:
            build_tag = st.text_input("Image tag", "kai:latest", key="k8s_build_tag")
        with bc2:
            build_push = st.checkbox("Push after build", key="k8s_build_push")
        build_submit = st.form_submit_button("Build Images")

    if build_submit:
        cmd = [sys.executable, "kai_cli.py", "build", "--tag", build_tag]
        if build_push:
            cmd.append("--push")
        _run_subprocess(cmd, "k8s_build_log", "k8s_build_proc")
        st.rerun()

    if _proc_is_running("k8s_build_proc"):
        st.info("Building Docker images...")
        st.code(st.session_state.get("k8s_build_log", ""), language="text")
        time.sleep(2)
        st.rerun()
    elif st.session_state.get("k8s_build_log"):
        proc = st.session_state.get("k8s_build_proc")
        if proc and proc.returncode == 0:
            st.success("Build complete!")
        elif proc:
            st.error(f"Build failed (exit code {proc.returncode})")
        st.code(st.session_state.get("k8s_build_log", ""), language="text")

    st.markdown("---")

    # ---- Step 3: Deploy ----
    st.header("Step 3: Deploy Pipeline")
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        deploy_chunks = st.number_input("Num chunks", 1, 8, 3, key="k8s_deploy_chunks")
    with dc2:
        deploy_model_type = st.selectbox("Model type", ["transformer", "cnn"], key="k8s_deploy_model")
    with dc3:
        st.markdown("")
        if st.button("Deploy All", type="primary", key="k8s_deploy_btn"):
            with st.spinner("Deploying to Kubernetes..."):
                try:
                    _load_controller = _get_controller()
                    _load_controller.deploy_all(
                        num_chunks=deploy_chunks,
                        model_type=deploy_model_type,
                    )
                    st.success("Deployment complete!")
                except Exception as e:
                    st.error(f"Deploy failed: {e}")

    st.markdown("---")

    # ---- Step 4: Status ----
    st.header("Step 4: Pod Status")
    if st.button("Refresh Status", key="k8s_status_btn"):
        try:
            ctrl = _get_controller()
            status = ctrl.get_status()
            st.session_state["k8s_status"] = status
        except Exception as e:
            st.error(f"Status check failed: {e}")

    status = st.session_state.get("k8s_status")
    if status:
        pods = status.get("pods", [])
        if pods:
            st.dataframe(pd.DataFrame(pods), hide_index=True)
        else:
            st.info("No KAI pods found.")

    st.markdown("---")

    # ---- Step 5: Teardown ----
    st.header("Step 5: Teardown")
    st.warning("This will delete all KAI resources from the cluster.")
    if st.button("Teardown All", key="k8s_teardown_btn"):
        try:
            ctrl = _get_controller()
            ctrl.teardown()
            st.success("Teardown complete!")
            st.session_state.pop("k8s_status", None)
        except Exception as e:
            st.error(f"Teardown failed: {e}")


def _get_controller():
    """Lazy-load and cache KAIController."""
    if "k8s_controller" not in st.session_state:
        import importlib.util as _ilu
        _controller_path = os.path.join(_PROJECT_ROOT, "kubernetes", "controller.py")
        _spec = _ilu.spec_from_file_location("kai_controller", _controller_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        st.session_state["k8s_controller"] = _mod.KAIController()
    return st.session_state["k8s_controller"]


# -------------------------------------------------------------------
# Page 5: Benchmark
# -------------------------------------------------------------------

def page_benchmark():
    st.title("Benchmark & Energy Analysis")

    # --- Config ---
    st.subheader("Configuration")

    col1, col2 = st.columns(2)
    with col1:
        bench_mode = st.selectbox("Mode", ["local", "kubernetes", "both"], key="bench_mode")
        bench_model = st.selectbox("Model type", ["transformer", "cnn"], key="bench_model")
        bench_hf = st.text_input("HF model (optional, overrides model type)", key="bench_hf")
    with col2:
        bench_iter = st.number_input("Iterations", 1, 500, 10, key="bench_iter")
        bench_batch = st.number_input("Batch size", 1, 64, 8, key="bench_batch")
        bench_chunks = st.number_input("Num chunks (K8s)", 1, 8, 2, key="bench_chunks")

    col3, col4 = st.columns(2)
    with col3:
        bench_rate = st.slider("Sampling rate (s)", 0.1, 5.0, 1.0, 0.1, key="bench_rate")
        bench_outdir = st.text_input("Output dir", "logs", key="bench_outdir")
    with col4:
        bench_deas = st.checkbox("Enable DEAS", key="bench_deas")
        if bench_deas:
            bench_cooldown = st.number_input("DEAS cooldown (s)", 5.0, 120.0, 30.0, key="bench_cooldown")
        else:
            bench_cooldown = 30.0

    st.markdown("---")

    # --- Run ---
    _init("bench_running", False)

    if st.button("Run Benchmark", type="primary", disabled=st.session_state.get("bench_running", False)):
        cmd = [
            sys.executable, "kai_cli.py", "benchmark",
            "--mode", bench_mode,
            "--model", bench_model,
            "--num-chunks", str(bench_chunks),
            "--iterations", str(bench_iter),
            "--batch-size", str(bench_batch),
            "--output-dir", bench_outdir,
            "--sampling-rate", str(bench_rate),
        ]
        if bench_hf:
            cmd += ["--hf-model", bench_hf]
        if bench_deas:
            cmd += ["--enable-deas", "--deas-cooldown", str(bench_cooldown)]

        st.session_state["bench_running"] = True
        st.session_state["bench_result"] = None
        _run_subprocess(cmd, "bench_log", "bench_proc")
        st.rerun()

    # --- Process monitoring ---
    if st.session_state.get("bench_running"):
        if _proc_is_running("bench_proc"):
            st.info("Benchmark running...")
            st.code(st.session_state.get("bench_log", ""), language="text")
            time.sleep(2)
            st.rerun()
        else:
            st.session_state["bench_running"] = False
            proc = st.session_state.get("bench_proc")
            if proc and proc.returncode == 0:
                st.success("Benchmark complete!")
                # Load the newest result file
                result_files = _find_result_files(bench_outdir)
                if result_files:
                    st.session_state["bench_result"] = _load_json(result_files[0])
            else:
                st.error(f"Benchmark failed (exit code {proc.returncode if proc else '?'})")
            st.code(st.session_state.get("bench_log", ""), language="text")

    # --- Results ---
    result = st.session_state.get("bench_result")
    if result:
        st.markdown("---")
        st.subheader("Results")

        mode = _detect_mode(result)
        if mode == "combined":
            local_r = result.get("local", {})
            k8s_r = result.get("kubernetes", {})
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Local Latency", f"{local_r.get('avg_latency_ms', 0):.2f} ms")
            rc2.metric("K8s Latency", f"{k8s_r.get('avg_e2e_latency_ms', 0):.2f} ms")
            rc3.metric("K8s Throughput", f"{k8s_r.get('throughput_inferences_per_sec', 0):.2f} inf/s")
        elif mode == "local":
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Avg Latency", f"{result.get('avg_latency_ms', 0):.2f} ms")
            rc2.metric("Throughput", f"{result.get('throughput_inferences_per_sec', 0):.2f} inf/s")
            rc3.metric("Avg Power", f"{result.get('avg_power_w', 0):.2f} W")
            rc4.metric("Energy", f"{result.get('total_energy_wh', 0):.6f} Wh")
        elif mode == "kubernetes":
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Avg E2E Latency", f"{result.get('avg_e2e_latency_ms', 0):.2f} ms")
            rc2.metric("Throughput", f"{result.get('throughput_inferences_per_sec', 0):.2f} inf/s")
            rc3.metric("Total Power", f"{result.get('total_avg_power_w', 0):.2f} W")
            rc4.metric("Energy", f"{result.get('total_energy_wh', 0):.6f} Wh")

        with st.expander("Raw Result JSON"):
            st.json(result)


# -------------------------------------------------------------------
# Page 6: Monitor
# -------------------------------------------------------------------

def page_monitor():
    st.title("Live GPU Monitor")

    _init("mon_active", False)

    col1, col2, col3 = st.columns(3)
    with col1:
        mon_interval = st.slider("Sampling interval (s)", 0.1, 5.0, 1.0, 0.1, key="mon_interval")
    with col2:
        enable_threshold = st.checkbox("Enable threshold service", key="mon_threshold")
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=True, key="mon_auto_refresh")
        refresh_rate = st.selectbox("Refresh (s)", [1, 2, 5, 10], index=1, key="mon_refresh_rate")

    # --- Start / Stop ---
    scol1, scol2, _ = st.columns([1, 1, 4])
    with scol1:
        if st.button("Start Monitoring", disabled=st.session_state.get("mon_active", False)):
            try:
                from monitoring.metrics import MetricsCollector
                collector = MetricsCollector(
                    interval=mon_interval,
                    enable_threshold=enable_threshold,
                )
                collector.start()
                st.session_state["mon_collector"] = collector
                st.session_state["mon_active"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start monitoring: {e}")
    with scol2:
        if st.button("Stop Monitoring", disabled=not st.session_state.get("mon_active", False)):
            collector = st.session_state.get("mon_collector")
            if collector:
                collector.stop()
            st.session_state["mon_active"] = False
            st.session_state.pop("mon_collector", None)
            st.rerun()

    st.markdown("---")

    # --- Live data ---
    collector = st.session_state.get("mon_collector")
    if not collector:
        st.info("Press 'Start Monitoring' to begin collecting GPU metrics.")
        return

    # Read GPU samples
    gpu_mon = getattr(collector, "_gpu_monitor", None)
    if gpu_mon:
        samples = gpu_mon.get_recent_samples(60)
    else:
        samples = []

    if samples:
        latest = samples[-1]

        # Metric cards
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Power", f"{latest.get('power_w', 0):.1f} W")
        mc2.metric("Utilization", f"{latest.get('gpu_utilization_pct', 0):.1f}%")
        mc3.metric("Temperature", f"{latest.get('gpu_temperature_c', 0):.0f} C")
        mc4.metric("Memory Used", f"{latest.get('gpu_memory_used_mb', 0):.0f} MB")

        # Threshold status
        if enable_threshold:
            ts = getattr(collector, "_threshold_service", None)
            if ts:
                level = ts.get_current_level()
                if level:
                    level_str = level.value
                    if level_str == "optimal":
                        st.success(f"Threshold: {level_str.upper()}")
                    elif level_str == "warning":
                        st.warning(f"Threshold: {level_str.upper()}")
                    else:
                        st.error(f"Threshold: {level_str.upper()}")

        # Charts
        chart_col1, chart_col2 = st.columns(2)

        power_data = [s.get("power_w", 0) for s in samples]
        util_data = [s.get("gpu_utilization_pct", 0) for s in samples]
        temp_data = [s.get("gpu_temperature_c", 0) for s in samples]

        with chart_col1:
            st.subheader("Power (W)")
            st.line_chart(pd.DataFrame({"Power (W)": power_data}))

            st.subheader("Temperature (C)")
            st.line_chart(pd.DataFrame({"Temp (C)": temp_data}))

        with chart_col2:
            st.subheader("GPU Utilization (%)")
            st.line_chart(pd.DataFrame({"Utilization (%)": util_data}))

            st.subheader("Memory (MB)")
            mem_data = [s.get("gpu_memory_used_mb", 0) for s in samples]
            st.line_chart(pd.DataFrame({"Memory (MB)": mem_data}))

        # Event history
        if enable_threshold:
            bus = getattr(collector, "event_bus", None)
            if bus:
                events = bus.get_event_history(20)
                if events:
                    st.subheader("Recent Threshold Events")
                    event_rows = []
                    for evt in events:
                        d = evt.to_dict() if hasattr(evt, "to_dict") else {"event": str(evt)}
                        event_rows.append(d)
                    st.dataframe(pd.DataFrame(event_rows), hide_index=True)
    else:
        st.info("Waiting for GPU samples...")

    # Auto-refresh
    if auto_refresh and st.session_state.get("mon_active"):
        time.sleep(refresh_rate)
        st.rerun()


# -------------------------------------------------------------------
# Page 7: Analysis
# -------------------------------------------------------------------

def page_analysis():
    st.title("KAI - Experiment Analysis")

    # --- Sidebar controls for this page ---
    result_files = _find_result_files(LOGS_DIR)

    if not result_files:
        st.info(
            f"No result files found in `{LOGS_DIR}/`. "
            "Run a benchmark first from the Benchmark page."
        )
        return

    file_labels = [os.path.basename(f) for f in result_files]
    selected_idx = st.selectbox(
        "Select result file",
        range(len(file_labels)),
        format_func=lambda i: file_labels[i],
        key="analysis_file",
    )
    selected_file = result_files[selected_idx]
    data = _load_json(selected_file)
    detected_mode = _detect_mode(data)

    st.caption(f"**Mode:** {detected_mode} | **File:** `{os.path.basename(selected_file)}`")

    # Split data
    local_data: Optional[Dict[str, Any]] = None
    k8s_data: Optional[Dict[str, Any]] = None

    if detected_mode == "combined":
        local_data = data.get("local")
        k8s_data = data.get("kubernetes")
    elif detected_mode == "local":
        local_data = data
    elif detected_mode == "kubernetes":
        k8s_data = data

    # Comparison loader
    compare_enabled = st.checkbox("Load second file for comparison", key="analysis_compare")
    if compare_enabled and len(result_files) > 1:
        other_idx = st.selectbox(
            "Compare with",
            range(len(file_labels)),
            format_func=lambda i: file_labels[i],
            key="analysis_compare_file",
        )
        other_data = _load_json(result_files[other_idx])
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

    mode_filter = st.radio("Display mode", ["Both", "Local only", "K8s only"], key="analysis_filter", horizontal=True)
    if mode_filter == "Local only":
        k8s_data = None
    elif mode_filter == "K8s only":
        local_data = None

    has_local = local_data is not None
    has_k8s = k8s_data is not None

    if not has_local and not has_k8s:
        st.warning("No data to display.")
        return

    st.markdown("---")

    # --- Summary Metrics ---
    st.header("Summary Metrics")

    if has_local and has_k8s:
        _analysis_summary_both(local_data, k8s_data)
    elif has_local:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Latency", f"{local_data.get('avg_latency_ms', 0):.2f} ms")
        c2.metric("Throughput", f"{local_data.get('throughput_inferences_per_sec', 0):.2f} inf/s")
        c3.metric("Avg Power", f"{local_data.get('avg_power_w', 0):.2f} W")
        c4.metric("Energy", f"{local_data.get('total_energy_wh', 0):.6f} Wh")
    elif has_k8s:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg E2E Latency", f"{k8s_data.get('avg_e2e_latency_ms', 0):.2f} ms")
        c2.metric("Throughput", f"{k8s_data.get('throughput_inferences_per_sec', 0):.2f} inf/s")
        c3.metric("Total Power", f"{k8s_data.get('total_avg_power_w', 0):.2f} W")
        c4.metric("Energy", f"{k8s_data.get('total_energy_wh', 0):.6f} Wh")

    # --- GPU Power ---
    st.header("GPU Power Consumption Over Time")
    power_chart_data = _collect_power_chart_data(local_data, k8s_data)
    if power_chart_data:
        max_len = max(len(v) for v in power_chart_data.values())
        chart_df = pd.DataFrame({
            k: v + [None] * (max_len - len(v)) for k, v in power_chart_data.items()
        })
        st.line_chart(chart_df)
    else:
        st.info("No GPU power sample data available.")

    # --- GPU Utilization ---
    st.header("GPU Utilization Over Time")
    util_chart_data = _collect_util_chart_data(local_data, k8s_data)
    if util_chart_data:
        max_len = max(len(v) for v in util_chart_data.values())
        chart_df = pd.DataFrame({
            k: v + [None] * (max_len - len(v)) for k, v in util_chart_data.items()
        })
        st.line_chart(chart_df)
    else:
        st.info("No GPU utilization data available.")

    # --- Latency ---
    st.header("Latency Comparison")
    if has_local and has_k8s:
        lat_c1, lat_c2 = st.columns(2)
        with lat_c1:
            st.subheader("Average Latency")
            lat_df = pd.DataFrame({
                "Mode": ["Local", "Kubernetes"],
                "Latency (ms)": [
                    local_data.get("avg_latency_ms", 0),
                    k8s_data.get("avg_e2e_latency_ms", 0),
                ],
            })
            st.bar_chart(lat_df.set_index("Mode"))
        with lat_c2:
            st.subheader("Latency Distribution")
            dist = {}
            ll = local_data.get("inference_latencies_ms", [])
            kl = k8s_data.get("e2e_latencies_ms", [])
            if ll or kl:
                ml = max(len(ll), len(kl))
                if ll:
                    dist["Local"] = ll + [None] * (ml - len(ll))
                if kl:
                    dist["Kubernetes"] = kl + [None] * (ml - len(kl))
                st.line_chart(pd.DataFrame(dist))
    elif has_local:
        lats = local_data.get("inference_latencies_ms", [])
        if lats:
            st.line_chart(pd.DataFrame({"Local Latency (ms)": lats}))
    elif has_k8s:
        lats = k8s_data.get("e2e_latencies_ms", [])
        if lats:
            st.line_chart(pd.DataFrame({"K8s E2E Latency (ms)": lats}))

    # --- Energy ---
    st.header("Energy Comparison")
    if has_local and has_k8s:
        ec1, ec2 = st.columns(2)
        with ec1:
            st.subheader("Total Energy")
            edf = pd.DataFrame({
                "Mode": ["Local", "Kubernetes"],
                "Energy (Wh)": [
                    local_data.get("total_energy_wh", 0),
                    k8s_data.get("total_energy_wh", 0),
                ],
            })
            st.bar_chart(edf.set_index("Mode"))
        with ec2:
            st.subheader("Energy per Inference")
            epdf = pd.DataFrame({
                "Mode": ["Local", "Kubernetes"],
                "Wh": [
                    local_data.get("energy_per_inference_wh", 0),
                    k8s_data.get("energy_per_inference_wh", 0),
                ],
            })
            st.bar_chart(epdf.set_index("Mode"))
    elif has_local:
        c1, c2 = st.columns(2)
        c1.metric("Total Energy", f"{local_data.get('total_energy_wh', 0):.6f} Wh")
        c2.metric("Energy/Inference", f"{local_data.get('energy_per_inference_wh', 0):.8f} Wh")
    elif has_k8s:
        c1, c2 = st.columns(2)
        c1.metric("Total Energy", f"{k8s_data.get('total_energy_wh', 0):.6f} Wh")
        c2.metric("Energy/Inference", f"{k8s_data.get('energy_per_inference_wh', 0):.8f} Wh")

    # --- Per-Chunk Latency ---
    if has_k8s:
        chunk_lats = k8s_data.get("per_chunk_avg_latency_ms", [])
        if chunk_lats:
            st.header("Per-Chunk Latency (Kubernetes)")
            cdf = pd.DataFrame({
                "Chunk": [f"Chunk {i}" for i in range(len(chunk_lats))],
                "Avg Latency (ms)": chunk_lats,
            })
            st.bar_chart(cdf.set_index("Chunk"))
            st.caption(f"Network overhead: {k8s_data.get('avg_network_overhead_ms', 0):.2f} ms")

    # --- Migration Energy ---
    if has_k8s:
        mig_events = k8s_data.get("migration_events", [])
        if mig_events:
            st.header("Migration Energy Impact")
            event_rows = []
            for evt in mig_events:
                event_rows.append({
                    "Chunk": evt.get("chunk_id", "N/A"),
                    "Source": evt.get("source_node", "N/A"),
                    "Target": evt.get("target_node", "N/A"),
                    "Duration (ms)": f"{evt.get('duration_ms', 0):.1f}",
                    "Reason": evt.get("reason", "N/A"),
                })
            st.dataframe(pd.DataFrame(event_rows), hide_index=True)
            total_me = k8s_data.get("total_migration_energy_wh", 0)
            st.caption(f"Total migration energy: {total_me:.6f} Wh across {len(mig_events)} migration(s)")

    # --- VRAM vs RAM ---
    if has_k8s:
        off = k8s_data.get("offloading_stats", {})
        gl = off.get("gpu_latencies_ms", [])
        cl = off.get("cpu_latencies_ms", [])
        if gl and cl:
            st.header("VRAM vs RAM Execution Trade-off")
            nc = min(len(gl), len(cl))
            tdf = pd.DataFrame({
                "GPU (VRAM)": gl[:nc],
                "CPU (RAM/Disk)": cl[:nc],
            }, index=[f"Chunk {i}" for i in range(nc)])
            st.bar_chart(tdf)
            ms = off.get("memory_saved_mb", [])
            if ms:
                st.caption(f"Total VRAM saved: {sum(ms):.0f} MB")

    # --- Config & Raw ---
    with st.expander("Experiment Configuration"):
        st.json(data.get("experiment_config", data.get("local", {}).get("experiment_config", {})))

    with st.expander("Raw JSON Data"):
        st.json(data)


# Analysis helper functions

def _analysis_summary_both(local_data, k8s_data):
    """Render comparison summary table."""
    def _ratio(a, b):
        return f"{a / b:.2f}" if b != 0 else "N/A"

    rows = {
        "Metric": [
            "Avg Latency (ms)", "Throughput (inf/s)", "Avg GPU Power (W)",
            "Total Energy (Wh)", "Energy/Inference (Wh)", "Network Overhead (ms)",
        ],
        "Local": [
            f"{local_data.get('avg_latency_ms', 0):.2f}",
            f"{local_data.get('throughput_inferences_per_sec', 0):.2f}",
            f"{local_data.get('avg_power_w', 0):.2f}",
            f"{local_data.get('total_energy_wh', 0):.6f}",
            f"{local_data.get('energy_per_inference_wh', 0):.8f}",
            "0.00",
        ],
        "Kubernetes": [
            f"{k8s_data.get('avg_e2e_latency_ms', 0):.2f}",
            f"{k8s_data.get('throughput_inferences_per_sec', 0):.2f}",
            f"{k8s_data.get('total_avg_power_w', 0):.2f}",
            f"{k8s_data.get('total_energy_wh', 0):.6f}",
            f"{k8s_data.get('energy_per_inference_wh', 0):.8f}",
            f"{k8s_data.get('avg_network_overhead_ms', 0):.2f}",
        ],
        "Ratio (K8s/Local)": [
            _ratio(k8s_data.get("avg_e2e_latency_ms", 0), local_data.get("avg_latency_ms", 1)),
            _ratio(k8s_data.get("throughput_inferences_per_sec", 0), local_data.get("throughput_inferences_per_sec", 1)),
            _ratio(k8s_data.get("total_avg_power_w", 0), local_data.get("avg_power_w", 1)),
            _ratio(k8s_data.get("total_energy_wh", 0), local_data.get("total_energy_wh", 1)),
            _ratio(k8s_data.get("energy_per_inference_wh", 0), local_data.get("energy_per_inference_wh", 1)),
            "N/A",
        ],
    }
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def _collect_power_chart_data(local_data, k8s_data):
    data = {}
    if local_data:
        samples = local_data.get("gpu_samples", [])
        if samples:
            data["Local"] = [s.get("power_w", 0) for s in samples]
    if k8s_data:
        for node in k8s_data.get("node_metrics", []):
            ns = node.get("gpu_samples", [])
            if ns:
                label = f"K8s ({node.get('node_name', 'node')})"
                data[label] = [s.get("power_w", 0) for s in ns]
    return data


def _collect_util_chart_data(local_data, k8s_data):
    data = {}
    if local_data:
        samples = local_data.get("gpu_samples", [])
        if samples:
            data["Local"] = [s.get("gpu_utilization_pct", 0) for s in samples]
    if k8s_data:
        for node in k8s_data.get("node_metrics", []):
            ns = node.get("gpu_samples", [])
            if ns:
                label = f"K8s ({node.get('node_name', 'node')})"
                data[label] = [s.get("gpu_utilization_pct", 0) for s in ns]
    return data


# ===================================================================
# SIDEBAR NAVIGATION & DISPATCH
# ===================================================================

st.sidebar.title("KAI Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Run Inference",
        "Cluster Setup",
        "Kubernetes Deploy",
        "Benchmark",
        "Monitor",
        "Analysis",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("KAI - Distributed AI Inference Platform")

# Dispatch
if page == "Home":
    page_home()
elif page == "Run Inference":
    page_run_inference()
elif page == "Cluster Setup":
    page_cluster_setup()
elif page == "Kubernetes Deploy":
    page_k8s_deploy()
elif page == "Benchmark":
    page_benchmark()
elif page == "Monitor":
    page_monitor()
elif page == "Analysis":
    page_analysis()
