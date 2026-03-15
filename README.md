# KAI — Kubernetes AI Distributed Inference Platform

## Project Overview

KAI is a platform that enables running **large AI models on clusters of low-end PCs** using Kubernetes. Each node in the cluster loads only the model layers it is responsible for, so no single machine needs enough VRAM or RAM for the entire model.

### Key Capabilities

- **Distributed LLM Inference** — Run HuggingFace models (GPT-2, Phi-2, LLaMA, Mistral, etc.) split across multiple nodes.
- **Layer-Wise Chunking** — Models are partitioned at transformer block boundaries. Each chunk loads only its weights.
- **Smart Auto-Partitioning** — Automatically detects GPU VRAM and RAM on each node and assigns layers proportionally.
- **Text Generation** — Full autoregressive generation pipeline with temperature, top-k, top-p sampling, and streaming output.
- **Energy Benchmarking** — Measures GPU power draw, CPU usage, and inference latency to compare local vs. Kubernetes deployment costs.
- **Real-Time Energy Instrumentation** — Sub-100ms GPU sampling, ring buffers, TDP auto-detection, trapezoidal energy integration, and power threshold alerts via an async event bus.
- **Dynamic Energy-Aware Scheduling (DEAS)** — Automatically migrates chunks away from overheating nodes using a 5-step Pause/Checkpoint/Migrate/Relink/Resume workflow with configurable cooldown.
- **CPU/Disk Offloading** — FlexGen-style tiered weight management (GPU VRAM → System RAM → Disk) with double-buffered prefetching to hide transfer latency.
- **Single-Command CLI** — `python kai_cli.py run --model <name> --prompt "Hello" --max-tokens 100`
- **Quantization** — Optional 4-bit (NF4) and 8-bit (INT8) quantization via bitsandbytes to reduce memory per chunk.
- **Docker Build & Prepare** — `kai build` builds all Docker images; `kai prepare` downloads, chunks, and saves weights for K8s deployment.

### Quick Example

```bash
# Scan your hardware
python kai_cli.py scan

# Preview how a model would be split across 3 nodes
python kai_cli.py partition --model microsoft/phi-2 --num-nodes 3

# Run distributed inference
python kai_cli.py run --model sshleifer/tiny-gpt2 --prompt "Once upon a time" --max-tokens 50 --stream

# Run with 4-bit quantization to reduce VRAM usage
python kai_cli.py run --model sshleifer/tiny-gpt2 --prompt "Hello" --quantize 4bit

# Prepare chunk weights for K8s deployment
python kai_cli.py prepare --model microsoft/phi-2 --num-chunks 3 --output-dir data/chunks

# Build all Docker images
python kai_cli.py build

# Run energy benchmark (original KAI workflow)
python kai_cli.py benchmark --model transformer --mode local

# Benchmark a HuggingFace model directly
python kai_cli.py benchmark --hf-model sshleifer/tiny-gpt2 --mode local

# Benchmark with high-frequency GPU sampling (100ms)
python kai_cli.py benchmark --hf-model sshleifer/tiny-gpt2 --mode local --sampling-rate 0.1

# Run with CPU/disk offloading for oversized models
python kai_cli.py run --model sshleifer/tiny-gpt2 --prompt "Hello" --offload --gpu-budget-mb 2000
```

---

## Table of Contents

- [Quick Example](#quick-example)
- [How It Works](#how-it-works)
- [Gap Analysis](#gap-analysis)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [CLI Reference](#cli-reference)
- [Quick Start — Local Mode](#quick-start--local-mode)
- [Quick Start — Kubernetes Mode](#quick-start--kubernetes-mode)
- [Analysis & Plotting](#analysis--plotting)
- [Dashboard Usage](#dashboard-usage)
- [Key Metrics](#key-metrics)
- [Technology Stack](#technology-stack)
- [Integration Testing Checklist](#integration-testing-checklist)

---

## How It Works

```
                    ┌─────────────┐
                    │  kai_cli.py │
                    │  (CLI)      │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Node A   │ │ Node B   │ │ Node C   │
        │ (GPU 4GB)│ │ (GPU 6GB)│ │ (CPU)    │
        │          │ │          │ │          │
        │ Embed    │ │ Layer 8  │ │ Layer 20 │
        │ Layer 0-7│ │ - 19     │ │ - 31     │
        │          │ │          │ │ Norm     │
        │          │ │          │ │ LM Head  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │  gRPC       │  gRPC      │
             └─────────────┴────────────┘
                    Pipeline
```

1. **HFModelLoader** loads a HuggingFace model's layer structure without loading full weights.
2. **ResourceDetector** scans cluster nodes for GPU VRAM, RAM, and CPU cores.
3. **AutoPartitioner** assigns layers to nodes proportional to their memory.
4. **LayerChunker** creates chunks that each load only their assigned weights.
5. **DistributedGenerator** chains chunks via gRPC for autoregressive text generation.

---

## Gap Analysis

### Problem Statement

The adoption of Kubernetes for AI/ML inference workloads is increasing rapidly, driven by scalability, fault tolerance, and resource management benefits. However, there is a significant lack of empirical data on the energy overhead introduced by Kubernetes orchestration compared to running inference directly on bare metal or a single GPU. Organizations making deployment decisions have no standardized way to quantify this trade-off.

### Identified Gaps

#### 1. Lack of Energy-Aware Benchmarking Tools for AI Inference

Most existing AI benchmarking frameworks (MLPerf, AI Benchmark) focus on throughput and latency. They do not measure or report energy consumption as a first-class metric. There is no widely available tool that correlates GPU power draw with inference performance across different deployment topologies.

#### 2. No Standardized Comparison Between Local and Orchestrated Execution

Current literature and tooling treat local GPU inference and Kubernetes-based inference as separate domains. There is no unified platform that runs the exact same model in both environments under identical conditions and produces directly comparable energy and performance metrics.

#### 3. Hidden Costs of Model Distribution Are Unquantified

When a model is split across multiple nodes in Kubernetes:
- Intermediate tensors must be serialized and transmitted over the network (gRPC overhead).
- Each chunk runs in a separate container with its own runtime overhead.
- Kubernetes scheduling, service discovery, and pod lifecycle management consume additional resources.

These costs exist but are rarely measured in isolation or attributed to specific components.

#### 4. GPU Power Monitoring Is Not Integrated Into Inference Pipelines

GPU power consumption data (via NVIDIA NVML) is available but is typically collected ad hoc using command-line tools like `nvidia-smi`. It is not systematically recorded alongside inference metrics (latency, throughput) in a way that enables time-correlated analysis.

#### 5. No Decision Framework for Energy-Efficient Deployment

Organizations deploying AI models lack data to answer questions such as:
- How much additional energy does Kubernetes orchestration consume for a given model?
- At what scale does the energy overhead of distribution become acceptable?
- What is the energy cost per inference in each deployment mode?
- Is the operational flexibility of Kubernetes worth the energy premium?

#### 6. Absence of Reproducible Experiment Infrastructure

Ad hoc benchmarking produces inconsistent results. There is no turnkey system that automates the full cycle of: deploy model, run controlled inference, collect synchronized metrics, tear down environment, and produce analysis — for both local and distributed modes.

### Summary Table

| Gap | Current State | What KAI Addresses |
|-----|--------------|-------------------|
| Energy-aware AI benchmarking | Not available in standard tools | Measures GPU power, energy (Wh), and correlates with latency/throughput |
| Local vs. K8s comparison | No unified platform exists | Runs identical models in both modes under same conditions |
| Distribution overhead quantification | Overhead exists but is unmeasured | Isolates network, serialization, and orchestration costs |
| Integrated GPU power monitoring | Manual, ad hoc nvidia-smi usage | Automated NVML-based sampling synchronized with inference runs |
| Energy-efficient deployment guidance | No data-driven framework | Produces comparative metrics and efficiency ratios |
| Reproducible experiment infrastructure | Ad hoc scripts, inconsistent setups | Automated experiment runner with structured logging and analysis |

---

## Project Goals

### Primary Goals

1. **Quantify the energy cost of AI inference** under local (single-GPU) execution by measuring GPU power draw, total energy consumption, and inference latency.

2. **Quantify the energy cost of AI inference** under Kubernetes-based distributed execution, including all overhead introduced by containerization, orchestration, and network communication.

3. **Produce a direct, controlled comparison** between the two deployment modes using identical models, inputs, and iteration counts.

4. **Build an automated, reproducible experiment pipeline** that handles deployment, execution, monitoring, data collection, teardown, and analysis without manual intervention.

### Secondary Goals

5. **Develop a model chunking mechanism** that splits a PyTorch model into independent layer groups, each deployable as a standalone microservice.

6. **Implement real-time GPU and CPU monitoring** with 1-second sampling resolution, producing time-series data correlated with inference events.

7. **Create a visualization dashboard** for interactive exploration of experiment results.

8. **Generate publication-quality analysis** including derived metrics (energy per inference, performance per watt) and comparative plots.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| NVIDIA GPU | Any CUDA-capable GPU (tested with RTX 3050 Ti) |
| NVIDIA Drivers | Compatible with CUDA 12.x |
| CUDA Toolkit | 12.2 or later |
| Docker | With NVIDIA Container Toolkit (for building images) |
| Kubernetes | Cluster with GPU nodes and NVIDIA device plugin (for K8s mode) |
| Python | 3.10 or later |
| OS | Linux or Windows (tested on Windows with PowerShell) |

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd KAI

# Install Python dependencies
pip install -r requirements.txt
```

The `requirements.txt` includes: PyTorch, pynvml, psutil, grpcio, grpcio-tools, protobuf, kubernetes, streamlit, matplotlib, pandas, numpy.

### Verify Installation

```bash
# Check GPU is accessible
python -c "import pynvml; pynvml.nvmlInit(); print('GPU:', pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0)))"

# Check PyTorch CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Project Structure

```
KAI/
├── model/                        # AI model definitions and chunking
│   ├── transformer.py            #   Transformer encoder model
│   ├── cnn.py                    #   CNN classification model
│   ├── chunker.py                #   Model splitting into N sequential chunks
│   ├── chunk_server.py           #   gRPC server for a single model chunk (+ Pause/Checkpoint/Resume)
│   ├── gateway.py                #   HTTP gateway chaining chunk services (+ /relink, /topology)
│   ├── hf_loader.py              #   HuggingFace model loader (layer extraction)
│   ├── layer_chunker.py          #   Layer-wise model splitting for distributed inference
│   ├── weight_utils.py           #   Partial weight loading from HF checkpoints
│   ├── quantizer.py              #   4-bit/8-bit quantization via bitsandbytes
│   ├── generation.py             #   Autoregressive text generation across chunks (+ offloaded prefetch)
│   ├── resource_detector.py      #   GPU/CPU/RAM detection (local + K8s nodes)
│   ├── auto_partitioner.py       #   Smart layer-to-node assignment
│   ├── deas_scheduler.py         #   Dynamic Energy-Aware Scheduler (Phase 21)
│   ├── tiered_weight_manager.py  #   GPU/RAM/Disk tiered weight placement (Phase 22)
│   └── prefetch_engine.py        #   Double-buffered async weight prefetching (Phase 22)
│
├── monitoring/                   # Power and performance monitoring
│   ├── gpu_monitor.py            #   NVML-based GPU power/util/temp sampling (+ ring buffer, TDP)
│   ├── cpu_monitor.py            #   psutil-based CPU usage sampling
│   ├── metrics.py                #   Unified MetricsCollector (GPU + CPU + latency + trapezoidal energy)
│   ├── monitor_service.py        #   HTTP service wrapping MetricsCollector (+ /threshold, /events)
│   ├── event_bus.py              #   Async pub/sub for power threshold events (Phase 20)
│   └── threshold_service.py      #   TDP-relative power threshold evaluation (Phase 20)
│
├── experiments/                  # Experiment runners
│   ├── local_runner.py           #   Single-GPU inference experiment
│   ├── k8s_runner.py             #   Kubernetes distributed inference experiment
│   └── experiment_runner.py      #   Unified orchestrator (local, k8s, or both)
│
├── analysis/                     # Post-experiment analysis
│   ├── analyzer.py               #   Metrics computation, comparison, migration energy (Phase 23)
│   └── plots.py                  #   Matplotlib visualization (10 plot types incl. migration/offloading)
│
├── dashboard/                    # Web dashboard
│   ├── unified_app.py           #   Unified 7-page Streamlit dashboard (default)
│   └── app.py                    #   Legacy analysis-only Streamlit dashboard
│
├── kubernetes/                   # Kubernetes configuration
│   ├── controller.py             #   Python K8s controller (deploy/status/teardown + DEAS + thresholds)
│   ├── deployments/              #   Deployment YAMLs (chunk, gateway, monitor)
│   ├── services/                 #   Service YAMLs (ClusterIP, NodePort)
│   └── gpu-resource-quota.yaml   #   GPU resource quota for namespace
│
├── docker/                       # Container images
│   ├── Dockerfile.chunk          #   Model chunk microservice (CUDA base)
│   ├── Dockerfile.gateway        #   Inference gateway
│   ├── Dockerfile.monitor        #   Monitoring DaemonSet
│   └── docker-compose.yml        #   Local multi-container testing
│
├── proto/                        # gRPC definitions
│   └── inference.proto           #   InferenceService with Infer + HealthCheck + Pause/Checkpoint/Resume RPCs
│
├── tests/                        # Test suites
│   ├── test_integration.py       #   25 integration tests (Phases 1-13)
│   ├── test_distributed.py       #   30 integration tests (Phases 14-18)
│   ├── test_phase19.py           #   27 integration tests (Phase 19)
│   ├── test_phase20.py           #   ~15 tests (Phase 20: instrumentation, event bus, thresholds)
│   ├── test_phase21.py           #   ~19 tests (Phase 21: EER, DEAS, migration, relinking)
│   ├── test_phase22.py           #   ~14 tests (Phase 22: tiered weights, prefetching, offloading)
│   └── test_phase23.py           #   ~14 tests (Phase 23: validation, analysis, plots)
│
├── logs/                         # Experiment output (JSON)
├── docs/                         # Phase documentation
├── kai_cli.py                    # Unified CLI (run, scan, partition, benchmark, dashboard)
├── requirements.txt
├── BUILD_GUIDE.md
└── README.md
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT RUNNER                        │
│              experiments/experiment_runner.py                    │
│         (CLI: --mode local | kubernetes | both)                 │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
       ┌───────▼───────┐         ┌────────▼────────┐
       │  LOCAL RUNNER  │         │   K8S RUNNER     │
       │ local_runner.py│         │  k8s_runner.py   │
       └───────┬───────┘         └────────┬─────────┘
               │                          │
       ┌───────▼───────┐         ┌────────▼─────────┐
       │  Full Model    │         │  K8s Controller   │
       │  (single GPU)  │         │  controller.py    │
       │                │         │                   │
       │  ┌──────────┐  │         │  ┌─────────────┐  │
       │  │model.eval│  │         │  │Deploy Chunks │  │
       │  │  (GPU)   │  │         │  │Deploy Gateway│  │
       │  └──────────┘  │         │  │Deploy Monitor│  │
       └───────┬───────┘         │  └──────┬──────┘  │
               │                  └─────────┼─────────┘
               │                            │
       ┌───────▼───────┐         ┌──────────▼──────────┐
       │ MetricsCollect │         │   Kubernetes Cluster │
       │ gpu_monitor.py │         │                      │
       │ cpu_monitor.py │         │  ┌──────┐ ┌──────┐  │
       │  metrics.py    │         │  │Chunk0│→│Chunk1│→ │
       └───────┬───────┘         │  └──────┘ └──────┘  │
               │                  │       ↕ gRPC         │
       ┌───────▼───────┐         │  ┌──────────────┐   │
       │  logs/*.json   │         │  │   Gateway     │   │
       └───────────────┘         │  │  (HTTP→gRPC) │   │
                                  │  └──────────────┘   │
                                  │  ┌──────────────┐   │
                                  │  │Monitor (NVML)│   │
                                  │  └──────┬───────┘   │
                                  └─────────┼───────────┘
                                            │
                                  ┌─────────▼───────────┐
                                  │    logs/*.json       │
                                  └─────────────────────┘

                    ┌───────────────────────────────────┐
                    │          ANALYSIS LAYER            │
                    │                                   │
                    │  analyzer.py  →  CSV / JSON       │
                    │  plots.py     →  8 PNG plots      │
                    │  dashboard/   →  Streamlit app    │
                    └───────────────────────────────────┘
```

### Data Flow

1. **Experiment Runner** selects mode and calls the appropriate runner.
2. **Local Runner** loads the full model on one GPU, runs inference N times, collects power/CPU metrics via `MetricsCollector`, and writes results to `logs/`.
3. **K8s Runner** uses the controller to deploy model chunks as gRPC microservices, sends inference requests through the gateway, collects metrics from the monitor DaemonSet, and writes results to `logs/`.
4. **Analysis** loads result JSON files, computes derived metrics (energy, efficiency, percentiles), and produces comparison tables and plots.
5. **Dashboard** provides interactive web-based exploration of all results.

### Communication Protocols

| Path | Protocol | Details |
|------|----------|---------|
| Client → Gateway | HTTP | POST `/infer`, GET `/health`, GET `/topology`, POST `/relink` |
| Gateway → Chunks | gRPC | Binary tensor serialization (256 MB max) |
| Client → Monitor | HTTP | GET `/metrics`, POST `/start`, `/stop`, GET `/metrics/threshold`, `/metrics/events` |
| Chunks ↔ Chunks | gRPC | Sequential pipeline (Chunk 0 → 1 → ... → N) |
| Chunks (migration) | gRPC | `Pause`, `Checkpoint`, `Resume` RPCs for live migration |

---

## Quick Start — Local Mode

Run inference on a single GPU and collect energy metrics:

```bash
# Basic run (Transformer model, 50 iterations)
python -m experiments.local_runner

# Custom configuration
python -m experiments.local_runner \
    --model transformer \
    --iterations 100 \
    --batch-size 16 \
    --warmup 10 \
    --monitor-interval 0.5
```

Results are saved to `logs/local_results_<timestamp>.json` containing:
- Average inference latency (ms)
- Average GPU power (W)
- Total energy consumed (Wh)
- Throughput (inferences/sec)
- Per-sample GPU power and utilization time-series
- Per-sample CPU utilization time-series
- Per-iteration inference latencies

---

## Quick Start — Kubernetes Mode

Deploy the chunked model to a Kubernetes cluster and run distributed inference:

### 1. Build Docker Images

```bash
# Build chunk server image
docker build -f docker/Dockerfile.chunk -t kai-chunk:latest .

# Build gateway image
docker build -f docker/Dockerfile.gateway -t kai-gateway:latest .

# Build monitor image
docker build -f docker/Dockerfile.monitor -t kai-monitor:latest .
```

### 2. Deploy to Kubernetes

```bash
# Deploy all components (3 chunks)
python -m kubernetes.controller deploy --num-chunks 3 --model transformer --wait

# Check status
python -m kubernetes.controller status

# Check gateway health
python -m kubernetes.controller health
```

### 3. Run Experiment

```bash
python -m experiments.k8s_runner \
    --model transformer \
    --num-chunks 3 \
    --iterations 100 \
    --batch-size 16
```

### 4. Teardown

```bash
python -m kubernetes.controller teardown
```

Results are saved to `logs/k8s_results_<timestamp>.json` containing:
- Average end-to-end latency (ms)
- Average gateway latency (ms)
- Average network overhead (ms)
- Total GPU power across all nodes (W)
- Total energy consumed (Wh)
- Per-chunk average latency
- Per-node GPU metrics

---

## Running Both Modes

Execute local and Kubernetes experiments back-to-back with a single command:

```bash
python -m experiments.experiment_runner \
    --mode both \
    --model transformer \
    --num-chunks 3 \
    --iterations 100 \
    --batch-size 16
```

This produces `logs/experiment_<timestamp>.json` with both result sets and a comparison section including ratios (K8s/Local) for all key metrics.

---

## Analysis & Plotting

### Metrics Analysis

```bash
# Analyze a combined experiment file
python -m analysis.analyzer --combined logs/experiment_<timestamp>.json

# Analyze separate files
python -m analysis.analyzer \
    --local logs/local_results_<ts>.json \
    --k8s logs/k8s_results_<ts>.json \
    --output analysis/output
```

The analyzer computes:
- Latency percentiles (p50, p90, p95, p99) and standard deviation
- Energy per inference
- Performance per watt
- Side-by-side comparison with ratio (K8s / Local)
- Exports CSV and JSON summaries

### Plot Generation

```bash
# Generate all plots
python -m analysis.plots --combined logs/experiment_<timestamp>.json --output analysis/output

# Or from separate files
python -m analysis.plots \
    --local logs/local_results_<ts>.json \
    --k8s logs/k8s_results_<ts>.json
```

Ten plot types are generated as PNG files:

| Plot | Description |
|------|-------------|
| Power over time | GPU power draw line chart (both modes overlaid) |
| GPU utilization over time | Utilization percentage line chart |
| Latency comparison | Bar chart of average latency |
| Energy comparison | Bar chart of total energy consumed |
| Energy efficiency | Bar chart of energy per inference |
| Throughput comparison | Bar chart of inference rate |
| Latency distribution | Box plot of per-iteration latencies |
| Per-chunk latency | Bar chart of K8s chunk processing times |
| Migration energy impact | Power timeline with migration windows highlighted (Phase 23) |
| VRAM vs RAM tradeoff | Grouped bar chart of GPU vs CPU execution time per chunk (Phase 23) |

---

## Dashboard Usage

KAI includes a unified 7-page web dashboard for managing every aspect of the platform — from running local AI models to deploying on Kubernetes clusters.

### Launching the Dashboard

```bash
# Default: launches the unified dashboard
python kai_cli.py dashboard

# Specify a custom port
python kai_cli.py dashboard --port 8502

# Legacy analysis-only dashboard
python kai_cli.py dashboard --legacy
```

The dashboard opens in your browser at `http://localhost:8501`.

### Page 1: Home — System Overview

Scans your local machine and displays:

- **GPU** — Name, VRAM (via NVML/pynvml)
- **System RAM** — Total physical memory
- **CPU Cores** — Physical core count
- **Model Compatibility Table** — Shows which popular models (GPT-2 through Llama-2-70B) fit in your available memory
- **Kubernetes Cluster Check** — Expander that probes your K8s cluster (if connected) and shows node resources

**First thing to do:** Click **"Rescan System"** to verify your GPU is detected. If GPU shows "none", install pynvml: `pip install pynvml`.

### Page 2: Run Inference — Generate Text

Run AI models directly from the dashboard:

1. **Select a model** from the dropdown (GPT-2, Phi-2, Gemma-2B, Falcon-7B, Mistral-7B, Llama-2-7B, Qwen2-7B) or enter a custom HuggingFace model name
2. **Load Model Info** — fetches parameter count, FP16/FP32/INT8 size estimates
3. **Configure generation** — prompt, max tokens, temperature, top-k, top-p, repetition penalty
4. **Choose device** — `cpu`, `cuda:0`, or `auto`
5. **Set dtype** — float16, bfloat16, or float32
6. **Offloading options** — Enable FlexGen-style GPU→RAM→Disk offloading for models too large for your GPU
7. **Partition Preview** — See how the model would be split across chunks
8. Click **"Generate"** — output streams in real-time

**Recommended first test:** Select `sshleifer/tiny-gpt2` (smallest model, ~250 MB), set device to `cpu`, and generate a few tokens to verify everything works.

### Page 3: Cluster Setup — Resource Discovery

Scan and inspect cluster resources:

- **Local mode** — Scans your machine via NVML + psutil
- **Kubernetes mode** — Queries K8s API for all nodes, automatically augments GPU info from your local machine when using Docker Desktop (which doesn't expose `nvidia.com/gpu` resources by default)
- **Node Table** — Shows all discovered nodes with GPU type, VRAM, RAM, and usable memory
- **Model Compatibility** — Which models fit in your cluster's combined memory
- **Custom Model Check** — Enter any HuggingFace model name to check if it fits

**Docker Desktop users:** The dashboard auto-detects Docker Desktop and augments K8s node data with your local GPU info via NVML. For production K8s clusters, install the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) for native GPU scheduling.

### Page 4: Kubernetes Deploy — Full Deployment Pipeline

Five-step deployment workflow:

1. **Prepare Weights** — Downloads a HuggingFace model, splits it into N chunks, saves weight files. Configure model name, chunk count, dtype, and output directory.
2. **Build Docker Images** — Builds chunk server, gateway, and monitor Docker images. Optionally pushes to a registry.
3. **Deploy Pipeline** — Deploys all K8s resources (chunk pods, gateway, monitor DaemonSet). Configure chunk count and model type.
4. **Pod Status** — Shows current status of all KAI pods in the cluster.
5. **Teardown** — Deletes all KAI resources from the cluster.

Each step shows real-time subprocess output so you can monitor progress.

### Page 5: Benchmark — Energy Analysis

Run controlled experiments comparing local vs. Kubernetes inference:

- **Mode** — `local`, `kubernetes`, or `both`
- **Model type** — `transformer` or `cnn` (built-in models), or specify a HuggingFace model name
- **Iterations** — Number of inference runs (more = more statistically reliable)
- **Batch size** — Input tensor batch size
- **Sampling rate** — GPU power sampling interval (0.1s = 100ms for high-frequency data)
- **DEAS** — Enable Dynamic Energy-Aware Scheduling with configurable cooldown
- **Output** — Results saved as JSON in the specified output directory

Click **"Run Benchmark"** and watch real-time logs. Results appear with metric cards (latency, throughput, power, energy) and raw JSON.

### Page 6: Monitor — Live GPU Metrics

Real-time GPU monitoring dashboard:

- **Start/Stop** monitoring with configurable sampling interval
- **Threshold service** — Enables power threshold alerts (optimal/warning/critical based on TDP)
- **Live metric cards** — Power (W), Utilization (%), Temperature (C), Memory Used (MB)
- **Live charts** — Power, temperature, utilization, and memory usage over time (last 60 samples)
- **Event history** — Recent threshold crossing events (when threshold service is enabled)
- **Auto-refresh** — Configurable 1-10 second refresh rate

### Page 7: Analysis — Experiment Results

Interactive visualization of benchmark results:

- **File selector** — Pick any result JSON from the logs directory
- **Mode detection** — Automatically detects local, kubernetes, or combined results
- **Comparison mode** — Load a second file to compare two experiments side-by-side
- **Summary metrics** — KPI cards or comparison table with K8s/Local ratios
- **Charts** — GPU power over time, utilization over time, latency comparison, energy comparison, per-chunk latency, migration energy impact, VRAM vs RAM trade-off
- **Raw data** — Collapsible JSON viewer for full inspection

### Custom Logs Directory

```bash
# Use a custom results directory
KAI_LOGS_DIR=path/to/results python kai_cli.py dashboard
```

---

## CLI Reference

### KAI CLI (Distributed Inference)

```
python kai_cli.py <command> [OPTIONS]

Commands:
  run          Generate text with a distributed HuggingFace model
  scan         Detect local GPU/CPU/RAM resources
  partition    Preview how a model would be split across nodes
  benchmark    Run energy benchmarking (original KAI workflow)
  dashboard    Launch the Streamlit dashboard
  build        Build Docker images for chunk/gateway/monitor
  prepare      Download model, chunk weights, save for K8s deployment
```

#### `run` — Generate Text

```
python kai_cli.py run --model <hf_model> --prompt "text" [OPTIONS]

Options:
  --model         HuggingFace model name          (required)
  --prompt        Input text prompt                (required)
  --max-tokens    Maximum tokens to generate       (default: 100)
  --temperature   Sampling temperature             (default: 0.7)
  --top-k         Top-k sampling                   (default: 50)
  --top-p         Nucleus sampling                 (default: 0.9)
  --num-chunks    Number of model chunks           (default: 2)
  --stream        Stream output token by token
  --quantize      Quantization mode (4bit/8bit)    (default: none)
  --offload       Enable CPU/disk offloading for models exceeding GPU VRAM
  --gpu-budget-mb GPU VRAM budget in MB            (default: 0 = auto-detect)
  --disk-swap-dir Disk swap directory              (default: /tmp/kai_swap)
```

#### `scan` — Detect Resources

```
python kai_cli.py scan [--mode local|kubernetes]
```

#### `partition` — Preview Layer Split

```
python kai_cli.py partition --model <hf_model> --num-nodes N
```

#### `benchmark` — Energy Benchmarking

```
python kai_cli.py benchmark [OPTIONS]

Options:
  --model         transformer | cnn               (default: transformer)
  --hf-model      HuggingFace model name for HF benchmark
  --mode          local | kubernetes | both        (default: local)
  --iterations    Number of inference iterations   (default: 50)
  --sampling-rate GPU sampling interval in seconds (default: 1.0, e.g. 0.1 for 100ms)
```

#### `build` — Build Docker Images

```
python kai_cli.py build [OPTIONS]

Options:
  --tag           Base image tag                   (default: kai:latest)
  --push          Push images after building
```

#### `prepare` — Prepare Chunk Weights for K8s

```
python kai_cli.py prepare --model <hf_model> [OPTIONS]

Options:
  --model         HuggingFace model name           (required)
  --num-chunks    Number of chunks                  (default: 3)
  --output-dir    Output directory                  (default: data/chunks)
  --dtype         Weight dtype                      (default: float16)
  --quantize      Quantize weights (4bit/8bit)      (default: none)
```

#### `dashboard` — Streamlit Dashboard

```
python kai_cli.py dashboard [OPTIONS]

Options:
  --port          Server port                       (default: 8501)
  --legacy        Launch the legacy analysis-only dashboard instead of the unified dashboard
```

### Experiment Runner (Detailed Benchmarking)

```
python -m experiments.experiment_runner [OPTIONS]

Options:
  --mode          local | kubernetes | both    (default: both)
  --model         transformer | cnn            (default: transformer)
  --num-chunks    2-8                          (default: 3)
  --iterations    Number of inference runs     (default: 50)
  --batch-size    Input batch size             (default: 8)
  --warmup        Warmup iterations            (default: 5)
  --output-dir    Results directory            (default: logs)
  --device        Torch device for local mode  (default: auto)
  --monitor-interval  Sampling interval (s)    (default: 1.0)
  --gateway-url   Override K8s gateway URL     (default: auto-detect)
  --wait-timeout  Pod readiness timeout (s)    (default: 300)
  --no-teardown   Keep K8s resources after run
```

### Kubernetes Controller

```
python -m kubernetes.controller <command> [OPTIONS]

Commands:
  deploy    Deploy chunks, gateway, and monitor
            --num-chunks N  --model NAME  --wait  --timeout SECS
  status    Show pod/deployment status
  health    Check gateway health
            --gateway-url URL
  metrics   Collect metrics from monitor service
  teardown  Delete all KAI resources
```

### Analyzer

```
python -m analysis.analyzer [OPTIONS]

Options:
  --local      Path to local results JSON
  --k8s        Path to K8s results JSON
  --combined   Path to combined experiment JSON
  --output     Output directory (default: analysis/output)
```

### Plot Generator

```
python -m analysis.plots [OPTIONS]

Options:
  --local      Path to local results JSON
  --k8s        Path to K8s results JSON
  --combined   Path to combined experiment JSON
  --output     Output directory (default: analysis/output)
```

---

## Key Metrics

The following metrics are computed for each experiment run and used for comparison:

```
Total Energy (Wh)          = Trapezoidal integration of power samples (or avg × time fallback)
Energy per Inference (Wh)  = Total Energy (Wh) / Number of Inferences
Throughput (inf/s)         = Number of Inferences / Total Runtime (s)
Performance per Watt       = Throughput / Average Power (W)
Energy Efficiency Ratio    = Throughput / Average Power (EER, Phase 21)
Network Overhead (K8s)     = End-to-End Latency − Sum of Chunk Compute Times
Migration Energy (Wh)      = Trapezoidal integration during migration window (Phase 23)
Offloading Overhead (%)    = (Offloaded Latency − Baseline Latency) / Baseline × 100 (Phase 23)
```

Latency percentiles (p50, p90, p95, p99) and standard deviation are also computed.

---

## Expected Outputs

### Data Outputs

| Output | Format | Description |
|--------|--------|-------------|
| Local experiment results | JSON | Per-run latency, GPU power samples, CPU usage, throughput, total energy |
| Kubernetes experiment results | JSON | Per-run latency (e2e + per-chunk), GPU power across nodes, network overhead, total energy |
| Combined experiment summary | JSON | Both modes with comparison ratios in a single file |

### Analysis Outputs

| Output | Format | Description |
|--------|--------|-------------|
| Summary comparison table | CSV | All metrics side-by-side for local vs. Kubernetes |
| Analysis export | JSON | Machine-readable comparison data |
| Visualization plots | 10 × PNG | Publication-quality charts (150 DPI) |
| Interactive dashboard | Streamlit | Web-based exploration of all metrics and plots |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| HuggingFace Models | transformers, accelerate, safetensors, sentencepiece |
| GPU Monitoring | pynvml (NVIDIA NVML) |
| CPU Monitoring | psutil |
| Inter-chunk Communication | gRPC + Protocol Buffers |
| Containerization | Docker (NVIDIA CUDA 12.2 base) |
| Orchestration | Kubernetes |
| K8s Client | `kubernetes` Python package |
| Dashboard | Streamlit |
| Plotting | Matplotlib |
| Quantization | bitsandbytes (4-bit NF4, 8-bit INT8) |
| Data Handling | Pandas, NumPy |
| Serialization | JSON, CSV |

---

## Integration Testing Checklist

### Energy Benchmarking (Phases 1-13) — 25 tests

- [x] Local mode runs end-to-end and produces logs
- [x] GPU monitoring records power, utilization, and temperature data correctly
- [x] CPU monitoring records per-core and overall utilization
- [x] Model chunking produces correct outputs (chunk chain output == full model output)
- [x] gRPC communication between chunks works (serialize → transmit → deserialize)
- [x] Docker images build successfully (chunk, gateway, monitor)
- [x] Kubernetes manifests deploy without errors
- [x] Kubernetes controller deploys, checks status, and tears down
- [x] Kubernetes mode runs end-to-end and produces logs
- [x] Analysis script processes both local and K8s result sets
- [x] Analyzer exports CSV and JSON summaries
- [x] All 8 plot types are generated as PNG files
- [x] Dashboard loads and displays data from result files
- [x] Unified experiment runner works in all three modes (local, kubernetes, both)
- [x] Combined results file includes comparison ratios

### Distributed LLM Inference (Phases 14-18) — 30 tests

- [x] HFModelLoader loads model config, tokenizer, and layer structure
- [x] Model architecture validation works for supported families
- [x] LayerChunker splits model layers into correct number of chunks
- [x] Embedding layer is always in chunk 0, norm+lm_head in last chunk
- [x] Memory-based chunking distributes layers by budget
- [x] DistributedGenerator produces text matching full model output
- [x] Streaming generation yields tokens incrementally
- [x] Temperature, top-k, and top-p sampling work correctly
- [x] ResourceDetector detects local GPU VRAM, RAM, and CPU cores
- [x] AutoPartitioner assigns layers proportional to node memory
- [x] AutoPartitioner validates plan feasibility
- [x] CLI `run` command generates text end-to-end
- [x] CLI `scan` command reports hardware resources
- [x] CLI `partition` command previews model split

### Gap Coverage & Production Readiness (Phase 19) — 27 tests

- [x] CLI `build` subcommand builds Docker images for all components
- [x] CLI `prepare` subcommand downloads model, chunks weights, saves manifest
- [x] Chunk weight saving/loading produces valid PyTorch checkpoints
- [x] HF model benchmarking via `--hf-model` flag in benchmark command
- [x] `_run_local_hf` runs HF model benchmark with energy monitoring
- [x] Shard-based weight loading for large models via WeightMapper
- [x] Automatic strategy selection (full-load vs shard-load) based on RAM
- [x] Quantizer module supports 4-bit and 8-bit estimation
- [x] Quantizer gracefully handles missing bitsandbytes
- [x] `--quantize` argument accepted by `run` and `prepare` commands
- [x] Dockerfile includes HuggingFace and bitsandbytes dependencies
- [x] requirements.txt includes bitsandbytes

### Real-Time Instrumentation & Event Bus (Phase 20) — ~15 tests

- [x] GPUMonitor accepts sub-second intervals (100ms)
- [x] Ring buffer bounded at configured buffer_size
- [x] get_recent_samples returns correct count
- [x] Trapezoidal energy integration more accurate than avg*time
- [x] Single-sample energy fallback works
- [x] EventBus delivers events to subscribers
- [x] EventBus level_filter works correctly
- [x] EventBus concurrent pub/sub is thread-safe
- [x] PowerThresholdService classifies < 70% TDP as OPTIMAL
- [x] PowerThresholdService classifies 70-80% TDP as WARNING
- [x] PowerThresholdService classifies >= 80% TDP as CRITICAL
- [x] monitor_service --help shows --sampling-rate flag

### Dynamic Scheduling & Migration (Phase 21) — ~19 tests

- [x] compute_summary includes energy_efficiency_ratio key
- [x] EER = throughput / power computed correctly
- [x] DEASScheduler importable with all data classes
- [x] compute_cluster_eer returns correct value
- [x] should_rebalance detects CRITICAL nodes
- [x] Cooldown prevents rapid migration triggers
- [x] plan_migration returns valid MigrationPlan objects
- [x] InferenceServicer has Pause/Checkpoint/Resume methods
- [x] Paused chunk rejects Infer calls with UNAVAILABLE
- [x] InferenceGateway has update_chunk_host method
- [x] GET /topology returns chain mapping
- [x] POST /relink succeeds

### CPU/Disk Offloading & Prefetching (Phase 22) — ~14 tests

- [x] TieredWeightManager importable with StorageTier enum
- [x] plan_placement fills GPU first, then RAM, then disk
- [x] Capacity tracking accurate after placement
- [x] get_tier_stats returns 3-tier list
- [x] PrefetchEngine importable with all methods
- [x] get_prefetch_stats returns expected keys
- [x] CLI --offload flag present in run --help
- [x] DistributedGenerator accepts prefetch_engine and weight_manager kwargs
- [x] _forward_all_chunks_offloaded method exists

### Validation & Energy Analysis (Phase 23) — ~14 tests

- [x] EventBus publish-to-callback latency < 500ms
- [x] ThresholdService detects CRITICAL within 500ms
- [x] Checkpoint + restore produces bit-exact tensors
- [x] State dict roundtrip is lossless
- [x] StorageTier enum values correct
- [x] plan_placement fills GPU first
- [x] PrefetchEngine stats structure correct
- [x] Power returns to baseline after migration spike (mock)
- [x] ExperimentAnalyzer has compute_migration_energy method
- [x] ExperimentAnalyzer has compute_offloading_overhead method
- [x] compute_offloading_overhead calculates correct delta and pct
- [x] ExperimentAnalyzer has analyse_deas method
- [x] plot_migration_energy_impact importable
- [x] plot_vram_ram_tradeoff importable

---

## License

This project is for academic and research purposes.
