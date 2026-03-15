# KAI Unified Dashboard — User Guide

Complete guide for the KAI Unified Control Dashboard, a 7-page Streamlit web application that lets you manage every aspect of the KAI distributed AI inference platform from a single interface.

---

## Prerequisites

Before launching the dashboard, ensure you have installed:

```bash
pip install streamlit pandas numpy pynvml psutil
```

Optional (for full functionality):

```bash
# For running AI models
pip install torch transformers accelerate safetensors sentencepiece

# For Kubernetes features
pip install kubernetes

# For quantization
pip install bitsandbytes

# For plot generation in Analysis page
pip install matplotlib
```

**Verify GPU detection:**

```bash
python -c "import pynvml; pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex(0); print(pynvml.nvmlDeviceGetName(h)); pynvml.nvmlShutdown()"
```

If this prints your GPU name (e.g. `NVIDIA GeForce RTX 3060 Ti`), you're ready.

---

## Launching the Dashboard

```bash
# Standard launch (opens browser automatically)
python kai_cli.py dashboard

# Custom port
python kai_cli.py dashboard --port 8502

# Or run directly with Streamlit
streamlit run dashboard/unified_app.py
```

The dashboard opens at **http://localhost:8501** (or your custom port).

Use the **sidebar** on the left to navigate between the 7 pages.

---

## Page-by-Page Guide

### 1. Home — System Overview

**Purpose:** Quick snapshot of your hardware and which AI models can run on it.

**How to use:**

1. The page auto-scans on first load. Click **"Rescan System"** to refresh.
2. Review the 4 metric cards at the top:
   - **GPU** — Your graphics card name (e.g. "NVIDIA GeForce RTX 3060 Ti")
   - **GPU VRAM** — Available video memory in MB
   - **System RAM** — Total physical RAM
   - **CPU Cores** — Physical core count
3. The **Model Compatibility** table shows whether popular models fit in your available memory.
4. Expand **"Kubernetes Cluster Check"** to test K8s connectivity and see cluster nodes.

**Troubleshooting:**

| Problem | Solution |
|---------|----------|
| GPU shows "none" | `pip install pynvml` and rescan |
| K8s check fails | Ensure Docker Desktop K8s is enabled or `kubectl` can reach your cluster |
| Low VRAM shown | Close other GPU-intensive apps (games, other models) |

---

### 2. Run Inference — Generate Text with AI Models

**Purpose:** Load and run HuggingFace language models for text generation.

**How to use:**

1. **Select a model** from the dropdown or choose "Custom..." to enter any HuggingFace model ID
2. Expand **"Model Info"** and click **"Load model info"** to see parameter count and memory requirements
3. **Configure generation settings:**
   - **Prompt** — The input text the model continues from
   - **Max tokens** — How many tokens to generate (more = longer output)
   - **Temperature** — Higher = more creative/random (0.0 = deterministic, 2.0 = very random)
   - **Top-k** — Limits token selection to top K most likely (50 is a good default)
   - **Top-p** — Nucleus sampling threshold (0.9 = consider tokens covering 90% probability)
   - **Repetition penalty** — Penalizes repeated tokens (1.1 is a good default)
4. **Choose device:** `cpu` (always works), `cuda:0` (requires CUDA GPU), `auto` (uses GPU if available)
5. **Choose dtype:** `float16` (recommended for GPU, half memory), `float32` (CPU default), `bfloat16` (newer GPUs)
6. Set **Num chunks** to 1 for single-device inference or 2+ to test the chunking pipeline
7. Click **"Generate"** — output appears in real-time if streaming is enabled

**Recommended models for testing:**

| Model | Size | Device | Notes |
|-------|------|--------|-------|
| `sshleifer/tiny-gpt2` | ~250 MB | CPU or GPU | Fastest, good for testing pipeline |
| `openai-community/gpt2` | ~500 MB | CPU or GPU | Classic GPT-2, decent quality |
| `microsoft/phi-2` | ~5.4 GB | GPU (FP16) | Strong quality for its size |
| `google/gemma-2b` | ~4 GB | GPU (FP16) | Google's efficient small model |

**Offloading (for large models):**

If a model doesn't fit in your GPU VRAM, expand **"Offloading Options"** and enable FlexGen-style offloading. This places some layers in system RAM or on disk. Set "GPU budget" to limit how much VRAM to use (0 = auto-detect).

---

### 3. Cluster Setup — Resource Discovery

**Purpose:** Scan your local machine or Kubernetes cluster to discover available compute resources.

**How to use:**

1. Choose **scan mode**:
   - **local** — Scans just your machine (GPU, RAM, CPU via NVML + psutil)
   - **kubernetes** — Queries K8s API for all cluster nodes
2. Click **"Scan Resources"**
3. Review:
   - **Aggregate metrics** — Total nodes, GPU nodes, total VRAM, total RAM
   - **Node table** — Per-node breakdown of resources
   - **Model compatibility** — Which models fit in the cluster's combined memory
4. **Check Custom Model** — Enter any HuggingFace model name to check if it fits

**Docker Desktop Kubernetes:**

Docker Desktop's built-in K8s doesn't expose GPU resources via the `nvidia.com/gpu` label. The dashboard automatically detects this and augments the K8s node data with your local GPU info (detected via NVML). You'll see an info banner explaining this.

For production clusters, install the [NVIDIA K8s device plugin](https://github.com/NVIDIA/k8s-device-plugin) to get native GPU resource reporting.

---

### 4. Kubernetes Deploy — Deployment Pipeline

**Purpose:** End-to-end Kubernetes deployment of a chunked AI model.

**Prerequisites:** Docker Desktop with Kubernetes enabled, or access to a K8s cluster.

**Steps:**

#### Step 1: Prepare Weights
- Enter the HuggingFace model name (e.g. `sshleifer/tiny-gpt2`)
- Set number of chunks (how many pods to split the model across)
- Choose dtype (float16 recommended)
- Click **"Prepare Weights"** — downloads the model and splits weights into chunk files
- Output directory defaults to `data/chunks`

#### Step 2: Build Docker Images
- Set image tag (e.g. `kai:latest`)
- Optionally enable "Push after build" for remote registries
- Click **"Build Images"** — builds chunk server, gateway, and monitor images

#### Step 3: Deploy
- Confirm chunk count and model type
- Click **"Deploy All"** — creates K8s deployments, services, and DaemonSets

#### Step 4: Pod Status
- Click **"Refresh Status"** to see all KAI pods and their states (Running, Pending, etc.)

#### Step 5: Teardown
- Click **"Teardown All"** to delete all KAI resources from the cluster
- This removes all deployments, services, and pods

Each step shows real-time logs so you can track progress.

---

### 5. Benchmark — Energy Analysis

**Purpose:** Run controlled experiments to measure inference performance and energy consumption.

**How to use:**

1. **Configure the experiment:**
   - **Mode** — `local` (single GPU), `kubernetes` (distributed), or `both` (runs both and compares)
   - **Model** — Built-in `transformer`/`cnn`, or enter a HuggingFace model name
   - **Iterations** — Number of inference runs (10-100 for quick tests, 500+ for publication)
   - **Batch size** — Input tensor batch size
   - **Num chunks** — For K8s mode, how many pods to split across
   - **Sampling rate** — GPU power sampling interval (0.1s for high-frequency, 1.0s for normal)
   - **DEAS** — Enable Dynamic Energy-Aware Scheduling (experimental)
2. Click **"Run Benchmark"**
3. Watch real-time logs during execution
4. When complete, see metric cards:
   - Average latency (ms)
   - Throughput (inferences/sec)
   - Average GPU power (W)
   - Total energy consumed (Wh)
5. Expand **"Raw Result JSON"** for full data

Results are saved to the output directory (default: `logs/`) as JSON files, which can then be loaded in the Analysis page.

---

### 6. Monitor — Live GPU Metrics

**Purpose:** Real-time GPU monitoring with threshold alerts.

**How to use:**

1. Set **sampling interval** (how often to read GPU metrics, 0.1-5.0 seconds)
2. Optionally enable **threshold service** for power alerts based on your GPU's TDP:
   - **Optimal** (green) — Below 70% TDP
   - **Warning** (yellow) — 70-80% TDP
   - **Critical** (red) — Above 80% TDP
3. Click **"Start Monitoring"**
4. View live metric cards: Power (W), Utilization (%), Temperature (C), Memory (MB)
5. View live charts updated every refresh cycle (configurable 1-10 seconds)
6. Enable **"Auto-refresh"** for continuous updates
7. Click **"Stop Monitoring"** when done

**Use cases:**
- Monitor GPU health during long inference runs
- Validate that power draw stays within acceptable limits
- Observe GPU utilization patterns to identify bottlenecks

---

### 7. Analysis — Experiment Results

**Purpose:** Visualize and compare benchmark results.

**How to use:**

1. Select a result JSON file from the dropdown (files from `logs/` directory)
2. The page auto-detects whether results are local, kubernetes, or combined
3. Optionally load a second file for comparison
4. Use the **display mode** radio to filter: Both, Local only, K8s only
5. Review sections:
   - **Summary Metrics** — KPI cards or comparison table with K8s/Local ratios
   - **GPU Power** — Power draw over time (overlaid for multiple modes)
   - **GPU Utilization** — Utilization percentage over time
   - **Latency Comparison** — Average latency bars and per-iteration distribution
   - **Energy Comparison** — Total energy and energy-per-inference
   - **Per-Chunk Latency** — K8s chunk processing times (identifies bottleneck chunks)
   - **Migration Energy** — DEAS migration event timeline and impact summary
   - **VRAM vs RAM Trade-off** — GPU vs CPU execution latency per chunk
6. Expand **"Experiment Configuration"** to see experiment parameters
7. Expand **"Raw JSON Data"** for full inspection

---

## Downloading Offline AI Models for Testing

Models are downloaded from HuggingFace the first time you use them. To pre-download for offline use:

### Method 1: Using the Dashboard

Go to **Run Inference** page, select a model, and click **"Load model info"**. This downloads the model config. To download full weights, click **"Generate"** — the model is cached locally in `~/.cache/huggingface/hub/`.

### Method 2: Using the CLI

```bash
# Download and cache a model (generates one token to trigger full download)
python kai_cli.py run --model sshleifer/tiny-gpt2 --prompt "test" --max-tokens 1 --device cpu
```

### Method 3: Using Python directly

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Downloads and caches model + tokenizer
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"Model cached at: ~/.cache/huggingface/hub/")
```

### Method 4: Using huggingface-cli

```bash
# Install the CLI
pip install huggingface_hub

# Download a specific model
huggingface-cli download sshleifer/tiny-gpt2
huggingface-cli download openai-community/gpt2
huggingface-cli download microsoft/phi-2

# Check cache location and size
huggingface-cli scan-cache
```

### Recommended Models for Testing

| Model | Download Size | Min VRAM (FP16) | Quality |
|-------|--------------|-----------------|---------|
| `sshleifer/tiny-gpt2` | ~50 MB | CPU-friendly | Low (test only) |
| `openai-community/gpt2` | ~500 MB | CPU-friendly | Moderate |
| `microsoft/phi-2` | ~5.4 GB | ~3 GB | Good |
| `google/gemma-2b` | ~4 GB | ~2.5 GB | Good |
| `tiiuae/falcon-7b` | ~14 GB | ~7 GB | Very good |
| `mistralai/Mistral-7B-v0.1` | ~14 GB | ~7 GB | Excellent |

### Offline Mode

Once downloaded, models are cached at `~/.cache/huggingface/hub/`. To run fully offline:

```bash
# Set environment variable to prevent any network requests
set HF_HUB_OFFLINE=1

# Now run inference using only cached models
python kai_cli.py run --model sshleifer/tiny-gpt2 --prompt "Hello" --max-tokens 50
```

### Cache Management

```bash
# See all cached models and their sizes
huggingface-cli scan-cache

# Delete a specific model from cache
huggingface-cli delete-cache
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Install NVIDIA ML bindings
pip install pynvml

# Verify
python -c "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlDeviceGetCount()); pynvml.nvmlShutdown()"
```

### Streamlit Won't Start

```bash
# Reinstall streamlit
pip install --upgrade streamlit

# Run directly
streamlit run dashboard/unified_app.py --server.port 8501
```

### K8s Connection Failed

```bash
# Check kubectl works
kubectl get nodes

# If using Docker Desktop, ensure Kubernetes is enabled in Settings > Kubernetes
```

### Model Download Fails

```bash
# Check internet connectivity and try with verbose logging
HF_HUB_VERBOSITY=debug python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')"

# Some models require authentication (Llama, etc.)
huggingface-cli login
```

### Out of Memory

- Use a smaller model (tiny-gpt2 or gpt2)
- Switch to `cpu` device
- Enable offloading in the Run Inference page
- Use `float16` dtype instead of `float32`
- Use quantization: `--quantize 4bit` in CLI
