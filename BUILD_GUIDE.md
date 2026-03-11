# KAI — Kubernetes AI Inference Energy Benchmarking Platform

## Build Guide: Development Process

---

## Overview

This platform measures energy consumption and performance of AI inference workloads in two scenarios:

1. **Local Mode** — Running an AI model on a single GPU (no Kubernetes)
2. **Kubernetes Mode** — Running the same AI model split into chunks across multiple nodes in a Kubernetes cluster

The system is written primarily in Python with supporting Docker and Kubernetes configuration.

---

## Final Project Structure

```
project/
├── model/                    # AI model definition and chunking logic
│   ├── __init__.py
│   ├── transformer.py        # Transformer model definition
│   ├── cnn.py                # CNN model definition
│   ├── chunker.py            # Model splitting/chunking utility
│   └── chunk_server.py       # gRPC server for serving a model chunk
│
├── monitoring/               # Power and performance monitoring
│   ├── __init__.py
│   ├── gpu_monitor.py        # NVIDIA NVML-based GPU power monitoring
│   ├── cpu_monitor.py        # CPU usage monitoring
│   └── metrics.py            # Unified metrics collection and export
│
├── kubernetes/               # Kubernetes configuration and controller
│   ├── deployments/
│   │   ├── chunk-deployment.yaml
│   │   ├── gateway-deployment.yaml
│   │   └── monitor-deployment.yaml
│   ├── services/
│   │   ├── chunk-service.yaml
│   │   ├── gateway-service.yaml
│   │   └── monitor-service.yaml
│   ├── gpu-resource-quota.yaml
│   └── controller.py         # Python K8s controller script
│
├── docker/                   # Dockerfiles
│   ├── Dockerfile.chunk      # Model chunk microservice
│   ├── Dockerfile.gateway    # Inference gateway
│   └── Dockerfile.monitor    # Monitoring service
│
├── experiments/              # Experiment runner scripts
│   ├── __init__.py
│   ├── local_runner.py       # Run inference locally on single GPU
│   ├── k8s_runner.py         # Run inference on Kubernetes cluster
│   └── experiment_runner.py  # Unified experiment orchestrator
│
├── dashboard/                # Streamlit visualization dashboard
│   └── app.py
│
├── logs/                     # Experiment output logs (CSV/JSON)
│   └── .gitkeep
│
├── analysis/                 # Post-experiment analysis and plotting
│   ├── __init__.py
│   ├── analyzer.py           # Compute energy, latency, efficiency
│   └── plots.py              # Matplotlib visualization
│
├── proto/                    # gRPC protocol buffer definitions
│   └── inference.proto
│
├── requirements.txt
├── BUILD_GUIDE.md
└── README.md
```

---

## Development Phases

---

### Phase 1: Project Scaffolding & Dependencies

**Goal:** Set up the project skeleton, virtual environment, and all dependency declarations.

**Tasks:**

1. Create the full directory structure as shown above.
2. Create `requirements.txt` with all dependencies:
   - `torch` and `torchvision` (PyTorch for model inference)
   - `pynvml` (NVIDIA NVML Python bindings for GPU monitoring)
   - `psutil` (CPU usage monitoring)
   - `grpcio` and `grpcio-tools` (gRPC for inter-chunk communication)
   - `protobuf` (protocol buffer serialization)
   - `kubernetes` (Python Kubernetes client)
   - `streamlit` (dashboard)
   - `matplotlib` (analysis plots)
   - `pandas` (data handling)
   - `numpy`
3. Create all `__init__.py` files for Python packages.
4. Create `.gitkeep` in `logs/` to track the empty directory.

**Deliverables:**
- Complete folder tree
- `requirements.txt`
- Empty `__init__.py` files

---

### Phase 2: AI Model Module

**Goal:** Define the AI models and implement the chunking mechanism that allows splitting a model into independent layer groups.

**Tasks:**

1. **`model/transformer.py`** — Implement a small Transformer model (e.g., a few-layer encoder) using PyTorch.
   - Define the model class with configurable number of layers.
   - Implement a `forward()` method.
   - Include a helper to generate dummy input tensors for benchmarking.

2. **`model/cnn.py`** — Implement a simple CNN model as an alternative workload.
   - Define convolution + fully connected layers.
   - Implement `forward()`.
   - Include dummy input generator.

3. **`model/chunker.py`** — Model chunking utility.
   - Accept a PyTorch model and a number of chunks.
   - Split the model's layers into `N` sequential groups.
   - Each chunk is a `nn.Sequential` sub-model that can run independently.
   - Provide `serialize_chunk()` and `load_chunk()` for saving/loading chunks.
   - Provide `get_chunk_info()` returning metadata (layer names, input/output shapes).

**Deliverables:**
- Working Transformer and CNN models
- Chunker that splits any sequential model into N independent parts
- Unit-testable: each chunk's output feeds into the next chunk's input

---

### Phase 3: gRPC Communication Layer

**Goal:** Define the gRPC protocol and implement the chunk server so model chunks can communicate intermediate tensors over the network.

**Tasks:**

1. **`proto/inference.proto`** — Define the protobuf service:
   - `InferenceService` with an `Infer` RPC method.
   - Request message: serialized tensor (bytes), chunk ID, metadata.
   - Response message: serialized output tensor (bytes), timing info.

2. **Generate Python gRPC stubs** — Use `grpcio-tools` to compile the proto file.
   - Generated files: `inference_pb2.py`, `inference_pb2_grpc.py`.

3. **`model/chunk_server.py`** — gRPC server for a single model chunk.
   - Load a specific chunk by ID.
   - Receive input tensor via gRPC, run inference on the chunk, return output tensor.
   - Log per-chunk inference latency.
   - Configurable port and chunk ID via environment variables.

**Deliverables:**
- `.proto` definition
- Generated gRPC stubs
- Chunk server that can serve any model chunk independently

---

### Phase 4: Power & Performance Monitoring Module

**Goal:** Build the monitoring subsystem that records GPU power, GPU utilization, GPU memory, CPU usage, and timestamps.

**Tasks:**

1. **`monitoring/gpu_monitor.py`** — GPU monitoring using NVML.
   - Initialize NVML on start, shutdown on exit.
   - Sample GPU metrics every 1 second (configurable):
     - Power draw (Watts)
     - GPU utilization (%)
     - GPU memory used (MB)
     - GPU temperature (°C)
     - Timestamp (ISO 8601)
   - Run sampling in a background thread.
   - Provide `start()`, `stop()`, and `get_samples()` methods.
   - Export samples to CSV or JSON.

2. **`monitoring/cpu_monitor.py`** — CPU monitoring using `psutil`.
   - Sample CPU utilization (%) per core and overall.
   - Sample every 1 second (configurable).
   - Run in a background thread alongside GPU monitor.
   - Provide `start()`, `stop()`, `get_samples()`.

3. **`monitoring/metrics.py`** — Unified metrics collector.
   - Combine GPU and CPU monitors.
   - Add inference-level metrics: latency per request, throughput.
   - Provide a single `MetricsCollector` class that starts/stops all monitors.
   - Export combined results to a structured JSON or CSV file in `logs/`.

**Deliverables:**
- GPU monitor with 1-second sampling via NVML
- CPU monitor with 1-second sampling via psutil
- Unified metrics collector that writes to `logs/`

---

### Phase 5: Local Execution Mode

**Goal:** Implement the local inference runner that executes the full model on a single GPU and collects all metrics.

**Tasks:**

1. **`experiments/local_runner.py`** — Local inference script.
   - Load the selected model (Transformer or CNN) onto a single GPU.
   - Generate dummy input data (configurable batch size).
   - Run inference N times (configurable iterations).
   - For each run:
     - Record start/end timestamps.
     - Compute inference latency.
   - Use `MetricsCollector` to capture GPU/CPU metrics during all runs.
   - After all runs, compute:
     - Average inference latency
     - Average GPU power consumption
     - Total energy consumption (Wh) = average power × total time / 3600
     - Throughput (inferences/second)
   - Save results to `logs/local_results_<timestamp>.json`.

**Deliverables:**
- Complete local inference pipeline
- Automated metrics collection and result logging

---

### Phase 6: Docker Setup

**Goal:** Create Docker images for the model chunk services, inference gateway, and monitoring service.

**Tasks:**

1. **`docker/Dockerfile.chunk`** — Model chunk microservice container.
   - Base image: `nvidia/cuda` with Python 3.
   - Install PyTorch, gRPC dependencies.
   - Copy model code and gRPC stubs.
   - Entrypoint: `chunk_server.py` with configurable `CHUNK_ID` and `PORT` env vars.

2. **`docker/Dockerfile.gateway`** — Inference gateway container.
   - Receives inference requests from the experiment runner.
   - Forwards input through the chain of chunk services via gRPC.
   - Returns final output and timing data.
   - Acts as the entry point for Kubernetes-mode inference.

3. **`docker/Dockerfile.monitor`** — Monitoring sidecar container.
   - Runs the GPU/CPU monitor.
   - Exposes metrics via a simple HTTP endpoint or writes to a shared volume.
   - Can be deployed as a sidecar or DaemonSet.

**Deliverables:**
- Three Dockerfiles, each buildable and runnable
- Docker Compose file (optional, for local multi-container testing)

---

### Phase 7: Kubernetes Configuration

**Goal:** Create all Kubernetes manifests to deploy the chunked model as a distributed inference pipeline.

**Tasks:**

1. **`kubernetes/deployments/chunk-deployment.yaml`** — One Deployment template per chunk.
   - Parameterized by chunk ID.
   - GPU resource requests/limits (e.g., `nvidia.com/gpu: 1`).
   - Environment variables: `CHUNK_ID`, `PORT`, `NEXT_CHUNK_HOST`.

2. **`kubernetes/deployments/gateway-deployment.yaml`** — Gateway Deployment.
   - Single pod that orchestrates the inference pipeline.
   - Knows the addresses of all chunk services.

3. **`kubernetes/deployments/monitor-deployment.yaml`** — Monitor DaemonSet or sidecar.
   - Runs GPU monitoring on each node with a GPU.

4. **`kubernetes/services/chunk-service.yaml`** — ClusterIP Service per chunk.
   - Exposes each chunk pod internally for gRPC communication.

5. **`kubernetes/services/gateway-service.yaml`** — NodePort or LoadBalancer service.
   - Exposes the gateway for external inference requests.

6. **`kubernetes/services/monitor-service.yaml`** — Monitor service.
   - Exposes metrics endpoint.

7. **`kubernetes/gpu-resource-quota.yaml`** — ResourceQuota for GPU allocation.

**Deliverables:**
- Full set of Kubernetes YAML manifests
- Configurable for different numbers of chunks/nodes

---

### Phase 8: Kubernetes Controller

**Goal:** Build the Python-based Kubernetes controller that automates deployment, monitoring, and data collection.

**Tasks:**

1. **`kubernetes/controller.py`** — Kubernetes orchestration script.
   - Use the `kubernetes` Python client library.
   - Functions:
     - `deploy_chunks(num_chunks)` — Create deployments and services for N chunks.
     - `wait_for_ready()` — Poll pod status until all are Running.
     - `send_inference_request(input_data)` — Call the gateway service.
     - `collect_metrics()` — Gather metrics from monitor pods.
     - `teardown()` — Delete all created resources.
   - Handle errors and retries for pod scheduling.
   - Log all operations.

**Deliverables:**
- Controller script that can deploy, run, collect, and teardown a full experiment

---

### Phase 9: Kubernetes Execution Mode

**Goal:** Implement the Kubernetes inference runner analogous to the local runner.

**Tasks:**

1. **`experiments/k8s_runner.py`** — Kubernetes inference script.
   - Use the controller to deploy the chunked model.
   - Wait for all pods to be ready.
   - Run inference N times through the gateway.
   - For each run:
     - Record end-to-end latency (including network overhead).
     - Record per-chunk latency from gateway responses.
   - Collect GPU metrics from all nodes via the monitor service.
   - After all runs, compute:
     - Average inference latency
     - Average GPU power across all nodes
     - Total energy consumption (sum across all GPUs)
     - Throughput
     - Network overhead (total latency − sum of chunk compute times)
   - Save results to `logs/k8s_results_<timestamp>.json`.
   - Teardown Kubernetes resources.

**Deliverables:**
- Complete Kubernetes inference pipeline
- Multi-node metrics aggregation
- Result logging

---

### Phase 10: Unified Experiment Runner

**Goal:** Create a single entry-point script that runs both local and Kubernetes experiments back-to-back and produces comparable results.

**Tasks:**

1. **`experiments/experiment_runner.py`** — Unified orchestrator.
   - CLI arguments:
     - `--mode`: `local`, `kubernetes`, or `both`
     - `--model`: `transformer` or `cnn`
     - `--num-chunks`: number of chunks for K8s mode
     - `--iterations`: number of inference runs
     - `--batch-size`: input batch size
     - `--output-dir`: directory for results (default: `logs/`)
   - Runs the selected experiment(s).
   - Produces a comparison summary if both modes are run.
   - Saves a combined results file: `logs/experiment_<timestamp>.json`.

**Deliverables:**
- Single CLI-based experiment runner
- Comparison output when both modes are executed

---

### Phase 11: Analysis Module

**Goal:** Build post-experiment analysis tools to compute derived metrics and generate visualizations.

**Tasks:**

1. **`analysis/analyzer.py`** — Metrics computation.
   - Load experiment result JSON files.
   - Compute:
     - Total energy usage (Wh) per experiment
     - Average GPU power (W) per experiment
     - Average inference latency (ms)
     - Throughput (inferences/sec)
     - Energy per inference (Wh/inference)
     - Performance-per-watt ratio
   - Compare local vs. Kubernetes results side-by-side.
   - Output a summary table (printed and saved as CSV).

2. **`analysis/plots.py`** — Matplotlib visualizations.
   - **Power over time:** Line plot of GPU power draw over the experiment duration (both modes overlaid).
   - **GPU utilization over time:** Line plot comparing utilization.
   - **Latency comparison:** Bar chart of average latency (local vs. K8s).
   - **Energy comparison:** Bar chart of total energy consumed.
   - **Energy efficiency:** Bar chart of energy-per-inference.
   - **Throughput comparison:** Bar chart.
   - Save all plots as PNG files in `analysis/output/`.

**Deliverables:**
- Analysis script with summary table
- 6+ publication-quality plots
- Saved outputs

---

### Phase 12: Streamlit Dashboard

**Goal:** Create an interactive web dashboard for real-time and post-experiment visualization.

**Tasks:**

1. **`dashboard/app.py`** — Streamlit application.
   - **Sidebar:** Select experiment results file, filter by mode.
   - **Power Consumption Panel:** Interactive line chart of GPU power over time.
   - **GPU Utilization Panel:** Interactive line chart.
   - **Latency Comparison Panel:** Bar chart comparing local vs. K8s.
   - **Energy Comparison Panel:** Bar chart.
   - **Summary Table:** Key metrics side-by-side.
   - Auto-refresh capability for live monitoring (optional).
   - Load data from `logs/` directory.

**Deliverables:**
- Fully functional Streamlit dashboard
- Runnable with `streamlit run dashboard/app.py`

---

### Phase 13: Documentation & Final Integration

**Goal:** Write documentation, ensure all components integrate cleanly, and verify the end-to-end workflow.

**Tasks:**

1. **`README.md`** — Project documentation.
   - Project overview and goals.
   - Prerequisites (NVIDIA GPU, CUDA, Docker, Kubernetes cluster).
   - Installation instructions.
   - Quick-start guide for local mode.
   - Quick-start guide for Kubernetes mode.
   - Dashboard usage.
   - Analysis instructions.
   - Architecture diagram (text-based).

2. **Integration testing checklist:**
   - [ ] Local mode runs end-to-end and produces logs.
   - [ ] GPU monitoring records data correctly.
   - [ ] Model chunking produces correct outputs (chunk chain output == full model output).
   - [ ] Docker images build successfully.
   - [ ] Kubernetes manifests deploy without errors.
   - [ ] gRPC communication between chunks works.
   - [ ] Kubernetes mode runs end-to-end and produces logs.
   - [ ] Analysis script processes both result sets.
   - [ ] Plots are generated.
   - [ ] Dashboard loads and displays data.

3. **Code quality:**
   - Ensure all modules have docstrings.
   - Consistent logging throughout.
   - Error handling at system boundaries.

**Deliverables:**
- Complete README
- Verified end-to-end pipeline

---

## Phase Dependency Graph

```
Phase 1  (Scaffolding)
   │
   ├── Phase 2  (AI Models)
   │      │
   │      ├── Phase 3  (gRPC Layer)
   │      │      │
   │      │      ├── Phase 6  (Docker)
   │      │      │      │
   │      │      │      └── Phase 7  (K8s YAML)
   │      │      │             │
   │      │      │             ├── Phase 8  (K8s Controller)
   │      │      │             │      │
   │      │      │             │      └── Phase 9  (K8s Runner)
   │      │      │             │             │
   │      │      │             │             └─┐
   │      │      │             │               │
   │      ├── Phase 4  (Monitoring)            │
   │      │      │                             │
   │      │      └── Phase 5  (Local Runner)   │
   │      │             │                      │
   │      │             └──────────────────────┤
   │      │                                    │
   │      │                    Phase 10  (Unified Runner)
   │      │                        │
   │      │                        ├── Phase 11  (Analysis)
   │      │                        │
   │      │                        └── Phase 12  (Dashboard)
   │      │
   │      └── Phase 13  (Documentation & Integration)
```

---

## Technology Stack Summary

| Component          | Technology                        |
|--------------------|-----------------------------------|
| Language           | Python 3.10+                      |
| Deep Learning      | PyTorch                           |
| GPU Monitoring     | pynvml (NVIDIA NVML)              |
| CPU Monitoring     | psutil                            |
| Inter-chunk Comm   | gRPC + Protocol Buffers           |
| Containerization   | Docker (NVIDIA CUDA base images)  |
| Orchestration      | Kubernetes                        |
| K8s Client         | `kubernetes` Python package       |
| Dashboard          | Streamlit                         |
| Plotting           | Matplotlib                        |
| Data Handling      | Pandas, NumPy                     |
| Serialization      | JSON, CSV                         |

---

## Key Design Decisions

1. **gRPC for tensor passing:** Chosen over REST because gRPC supports efficient binary serialization, which is critical for large intermediate tensors between model chunks.

2. **NVML over nvidia-smi:** Direct NVML API calls via `pynvml` are faster and more programmatic than parsing `nvidia-smi` CLI output. They allow precise 1-second sampling without subprocess overhead.

3. **Sequential chunk pipeline:** Model chunks are arranged in a linear pipeline (chunk 1 → chunk 2 → ... → chunk N). This matches the sequential nature of Transformer and CNN architectures and simplifies the communication pattern.

4. **Separate monitoring containers:** Running GPU monitoring as a sidecar or DaemonSet ensures that monitoring does not interfere with inference performance and can capture node-level metrics.

5. **JSON for result storage:** JSON is human-readable, easily loadable in Python, and compatible with both the analysis module and the Streamlit dashboard.

---

## Notes for Development

- **GPU Requirement:** An NVIDIA GPU with CUDA support is required for local mode. Kubernetes mode requires a cluster with GPU-equipped nodes and the NVIDIA device plugin installed.
- **Chunk Count Flexibility:** The system should support 2–8 chunks, configurable at experiment time.
- **Idempotent Kubernetes Deployments:** The controller should handle re-deployments gracefully (delete existing resources before creating new ones).
- **Reproducibility:** All experiments log their configuration (model type, batch size, iterations, chunk count) alongside results for reproducibility.

---

## Pivot: Distributed Inference for Large Models on Low-End Hardware

> **Phases 14–18** extend KAI from an energy-benchmarking tool into a
> **distributed inference platform** that can run large HuggingFace models
> (e.g., LLaMA 7B, Mistral 7B, Phi-2) on clusters of low-end PCs using
> Kubernetes. Each node loads only the layers it is responsible for, so no
> single machine needs enough VRAM/RAM for the full model.

---

### Phase 14: HuggingFace Large Model Support

**Goal:** Load real HuggingFace models (LLMs) and integrate them into the KAI pipeline alongside the existing benchmark models.

**Tasks:**

1. **`model/hf_loader.py`** — HuggingFace model loader.
   - Accept a HuggingFace model name or local path (e.g., `microsoft/phi-2`, `meta-llama/Llama-2-7b-hf`).
   - Load the tokenizer via `transformers.AutoTokenizer`.
   - Load the model configuration and layer structure via `transformers.AutoModelForCausalLM` or `AutoConfig`.
   - Support loading with `torch_dtype=torch.float16` / `bfloat16` to reduce memory.
   - Provide `get_layer_list()` returning an ordered list of `(name, module)` for all model layers (embeddings, transformer blocks, output head).
   - Provide `get_tokenizer()`, `get_config()` helper methods.
   - **Memory-safe loading:** Support `device_map="meta"` or `low_cpu_mem_usage=True` to avoid loading full weights into RAM during layer enumeration.
   - Validate that the model is a causal LM or encoder-decoder; refuse unsupported architectures.

2. **Update `model/__init__.py`** — Expose the new module.

3. **Update `requirements.txt`** — Add:
   - `transformers>=4.36.0`
   - `accelerate>=0.25.0`
   - `sentencepiece>=0.1.99`
   - `safetensors>=0.4.0`

**Deliverables:**
- `model/hf_loader.py` with `HFModelLoader` class
- Able to enumerate layers of any HuggingFace causal LM without loading full weights
- Tokenizer integration for encoding text prompts → token IDs

---

### Phase 15: Layer-Wise Distributed Chunking

**Goal:** Split a large HuggingFace model layer-by-layer across nodes so each node holds only its assigned layers in memory.

**Tasks:**

1. **`model/layer_chunker.py`** — Layer-aware chunking for HuggingFace models.
   - Accept the output of `HFModelLoader.get_layer_list()`.
   - Split layers into N groups (configurable).
   - Each chunk loads only its assigned layers' weights from disk/hub — NOT the full model.
   - Use `safetensors` shard loading or `accelerate`'s `load_checkpoint_in_model` to load only specific layers.
   - Each chunk is a `LayerChunk(nn.Module)` that:
     - Holds its assigned layers.
     - Accepts hidden states as input, returns hidden states as output.
     - Handles embedding layer (chunk 0 only) and LM head (last chunk only).
   - Provide `estimate_chunk_memory(chunk)` → estimated VRAM in MB.
   - Provide `save_chunk_weights(chunk, path)` and `load_chunk_weights(chunk, path)`.

2. **`model/weight_utils.py`** — Utility for partial weight loading.
   - Map HuggingFace checkpoint shards to specific layer indices.
   - Load only needed shard files for a given chunk.
   - Support both `.safetensors` and `.bin` weight formats.
   - Avoid loading the entire model into memory at any point.

**Deliverables:**
- Layer-wise chunking that loads only needed weights per chunk
- Memory estimation per chunk
- Chunk serialization for deployment to nodes

---

### Phase 16: Text Generation Inference Pipeline

**Goal:** Implement a text generation pipeline that chains chunk services to perform autoregressive token generation across distributed nodes.

**Tasks:**

1. **`model/generation.py`** — Distributed text generation engine.
   - Implement autoregressive generation loop:
     1. Tokenize input prompt.
     2. Forward through all chunks (chunk 0 → chunk 1 → ... → chunk N).
     3. Chunk N returns logits; sample next token.
     4. Append token to sequence; repeat until stop condition.
   - Support generation parameters:
     - `max_new_tokens`: maximum tokens to generate.
     - `temperature`: sampling temperature.
     - `top_k`: top-k sampling.
     - `top_p`: nucleus sampling.
     - `repetition_penalty`: penalize repeated tokens.
   - Implement KV-cache forwarding:
     - Each chunk maintains a key-value cache for its layers.
     - On subsequent tokens, only the new token is forwarded (not the full sequence).
     - Cache is stored on the chunk's device, passed by reference (not over network).
   - Streaming support: yield tokens as they are generated.

2. **Update `model/chunk_server.py`** — Extend the gRPC servicer:
   - Add `InferCausal` RPC method for causal LM chunks.
   - Handle KV-cache state across calls within a session.
   - Accept `session_id` to maintain per-request caches.
   - Add `ClearCache` RPC to free memory when generation is done.

3. **Update `proto/inference.proto`** — Add new messages:
   - `CausalInferRequest`: token IDs, session_id, generation step.
   - `CausalInferResponse`: output logits/hidden states, cache status.
   - `ClearCacheRequest` / `ClearCacheResponse`.

4. **Update `model/gateway.py`** — Add generation endpoint:
   - `POST /generate` accepts `{"prompt": "...", "max_tokens": 100, ...}`.
   - Orchestrates the generation loop across chunk services.
   - Returns generated text (and optionally streams tokens via SSE).

**Deliverables:**
- End-to-end text generation across distributed chunks
- KV-cache support for efficient autoregressive decoding
- Streaming token output
- Generation parameter support (temperature, top_k, top_p)

---

### Phase 17: Smart Resource Detection & Auto-Partitioning

**Goal:** Automatically detect available resources on each Kubernetes node and partition the model proportionally.

**Tasks:**

1. **`model/resource_detector.py`** — Node resource scanner.
   - Query each Kubernetes node for:
     - Available GPU VRAM (via NVML if GPU present, else 0).
     - Available system RAM.
     - CPU core count.
   - Also detect GPU type (e.g., RTX 3050 Ti 4GB vs. GTX 1060 6GB).
   - Return a sorted list of nodes with their capabilities.
   - Support CPU-only nodes (layers can run on CPU with reduced speed).

2. **`model/auto_partitioner.py`** — Intelligent model partitioner.
   - Given the model's layer list and each node's capability:
     - Estimate memory per layer (parameters × dtype size × ~1.2 overhead).
     - Assign layers to nodes proportional to their available memory.
     - Nodes with more VRAM get more layers.
     - Ensure every layer is assigned to exactly one node.
   - Handle edge cases:
     - Model too large for the cluster → clear error message with requirements.
     - Only one node available → all layers on that node.
     - Mixed GPU/CPU nodes → GPU nodes get more layers.
   - Output a partition plan: `{node_id: [layer_start, layer_end], ...}`.

3. **Update `kubernetes/controller.py`** — Enhanced deployment:
   - Use the partition plan to create per-node deployments.
   - Pass `LAYER_START` and `LAYER_END` environment variables to chunk containers.
   - Deploy chunk containers with appropriate resource limits matching node capabilities.

**Deliverables:**
- Automatic node capability detection
- Proportional model partitioning based on real hardware
- Partition plan generation and validation

---

### Phase 18: End-to-End CLI & Integration

**Goal:** Provide a unified CLI that lets users run large models on their K8s cluster with a single command, plus update all documentation.

**Tasks:**

1. **`kai_cli.py`** — Main entry point (root of project).
   - Subcommands:
     - `kai run --model <hf_model_name> --prompt "Hello" --max-tokens 100`
       → Downloads model, partitions across cluster, generates text.
     - `kai scan` → Detect cluster resources and show capabilities.
     - `kai partition --model <name> --preview`
       → Show how the model would be split without deploying.
     - `kai benchmark --model <name> --mode both`
       → Run the original energy benchmarking workflow.
     - `kai dashboard` → Launch the Streamlit dashboard.
   - Progress indicators for model download, partitioning, deployment.
   - Clean error messages for common failures (no cluster, OOM, etc.).

2. **Update `docker/Dockerfile.chunk`** — Support HuggingFace models:
   - Install `transformers`, `accelerate`, `safetensors`.
   - Accept `HF_MODEL_NAME`, `LAYER_START`, `LAYER_END` env vars.
   - Load only assigned layers on startup.

3. **Update `README.md`** — Complete documentation rewrite:
   - New project description: distributed inference platform for large AI models.
   - Quick-start: "Run LLaMA 7B on 3 budget PCs."
   - Architecture diagram showing layer distribution.
   - Resource requirements table.
   - Comparison with Petals, AirLLM, Ollama.

4. **Update integration tests** — Add tests for new modules:
   - Test HFModelLoader with a small model (e.g., `sshleifer/tiny-gpt2`).
   - Test layer chunking produces valid chunks.
   - Test generation pipeline end-to-end with tiny model.
   - Test resource detector with mock data.
   - Test auto-partitioner with various node configurations.

**Deliverables:**
- Single-command CLI for distributed LLM inference
- Updated Docker images for HuggingFace model serving
- Complete documentation and tests

---

### Phase 19: Gap Coverage & Production Readiness

**Goal:** Close 5 remaining gaps to make KAI fully operational end-to-end without manual steps, support real HuggingFace models in benchmarks, handle 70B+ parameter models, and add quantization support.

**Tasks:**

1. **`kai_cli.py build`** — Auto-build Docker images.
   - Add `build` subcommand that runs `docker build` for all 3 images (chunk, gateway, monitor).
   - Accepts `--tag` for custom image tags.
   - Optionally pushes to a container registry with `--push`.
   - Validates Docker is installed and accessible.

2. **`kai_cli.py prepare`** — Save chunk weights for K8s deployment.
   - Add `prepare` subcommand that downloads a model, splits it into chunks, and saves each chunk's weights to disk.
   - Accepts `--model`, `--num-chunks`, `--output-dir`, `--dtype`.
   - Output directory structure: `<output_dir>/chunk_0.pt`, `chunk_1.pt`, etc.
   - These files are what K8s chunk pods mount from a shared volume (NFS/PVC).
   - Uses shard-based loading (`WeightMapper`) for large models to avoid OOM.

3. **Extend `benchmark` to support HuggingFace models.**
   - Add `--hf-model` argument to `kai_cli.py benchmark`.
   - When `--hf-model` is provided, use the HuggingFace model for both local and K8s benchmarks instead of the dummy transformer/CNN.
   - Local mode: loads model on single GPU, runs generation, records energy.
   - K8s mode: deploys chunks via controller, runs generation through gateway, records energy.
   - Comparison output includes energy/power ratios proving distributed = lower power.

4. **Integrate shard-based weight loading into CLI for 70B+ models.**
   - Update `_load_real_weights()` in `kai_cli.py` to use `WeightMapper.load_state_dict_for_layers()` when the full model would exceed available RAM.
   - Detect available system memory and choose between full-model load (fast, for small models) vs. shard-by-shard load (memory-safe, for large models).
   - Each chunk loads only its own weight shards — never the full model.

5. **Add quantization support via `bitsandbytes`.**
   - Create `model/quantizer.py` — Quantization utility.
     - Support 4-bit (NF4) and 8-bit (INT8) quantization via `bitsandbytes`.
     - `quantize_chunk(chunk, mode="4bit"|"8bit")` quantizes a LayerChunk in-place.
     - `estimate_quantized_memory(chunk, mode)` returns expected memory after quantization.
   - Add `--quantize` argument to `kai_cli.py run` and `kai_cli.py prepare`.
   - When enabled, each chunk is quantized after weight loading.
   - Reduces memory per chunk by 4-8x while maintaining near-full-precision output.
   - Update `requirements.txt` with `bitsandbytes>=0.41.0`.

6. **Update Docker images for HuggingFace support.**
   - Update `Dockerfile.chunk` to install `transformers`, `accelerate`, `safetensors`, `bitsandbytes`.
   - Add `HF_MODEL_NAME`, `LAYER_START`, `LAYER_END`, `QUANTIZE` env vars.
   - Chunk container loads only its assigned layers on startup using `WeightMapper`.

7. **Add integration tests for all new functionality.**
   - Test `build` command (validates Docker command construction).
   - Test `prepare` command (saves and reloads chunk weights).
   - Test shard-based loading with a small model.
   - Test quantization (4-bit/8-bit) on a small model.
   - Test HF benchmark mode.

**Deliverables:**
- `kai_cli.py build` and `kai_cli.py prepare` commands
- HuggingFace model support in benchmark mode
- Shard-based weight loading for 70B+ models
- `model/quantizer.py` with 4-bit and 8-bit quantization
- Updated Docker images
- Integration tests for all new features

---

## Updated Phase Dependency Graph

```
Phase 1-13  (Original KAI -- Energy Benchmarking)
   |         (All completed)
   |
   +-- Phase 14  (HuggingFace Model Support)
          |
          +-- Phase 15  (Layer-Wise Chunking)
          |      |
          |      +-- Phase 16  (Generation Pipeline)
          |      |
          |      +-- Phase 17  (Smart Resource Detection)
          |             |
          |             +-- Phase 18  (E2E CLI & Integration)
          |                    |
          |                    +-- Phase 19  (Gap Coverage & Production Readiness)
```

---

## Updated Technology Stack

| Component            | Technology                                |
|----------------------|-------------------------------------------|
| Language             | Python 3.10+                              |
| Deep Learning        | PyTorch                                   |
| LLM Models           | HuggingFace Transformers + Accelerate     |
| Weight Format        | SafeTensors                               |
| Tokenization         | HuggingFace Tokenizers / SentencePiece    |
| Quantization         | bitsandbytes (4-bit NF4, 8-bit INT8)      |
| GPU Monitoring       | pynvml (NVIDIA NVML)                      |
| CPU Monitoring       | psutil                                    |
| Inter-chunk Comm     | gRPC + Protocol Buffers                   |
| Containerization     | Docker (NVIDIA CUDA base images)          |
| Orchestration        | Kubernetes                                |
| K8s Client           | `kubernetes` Python package               |
| Dashboard            | Streamlit                                 |
| Plotting             | Matplotlib                                |
| Data Handling        | Pandas, NumPy                             |

---

*This guide will be updated as development progresses. Await instruction to begin implementation.*
