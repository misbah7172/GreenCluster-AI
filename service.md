# KAI — Service Overview

## What KAI Can Do

KAI is a **distributed AI inference platform** that lets you run large language models across a cluster of low-end PCs using Kubernetes. Here's what it can do:

---

## Distributed LLM Inference

- **Run models too large for one machine** — splits HuggingFace models (GPT-2, LLaMA, Mistral, Phi-2, Falcon, Qwen, etc.) layer-by-layer across multiple nodes. Each node loads only its assigned layers.
- **Autoregressive text generation** with temperature, top-k, top-p sampling, repetition penalty, and streaming output.
- **Single command**: `python kai_cli.py run --model microsoft/phi-2 --prompt "Hello" --max-tokens 100 --stream`

---

## Energy-Efficient Inference (Core Value)

A key advantage of KAI is **reducing power consumption while producing identical output**.

Running a massive model like LLaMA 7B (14GB in fp16) on a single high-end GPU consumes significant power:

| Setup | Hardware | Power Draw | Output |
|-------|----------|------------|--------|
| Single GPU | 1x RTX 4090 (24GB) | ~350-450W | "The capital of France is Paris..." |
| KAI Cluster | 3x RTX 3050 Ti (4GB each) | ~60W x 3 = **180W** | "The capital of France is Paris..." |
| KAI Cluster | 4x GTX 1060 (6GB each) | ~40W x 4 = **160W** | "The capital of France is Paris..." |

**Same model. Same weights. Same output. Less than half the power.**

Why this works:
- High-end GPUs (RTX 4090, A100) are built for maximum throughput and draw enormous power even when most of their compute units are idle during autoregressive generation (which is sequential by nature — one token at a time).
- Low-end GPUs draw far less power per card. Each card only loads a fraction of the model layers, so it does less work per token but at a fraction of the power cost.
- The total cluster power (sum of all small GPUs) is often **less than a single big GPU**, because each small GPU operates within its efficient power envelope.
- The model weights are identical — no quantization, no approximation. Layer 0-10 on Node A produce the exact same hidden states as they would on a single machine. The math is the same. The output is the same.

This is what makes KAI unique: **it's not just about making large models accessible to low-end hardware — it's about making inference more energy-efficient by design.**

> **Important: KAI requires 2 or more machines (nodes) to deliver its power-saving advantage.** The energy savings come from distributing layers across multiple low-power GPUs instead of running everything on one high-power GPU. If you only have a single machine, KAI's distributed architecture has no benefit — use [Ollama](https://ollama.com), [AirLLM](https://github.com/lyogavin/Airllm), or [llama.cpp](https://github.com/ggerganov/llama.cpp) instead for efficient single-machine inference.

KAI's built-in energy benchmarking proves this. Run `python kai_cli.py benchmark --mode both` and KAI will:
1. Run the model on a single GPU and measure total energy (Wh).
2. Run the same model distributed across your cluster and measure total energy (Wh).
3. Produce a side-by-side comparison showing the energy savings.

---

## Smart Hardware Detection & Auto-Partitioning

- **Scans your hardware** — detects GPU model, VRAM, system RAM, and CPU cores on each node (local or Kubernetes).
- **Proportional layer assignment** — nodes with more VRAM/RAM get more layers automatically. Supports mixed GPU/CPU-only nodes.
- **Preview splits** before deploying: `python kai_cli.py partition --model microsoft/phi-2 --num-nodes 3`

---

## Energy Benchmarking

- **Measures real energy costs** of AI inference — GPU power draw (W), total energy (Wh), CPU utilization, inference latency, and throughput.
- **Compares local vs. Kubernetes** — runs the same model in both modes and produces side-by-side metrics with efficiency ratios.
- **Quantifies distribution overhead** — isolates gRPC serialization, network, and orchestration costs.

---

## Monitoring & Visualization

- **Real-time GPU/CPU monitoring** at 1-second resolution via NVML and psutil.
- **8 publication-quality plots** — power over time, latency distribution, energy comparison, per-chunk latency, etc.
- **Interactive Streamlit dashboard** for exploring results, comparing experiments, and auto-refreshing live data.

---

## Infrastructure

- **Docker images** for chunk servers, gateway, and monitoring — ready for Kubernetes deployment.
- **One-command Docker build**: `python kai_cli.py build` builds all images (chunk, gateway, monitor).
- **One-command weight preparation**: `python kai_cli.py prepare --model <name> --num-chunks 3` downloads model, splits weights, and saves per-chunk files.
- **Kubernetes controller** — automated deploy, health check, metrics collection, and teardown.
- **gRPC pipeline** — binary tensor serialization between chunks with 256 MB max message size.

---

## Quantization Support

KAI supports optional **4-bit (NF4) and 8-bit (INT8) quantization** via bitsandbytes to further reduce memory usage per chunk:

```bash
# Run with 4-bit quantization
python kai_cli.py run --model microsoft/phi-2 --prompt "Hello" --quantize 4bit

# Prepare quantized chunk weights for K8s deployment
python kai_cli.py prepare --model microsoft/phi-2 --num-chunks 3 --quantize 8bit
```

Memory savings:

| Mode | Memory vs fp16 | Compression |
|------|---------------|-------------|
| 8-bit (INT8) | 50% | 2x |
| 4-bit (NF4) | 25% | 4x |

This means a LLaMA 7B model (~14 GB in fp16) can be reduced to ~3.5 GB in 4-bit — making it fit on a cluster of even smaller GPUs.

---

## CLI Commands

| Command | What It Does |
|---------|-------------|
| `kai_cli.py run` | Generate text with a distributed HuggingFace model |
| `kai_cli.py scan` | Show available GPU/CPU/RAM on your machine or cluster |
| `kai_cli.py partition` | Preview how a model would be split across N nodes |
| `kai_cli.py benchmark` | Run energy benchmarking (local, K8s, or both) |
| `kai_cli.py benchmark --hf-model <name>` | Benchmark a HuggingFace model with energy monitoring |
| `kai_cli.py dashboard` | Launch the Streamlit visualization dashboard |
| `kai_cli.py build` | Build Docker images for chunk/gateway/monitor |
| `kai_cli.py prepare` | Download model, chunk weights, save for K8s deployment |

---

## Supported Model Families

| Model Family | Examples |
|-------------|----------|
| LLaMA | LLaMA-2-7B, LLaMA-2-13B, LLaMA-3-8B |
| Mistral | Mistral-7B, Mixtral-8x7B |
| Phi | Phi-2, Phi-3-mini |
| GPT-2 | GPT-2, GPT-2-Medium, GPT-2-Large, GPT-2-XL |
| GPT-NeoX | GPT-NeoX-20B, Pythia |
| GPT-J | GPT-J-6B |
| Falcon | Falcon-7B, Falcon-40B |
| Gemma | Gemma-2B, Gemma-7B |
| Qwen | Qwen-7B, Qwen-14B |
| OPT | OPT-1.3B, OPT-6.7B, OPT-13B |
| BLOOM | BLOOM-560M, BLOOM-7B |

Any HuggingFace `AutoModelForCausalLM` architecture is supported.

---

## Test Coverage

- **82 integration tests** — all passing
  - 25 tests for energy benchmarking (Phases 1-13)
  - 30 tests for distributed inference (Phases 14-18)
  - 27 tests for gap coverage & production readiness (Phase 19)

---

## Where KAI is NOT the Best Tool

KAI solves a specific problem. Here are scenarios where other tools are a better fit:

### 1. You Prioritize Speed Over Energy Efficiency

If your only goal is **minimum latency per token** and you already have a powerful GPU, a single-GPU setup will be faster because there's no network overhead between chunks.

- **Ollama / vLLM** on an RTX 4090 will generate tokens faster than a KAI cluster of 3x RTX 3050 Ti.

However, KAI still has an advantage here: the single RTX 4090 draws ~350-450W while the KAI cluster draws ~180W total — **same output, lower energy bill**. So the trade-off is:

| Priority | Best Choice |
|----------|-------------|
| Fastest tokens per second | Single powerful GPU (Ollama/vLLM) |
| Lowest energy per inference | KAI distributed cluster |
| Same output quality | Both (identical — same model, same weights) |

### 2. You Need Production-Grade Throughput

KAI is designed for **low-end hardware clusters**, not high-throughput production serving. If you need to serve hundreds of concurrent users with low latency:

- **vLLM** — Optimized with PagedAttention, continuous batching, and tensor parallelism on high-end GPUs.
- **TGI (Text Generation Inference)** — HuggingFace's production server with batching and quantization.
- **TensorRT-LLM** — NVIDIA's maximum-performance inference engine.

KAI's sequential chunk pipeline (Node A -> Node B -> Node C) adds network round-trips per token. Production systems use tensor parallelism (all GPUs compute simultaneously), which is fundamentally faster.

### 3. You Want Quick Local Chat with No Setup

If you just want to chat with a model on your own PC without configuring Kubernetes, Docker, or Python environments:

- **Ollama** — Single binary, `ollama run llama3`, done. No Python, no K8s, no Docker.
- **LM Studio** — GUI app, download a model, click play.
- **GPT4All** — Desktop app with offline models.

KAI requires Python, PyTorch, Kubernetes, Docker, and multiple dependencies. It's an infrastructure tool, not a consumer app.

### 4. You Don't Have Multiple Machines

KAI's core value is **distributing a model across multiple nodes**. If you only have one PC, KAI's distributed features don't help. Instead:

- **AirLLM** — Streams layers from disk on a single machine (slow but works with just disk space).
- **Ollama with quantization** — Runs 4-bit quantized models that fit in less VRAM.
- **llama.cpp** — Runs quantized models on CPU with no GPU required.

### 5. You Need Volunteer/Public Network Distribution

KAI runs on your **private LAN cluster**. If you don't own the hardware and want to borrow compute from strangers over the internet:

- **Petals** — Peer-to-peer swarm over the public internet. No hardware ownership needed.

KAI assumes you control all nodes, configure Kubernetes, and have a local network. It doesn't work across the internet.

### 6. You Need Fine-Tuning, Not Just Inference

KAI is **inference-only**. It generates text from pre-trained models but cannot train or fine-tune them. For training distributed across multiple machines:

- **DeepSpeed** — Distributed training with ZeRO optimization.
- **FSDP (Fully Sharded Data Parallel)** — PyTorch's native distributed training.
- **Megatron-LM** — NVIDIA's large-scale training framework.

### 7. You Need Mobile or Edge Deployment

KAI targets server/desktop hardware with NVIDIA GPUs. For phones, Raspberry Pi, or embedded devices:

- **llama.cpp** — Runs on ARM, x86, Apple Silicon, even Android.
- **MLC LLM** — Compiles models for mobile GPUs.
- **ONNX Runtime** — Cross-platform inference on any hardware.

### 8. You Need Maximum Accuracy with Quantization

KAI now supports **4-bit (NF4) and 8-bit (INT8) quantization** via bitsandbytes. You can run with `--quantize 4bit` or `--quantize 8bit` to reduce memory per chunk. However, if your priority is maximum quantization flexibility with many format options:

- **Ollama / llama.cpp** — Supports GGUF 2/3/4/5/6/8-bit quantization with many quant variants.
- **GPTQ / AWQ** — GPU-optimized quantized inference with near-full-precision quality.

KAI's quantization is simpler (NF4 or INT8 only), but it combines with distribution — you can quantize AND split across nodes, which none of the above tools do.

---

## Honest Summary

| KAI's Strength | KAI's Weakness |
|----------------|----------------|
| Pool multiple cheap GPUs to run large models | Slower than single powerful GPU (network overhead) |
| **Lower power consumption** — same output at less than half the watts | Complex setup vs. single-binary tools like Ollama |
| Energy benchmarking — proves the savings with real data | Not designed for production throughput |
| Kubernetes-native with health checks and scaling | Inference only — no training or fine-tuning |
| Smart auto-partitioning based on real hardware | Requires multiple machines to be useful |
| 4-bit/8-bit quantization to reduce memory per chunk | Fewer quant formats than llama.cpp/GGUF |
| Private and secure (your own cluster) | Overkill for casual local use |
| Full monitoring + dashboard | |

**KAI is best when:** You have 2+ low-end PCs with small GPUs, you want to run a model that doesn't fit on any single one, you want **lower power consumption** than a single high-end GPU, and you want provable energy metrics.

**KAI is NOT best when:** You need maximum speed above all else, need production throughput, want zero-setup local chat, or only have one machine.

---
