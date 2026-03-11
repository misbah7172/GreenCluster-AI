"""
KAI CLI — Main entry point for distributed LLM inference on low-end hardware.

Run large AI models across a Kubernetes cluster of budget PCs. Each node
loads only the layers it is responsible for, so no single machine needs
enough VRAM/RAM for the full model.

Subcommands::

    kai run      — Download a model, partition across cluster, generate text.
    kai scan     — Detect cluster resources and show capabilities.
    kai partition — Preview how a model would be split (dry-run).
    kai benchmark — Run the original energy benchmarking workflow.
    kai dashboard — Launch the Streamlit dashboard.

Usage::

    python kai_cli.py run --model sshleifer/tiny-gpt2 --prompt "Hello" --max-tokens 50
    python kai_cli.py scan
    python kai_cli.py partition --model microsoft/phi-2 --num-nodes 3
    python kai_cli.py benchmark --model transformer --mode local
    python kai_cli.py dashboard
"""

import argparse
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger("kai")


def cmd_run(args):
    """Run distributed inference: download model, partition, generate text."""
    from model.hf_loader import HFModelLoader
    from model.layer_chunker import LayerChunker, LayerChunk
    from model.generation import DistributedGenerator
    from model.resource_detector import ResourceDetector

    print(f"[KAI] Loading model: {args.model}")
    loader = HFModelLoader(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )

    # Validate architecture
    try:
        loader.validate_architecture()
    except ValueError as e:
        print(f"[KAI] Error: {e}")
        sys.exit(1)

    # Get model info
    size_est = loader.get_model_size_estimate()
    dtype_key = "float16_mb" if args.dtype in ("float16", "fp16") else "float32_mb"
    est_mb = size_est.get(dtype_key, size_est["float32_mb"])
    print(f"[KAI] Model: ~{size_est['params_millions']:.0f}M params, ~{est_mb:.0f} MB ({args.dtype})")

    # Scan resources
    print("[KAI] Scanning resources...")
    detector = ResourceDetector(mode=args.resource_mode)
    nodes = detector.scan()
    total_usable = sum(n.usable_memory_mb for n in nodes)
    print(f"[KAI] Cluster: {len(nodes)} node(s), {total_usable:.0f} MB usable")

    # Create chunks
    num_chunks = args.num_chunks or len(nodes)
    print(f"[KAI] Partitioning model into {num_chunks} chunks...")
    chunker = LayerChunker(loader)

    if len(nodes) > 1:
        memory_budgets = [n.usable_memory_mb for n in nodes[:num_chunks]]
        chunks = chunker.create_chunks_by_memory(memory_budgets)
    else:
        chunks = chunker.create_chunks(num_chunks)

    for c in chunks:
        print(f"  Chunk {c.chunk_id}: {c.layer_names} (~{c.estimate_memory_mb():.0f} MB)")

    # Load real weights into chunks
    print("[KAI] Loading model weights...")
    _load_real_weights(loader, chunks, args.device)

    # Generate
    tokenizer = loader.get_tokenizer()
    gen = DistributedGenerator(chunks, tokenizer, device=args.device)

    print(f"[KAI] Generating (max_tokens={args.max_tokens}, temp={args.temperature})...")
    print("---")

    if args.stream:
        for token_text in gen.generate_stream(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ):
            print(token_text, end="", flush=True)
        print("\n---")
    else:
        result = gen.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(result)
        print("---")

    print("[KAI] Done.")


def cmd_scan(args):
    """Scan cluster resources."""
    from model.resource_detector import ResourceDetector

    print(f"[KAI] Scanning resources (mode={args.mode})...")
    detector = ResourceDetector(mode=args.mode)
    summary = detector.scan_summary()

    print(f"\nCluster Summary:")
    print(f"  Nodes: {summary['num_nodes']}")
    print(f"  GPU nodes: {summary['gpu_nodes']}")
    print(f"  CPU-only nodes: {summary['cpu_only_nodes']}")
    print(f"  Total GPU VRAM: {summary['total_gpu_vram_mb']:.0f} MB")
    print(f"  Total RAM: {summary['total_ram_mb']:.0f} MB")
    print(f"  Total usable: {summary['total_usable_mb']:.0f} MB")
    print()

    for node in summary["nodes"]:
        gpu_str = f"{node['gpu_type']} ({node['gpu_vram_mb']:.0f} MB)" if node["gpu_type"] != "none" else "none"
        print(f"  {node['name']}:")
        print(f"    GPU: {gpu_str}")
        print(f"    RAM: {node['ram_mb']:.0f} MB")
        print(f"    Usable for model: {node['usable_mb']:.0f} MB")

    # Estimate which models can fit
    print("\nModel Compatibility (approximate):")
    model_sizes = {
        "GPT-2 (124M)": 250,
        "Phi-2 (2.7B)": 5400,
        "Llama-2-7B": 14000,
        "Mistral-7B": 14000,
        "Llama-2-13B": 26000,
        "Llama-2-70B": 140000,
    }
    total_usable = summary["total_usable_mb"]
    for model_name, size_mb in model_sizes.items():
        fits = "YES" if total_usable >= size_mb else "NO"
        print(f"  {model_name} (~{size_mb} MB fp16): {fits}")


def cmd_partition(args):
    """Preview model partitioning without deploying."""
    from model.hf_loader import HFModelLoader
    from model.resource_detector import ResourceDetector, NodeInfo
    from model.auto_partitioner import AutoPartitioner

    print(f"[KAI] Loading model config: {args.model}")
    loader = HFModelLoader(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )

    if args.mode == "local":
        detector = ResourceDetector(mode="local")
        nodes = detector.scan()
        # Simulate multiple nodes by splitting local resources
        if args.num_nodes > 1:
            local = nodes[0]
            per_node_mb = local.usable_memory_mb / args.num_nodes
            nodes = [
                NodeInfo(
                    name=f"virtual-node-{i}",
                    gpu_vram_mb=local.gpu_vram_mb / args.num_nodes if local.has_gpu else 0,
                    gpu_type=local.gpu_type,
                    ram_mb=local.ram_mb / args.num_nodes,
                    cpu_cores=max(1, local.cpu_cores // args.num_nodes),
                    has_gpu=local.has_gpu,
                )
                for i in range(args.num_nodes)
            ]
    else:
        detector = ResourceDetector(mode="kubernetes")
        nodes = detector.scan()

    partitioner = AutoPartitioner()
    plan = partitioner.create_plan(loader, nodes)
    print()
    print(plan.summary())
    print()

    issues = partitioner.validate_plan(plan)
    if issues:
        print("Warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Plan validation: OK")


def cmd_benchmark(args):
    """Run the original energy benchmarking workflow."""
    from experiments.experiment_runner import run_experiment

    print(f"[KAI] Running benchmark: mode={args.mode}, model={args.model}")
    results = run_experiment(
        mode=args.mode,
        model_type=args.model,
        num_chunks=args.num_chunks,
        iterations=args.iterations,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    print("[KAI] Benchmark complete. Results saved to:", args.output_dir)


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    print("[KAI] Launching dashboard...")
    cmd = [
        sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
        "--server.headless", "true",
        "--server.port", str(args.port),
    ]
    subprocess.run(cmd)


def _load_real_weights(loader, chunks, device):
    """Load actual model weights into chunk modules.

    For local single-machine mode, loads the full model then distributes
    layers to chunks. For large models on limited hardware, this would
    be replaced with shard-based loading via weight_utils.
    """
    from transformers import AutoModelForCausalLM

    model_name = loader.model_name
    dtype = loader.torch_dtype

    # Try loading with real weights
    try:
        real_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=loader.trust_remote_code,
            token=loader.token,
        )
        real_model.eval()
    except Exception as e:
        logger.warning("Could not load full model: %s. Using meta weights.", e)
        return

    # Detect structure and get real layers
    from model.hf_loader import HFModelLoader
    embed, blocks, norm, lm_head = HFModelLoader._detect_structure(real_model)

    # Map layer names to real modules
    real_layer_map = {"embed": embed}
    for i, block in enumerate(blocks):
        real_layer_map[f"layer_{i}"] = block
    if norm is not None:
        real_layer_map["norm"] = norm
    if lm_head is not None:
        real_layer_map["lm_head"] = lm_head

    # Replace chunk modules with real-weight versions
    for chunk in chunks:
        for name in chunk.layer_names:
            if name in real_layer_map:
                chunk.layers[name] = real_layer_map[name]
        chunk.to(device)
        chunk.eval()


def main():
    parser = argparse.ArgumentParser(
        prog="kai",
        description="KAI — Run large AI models on clusters of low-end PCs",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Generate text with a distributed model")
    run_parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    run_parser.add_argument("--prompt", required=True, help="Input text prompt")
    run_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    run_parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    run_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    run_parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    run_parser.add_argument("--num-chunks", type=int, default=None, help="Number of chunks (default: auto)")
    run_parser.add_argument("--dtype", default="float16", help="Weight dtype (float16/bfloat16/float32)")
    run_parser.add_argument("--device", default="cpu", help="Compute device (cpu/cuda:0)")
    run_parser.add_argument("--stream", action="store_true", help="Stream tokens as generated")
    run_parser.add_argument("--resource-mode", default="local", help="Resource scan mode (local/kubernetes)")
    run_parser.add_argument("--trust-remote-code", action="store_true")
    run_parser.add_argument("--token", default=None, help="HuggingFace token for gated models")
    run_parser.set_defaults(func=cmd_run)

    # --- scan ---
    scan_parser = subparsers.add_parser("scan", help="Scan cluster resources")
    scan_parser.add_argument("--mode", default="local", choices=["local", "kubernetes"])
    scan_parser.set_defaults(func=cmd_scan)

    # --- partition ---
    part_parser = subparsers.add_parser("partition", help="Preview model partitioning")
    part_parser.add_argument("--model", required=True, help="HuggingFace model name")
    part_parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to simulate")
    part_parser.add_argument("--dtype", default="float16", help="Weight dtype")
    part_parser.add_argument("--mode", default="local", choices=["local", "kubernetes"])
    part_parser.add_argument("--trust-remote-code", action="store_true")
    part_parser.add_argument("--token", default=None)
    part_parser.set_defaults(func=cmd_partition)

    # --- benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Run energy benchmark (original KAI)")
    bench_parser.add_argument("--mode", default="local", choices=["local", "kubernetes", "both"])
    bench_parser.add_argument("--model", default="transformer", choices=["transformer", "cnn"])
    bench_parser.add_argument("--num-chunks", type=int, default=2)
    bench_parser.add_argument("--iterations", type=int, default=10)
    bench_parser.add_argument("--batch-size", type=int, default=8)
    bench_parser.add_argument("--output-dir", default="logs")
    bench_parser.set_defaults(func=cmd_benchmark)

    # --- dashboard ---
    dash_parser = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash_parser.add_argument("--port", type=int, default=8501)
    dash_parser.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
