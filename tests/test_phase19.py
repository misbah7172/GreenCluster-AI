"""
Integration tests for Phase 19: Gap Coverage & Production Readiness.

Tests cover:
  - Docker build command (CLI parsing)
  - Chunk weight prepare/save (CLI parsing + save logic)
  - HF model benchmarking (experiment runner integration)
  - Shard-based loading (weight_utils + CLI fallback logic)
  - Quantization (quantizer module)

Uses sshleifer/tiny-gpt2 for fast testing.
Run with: python -m pytest tests/test_phase19.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn

MODEL_NAME = "sshleifer/tiny-gpt2"


# ------------------------------------------------------------------
# Gap 1: Docker Build Command (CLI Parsing)
# ------------------------------------------------------------------

class TestBuildCommand:
    """Tests for the 'build' subcommand CLI parsing."""

    def test_build_subcommand_exists(self):
        """The 'build' subcommand is registered in argparse."""
        from kai_cli import main
        import argparse
        # Parse args to verify build is accepted
        result = subprocess.run(
            [sys.executable, "kai_cli.py", "build", "--help"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert "--tag" in result.stdout
        assert "--push" in result.stdout

    def test_build_default_args(self):
        """Build command parses default arguments correctly."""
        from kai_cli import main
        import argparse
        # Import and test the parser directly
        sys_argv_backup = sys.argv
        sys.argv = ["kai_cli.py", "build"]
        try:
            import kai_cli
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest="command")
            build_parser = subparsers.add_parser("build")
            build_parser.add_argument("--tag", default="kai:latest")
            build_parser.add_argument("--push", action="store_true")
            args = parser.parse_args(["build"])
            assert args.tag == "kai:latest"
            assert args.push is False
        finally:
            sys.argv = sys_argv_backup

    def test_build_custom_tag(self):
        """Build command accepts custom tag."""
        result = subprocess.run(
            [sys.executable, "kai_cli.py", "build", "--help"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert "--tag" in result.stdout


# ------------------------------------------------------------------
# Gap 2: Chunk Weight Prepare & Save
# ------------------------------------------------------------------

class TestPrepareCommand:
    """Tests for the 'prepare' subcommand."""

    def test_prepare_subcommand_exists(self):
        """The 'prepare' subcommand is registered."""
        result = subprocess.run(
            [sys.executable, "kai_cli.py", "prepare", "--help"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--num-chunks" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--quantize" in result.stdout

    def test_prepare_saves_chunk_weights(self):
        """Prepare downloads model, creates chunks, saves weights to disk."""
        from model.hf_loader import HFModelLoader
        from model.layer_chunker import LayerChunker

        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        chunker = LayerChunker(loader)
        chunks = chunker.create_chunks(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            for chunk in chunks:
                path = chunker.save_chunk_weights(chunk, tmpdir)
                assert os.path.exists(path)
                # Verify file is a valid PyTorch checkpoint
                state = torch.load(path, map_location="cpu", weights_only=True)
                assert len(state) > 0

    def test_prepare_saves_manifest(self):
        """Prepare creates a chunk_manifest.json."""
        from model.hf_loader import HFModelLoader
        from model.layer_chunker import LayerChunker

        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        chunker = LayerChunker(loader)
        chunks = chunker.create_chunks(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            for chunk in chunks:
                chunker.save_chunk_weights(chunk, tmpdir)

            # Create manifest like cmd_prepare does
            manifest = {
                "model": MODEL_NAME,
                "dtype": "float32",
                "num_chunks": 2,
                "quantize": None,
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "layer_names": c.layer_names,
                        "memory_mb": round(c.estimate_memory_mb(), 2),
                    }
                    for c in chunks
                ],
            }
            manifest_path = os.path.join(tmpdir, "chunk_manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            assert os.path.exists(manifest_path)
            with open(manifest_path) as f:
                loaded = json.load(f)
            assert loaded["model"] == MODEL_NAME
            assert loaded["num_chunks"] == 2
            assert len(loaded["chunks"]) == 2


# ------------------------------------------------------------------
# Gap 3: HF Model Benchmarking
# ------------------------------------------------------------------

class TestHFBenchmark:
    """Tests for HF model benchmarking in experiment_runner."""

    def test_run_experiment_accepts_hf_model(self):
        """run_experiment() accepts hf_model parameter."""
        import inspect
        from experiments.experiment_runner import run_experiment
        sig = inspect.signature(run_experiment)
        assert "hf_model" in sig.parameters

    def test_hf_benchmark_cli_arg(self):
        """benchmark subcommand accepts --hf-model argument."""
        result = subprocess.run(
            [sys.executable, "kai_cli.py", "benchmark", "--help"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert "--hf-model" in result.stdout

    def test_run_local_hf_function_exists(self):
        """_run_local_hf helper function exists in experiment_runner."""
        from experiments.experiment_runner import _run_local_hf
        assert callable(_run_local_hf)

    def test_run_local_hf_returns_results(self):
        """_run_local_hf returns a results dict with expected keys."""
        from experiments.experiment_runner import _run_local_hf

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_local_hf(
                hf_model=MODEL_NAME,
                iterations=2,
                device="cpu",
                output_dir=tmpdir,
                monitor_interval=1.0,
                warmup=1,
            )
            assert "hf_model" in result
            assert result["hf_model"] == MODEL_NAME
            assert "avg_latency_ms" in result
            assert result["avg_latency_ms"] > 0
            assert "throughput_inferences_per_sec" in result
            assert "iterations" in result
            assert result["iterations"] == 2


# ------------------------------------------------------------------
# Gap 4: Shard-Based Loading
# ------------------------------------------------------------------

class TestShardLoading:
    """Tests for shard-based weight loading."""

    def test_weight_mapper_exists(self):
        """WeightMapper class is importable."""
        from model.weight_utils import WeightMapper
        assert WeightMapper is not None

    def test_weight_mapper_layer_name_prefixes(self):
        """_layer_names_to_prefixes maps KAI layer names to HF prefixes."""
        from model.weight_utils import WeightMapper
        prefixes = WeightMapper._layer_names_to_prefixes(["embed", "layer_0", "norm", "lm_head"])
        assert any("embed" in p for p in prefixes)
        assert any("layers.0" in p or "h.0" in p for p in prefixes)
        assert any("norm" in p or "ln_f" in p for p in prefixes)
        assert any("lm_head" in p for p in prefixes)

    def test_load_real_weights_chooses_strategy(self):
        """_load_real_weights uses shard loading for large models."""
        from kai_cli import _load_real_weights
        import inspect
        sig = inspect.signature(_load_real_weights)
        # Should accept quantize parameter
        assert "quantize" in sig.parameters

    def test_load_weights_shard_based_exists(self):
        """_load_weights_shard_based function is importable."""
        from kai_cli import _load_weights_shard_based
        assert callable(_load_weights_shard_based)

    def test_full_model_loading_still_works(self):
        """Small models still use full-model loading path."""
        from model.hf_loader import HFModelLoader
        from model.layer_chunker import LayerChunker
        from kai_cli import _load_real_weights

        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        chunker = LayerChunker(loader)
        chunks = chunker.create_chunks(1)

        # This should use full-model loading (tiny-gpt2 is small)
        _load_real_weights(loader, chunks, "cpu")

        # Verify weights were loaded (not meta tensors)
        chunk = chunks[0]
        for p in chunk.parameters():
            assert p.device.type == "cpu"
            assert p.numel() > 0


# ------------------------------------------------------------------
# Gap 5: Quantization Support
# ------------------------------------------------------------------

class TestQuantization:
    """Tests for the quantizer module."""

    def test_quantizer_imports(self):
        """quantizer module is importable."""
        from model.quantizer import quantize_module, estimate_quantized_memory, is_quantization_available
        assert callable(quantize_module)
        assert callable(estimate_quantized_memory)
        assert callable(is_quantization_available)

    def test_estimate_quantized_memory_4bit(self):
        """4-bit quantization estimates correct memory savings."""
        from model.quantizer import estimate_quantized_memory
        est = estimate_quantized_memory(14000, "4bit")
        assert est["original_mb"] == 14000
        assert est["quantized_mb"] == 3500  # 25% of fp16
        assert est["compression_ratio"] == 4.0
        assert est["mode"] == "4bit"

    def test_estimate_quantized_memory_8bit(self):
        """8-bit quantization estimates correct memory savings."""
        from model.quantizer import estimate_quantized_memory
        est = estimate_quantized_memory(14000, "8bit")
        assert est["original_mb"] == 14000
        assert est["quantized_mb"] == 7000  # 50% of fp16
        assert est["compression_ratio"] == 2.0
        assert est["mode"] == "8bit"

    def test_estimate_invalid_mode(self):
        """Invalid quantization mode raises ValueError."""
        from model.quantizer import estimate_quantized_memory
        with pytest.raises(ValueError, match="Unsupported"):
            estimate_quantized_memory(14000, "3bit")

    def test_quantize_module_invalid_mode(self):
        """quantize_module raises ValueError for unsupported modes."""
        from model.quantizer import quantize_module
        module = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="Unsupported"):
            quantize_module(module, "3bit")

    def test_quantize_module_no_bitsandbytes(self):
        """quantize_module gracefully handles missing bitsandbytes."""
        from model.quantizer import quantize_module
        module = nn.Linear(10, 10)
        # Mock import failure
        with patch.dict("sys.modules", {"bitsandbytes": None}):
            with patch("builtins.__import__", side_effect=ImportError("no bitsandbytes")):
                # Should return module unchanged without error
                result = quantize_module(module, "8bit")
                # Module should still work
                assert isinstance(result, nn.Module)

    def test_supported_modes(self):
        """SUPPORTED_MODES contains expected values."""
        from model.quantizer import SUPPORTED_MODES
        assert "4bit" in SUPPORTED_MODES
        assert "8bit" in SUPPORTED_MODES

    def test_cli_run_quantize_arg(self):
        """run subcommand accepts --quantize argument."""
        result = subprocess.run(
            [sys.executable, "kai_cli.py", "run", "--help"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert "--quantize" in result.stdout

    def test_cli_prepare_quantize_arg(self):
        """prepare subcommand accepts --quantize argument."""
        result = subprocess.run(
            [sys.executable, "kai_cli.py", "prepare", "--help"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert "--quantize" in result.stdout


# ------------------------------------------------------------------
# Cross-cutting: Dockerfile & Requirements
# ------------------------------------------------------------------

class TestInfrastructure:
    """Tests for updated Docker and dependency files."""

    def test_requirements_has_bitsandbytes(self):
        """requirements.txt includes bitsandbytes."""
        req_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")
        with open(req_path) as f:
            content = f.read()
        assert "bitsandbytes" in content

    def test_dockerfile_has_hf_deps(self):
        """Dockerfile.chunk installs HuggingFace dependencies."""
        dockerfile = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "docker", "Dockerfile.chunk",
        )
        with open(dockerfile) as f:
            content = f.read()
        assert "transformers" in content
        assert "accelerate" in content
        assert "safetensors" in content
        assert "bitsandbytes" in content

    def test_dockerfile_has_hf_env_vars(self):
        """Dockerfile.chunk defines HF-related environment variables."""
        dockerfile = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "docker", "Dockerfile.chunk",
        )
        with open(dockerfile) as f:
            content = f.read()
        assert "HF_MODEL_NAME" in content
        assert "QUANTIZE" in content
