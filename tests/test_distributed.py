"""
Integration tests for Phases 14–18: Distributed LLM inference.

Uses sshleifer/tiny-gpt2 (a tiny GPT-2 model) for fast testing.
Run with: python -m pytest tests/test_distributed.py -v
"""

import tempfile
import pytest
import torch

# ------------------------------------------------------------------
# Phase 14: HFModelLoader
# ------------------------------------------------------------------

MODEL_NAME = "sshleifer/tiny-gpt2"


class TestHFModelLoader:
    """Tests for model.hf_loader.HFModelLoader."""

    def test_load_config(self):
        from model.hf_loader import HFModelLoader
        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        config = loader.get_config()
        assert "vocab_size" in config
        assert config["vocab_size"] == 50257

    def test_load_tokenizer(self):
        from model.hf_loader import HFModelLoader
        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        tok = loader.get_tokenizer()
        ids = tok("Hello", return_tensors="pt")
        assert ids["input_ids"].shape[0] == 1
        assert ids["input_ids"].shape[1] > 0

    def test_get_layer_list(self):
        from model.hf_loader import HFModelLoader
        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        layers = loader.get_layer_list()
        assert len(layers) >= 4  # embed + blocks + norm + lm_head
        names = [n for n, _ in layers]
        assert "embed" in names
        assert "lm_head" in names

    def test_get_num_layers(self):
        from model.hf_loader import HFModelLoader
        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        assert loader.get_num_layers() == 5  # embed + 2 blocks + norm + lm_head

    def test_model_size_estimate(self):
        from model.hf_loader import HFModelLoader
        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        est = loader.get_model_size_estimate()
        assert "params_millions" in est
        assert est["float32_mb"] > 0

    def test_validate_architecture(self):
        from model.hf_loader import HFModelLoader
        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        assert loader.validate_architecture() is True

    def test_invalid_dtype(self):
        from model.hf_loader import HFModelLoader
        with pytest.raises(ValueError, match="Unsupported dtype"):
            HFModelLoader(MODEL_NAME, dtype="int4")


# ------------------------------------------------------------------
# Phase 15: LayerChunker
# ------------------------------------------------------------------

class TestLayerChunker:
    """Tests for model.layer_chunker.LayerChunker."""

    def _make_chunker(self):
        from model.hf_loader import HFModelLoader
        from model.layer_chunker import LayerChunker
        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        return LayerChunker(loader)

    def test_create_2_chunks(self):
        chunker = self._make_chunker()
        chunks = chunker.create_chunks(2)
        assert len(chunks) == 2
        assert chunks[0].has_embed is True
        assert chunks[-1].has_lm_head is True

    def test_create_single_chunk(self):
        chunker = self._make_chunker()
        chunks = chunker.create_chunks(1)
        assert len(chunks) == 1
        assert chunks[0].has_embed is True
        assert chunks[0].has_lm_head is True

    def test_invalid_num_chunks(self):
        chunker = self._make_chunker()
        with pytest.raises(ValueError):
            chunker.create_chunks(0)

    def test_too_many_chunks(self):
        chunker = self._make_chunker()
        with pytest.raises(ValueError):
            chunker.create_chunks(100)

    def test_memory_based_chunking(self):
        chunker = self._make_chunker()
        chunks = chunker.create_chunks_by_memory([2000.0, 4000.0])
        assert len(chunks) == 2

    def test_partition_preview(self):
        chunker = self._make_chunker()
        preview = chunker.get_partition_preview(2)
        assert len(preview) == 2
        assert preview[0]["has_embed"] is True
        assert preview[-1]["has_lm_head"] is True

    def test_save_load_chunks(self):
        chunker = self._make_chunker()
        chunks = chunker.create_chunks(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            for c in chunks:
                chunker.save_chunk_weights(c, tmpdir)
            chunks2 = chunker.create_chunks(2)
            for c in chunks2:
                chunker.load_chunk_weights(c, tmpdir)

    def test_layer_names_coverage(self):
        """All layers from the model appear in exactly one chunk."""
        from model.hf_loader import HFModelLoader
        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        all_names = [n for n, _ in loader.get_layer_list()]

        chunker = self._make_chunker()
        chunks = chunker.create_chunks(2)
        chunk_names = []
        for c in chunks:
            chunk_names.extend(c.layer_names)
        assert sorted(chunk_names) == sorted(all_names)


# ------------------------------------------------------------------
# Phase 16: DistributedGenerator
# ------------------------------------------------------------------

class TestDistributedGenerator:
    """Tests for model.generation.DistributedGenerator."""

    def _make_generator(self):
        from model.hf_loader import HFModelLoader, _GPT2EmbedWrapper
        from model.layer_chunker import LayerChunker, LayerChunk
        from model.generation import DistributedGenerator
        from transformers import AutoModelForCausalLM

        loader = HFModelLoader(MODEL_NAME, dtype="float32")
        tokenizer = loader.get_tokenizer()

        # Load real model for actual generation
        real_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        real_model.eval()

        chunk0_layers = [
            ("embed", _GPT2EmbedWrapper(
                real_model.transformer.wte,
                real_model.transformer.wpe,
                real_model.transformer.drop,
            )),
            ("layer_0", real_model.transformer.h[0]),
        ]
        chunk1_layers = [
            ("layer_1", real_model.transformer.h[1]),
            ("norm", real_model.transformer.ln_f),
            ("lm_head", real_model.lm_head),
        ]

        chunks = [
            LayerChunk(0, 2, chunk0_layers),
            LayerChunk(1, 2, chunk1_layers),
        ]
        for c in chunks:
            c.eval()

        return DistributedGenerator(chunks, tokenizer, device="cpu"), real_model, tokenizer

    def test_generate_returns_string(self):
        gen, _, _ = self._make_generator()
        result = gen.generate("Hello", max_new_tokens=5, temperature=0)
        assert isinstance(result, str)
        assert len(result) > len("Hello")

    def test_generate_stream(self):
        gen, _, _ = self._make_generator()
        tokens = list(gen.generate_stream("Hi", max_new_tokens=3, temperature=0))
        assert len(tokens) >= 2  # at least prompt + 1 token

    def test_generate_greedy_deterministic(self):
        gen, _, _ = self._make_generator()
        r1 = gen.generate("Test", max_new_tokens=10, temperature=0)
        r2 = gen.generate("Test", max_new_tokens=10, temperature=0)
        assert r1 == r2

    def test_chunk_chain_matches_full_model(self):
        gen, real_model, tokenizer = self._make_generator()
        input_ids = tokenizer.encode("Test input", return_tensors="pt")

        with torch.no_grad():
            full_logits = real_model(input_ids).logits
            x = input_ids
            for c in gen.chunks:
                x = c(x)
            chunk_logits = x

        assert (full_logits - chunk_logits).abs().max().item() < 1e-4

    def test_stop_strings(self):
        gen, _, _ = self._make_generator()
        # Generate enough tokens then check stop
        result = gen.generate("A", max_new_tokens=50, temperature=0, stop_strings=["stairs"])
        # Should stop when "stairs" appears — or generate max tokens if never appears
        assert isinstance(result, str)


# ------------------------------------------------------------------
# Phase 17: ResourceDetector & AutoPartitioner
# ------------------------------------------------------------------

class TestResourceDetector:
    """Tests for model.resource_detector.ResourceDetector."""

    def test_local_scan(self):
        from model.resource_detector import ResourceDetector
        detector = ResourceDetector(mode="local")
        nodes = detector.scan()
        assert len(nodes) == 1
        assert nodes[0].cpu_cores > 0
        assert nodes[0].ram_mb > 0

    def test_scan_summary(self):
        from model.resource_detector import ResourceDetector
        detector = ResourceDetector(mode="local")
        summary = detector.scan_summary()
        assert summary["num_nodes"] == 1
        assert summary["total_ram_mb"] > 0

    def test_invalid_mode(self):
        from model.resource_detector import ResourceDetector
        with pytest.raises(ValueError):
            ResourceDetector(mode="invalid")

    def test_usable_memory(self):
        from model.resource_detector import NodeInfo
        gpu_node = NodeInfo(name="gpu", gpu_vram_mb=8192, has_gpu=True, ram_mb=16384)
        assert gpu_node.usable_memory_mb == 8192 - 500

        cpu_node = NodeInfo(name="cpu", ram_mb=16384, has_gpu=False)
        assert cpu_node.usable_memory_mb == 16384 * 0.7


class TestAutoPartitioner:
    """Tests for model.auto_partitioner.AutoPartitioner."""

    def _make_loader(self):
        from model.hf_loader import HFModelLoader
        return HFModelLoader(MODEL_NAME, dtype="float32")

    def test_single_node_plan(self):
        from model.auto_partitioner import AutoPartitioner
        from model.resource_detector import NodeInfo
        loader = self._make_loader()
        node = NodeInfo(name="n1", gpu_vram_mb=8192, has_gpu=True, ram_mb=16384)
        plan = AutoPartitioner().create_plan(loader, [node])
        assert plan.feasible is True
        assert len(plan.assignments) == 1

    def test_multi_node_plan(self):
        from model.auto_partitioner import AutoPartitioner
        from model.resource_detector import NodeInfo
        loader = self._make_loader()
        nodes = [
            NodeInfo(name="n1", gpu_vram_mb=4096, has_gpu=True, ram_mb=8192),
            NodeInfo(name="n2", gpu_vram_mb=8192, has_gpu=True, ram_mb=16384),
        ]
        plan = AutoPartitioner().create_plan(loader, nodes)
        assert plan.feasible is True
        assert len(plan.assignments) == 2

    def test_plan_validation(self):
        from model.auto_partitioner import AutoPartitioner
        from model.resource_detector import NodeInfo
        loader = self._make_loader()
        nodes = [
            NodeInfo(name="n1", gpu_vram_mb=4096, has_gpu=True, ram_mb=8192),
            NodeInfo(name="n2", ram_mb=16384, has_gpu=False),
        ]
        plan = AutoPartitioner().create_plan(loader, nodes)
        issues = AutoPartitioner().validate_plan(plan)
        assert len(issues) == 0

    def test_plan_summary(self):
        from model.auto_partitioner import AutoPartitioner
        from model.resource_detector import NodeInfo
        loader = self._make_loader()
        node = NodeInfo(name="n1", gpu_vram_mb=8192, has_gpu=True, ram_mb=16384)
        plan = AutoPartitioner().create_plan(loader, [node])
        summary = plan.summary()
        assert "sshleifer/tiny-gpt2" in summary
        assert "Feasible: True" in summary


# ------------------------------------------------------------------
# Phase 18: CLI (argument parsing)
# ------------------------------------------------------------------

class TestCLI:
    """Tests for kai_cli.py argument parsing."""

    def test_scan_command(self):
        """Scan command runs without error."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "kai_cli.py", "scan", "--mode", "local"],
            capture_output=True, text=True, cwd=r"C:\CODE\KAI",
            env={**__import__("os").environ, "PYTHONPATH": r"C:\CODE\KAI"},
            timeout=30,
        )
        assert result.returncode == 0
        assert "Cluster Summary" in result.stdout

    def test_partition_command(self):
        """Partition preview command runs without error."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "kai_cli.py", "partition",
             "--model", MODEL_NAME, "--num-nodes", "2"],
            capture_output=True, text=True, cwd=r"C:\CODE\KAI",
            env={**__import__("os").environ, "PYTHONPATH": r"C:\CODE\KAI"},
            timeout=60,
        )
        assert result.returncode == 0
        assert "Partition Plan" in result.stdout
