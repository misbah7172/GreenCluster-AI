"""
Layer-aware chunking for HuggingFace models.

Splits a large HuggingFace causal LM layer-by-layer so each chunk loads
only its assigned layers' weights. No single node needs to hold the full
model in memory.

Usage::

    from model.hf_loader import HFModelLoader
    from model.layer_chunker import LayerChunker

    loader = HFModelLoader("microsoft/phi-2", dtype="float16")
    chunker = LayerChunker(loader)
    chunks = chunker.create_chunks(num_chunks=3)
    for c in chunks:
        print(c.chunk_id, c.layer_names, c.estimate_memory_mb())
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LayerChunk(nn.Module):
    """A contiguous slice of a HuggingFace model's layers.

    Attributes
    ----------
    chunk_id : int
        Zero-based chunk index.
    num_chunks : int
        Total number of chunks in the partition.
    layer_names : list[str]
        Names of the layers in this chunk.
    has_embed : bool
        True if this chunk contains the embedding layer (chunk 0).
    has_lm_head : bool
        True if this chunk contains the LM head (last chunk).
    """

    def __init__(
        self,
        chunk_id: int,
        num_chunks: int,
        layers: List[Tuple[str, nn.Module]],
    ):
        super().__init__()
        self.chunk_id = chunk_id
        self.num_chunks = num_chunks
        self.layer_names = [name for name, _ in layers]

        # Store layers as a ModuleDict so parameters are tracked
        self.layers = nn.ModuleDict()
        for name, module in layers:
            self.layers[name] = module

        self.has_embed = any(n == "embed" for n in self.layer_names)
        self.has_lm_head = any(n == "lm_head" for n in self.layer_names)
        self.has_norm = any(n == "norm" for n in self.layer_names)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass through this chunk's layers sequentially.

        For the embedding chunk, ``x`` should be integer token IDs.
        For intermediate chunks, ``x`` should be hidden states.
        For the final chunk, the output will be logits over vocabulary.
        """
        for name in self.layer_names:
            module = self.layers[name]
            if name == "embed":
                # Embedding: int token IDs → hidden states
                x = module(x)
            elif name == "lm_head":
                # LM head: hidden states → logits
                x = module(x)
            elif name == "norm":
                # Final layer norm
                x = module(x)
            else:
                # Transformer block: hidden states → hidden states
                x = module(x)
        return x

    def estimate_memory_mb(self) -> float:
        """Estimate memory usage of this chunk in MB (based on parameter sizes)."""
        total_bytes = 0
        for p in self.parameters():
            total_bytes += p.numel() * p.element_size()
        # Add ~20% overhead for optimizer states, activations, etc.
        return (total_bytes * 1.2) / (1024 ** 2)

    def param_count(self) -> int:
        """Total number of parameters in this chunk."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        params_m = self.param_count() / 1e6
        return (
            f"LayerChunk(id={self.chunk_id}/{self.num_chunks}, "
            f"layers={self.layer_names}, params={params_m:.1f}M)"
        )


class LayerChunker:
    """Splits a HuggingFace model into LayerChunks for distributed inference.

    Parameters
    ----------
    loader : HFModelLoader
        An initialised HFModelLoader with model name and dtype.
    """

    def __init__(self, loader):
        self.loader = loader
        self._all_layers = None

    def create_chunks(self, num_chunks: int) -> List[LayerChunk]:
        """Split the model into ``num_chunks`` contiguous LayerChunks.

        The embedding layer is always in chunk 0.
        The norm and lm_head are always in the last chunk.
        Transformer blocks are distributed as evenly as possible.

        Parameters
        ----------
        num_chunks : int
            Number of chunks (must be >= 1 and <= num_transformer_blocks + 2).

        Returns
        -------
        list[LayerChunk]
            Ordered list of chunks.
        """
        all_layers = self._get_layers()

        # Separate special layers from transformer blocks
        embed_layers = []
        block_layers = []
        tail_layers = []  # norm + lm_head

        for name, mod in all_layers:
            if name == "embed":
                embed_layers.append((name, mod))
            elif name in ("norm", "lm_head"):
                tail_layers.append((name, mod))
            else:
                block_layers.append((name, mod))

        num_blocks = len(block_layers)
        if num_chunks < 1:
            raise ValueError("num_chunks must be >= 1")
        if num_chunks > num_blocks + 2:
            raise ValueError(
                f"num_chunks ({num_chunks}) exceeds available layers "
                f"({num_blocks} blocks + embed + tail)"
            )

        if num_chunks == 1:
            # Single chunk — everything in one
            return [LayerChunk(0, 1, embed_layers + block_layers + tail_layers)]

        # Distribute blocks across chunks.
        # Chunk 0 always gets embed + some blocks.
        # Last chunk always gets some blocks + tail.
        # This is a simplification; Phase 17 does smarter partitioning.
        blocks_per_chunk = self._distribute(num_blocks, num_chunks)

        chunks = []
        block_idx = 0
        for i in range(num_chunks):
            chunk_layers = []

            # Chunk 0 gets the embedding
            if i == 0:
                chunk_layers.extend(embed_layers)

            # Assign this chunk's share of transformer blocks
            n = blocks_per_chunk[i]
            chunk_layers.extend(block_layers[block_idx:block_idx + n])
            block_idx += n

            # Last chunk gets norm + lm_head
            if i == num_chunks - 1:
                chunk_layers.extend(tail_layers)

            chunks.append(LayerChunk(i, num_chunks, chunk_layers))

        logger.info(
            "Created %d chunks: %s",
            num_chunks,
            [(c.chunk_id, c.layer_names) for c in chunks],
        )
        return chunks

    def create_chunks_by_memory(
        self,
        node_memory_mb: List[float],
    ) -> List[LayerChunk]:
        """Split the model proportionally based on per-node memory budgets.

        Parameters
        ----------
        node_memory_mb : list[float]
            Available memory in MB for each node. The number of chunks
            equals ``len(node_memory_mb)``.

        Returns
        -------
        list[LayerChunk]
            Chunks assigned proportionally to node capability.
        """
        all_layers = self._get_layers()

        embed_layers = []
        block_layers = []
        tail_layers = []

        for name, mod in all_layers:
            if name == "embed":
                embed_layers.append((name, mod))
            elif name in ("norm", "lm_head"):
                tail_layers.append((name, mod))
            else:
                block_layers.append((name, mod))

        num_nodes = len(node_memory_mb)
        num_blocks = len(block_layers)

        if num_nodes < 1:
            raise ValueError("Must have at least 1 node")
        if num_nodes > num_blocks:
            raise ValueError(f"More nodes ({num_nodes}) than blocks ({num_blocks})")

        # Distribute blocks proportionally to memory
        total_mem = sum(node_memory_mb)
        blocks_per_node = []
        remaining_blocks = num_blocks
        for i, mem in enumerate(node_memory_mb):
            if i == num_nodes - 1:
                # Last node gets whatever is left
                blocks_per_node.append(remaining_blocks)
            else:
                share = max(1, round(num_blocks * mem / total_mem))
                share = min(share, remaining_blocks - (num_nodes - i - 1))
                blocks_per_node.append(share)
                remaining_blocks -= share

        chunks = []
        block_idx = 0
        for i in range(num_nodes):
            chunk_layers = []
            if i == 0:
                chunk_layers.extend(embed_layers)
            n = blocks_per_node[i]
            chunk_layers.extend(block_layers[block_idx:block_idx + n])
            block_idx += n
            if i == num_nodes - 1:
                chunk_layers.extend(tail_layers)
            chunks.append(LayerChunk(i, num_nodes, chunk_layers))

        logger.info(
            "Memory-based chunking: %s",
            [(c.chunk_id, len(c.layer_names), f"{c.estimate_memory_mb():.0f}MB")
             for c in chunks],
        )
        return chunks

    def get_partition_preview(self, num_chunks: int) -> List[Dict]:
        """Preview how the model would be partitioned without allocating chunks.

        Returns
        -------
        list[dict]
            Per-chunk info: chunk_id, layer_names, estimated_memory_mb, param_count.
        """
        chunks = self.create_chunks(num_chunks)
        preview = []
        for c in chunks:
            preview.append({
                "chunk_id": c.chunk_id,
                "layer_names": c.layer_names,
                "num_layers": len(c.layer_names),
                "estimated_memory_mb": round(c.estimate_memory_mb(), 2),
                "param_count": c.param_count(),
                "has_embed": c.has_embed,
                "has_lm_head": c.has_lm_head,
            })
        return preview

    # ------------------------------------------------------------------
    # Chunk persistence
    # ------------------------------------------------------------------

    def save_chunk_weights(self, chunk: LayerChunk, output_dir: str) -> str:
        """Save a chunk's weights to disk.

        Parameters
        ----------
        chunk : LayerChunk
            The chunk whose weights to save.
        output_dir : str
            Directory to write the file into.

        Returns
        -------
        str
            Path to the saved file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"layer_chunk_{chunk.chunk_id}.pt"
        torch.save(chunk.state_dict(), path)
        logger.info("Saved chunk %d weights to %s", chunk.chunk_id, path)
        return str(path)

    def load_chunk_weights(self, chunk: LayerChunk, weights_dir: str) -> LayerChunk:
        """Load saved weights into a chunk.

        Parameters
        ----------
        chunk : LayerChunk
            The chunk to load weights into.
        weights_dir : str
            Directory containing saved chunk files.

        Returns
        -------
        LayerChunk
            The chunk with loaded weights.
        """
        path = Path(weights_dir) / f"layer_chunk_{chunk.chunk_id}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        chunk.load_state_dict(state)
        chunk.eval()
        logger.info("Loaded chunk %d weights from %s", chunk.chunk_id, path)
        return chunk

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_layers(self):
        if self._all_layers is None:
            self._all_layers = self.loader.get_layer_list()
        return self._all_layers

    @staticmethod
    def _distribute(total: int, num_groups: int) -> List[int]:
        """Distribute ``total`` items into ``num_groups`` as evenly as possible."""
        base = total // num_groups
        remainder = total % num_groups
        result = []
        for i in range(num_groups):
            result.append(base + (1 if i < remainder else 0))
        return result
