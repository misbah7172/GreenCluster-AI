"""
Weight utilities for partial model loading.

Maps HuggingFace checkpoint shards to specific layer indices so that
each distributed chunk loads only the weight files it needs — avoiding
loading the entire model into memory.

Usage::

    from model.weight_utils import WeightMapper

    mapper = WeightMapper("meta-llama/Llama-2-7b-hf")
    shard_map = mapper.get_shard_map()
    # => {"model-00001-of-00002.safetensors": ["layer_0", "layer_1", ...], ...}

    mapper.load_chunk_weights(chunk, device="cuda:0")
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WeightMapper:
    """Maps HuggingFace checkpoint shards to model layers.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model name or local path.
    cache_dir : str, optional
        Local directory where model files are cached.
    token : str, optional
        HuggingFace authentication token.
    """

    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.token = token
        self._index = None
        self._model_path = None

    def get_model_path(self) -> str:
        """Return the local filesystem path to the model files.

        Downloads the model if not already cached.
        """
        if self._model_path is not None:
            return self._model_path

        from huggingface_hub import snapshot_download

        # If it's already a local path, use it directly
        if os.path.isdir(self.model_name_or_path):
            self._model_path = self.model_name_or_path
            return self._model_path

        logger.info("Resolving model path for '%s'…", self.model_name_or_path)
        self._model_path = snapshot_download(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            token=self.token,
            ignore_patterns=["*.gguf", "*.ggml", "original/**"],
        )
        logger.info("Model cached at: %s", self._model_path)
        return self._model_path

    def get_weight_index(self) -> Dict:
        """Load the weight-to-shard index.

        For sharded models this reads ``model.safetensors.index.json``
        or ``pytorch_model.bin.index.json``.
        For single-file models, all weights map to a single file.

        Returns
        -------
        dict
            ``{"weight_map": {"param_name": "shard_file", ...}}``
        """
        if self._index is not None:
            return self._index

        model_path = self.get_model_path()

        # Try safetensors index first, then pytorch bin index
        for index_name in [
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ]:
            index_file = os.path.join(model_path, index_name)
            if os.path.exists(index_file):
                with open(index_file, "r") as f:
                    self._index = json.load(f)
                logger.info("Loaded weight index from %s", index_name)
                return self._index

        # No index file — single-file model
        for fname in os.listdir(model_path):
            if fname.endswith(".safetensors") and not fname.endswith(".index.json"):
                single_file = fname
                break
            elif fname == "pytorch_model.bin":
                single_file = fname
                break
        else:
            raise FileNotFoundError(
                f"No weight files found in {model_path}. "
                "Expected .safetensors or pytorch_model.bin."
            )

        # Build a synthetic index for single-file models
        self._index = self._build_single_file_index(model_path, single_file)
        return self._index

    def get_shard_for_layers(self, layer_names: List[str]) -> Set[str]:
        """Determine which shard files are needed for the given layers.

        Parameters
        ----------
        layer_names : list[str]
            Layer names as used in the LayerChunk (e.g., "embed", "layer_0", "norm", "lm_head").

        Returns
        -------
        set[str]
            Set of shard filenames needed for these layers.
        """
        index = self.get_weight_index()
        weight_map = index.get("weight_map", {})

        # Map LayerChunk names → HF parameter name prefixes
        needed_prefixes = self._layer_names_to_prefixes(layer_names)

        needed_shards = set()
        for param_name, shard_file in weight_map.items():
            for prefix in needed_prefixes:
                if param_name.startswith(prefix):
                    needed_shards.add(shard_file)
                    break

        logger.info(
            "Layers %s need shards: %s",
            layer_names, sorted(needed_shards),
        )
        return needed_shards

    def load_state_dict_for_layers(
        self,
        layer_names: List[str],
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Load only the weight tensors for the specified layers.

        Parameters
        ----------
        layer_names : list[str]
            Layers to load weights for.
        device : str
            Target device for tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            State dict restricted to the requested layers.
        """
        model_path = self.get_model_path()
        needed_shards = self.get_shard_for_layers(layer_names)
        needed_prefixes = self._layer_names_to_prefixes(layer_names)

        state_dict = {}
        for shard_file in sorted(needed_shards):
            shard_path = os.path.join(model_path, shard_file)
            logger.info("Loading shard: %s", shard_file)

            if shard_file.endswith(".safetensors"):
                shard_tensors = self._load_safetensors(shard_path, device)
            else:
                shard_tensors = torch.load(
                    shard_path, map_location=device, weights_only=True
                )

            # Filter to only needed parameters
            for param_name, tensor in shard_tensors.items():
                for prefix in needed_prefixes:
                    if param_name.startswith(prefix):
                        state_dict[param_name] = tensor
                        break

        logger.info(
            "Loaded %d parameters for layers %s",
            len(state_dict), layer_names,
        )
        return state_dict

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_safetensors(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Load tensors from a safetensors file."""
        from safetensors.torch import load_file
        return load_file(path, device=device)

    @staticmethod
    def _build_single_file_index(model_path: str, filename: str) -> Dict:
        """Build a weight_map from a single-file checkpoint."""
        filepath = os.path.join(model_path, filename)

        if filename.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(filepath, framework="pt") as f:
                keys = f.keys()
        else:
            state = torch.load(filepath, map_location="cpu", weights_only=True)
            keys = state.keys()
            del state

        weight_map = {k: filename for k in keys}
        return {"weight_map": weight_map}

    @staticmethod
    def _layer_names_to_prefixes(layer_names: List[str]) -> List[str]:
        """Convert LayerChunk layer names to HuggingFace parameter prefixes.

        Mapping:
          "embed"    → "model.embed_tokens", "transformer.wte", "transformer.wpe", etc.
          "layer_N"  → "model.layers.N", "transformer.h.N", etc.
          "norm"     → "model.norm", "transformer.ln_f", etc.
          "lm_head"  → "lm_head", "embed_out"
        """
        prefixes = []
        for name in layer_names:
            if name == "embed":
                prefixes.extend([
                    "model.embed_tokens.",
                    "transformer.wte.",
                    "transformer.wpe.",
                    "transformer.word_embeddings.",
                    "transformer.word_embeddings_layernorm.",
                    "gpt_neox.embed_in.",
                    "model.decoder.embed_tokens.",
                    "model.decoder.embed_positions.",
                    "model.decoder.project_in.",
                ])
            elif name.startswith("layer_"):
                idx = name.split("_", 1)[1]
                prefixes.extend([
                    f"model.layers.{idx}.",
                    f"transformer.h.{idx}.",
                    f"gpt_neox.layers.{idx}.",
                    f"model.decoder.layers.{idx}.",
                ])
            elif name == "norm":
                prefixes.extend([
                    "model.norm.",
                    "transformer.ln_f.",
                    "gpt_neox.final_layer_norm.",
                    "model.decoder.final_layer_norm.",
                ])
            elif name == "lm_head":
                prefixes.extend([
                    "lm_head.",
                    "embed_out.",
                ])
        return prefixes
