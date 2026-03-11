"""
HuggingFace model loader for KAI distributed inference.

Loads HuggingFace causal language models and exposes their layer structure
for distributed chunking across Kubernetes nodes. Supports memory-safe
loading so the full model weights never need to reside in RAM at once.

Usage::

    from model.hf_loader import HFModelLoader

    loader = HFModelLoader("microsoft/phi-2", dtype="float16")
    layers = loader.get_layer_list()
    tokenizer = loader.get_tokenizer()
    config = loader.get_config()
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Supported model architecture prefixes
_SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "GPT2LMHeadModel",
    "GPTNeoForCausalLM",
    "GPTNeoXForCausalLM",
    "GPTJForCausalLM",
    "FalconForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Qwen2ForCausalLM",
    "StableLmForCausalLM",
    "OPTForCausalLM",
    "BloomForCausalLM",
}

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class HFModelLoader:
    """Loads a HuggingFace causal LM and exposes its layer structure.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or local path
        (e.g. ``"microsoft/phi-2"``, ``"meta-llama/Llama-2-7b-hf"``).
    dtype : str
        Weight dtype — ``"float16"``, ``"bfloat16"``, or ``"float32"``.
    trust_remote_code : bool
        Whether to trust remote code from the model repo.
    token : str, optional
        HuggingFace access token for gated models.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        trust_remote_code: bool = False,
        token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.token = token

        if dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Choose from: {list(_DTYPE_MAP.keys())}"
            )
        self.torch_dtype = _DTYPE_MAP[dtype]

        self._tokenizer = None
        self._config = None
        self._model = None
        self._layer_list: Optional[List[Tuple[str, nn.Module]]] = None

        logger.info("HFModelLoader initialised for '%s' (dtype=%s)", model_name, dtype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> Dict:
        """Return the model configuration as a dictionary."""
        if self._config is None:
            self._load_config()
        return self._config.to_dict()

    def get_tokenizer(self):
        """Return the HuggingFace tokenizer for this model.

        Returns
        -------
        transformers.PreTrainedTokenizerBase
        """
        if self._tokenizer is None:
            self._load_tokenizer()
        return self._tokenizer

    def get_layer_list(self) -> List[Tuple[str, nn.Module]]:
        """Return an ordered list of ``(name, module)`` for all model layers.

        The list is structured as:
          - ``("embed", embedding_module)``  — token + position embeddings
          - ``("layer_0", block_0)``, ..., ``("layer_N", block_N)`` — transformer blocks
          - ``("norm", final_norm)``  — final layer norm
          - ``("lm_head", lm_head)``  — output projection to vocabulary

        Layers are returned on the ``meta`` device so no real memory is used
        until weights are explicitly loaded by the chunker.
        """
        if self._layer_list is None:
            self._build_layer_list()
        return list(self._layer_list)  # return a copy

    def get_num_layers(self) -> int:
        """Return the total number of distributable layers."""
        return len(self.get_layer_list())

    def get_model_size_estimate(self) -> Dict[str, float]:
        """Estimate model size in MB for different dtypes."""
        config = self.get_config()
        # Rough parameter count estimation
        num_params = self._estimate_param_count(config)
        return {
            "params_millions": num_params / 1e6,
            "float32_mb": num_params * 4 / (1024 ** 2),
            "float16_mb": num_params * 2 / (1024 ** 2),
            "int8_mb": num_params * 1 / (1024 ** 2),
        }

    def validate_architecture(self) -> bool:
        """Check that the model architecture is supported.

        Raises
        ------
        ValueError
            If the architecture is not a supported causal LM.
        """
        config = self.get_config()
        architectures = config.get("architectures", [])
        if not architectures:
            raise ValueError(
                f"Model '{self.model_name}' does not specify any architectures "
                f"in its config. Cannot determine if it is a causal LM."
            )
        arch = architectures[0]
        if arch not in _SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Architecture '{arch}' is not supported. "
                f"Supported: {sorted(_SUPPORTED_ARCHITECTURES)}"
            )
        logger.info("Architecture '%s' is supported.", arch)
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_config(self):
        from transformers import AutoConfig

        logger.info("Loading config for '%s'…", self.model_name)
        self._config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            token=self.token,
        )

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        logger.info("Loading tokenizer for '%s'…", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            token=self.token,
        )
        # Ensure pad_token is set (many LLMs lack one)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.info("Set pad_token = eos_token ('%s')", self._tokenizer.eos_token)

    def _load_model_meta(self):
        """Load the model on the ``meta`` device (no real memory used)."""
        from transformers import AutoModelForCausalLM

        if self._config is None:
            self._load_config()

        logger.info("Loading model '%s' on meta device…", self.model_name)
        self._model = AutoModelForCausalLM.from_config(
            self._config,
            dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        # Move to meta device — zero memory footprint
        self._model.to(torch.device("meta"))

    def _build_layer_list(self):
        """Decompose the model into an ordered layer list."""
        if self._model is None:
            self._load_model_meta()

        model = self._model
        layers = []

        # Detect the model structure — different HF model families use
        # slightly different attribute names.
        embed, blocks, norm, lm_head = self._detect_structure(model)

        # 1. Embedding layer(s)
        layers.append(("embed", embed))

        # 2. Transformer blocks
        for i, block in enumerate(blocks):
            layers.append((f"layer_{i}", block))

        # 3. Final norm
        if norm is not None:
            layers.append(("norm", norm))

        # 4. LM head
        if lm_head is not None:
            layers.append(("lm_head", lm_head))

        self._layer_list = layers
        logger.info(
            "Decomposed '%s' into %d layers: [%s]",
            self.model_name,
            len(layers),
            ", ".join(n for n, _ in layers),
        )

    @staticmethod
    def _detect_structure(model):
        """Auto-detect embedding, transformer blocks, norm, and lm_head.

        Supports the common HuggingFace structures:
          - model.model.embed_tokens / model.model.layers / model.model.norm / model.lm_head  (LLaMA, Mistral)
          - model.transformer.wte + wpe / model.transformer.h / model.transformer.ln_f / model.lm_head  (GPT-2)
          - model.gpt_neox.embed_in / model.gpt_neox.layers / model.gpt_neox.final_layer_norm / model.embed_out  (GPT-NeoX)
        """
        # --- LLaMA / Mistral / Phi / Gemma / Qwen style ---
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            inner = model.model
            embed = inner.embed_tokens
            blocks = list(inner.layers)
            norm = getattr(inner, "norm", getattr(inner, "final_layernorm", None))
            lm_head = getattr(model, "lm_head", None)
            return embed, blocks, norm, lm_head

        # --- GPT-2 style ---
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            tfm = model.transformer
            # Combine wte + wpe into a wrapper
            embed = _GPT2EmbedWrapper(tfm.wte, getattr(tfm, "wpe", None), getattr(tfm, "drop", None))
            blocks = list(tfm.h)
            norm = getattr(tfm, "ln_f", None)
            lm_head = getattr(model, "lm_head", None)
            return embed, blocks, norm, lm_head

        # --- GPT-NeoX style ---
        if hasattr(model, "gpt_neox"):
            inner = model.gpt_neox
            embed = inner.embed_in
            blocks = list(inner.layers)
            norm = getattr(inner, "final_layer_norm", None)
            lm_head = getattr(model, "embed_out", None)
            return embed, blocks, norm, lm_head

        # --- OPT style ---
        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            decoder = model.model.decoder
            embed = _OPTEmbedWrapper(
                decoder.embed_tokens,
                getattr(decoder, "embed_positions", None),
                getattr(decoder, "project_in", None),
            )
            blocks = list(decoder.layers)
            norm = getattr(decoder, "final_layer_norm", None)
            lm_head = getattr(model, "lm_head", None)
            return embed, blocks, norm, lm_head

        # --- BLOOM style ---
        if hasattr(model, "transformer") and hasattr(model.transformer, "word_embeddings"):
            tfm = model.transformer
            embed = _BloomEmbedWrapper(tfm.word_embeddings, getattr(tfm, "word_embeddings_layernorm", None))
            blocks = list(tfm.h)
            norm = getattr(tfm, "ln_f", None)
            lm_head = getattr(model, "lm_head", None)
            return embed, blocks, norm, lm_head

        raise ValueError(
            f"Could not auto-detect model structure for {type(model).__name__}. "
            f"Please check if the architecture is supported."
        )

    @staticmethod
    def _estimate_param_count(config: dict) -> int:
        """Rough parameter count from config fields."""
        hidden = config.get("hidden_size", config.get("n_embd", 768))
        n_layers = config.get("num_hidden_layers", config.get("n_layer", 12))
        vocab = config.get("vocab_size", 32000)
        intermediate = config.get("intermediate_size", hidden * 4)

        # Embedding: vocab * hidden
        params = vocab * hidden
        # Per-layer: ~12 * hidden^2 (attention + FFN, approximate)
        params += n_layers * (4 * hidden * hidden + 2 * hidden * intermediate)
        # LM head (often tied with embedding): vocab * hidden
        params += vocab * hidden
        return params


# ------------------------------------------------------------------
# Embedding wrappers for architectures that split embeddings
# ------------------------------------------------------------------

class _GPT2EmbedWrapper(nn.Module):
    """Combines GPT-2's word + position embeddings."""
    def __init__(self, wte, wpe, drop):
        super().__init__()
        self.wte = wte
        self.wpe = wpe
        self.drop = drop

    def forward(self, input_ids, **kwargs):
        pos_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids)
        if self.wpe is not None:
            x = x + self.wpe(pos_ids)
        if self.drop is not None:
            x = self.drop(x)
        return x


class _OPTEmbedWrapper(nn.Module):
    """Combines OPT's token + position embeddings."""
    def __init__(self, embed_tokens, embed_positions, project_in):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions
        self.project_in = project_in

    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        if self.embed_positions is not None:
            # OPT position embeddings expect attention_mask; generate one
            pos = self.embed_positions(
                torch.ones_like(input_ids, dtype=torch.long)
            )
            x = x + pos
        if self.project_in is not None:
            x = self.project_in(x)
        return x


class _BloomEmbedWrapper(nn.Module):
    """Combines BLOOM's word embeddings + layer norm."""
    def __init__(self, word_embeddings, word_embeddings_layernorm):
        super().__init__()
        self.word_embeddings = word_embeddings
        self.word_embeddings_layernorm = word_embeddings_layernorm

    def forward(self, input_ids, **kwargs):
        x = self.word_embeddings(input_ids)
        if self.word_embeddings_layernorm is not None:
            x = self.word_embeddings_layernorm(x)
        return x
