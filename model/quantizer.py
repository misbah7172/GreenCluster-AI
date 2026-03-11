"""
Quantization utilities for KAI distributed inference.

Provides 4-bit (NF4) and 8-bit (INT8) quantization via bitsandbytes,
reducing memory usage per chunk so larger models fit on low-end GPUs.

Usage::

    from model.quantizer import quantize_module, estimate_quantized_memory

    # Quantize a module in-place
    quantized = quantize_module(module, mode="4bit")

    # Estimate memory savings
    est = estimate_quantized_memory(original_mb=14000, mode="4bit")
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SUPPORTED_MODES = ("4bit", "8bit")


def quantize_module(module: nn.Module, mode: str, device: str = "cuda:0") -> nn.Module:
    """Quantize a PyTorch module using bitsandbytes.

    Parameters
    ----------
    module : nn.Module
        The module to quantize.
    mode : str
        Quantization mode: ``"4bit"`` (NF4) or ``"8bit"`` (INT8).
    device : str
        Target device for quantized weights.

    Returns
    -------
    nn.Module
        The quantized module (modified in-place where possible).
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported quantization mode '{mode}'. Choose from: {SUPPORTED_MODES}")

    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.warning(
            "bitsandbytes not installed. Install with: pip install bitsandbytes>=0.41.0. "
            "Returning module without quantization."
        )
        return module

    logger.info("Quantizing module with mode=%s", mode)

    if mode == "8bit":
        _quantize_linear_layers_8bit(module, bnb, device)
    elif mode == "4bit":
        _quantize_linear_layers_4bit(module, bnb, device)

    return module


def _quantize_linear_layers_8bit(module: nn.Module, bnb, device: str) -> None:
    """Replace nn.Linear layers with bitsandbytes Int8 linear layers."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            has_bias = child.bias is not None
            new_layer = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=has_bias,
                has_fp16_weights=False,
            )
            new_layer.weight = bnb.nn.Int8Params(
                child.weight.data.to(torch.float16),
                requires_grad=False,
                has_fp16_weights=False,
            )
            if has_bias:
                new_layer.bias = nn.Parameter(child.bias.data.to(torch.float16))
            setattr(module, name, new_layer)
            logger.debug("Quantized %s to 8-bit", name)
        else:
            _quantize_linear_layers_8bit(child, bnb, device)


def _quantize_linear_layers_4bit(module: nn.Module, bnb, device: str) -> None:
    """Replace nn.Linear layers with bitsandbytes 4-bit (NF4) linear layers."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            has_bias = child.bias is not None
            new_layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=has_bias,
                compute_dtype=torch.float16,
                quant_type="nf4",
            )
            new_layer.weight = bnb.nn.Params4bit(
                child.weight.data.to(torch.float16),
                requires_grad=False,
                quant_type="nf4",
            )
            if has_bias:
                new_layer.bias = nn.Parameter(child.bias.data.to(torch.float16))
            setattr(module, name, new_layer)
            logger.debug("Quantized %s to 4-bit NF4", name)
        else:
            _quantize_linear_layers_4bit(child, bnb, device)


def estimate_quantized_memory(original_mb: float, mode: str) -> dict:
    """Estimate memory usage after quantization.

    Parameters
    ----------
    original_mb : float
        Original model size in MB (typically fp16).
    mode : str
        Quantization mode: ``"4bit"`` or ``"8bit"``.

    Returns
    -------
    dict
        Estimated sizes and savings.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from: {SUPPORTED_MODES}")

    if mode == "8bit":
        # INT8 is ~50% of fp16
        quantized_mb = original_mb * 0.5
    elif mode == "4bit":
        # NF4 is ~25% of fp16
        quantized_mb = original_mb * 0.25

    return {
        "original_mb": round(original_mb, 2),
        "quantized_mb": round(quantized_mb, 2),
        "savings_mb": round(original_mb - quantized_mb, 2),
        "compression_ratio": round(original_mb / quantized_mb, 2) if quantized_mb > 0 else 0,
        "mode": mode,
    }


def is_quantization_available() -> bool:
    """Check if bitsandbytes is installed and CUDA is available."""
    try:
        import bitsandbytes  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False
