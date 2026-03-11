"""
Distributed text generation engine for KAI.

Implements autoregressive token generation across distributed LayerChunks.
Each chunk processes its assigned layers and passes hidden states to the
next chunk. The final chunk returns logits from which the next token is
sampled.

Supports temperature, top-k, top-p sampling, repetition penalty, and
streaming output.

Usage::

    from model.hf_loader import HFModelLoader
    from model.layer_chunker import LayerChunker
    from model.generation import DistributedGenerator

    loader = HFModelLoader("sshleifer/tiny-gpt2", dtype="float32")
    chunker = LayerChunker(loader)
    chunks = chunker.create_chunks(2)
    tokenizer = loader.get_tokenizer()

    gen = DistributedGenerator(chunks, tokenizer)
    text = gen.generate("Once upon a time", max_new_tokens=50)
    print(text)

    # Streaming
    for token_text in gen.generate_stream("Hello", max_new_tokens=20):
        print(token_text, end="", flush=True)
"""

import logging
from typing import Dict, Generator, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DistributedGenerator:
    """Autoregressive text generator using distributed LayerChunks.

    Parameters
    ----------
    chunks : list[LayerChunk]
        Ordered list of LayerChunks from LayerChunker.
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer.
    device : str
        Device for computation (``"cpu"`` or ``"cuda:0"``).
    """

    def __init__(self, chunks, tokenizer, device: str = "cpu"):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        # Move all chunks to device and set eval mode
        for chunk in self.chunks:
            chunk.to(self.device)
            chunk.eval()

        self._eos_token_id = tokenizer.eos_token_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop_strings: Optional[List[str]] = None,
    ) -> str:
        """Generate text from a prompt.

        Parameters
        ----------
        prompt : str
            Input text prompt.
        max_new_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature (higher = more random).
        top_k : int
            Top-k sampling (0 = disabled).
        top_p : float
            Nucleus sampling threshold.
        repetition_penalty : float
            Penalty for repeated tokens (1.0 = no penalty).
        stop_strings : list[str], optional
            Stop generation when any of these strings appear.

        Returns
        -------
        str
            Generated text (prompt + completion).
        """
        tokens = []
        for token_text in self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_strings=stop_strings,
        ):
            tokens.append(token_text)
        return "".join(tokens)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop_strings: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Stream generated tokens one at a time.

        Yields
        ------
        str
            Each decoded token as it is generated.
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()

        # Yield the prompt first
        yield prompt

        generated_text = prompt

        for step in range(max_new_tokens):
            # Forward through all chunks
            logits = self._forward_all_chunks(generated_ids)

            # Get logits for the last token position
            next_logits = logits[:, -1, :]  # (batch=1, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, generated_ids, repetition_penalty
                )

            # Sample next token
            next_token_id = self._sample(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Check for EOS
            if next_token_id.item() == self._eos_token_id:
                logger.info("EOS token generated at step %d", step)
                break

            # Append to sequence
            generated_ids = torch.cat(
                [generated_ids, next_token_id.unsqueeze(0)], dim=1
            )

            # Decode the new token
            token_text = self.tokenizer.decode(
                next_token_id, skip_special_tokens=True
            )
            generated_text += token_text
            yield token_text

            # Check stop strings
            if stop_strings:
                for stop in stop_strings:
                    if stop in generated_text:
                        logger.info("Stop string '%s' found at step %d", stop, step)
                        return

        logger.info(
            "Generation complete: %d tokens total",
            generated_ids.shape[1],
        )

    def _forward_all_chunks(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Pass input through all chunks sequentially.

        The first chunk receives token IDs and outputs hidden states.
        Intermediate chunks pass hidden states through.
        The last chunk (with lm_head) outputs logits.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        x = input_ids
        with torch.no_grad():
            for chunk in self.chunks:
                x = chunk(x)
        return x

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Penalize tokens that have already been generated."""
        for token_id in generated_ids[0].unique():
            tid = token_id.item()
            if logits[0, tid] > 0:
                logits[0, tid] /= penalty
            else:
                logits[0, tid] *= penalty
        return logits

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Sample a single token from logits using temperature, top-k, and top-p.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(1, vocab_size)``.

        Returns
        -------
        torch.Tensor
            Scalar token ID.
        """
        # Temperature
        if temperature <= 0:
            # Greedy
            return logits.argmax(dim=-1)
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_val = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = torch.where(logits < kth_val, torch.full_like(logits, -float("inf")), logits)

        # Top-p (nucleus) filtering
        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above top_p
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = -float("inf")

            # Scatter back to original order
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.squeeze(-1)
