"""
Tokenization Utilities

Shared utilities for tokenization and generation to eliminate code duplication.
"""

from typing import List
import torch
from transformers import PreTrainedTokenizer


def tokenize_and_generate(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    device: torch.device,
    max_source_length: int = 256,
    max_new_tokens: int = 128,
) -> torch.Tensor:
    """
    Tokenize inputs and generate outputs.
    
    Args:
        model: The model to use for generation.
        tokenizer: Tokenizer for encoding inputs.
        prompts: List of input prompts.
        device: Device to run generation on.
        max_source_length: Maximum length for input sequences.
        max_new_tokens: Maximum number of tokens to generate.
    
    Returns:
        Generated token IDs.
    """
    inputs = tokenizer(
        prompts,
        max_length=max_source_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    return outputs
