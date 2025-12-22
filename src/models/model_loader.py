"""
Model Loader Module

Loads a Seq2Seq model from HuggingFace and injects LoRA adapters.
"""

from typing import Tuple, List, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_model_with_lora(
    model_path: str,
    lora_r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a Seq2Seq model and inject LoRA adapters.
    
    Args:
        model_path: Path to the pretrained model (local or HuggingFace Hub).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA. 
                       Defaults to ["q_proj", "k_proj", "fc2"].
        dtype: Data type for model weights.
        device: Device to load model on ("cuda", "cuda:0", etc.).
    
    Returns:
        Tuple of (model with LoRA, tokenizer).
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "fc2"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Load base model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
    )
    
    # Inject LoRA
    model = get_peft_model(model, lora_config)
    
    # Move to device if specified
    if device:
        model = model.to(device)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return model, tokenizer


def get_lora_parameters(model: PreTrainedModel) -> List[torch.nn.Parameter]:
    """
    Get only the LoRA parameters from the model.
    
    Args:
        model: Model with LoRA adapters.
    
    Returns:
        List of LoRA parameters.
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_params.append(param)
    return lora_params


def count_parameters(model: PreTrainedModel) -> Tuple[int, int]:
    """
    Count trainable and total parameters.
    
    Args:
        model: The model to count parameters for.
    
    Returns:
        Tuple of (trainable_params, total_params).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
