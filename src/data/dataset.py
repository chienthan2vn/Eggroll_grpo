"""
Translation Dataset Module

Handles loading and preprocessing of translation data.
"""

from typing import Dict, List, Optional, Any
import json
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import PreTrainedTokenizer


class TranslationDataset(Dataset):
    """
    Dataset for Vietnamese-Korean translation.
    
    Expected JSON format:
    {
        "src": "Câu tiếng Việt",
        "prompt": "Câu tiếng Việt (optional, same as src)",
        "answer": "Câu tiếng Hàn"
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 256,
        max_target_length: int = 256,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON file or directory.
            tokenizer: Tokenizer for encoding text.
            max_source_length: Maximum source sequence length.
            max_target_length: Maximum target sequence length.
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Load data
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load data from JSON file."""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            # Assume it's a single example
            data = [data]
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Get source and target texts
        source_text = item.get("src") or item.get("prompt", "")
        target_text = item.get("answer", "")
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(0),
            "attention_mask": source_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
            "decoder_attention_mask": target_encoding["attention_mask"].squeeze(0),
            # Keep raw text for metric computation
            "source_text": source_text,
            "target_text": target_text,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    
    Handles both tensor and string fields.
    """
    result = {}
    
    # Tensor fields
    tensor_keys = ["input_ids", "attention_mask", "labels", "decoder_attention_mask"]
    for key in tensor_keys:
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])
    
    # String fields
    string_keys = ["source_text", "target_text"]
    for key in string_keys:
        if key in batch[0]:
            result[key] = [item[key] for item in batch]
    
    return result


def create_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_source_length: int = 256,
    max_target_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader for translation data.
    
    Args:
        data_path: Path to JSON data file.
        tokenizer: Tokenizer for encoding.
        batch_size: Batch size per GPU.
        max_source_length: Maximum source length.
        max_target_length: Maximum target length.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        distributed: Whether using distributed training.
        rank: Process rank for distributed.
        world_size: World size for distributed.
    
    Returns:
        DataLoader instance.
    """
    dataset = TranslationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
