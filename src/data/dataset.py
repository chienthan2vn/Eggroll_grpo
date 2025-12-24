"""
Translation Dataset Module

Handles loading of translation data using HuggingFace Datasets.
"""

import os
import json
from typing import Tuple, Dict, List, Any
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, DistributedSampler


def get_dataset(data_path: str) -> Tuple[Any, Any]:
    """
    Load dataset from JSON or disk.
    
    Args:
        data_path: Path to data file or directory.
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    try:
        # Try loading as JSON first
        src_text = []
        tgt_text = []
        
        with open(data_path, "r", encoding="utf-8") as f:
            file_content = json.load(f)
            
        src_text = [text.strip() for text in file_content.get('prompt', [])]
        tgt_text = [text.strip() for text in file_content.get('chosen', [])]
        
        train_data = {
            "prompt": src_text,
            "src": src_text,
            "answer": tgt_text,
        }
        
        # Eval data paths (can be extended as parameters if needed)
        eval_src_path = ""
        eval_tgt_path = ""
        
        eval_src = []
        eval_tgt = []
        
        # Guard logic: Only open files if paths are not empty string
        if eval_src_path and os.path.exists(eval_src_path):
            with open(eval_src_path, "r", encoding="utf-8") as f:
                eval_src = [text.strip() for text in f.readlines()]
                
        if eval_tgt_path and os.path.exists(eval_tgt_path):
            with open(eval_tgt_path, "r", encoding="utf-8") as f:
                eval_tgt = [text.strip() for text in f.readlines()]
        
        eval_data = {
            "prompt": eval_src,
            "src": eval_src,
            "answer": eval_tgt,
        }

        train_dataset = Dataset.from_dict(train_data)
        test_dataset = Dataset.from_dict(eval_data)
        
    except Exception as e:
        print(f"Failed to load as JSON (error: {e}). Trying load_from_disk...")
        try:
            loaded_dataset_dict = load_from_disk(data_path)
            # Handle case where keys might differ or dataset structure varies
            train_dataset = loaded_dataset_dict.get("train")
            test_dataset = loaded_dataset_dict.get("test")
        except Exception as disk_error:
            raise ValueError(f"Could not load dataset from {data_path} as JSON or disk. Errors: JSON({e}), Disk({disk_error})")

    return train_dataset, test_dataset


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Collate function for DataLoader.
    
    Returns a dictionary of lists.
    """
    return {
        "prompt": [item.get("prompt", "") for item in batch],
        "src": [item.get("src", "") for item in batch],
        "answer": [item.get("answer", "") for item in batch]
    }


def create_dataloader(
    dataset: Any,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader for a HF Dataset.
    
    Args:
        dataset: HuggingFace Dataset object.
        batch_size: Batch size per GPU.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        distributed: Whether using distributed training.
        rank: Process rank for distributed.
        world_size: World size for distributed.
    
    Returns:
        DataLoader instance.
    """
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle, # DistributedSampler handles shuffling
        )
        shuffle = False 
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
