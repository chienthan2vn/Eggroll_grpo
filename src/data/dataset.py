"""
Translation Dataset Module

Handles loading of raw translation data.
"""

from typing import Dict, List, Any
import json
from torch.utils.data import Dataset, DataLoader, DistributedSampler


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
    
    def __init__(self, data_path: str):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON file.
        """
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load data from JSON file."""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            data = [data]
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]
        
        src = item.get("src", "")
        prompt = item.get("prompt", "")
        if not prompt and src:
            prompt = src
            
        return {
            "prompt": prompt,
            "src": src,
            "answer": item.get("answer", "")
        }


def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Collate function for DataLoader.
    
    Returns a dictionary of lists.
    """
    result = {
        "prompt": [],
        "src": [],
        "answer": []
    }
    
    for item in batch:
        result["prompt"].append(item["prompt"])
        result["src"].append(item["src"])
        result["answer"].append(item["answer"])
        
    return result


def create_dataloader(
    data_path: str,
    batch_size: int = 8,
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
        batch_size: Batch size per GPU.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        distributed: Whether using distributed training.
        rank: Process rank for distributed.
        world_size: World size for distributed.
    
    Returns:
        DataLoader instance.
    """
    dataset = TranslationDataset(data_path=data_path)
    
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
