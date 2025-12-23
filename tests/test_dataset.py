"""
Tests for Translation Dataset
"""

import os
import json
import pytest
import torch
from src.data.dataset import TranslationDataset, create_dataloader

@pytest.fixture
def sample_data_file(tmp_path):
    data = [
        {"src": "Xin chào", "prompt": "Dịch sang tiếng Hàn: Xin chào", "answer": "안녕하세요"},
        {"src": "Cảm ơn", "answer": "감사합니다"},  # Missing prompt, should default to src
    ]
    p = tmp_path / "data.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(p)

def test_dataset_loading(sample_data_file):
    dataset = TranslationDataset(sample_data_file)
    assert len(dataset) == 2
    
    item0 = dataset[0]
    assert item0["prompt"] == "Dịch sang tiếng Hàn: Xin chào"
    assert item0["src"] == "Xin chào"
    assert item0["answer"] == "안녕하세요"
    
    item1 = dataset[1]
    assert item1["src"] == "Cảm ơn"
    assert item1["prompt"] == "Cảm ơn"  # Defaults to src
    assert item1["answer"] == "감사합니다"

def test_dataloader(sample_data_file):
    loader = create_dataloader(sample_data_file, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    assert isinstance(batch, dict)
    assert "prompt" in batch
    assert isinstance(batch["prompt"], list)
    assert len(batch["prompt"]) == 2
    assert batch["prompt"][0] == "Dịch sang tiếng Hàn: Xin chào"
    assert batch["prompt"][1] == "Cảm ơn"

def test_distributed_dataloader(sample_data_file):
    # Mock distributed environment variables if needed, 
    # but DistributedSampler can run without actual dist init if num_replicas is provided
    loader = create_dataloader(
        sample_data_file, 
        batch_size=1, 
        distributed=True, 
        rank=0, 
        world_size=2
    )
    # With 2 items and world_size=2, each rank gets 1 item
    assert len(loader) == 1
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
