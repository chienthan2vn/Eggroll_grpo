"""
Tests for Translation Dataset (HF Dataset version)
"""

import os
import json
import pytest
from src.data.dataset import get_dataset, create_dataloader

@pytest.fixture
def sample_data_file(tmp_path):
    # Format expected by get_dataset
    data = {
        "prompt": ["Dịch sang tiếng Hàn: Xin chào", "Cảm ơn"],
        "chosen": ["안녕하세요", "감사합니다"]
    }
    p = tmp_path / "data.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(p)

def test_get_dataset_json(sample_data_file):
    train_ds, test_ds = get_dataset(sample_data_file)
    
    # Train dataset should be loaded
    assert train_ds is not None
    assert len(train_ds) == 2
    assert train_ds[0]["prompt"] == "Dịch sang tiếng Hàn: Xin chào"
    assert train_ds[0]["src"] == "Dịch sang tiếng Hàn: Xin chào" # src mirrored from prompt
    assert train_ds[0]["answer"] == "안녕하세요"
    
    # Test dataset should be empty (0 rows) because eval paths are empty
    assert len(test_ds) == 0

def test_dataloader(sample_data_file):
    train_ds, _ = get_dataset(sample_data_file)
    loader = create_dataloader(train_ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    assert isinstance(batch, dict)
    assert "prompt" in batch
    assert isinstance(batch["prompt"], list)
    assert len(batch["prompt"]) == 2
    assert batch["prompt"][0] == "Dịch sang tiếng Hàn: Xin chào"
    assert batch["prompt"][1] == "Cảm ơn"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
