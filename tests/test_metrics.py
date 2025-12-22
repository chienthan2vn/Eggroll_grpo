"""
Test Metrics Module

Tests for fitness functions (BLEU, COMET).
"""

import pytest

from src.utils.metrics import BLEUFitness, compute_bleu


class TestBLEUFitness:
    """Tests for BLEU fitness function."""
    
    def test_perfect_match(self):
        """Test BLEU score for identical strings."""
        fitness = BLEUFitness()
        
        predictions = ["Hello world", "This is a test"]
        references = ["Hello world", "This is a test"]
        
        score = fitness(predictions, references)
        
        assert score == 1.0  # Perfect match = 100 BLEU / 100
    
    def test_no_match(self):
        """Test BLEU score for completely different strings."""
        fitness = BLEUFitness()
        
        predictions = ["apple banana cherry"]
        references = ["dog cat mouse elephant"]
        
        score = fitness(predictions, references)
        
        assert score < 0.1  # Should be very low
    
    def test_partial_match(self):
        """Test BLEU score for partially matching strings."""
        fitness = BLEUFitness()
        
        predictions = ["The cat sat on the mat"]
        references = ["The cat is sitting on a mat"]
        
        score = fitness(predictions, references)
        
        assert 0.0 < score < 1.0  # Should be partial
    
    def test_compute_bleu_convenience(self):
        """Test convenience function."""
        predictions = ["Hello world"]
        references = ["Hello world"]
        
        score = compute_bleu(predictions, references)
        
        assert score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
