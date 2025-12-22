"""
Test ES Core Module

Tests for the Evolutionary Strategies optimizer.
"""

import pytest
import torch

from src.es_engine.es_core import ESOptimizer, ESConfig


class TestESOptimizer:
    """Tests for ESOptimizer."""
    
    @pytest.fixture
    def simple_params(self):
        """Create simple test parameters."""
        return [torch.randn(10, 10, requires_grad=False) for _ in range(3)]
    
    @pytest.fixture
    def es_config(self):
        """Default ES config for tests."""
        return ESConfig(
            sigma=0.1,
            learning_rate=0.01,
            population_size=8,
            antithetic=True,
            rank_transform=True,
        )
    
    @pytest.fixture
    def device(self):
        """Test device."""
        return torch.device("cpu")
    
    def test_init(self, simple_params, es_config, device):
        """Test ESOptimizer initialization."""
        optimizer = ESOptimizer(simple_params, es_config, device)
        
        assert optimizer._total_size == 300  # 3 * 10 * 10
        assert len(optimizer._shapes) == 3
        assert all(s == (10, 10) for s in optimizer._shapes)
    
    def test_perturbation_deterministic(self, simple_params, es_config, device):
        """Test that same seed produces same perturbation."""
        optimizer = ESOptimizer(simple_params, es_config, device)
        
        noise1_pos, noise1_neg = optimizer.generate_perturbations(seed=42)
        noise2_pos, noise2_neg = optimizer.generate_perturbations(seed=42)
        
        assert torch.allclose(noise1_pos, noise2_pos)
        assert torch.allclose(noise1_neg, noise2_neg)
    
    def test_perturbation_different_seeds(self, simple_params, es_config, device):
        """Test that different seeds produce different perturbations."""
        optimizer = ESOptimizer(simple_params, es_config, device)
        
        noise1_pos, _ = optimizer.generate_perturbations(seed=42)
        noise2_pos, _ = optimizer.generate_perturbations(seed=123)
        
        assert not torch.allclose(noise1_pos, noise2_pos)
    
    def test_antithetic_sampling(self, simple_params, es_config, device):
        """Test that antithetic sampling produces mirrored noise."""
        optimizer = ESOptimizer(simple_params, es_config, device)
        
        pos_noise, neg_noise = optimizer.generate_perturbations(seed=42)
        
        assert torch.allclose(pos_noise, -neg_noise)
    
    def test_perturb_and_restore(self, simple_params, es_config, device):
        """Test that weights can be perturbed and restored."""
        optimizer = ESOptimizer(simple_params, es_config, device)
        
        original_weights = optimizer.get_flat_weights().clone()
        
        # Perturb
        pos_noise, _ = optimizer.generate_perturbations(seed=42)
        optimizer.perturb_weights(pos_noise, index=0)
        
        perturbed_weights = optimizer.get_flat_weights()
        assert not torch.allclose(original_weights, perturbed_weights)
        
        # Restore
        optimizer.restore_center_weights()
        restored_weights = optimizer.get_flat_weights()
        assert torch.allclose(original_weights, restored_weights)
    
    def test_step_changes_weights(self, simple_params, es_config, device):
        """Test that ES step changes weights based on fitness."""
        optimizer = ESOptimizer(simple_params, es_config, device)
        
        original_center = optimizer._center.clone()
        
        # Generate noise
        pos_noise, neg_noise = optimizer.generate_perturbations(seed=42)
        
        # Create fake fitnesses (higher for positive perturbations)
        fitnesses = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.0])
        
        # Step
        optimizer.step(fitnesses, pos_noise, neg_noise)
        
        new_center = optimizer._center
        assert not torch.allclose(original_center, new_center)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
