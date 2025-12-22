"""
Evolutionary Strategies Core

Implements the ES optimization algorithm for LoRA parameters.
Key concept: Instead of backprop, we use population-based optimization
where perturbations are evaluated and aggregated to update weights.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class ESConfig:
    """Configuration for ES optimizer."""
    sigma: float = 0.001  # Noise standard deviation
    learning_rate: float = 0.001  # Learning rate for weight updates
    population_size: int = 64  # Number of perturbations per generation
    antithetic: bool = True  # Use mirrored sampling (recommended)
    weight_decay: float = 0.0  # L2 regularization
    rank_transform: bool = True  # Use fitness rank transformation


class ESOptimizer:
    """
    Evolutionary Strategies Optimizer for LoRA parameters.
    
    This optimizer perturbs model weights with Gaussian noise,
    evaluates fitness, and updates weights based on the ES gradient estimate.
    
    The key insight: We don't need gradients! We estimate them via:
        gradient ≈ (1/σ) * Σ fitness_i * noise_i
    """
    
    def __init__(
        self,
        parameters: List[torch.nn.Parameter],
        config: ESConfig,
        device: torch.device,
    ):
        """
        Initialize ES optimizer.
        
        Args:
            parameters: List of parameters to optimize (LoRA weights).
            config: ES configuration.
            device: Device for computations.
        """
        self.parameters = list(parameters)
        self.config = config
        self.device = device
        
        # Flatten parameters for easier manipulation
        self._shapes = [p.shape for p in self.parameters]
        self._sizes = [p.numel() for p in self.parameters]
        self._total_size = sum(self._sizes)
        
        # Store center (mean) weights as a flat vector
        self._center = self._flatten_params()
        
        # For momentum (optional)
        self._momentum = torch.zeros_like(self._center)
        self.momentum_beta = 0.9
    
    def _flatten_params(self) -> torch.Tensor:
        """Flatten all parameters into a single vector."""
        return torch.cat([p.data.view(-1) for p in self.parameters])
    
    def _unflatten_params(self, flat: torch.Tensor) -> List[torch.Tensor]:
        """Convert flat vector back to list of tensors with original shapes."""
        tensors = []
        offset = 0
        for shape, size in zip(self._shapes, self._sizes):
            tensors.append(flat[offset:offset + size].view(shape))
            offset += size
        return tensors
    
    def _set_params(self, flat: torch.Tensor) -> None:
        """Set model parameters from flat vector."""
        offset = 0
        for param, size, shape in zip(self.parameters, self._sizes, self._shapes):
            param.data.copy_(flat[offset:offset + size].view(shape))
            offset += size
    
    def generate_perturbations(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate perturbation noise vectors.
        
        Uses deterministic seeding so workers can regenerate the same noise
        without needing to communicate the full noise vectors.
        
        Args:
            seed: Random seed for this generation.
        
        Returns:
            Tuple of (positive_noise, negative_noise) if antithetic, else just noise.
        """
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        
        half_pop = self.config.population_size // 2 if self.config.antithetic else self.config.population_size
        
        # Generate noise vectors
        noise = torch.randn(
            half_pop, self._total_size,
            generator=generator,
            device=self.device,
            dtype=self._center.dtype
        )
        
        if self.config.antithetic:
            # Mirrored sampling: for each noise, also use -noise
            return noise, -noise
        return noise, None
    
    def perturb_weights(self, noise: torch.Tensor, index: int) -> None:
        """
        Apply a specific perturbation to the model weights.
        
        Args:
            noise: Noise tensor (half_pop x total_size).
            index: Which noise vector to apply.
        """
        perturbed = self._center + self.config.sigma * noise[index]
        self._set_params(perturbed)
    
    def restore_center_weights(self) -> None:
        """Restore model weights to the center (unperturbed) state."""
        self._set_params(self._center)
    
    def compute_update(
        self,
        fitnesses: torch.Tensor,
        pos_noise: torch.Tensor,
        neg_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the ES gradient estimate.
        
        Args:
            fitnesses: Fitness scores for each perturbation.
                      Shape: (population_size,) where first half are positive,
                      second half are negative (if antithetic).
            pos_noise: Positive noise vectors.
            neg_noise: Negative noise vectors (if antithetic).
        
        Returns:
            Gradient estimate.
        """
        if self.config.rank_transform:
            # Rank-based fitness transformation (more robust)
            fitnesses = self._rank_transform(fitnesses)
        else:
            # Normalize fitnesses
            fitnesses = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        
        if self.config.antithetic and neg_noise is not None:
            half_pop = len(pos_noise)
            pos_fit = fitnesses[:half_pop]
            neg_fit = fitnesses[half_pop:]
            
            # Combined gradient estimate
            weighted_noise = (
                torch.einsum('i,ij->j', pos_fit, pos_noise) -
                torch.einsum('i,ij->j', neg_fit, pos_noise)
            )
            gradient = weighted_noise / (2 * half_pop * self.config.sigma)
        else:
            # Standard gradient estimate
            gradient = torch.einsum('i,ij->j', fitnesses, pos_noise) / (
                len(pos_noise) * self.config.sigma
            )
        
        return gradient
    
    def _rank_transform(self, fitnesses: torch.Tensor) -> torch.Tensor:
        """
        Transform fitnesses using rank-based normalization.
        Makes optimization more robust to outliers.
        """
        ranks = torch.argsort(torch.argsort(fitnesses)).float()
        ranks = ranks / (len(ranks) - 1) - 0.5  # Normalize to [-0.5, 0.5]
        return ranks
    
    def step(
        self,
        fitnesses: torch.Tensor,
        pos_noise: torch.Tensor,
        neg_noise: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Perform one ES update step.
        
        Args:
            fitnesses: Fitness scores for all perturbations.
            pos_noise: Positive noise vectors.
            neg_noise: Negative noise vectors (if antithetic).
        
        Returns:
            Mean fitness for logging.
        """
        # Compute gradient estimate
        gradient = self.compute_update(fitnesses, pos_noise, neg_noise)
        
        # Apply weight decay
        if self.config.weight_decay > 0:
            gradient = gradient - self.config.weight_decay * self._center
        
        # Update with momentum
        self._momentum = self.momentum_beta * self._momentum + (1 - self.momentum_beta) * gradient
        
        # Update center weights
        self._center = self._center + self.config.learning_rate * self._momentum
        
        # Apply to model
        self._set_params(self._center)
        
        return fitnesses.mean().item()
    
    def get_flat_weights(self) -> torch.Tensor:
        """Get current center weights as flat tensor."""
        return self._center.clone()
    
    def set_flat_weights(self, weights: torch.Tensor) -> None:
        """Set center weights from flat tensor."""
        self._center = weights.clone()
        self._set_params(self._center)
