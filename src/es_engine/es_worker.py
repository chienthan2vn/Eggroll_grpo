"""
ES Multi-GPU Worker

Handles distributed evaluation of perturbations across multiple GPUs.
Uses torch.distributed for communication.
"""

from typing import Callable
import torch
import torch.distributed as dist

from .es_core import ESOptimizer, ESConfig


class ESWorker:
    """
    Distributed ES worker for multi-GPU training.
    
    Each worker handles a subset of the population:
    - If population_size=64 and world_size=8, each worker evaluates 8 perturbations.
    - Workers generate their own noise deterministically using shared seeds.
    - Fitness scores are all-gathered for the ES update.
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        model: torch.nn.Module,
        es_optimizer: ESOptimizer,
        fitness_fn: Callable[..., torch.Tensor],
        device: torch.device,
    ):
        """
        Initialize ES worker.
        
        Args:
            rank: This worker's rank.
            world_size: Total number of workers.
            model: The model to optimize.
            es_optimizer: ES optimizer instance.
            fitness_fn: Function that evaluates fitness given model outputs.
            device: Device for this worker.
        """
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.es_optimizer = es_optimizer
        self.fitness_fn = fitness_fn
        self.device = device
        
        # Population split across workers
        self.pop_size = es_optimizer.config.population_size
        self.local_pop_size = self.pop_size // world_size
        
        assert self.pop_size % world_size == 0, (
            f"Population size {self.pop_size} must be divisible by world_size {world_size}"
        )
    
    def is_master(self) -> bool:
        """Check if this is the master worker."""
        return self.rank == 0
    
    def broadcast_weights(self) -> None:
        """Broadcast model weights from master to all workers."""
        flat_weights = self.es_optimizer.get_flat_weights()
        dist.broadcast(flat_weights, src=0)
        if not self.is_master():
            self.es_optimizer.set_flat_weights(flat_weights)
    
    def broadcast_seed(self, seed: Optional[int] = None) -> int:
        """
        Broadcast random seed from master to all workers.
        
        Args:
            seed: Seed value (only used on master).
        
        Returns:
            The broadcasted seed.
        """
        if seed is None and self.is_master():
            seed = torch.randint(0, 2**31, (1,)).item()
        
        seed_tensor = torch.tensor([seed if seed else 0], device=self.device)
        dist.broadcast(seed_tensor, src=0)
        return seed_tensor.item()
    
    def evaluate_perturbations(
        self,
        seed: int,
        dataloader: torch.utils.data.DataLoader,
        **fitness_kwargs,
    ) -> torch.Tensor:
        """
        Evaluate fitness for this worker's assigned perturbations.
        
        Args:
            seed: Seed for noise generation.
            dataloader: Data to evaluate on.
            **fitness_kwargs: Additional arguments for fitness function.
        
        Returns:
            Local fitness scores (shape: local_pop_size).
        """
        # Generate all noise (same on all workers due to shared seed)
        pos_noise, neg_noise = self.es_optimizer.generate_perturbations(seed)
        
        # Combine noise if antithetic
        all_noise = pos_noise if neg_noise is None else torch.cat([pos_noise, neg_noise], dim=0)
        
        # Determine which perturbations this worker handles
        start_idx = self.rank * self.local_pop_size
        end_idx = start_idx + self.local_pop_size
        
        local_fitnesses = []
        
        for i in range(start_idx, end_idx):
            # Apply perturbation
            self.es_optimizer.perturb_weights(all_noise, i)
            
            # Evaluate fitness
            fitness = self._evaluate_single(dataloader, **fitness_kwargs)
            local_fitnesses.append(fitness)
        
        # Restore center weights
        self.es_optimizer.restore_center_weights()
        
        return torch.tensor(local_fitnesses, device=self.device)
    
    def _evaluate_single(
        self,
        dataloader: torch.utils.data.DataLoader,
        **fitness_kwargs,
    ) -> float:
        """
        Evaluate fitness for a single perturbation.
        
        Args:
            dataloader: Data to evaluate on.
            **fitness_kwargs: Additional arguments for fitness function.
        
        Returns:
            Fitness score.
        """
        from src.utils import tokenize_and_generate
        
        self.model.eval()
        total_fitness = 0.0
        num_batches = 0
        
        if "tokenizer" not in fitness_kwargs:
            raise ValueError("Tokenizer required for evaluation but not found in kwargs")
        
        tokenizer = fitness_kwargs["tokenizer"]
        max_source_len = fitness_kwargs.get("max_source_length", 256)
        max_new_tokens = fitness_kwargs.get("max_new_tokens", 128)
        
        with torch.no_grad():
            for batch in dataloader:
                # Use shared tokenization utility
                outputs = tokenize_and_generate(
                    model=self.model,
                    tokenizer=tokenizer,
                    prompts=batch["prompt"],
                    device=self.device,
                    max_source_length=max_source_len,
                    max_new_tokens=max_new_tokens,
                )
                
                # Compute fitness
                fitness = self.fitness_fn(
                    predictions=outputs,
                    references=batch.get("answer"),
                    **fitness_kwargs,
                )
                total_fitness += fitness
                num_batches += 1
        
        return total_fitness / max(num_batches, 1)
    
    def gather_fitnesses(self, local_fitnesses: torch.Tensor) -> torch.Tensor:
        """
        Gather fitness scores from all workers.
        
        Args:
            local_fitnesses: This worker's fitness scores.
        
        Returns:
            All fitness scores (only meaningful on master).
        """
        # Gather all fitnesses
        all_fitnesses = [torch.zeros_like(local_fitnesses) for _ in range(self.world_size)]
        dist.all_gather(all_fitnesses, local_fitnesses)
        
        return torch.cat(all_fitnesses, dim=0)
    
    def step(
        self,
        seed: int,
        dataloader: torch.utils.data.DataLoader,
        **fitness_kwargs,
    ) -> float:
        """
        Perform one ES training step.
        
        Args:
            seed: Seed for noise generation.
            dataloader: Data to evaluate on.
            **fitness_kwargs: Additional arguments for fitness function.
        
        Returns:
            Mean fitness (only meaningful on master).
        """
        # Evaluate perturbations
        local_fitnesses = self.evaluate_perturbations(seed, dataloader, **fitness_kwargs)
        
        # Gather all fitnesses
        all_fitnesses = self.gather_fitnesses(local_fitnesses)
        
        # Perform ES update (on all workers, but result is the same)
        pos_noise, neg_noise = self.es_optimizer.generate_perturbations(seed)
        mean_fitness = self.es_optimizer.step(all_fitnesses, pos_noise, neg_noise)
        
        return mean_fitness
