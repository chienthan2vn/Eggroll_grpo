"""
ES Multi-GPU Worker

Handles distributed evaluation of perturbations across multiple GPUs.
Uses torch.distributed for communication.
"""

from typing import Dict, List, Optional, Callable, Any
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses import dataclass
import os

from .es_core import ESOptimizer, ESConfig


@dataclass
class WorkerConfig:
    """Configuration for ES worker."""
    world_size: int = 8  # Number of GPUs
    master_addr: str = "localhost"
    master_port: str = "12355"
    backend: str = "nccl"  # Use NCCL for GPU communication


def setup_distributed(rank: int, config: WorkerConfig) -> None:
    """
    Initialize distributed process group.
    
    Args:
        rank: Process rank (0 to world_size - 1).
        config: Worker configuration.
    """
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    
    dist.init_process_group(
        backend=config.backend,
        rank=rank,
        world_size=config.world_size,
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    dist.destroy_process_group()


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
        self.model.eval()
        total_fitness = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate outputs
                outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=fitness_kwargs.get("max_new_tokens", 128),
                    do_sample=False,  # Greedy for evaluation
                )
                
                # Compute fitness
                fitness = self.fitness_fn(
                    predictions=outputs,
                    references=batch.get("labels"),
                    tokenizer=fitness_kwargs.get("tokenizer"),
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


def run_worker(
    rank: int,
    world_size: int,
    model_factory: Callable[[], torch.nn.Module],
    es_config: ESConfig,
    fitness_fn: Callable,
    train_fn: Callable,
    **train_kwargs,
) -> None:
    """
    Entry point for each distributed worker.
    
    Args:
        rank: Worker rank.
        world_size: Total workers.
        model_factory: Function that creates the model.
        es_config: ES configuration.
        fitness_fn: Fitness evaluation function.
        train_fn: Main training function to run.
        **train_kwargs: Additional training arguments.
    """
    # Setup
    worker_config = WorkerConfig(world_size=world_size)
    setup_distributed(rank, worker_config)
    device = torch.device(f"cuda:{rank}")
    
    try:
        # Create model
        model = model_factory()
        model = model.to(device)
        
        # Get LoRA parameters
        lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
        
        # Create ES optimizer
        es_optimizer = ESOptimizer(lora_params, es_config, device)
        
        # Create worker
        worker = ESWorker(rank, world_size, model, es_optimizer, fitness_fn, device)
        
        # Run training
        train_fn(worker, **train_kwargs)
        
    finally:
        cleanup_distributed()
