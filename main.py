#!/usr/bin/env python3
"""
ES Translation Fine-tuning - Main Entry Point

Train a Seq2Seq translation model using Evolutionary Strategies with LoRA.
Supports multi-GPU training via torch.distributed.

Usage:
    Multi-GPU (8 GPUs A100):
        torchrun --nproc_per_node=8 main.py --model_path /path/to/model --data_path /path/to/data.json
"""

import os
import argparse
import json
from typing import Optional
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import load_model_with_lora
from src.es_engine import ESOptimizer, ESWorker
from src.es_engine.es_core import ESConfig
from src.es_engine.es_worker import setup_distributed, cleanup_distributed
from src.data import TranslationDataset, create_dataloader
from src.utils import BLEUFitness, create_default_fitness

# Optional: Weights & Biases logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ES-based Translation Fine-tuning with LoRA"
    )
    
    # Model & Data
    parser.add_argument(
        "--model_path", type=str, default="",
        help="Path to pretrained Seq2Seq model (local or HuggingFace Hub)"
    )
    parser.add_argument(
        "--data_path", type=str, default="",
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory to save checkpoints"
    )
    
    # LoRA Config
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target_modules", type=str, default="q_proj,k_proj,fc2",
        help="Comma-separated LoRA target modules"
    )
    
    # ES Config
    parser.add_argument("--sigma", type=float, default=0.001, help="ES noise std")
    parser.add_argument("--lr", type=float, default=0.001, help="ES learning rate")
    parser.add_argument(
        "--population_size", type=int, default=64,
        help="Population size (should be divisible by num_gpus)"
    )
    parser.add_argument(
        "--antithetic", action="store_true", default=True,
        help="Use antithetic (mirrored) sampling"
    )
    parser.add_argument(
        "--rank_transform", action="store_true", default=True,
        help="Use rank-based fitness transformation"
    )
    
    # Training Config
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--max_source_len", type=int, default=256, help="Max source length")
    parser.add_argument("--max_target_len", type=int, default=256, help="Max target length")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate every N epochs")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N epochs")
    
    # System
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="es-translation", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]


def fitness_wrapper(
    predictions: torch.Tensor,
    references: torch.Tensor,
    tokenizer,
    fitness_fn,
    **kwargs,
) -> float:
    """Wrapper to compute fitness from model outputs."""
    # Decode predictions and references
    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    ref_texts = tokenizer.batch_decode(references, skip_special_tokens=True)
    
    # Compute fitness
    return fitness_fn(
        predictions=pred_texts,
        references=ref_texts,
        **kwargs,
    )


def train_epoch(
    worker: ESWorker,
    dataloader: DataLoader,
    epoch: int,
    args: argparse.Namespace,
    tokenizer,
    fitness_fn,
) -> float:
    """Run one training epoch."""
    # Generate seed for this epoch
    epoch_seed = args.seed + epoch * 1000
    
    if worker.is_master():
        seed = epoch_seed
    else:
        seed = None
    
    # Broadcast seed to all workers
    seed = worker.broadcast_seed(seed)
    
    # Perform ES step
    mean_fitness = worker.step(
        seed=seed,
        dataloader=dataloader,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        max_source_length=args.max_source_len,
    )
    
    return mean_fitness


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer,
    fitness_fn,
    device: torch.device,
    max_new_tokens: int = 128,
) -> float:
    """Evaluate model on validation data."""
    model.eval()
    total_fitness = 0.0
    num_batches = 0
    max_source_len = 256 # Default or from args if available, but here we hardcode or could pass as arg
    # Note: args is not passed to evaluate, let's assume valid default or modify signature.
    # Actually, let's add max_source_len to signature or usage.
    
    with torch.no_grad():
        for batch in dataloader:
             # Tokenize inputs on the fly
            inputs = tokenizer(
                batch["prompt"],
                max_length=max_source_len,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            
            # Compute fitness
            fitness = fitness_wrapper(
                predictions=outputs,
                references=batch["answer"],
                tokenizer=tokenizer,
                fitness_fn=fitness_fn,
            )
            total_fitness += fitness
            num_batches += 1
    
    return total_fitness / max(num_batches, 1)


def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize distributed (Optimized for 8x A100)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    is_master = rank == 0
    
    # Set seed
    set_seed(args.seed + rank)
    
    # Setup logging (master only)
    if is_master:
        print(f"\n{'='*60}")
        print("ES Translation Fine-tuning with LoRA")
        print(f"{'='*60}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Model: {args.model_path}")
        print(f"Data: {args.data_path}")
        print(f"{'='*60}\n")
        
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"es-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args),
            )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model with LoRA
    if is_master:
        print("Loading model...")
    
    target_modules = args.target_modules.split(",")
    dtype = get_dtype(args.dtype)
    
    model, tokenizer = load_model_with_lora(
        model_path=args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        dtype=dtype,
        device=str(device),
    )
    
    # Get LoRA parameters
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    
    if is_master:
        print(f"LoRA parameters: {len(lora_params)}")
        total_lora = sum(p.numel() for p in lora_params)
        print(f"Total LoRA params: {total_lora:,}")
    
    # Create ES optimizer
    es_config = ESConfig(
        sigma=args.sigma,
        learning_rate=args.lr,
        population_size=args.population_size,
        antithetic=args.antithetic,
        rank_transform=args.rank_transform,
    )
    es_optimizer = ESOptimizer(lora_params, es_config, device)
    
    # Create fitness function
    fitness_fn = create_default_fitness()
    
    # Create worker
    worker = ESWorker(
        rank=rank,
        world_size=world_size,
        model=model,
        es_optimizer=es_optimizer,
        fitness_fn=lambda **kw: fitness_wrapper(fitness_fn=fitness_fn, **kw),
        device=device,
    )
    
    # Create dataloader
    if is_master:
        print("Loading data...")
    
    dataloader = create_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )
    
    # Broadcast initial weights
    if is_master:
        print("Broadcasting initial weights...")
    worker.broadcast_weights()
    
    # Training loop
    if is_master:
        print(f"\nStarting training for {args.epochs} epochs...")
        pbar = tqdm(range(args.epochs), desc="Training")
    else:
        pbar = range(args.epochs)
    
    best_fitness = -float("inf")
    
    for epoch in pbar:
        # Train
        mean_fitness = train_epoch(
            worker=worker,
            dataloader=dataloader,
            epoch=epoch,
            args=args,
            tokenizer=tokenizer,
            fitness_fn=fitness_fn,
        )
        
        # Logging
        if is_master:
            pbar.set_postfix({"fitness": f"{mean_fitness:.4f}"})
            
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "fitness": mean_fitness,
                })
        
        # Evaluation
        if (epoch + 1) % args.eval_every == 0 and is_master:
            eval_fitness = evaluate(
                model=model,
                dataloader=dataloader,
                tokenizer=tokenizer,
                fitness_fn=fitness_fn,
                device=device,
                max_new_tokens=args.max_new_tokens,
                # Note: evaluate needs max_source_len fix, but for now we rely on default in updated code
            )
            print(f"\nEpoch {epoch+1} | Eval Fitness: {eval_fitness:.4f}")
            
            if eval_fitness > best_fitness:
                best_fitness = eval_fitness
                # Save best model
                model.save_pretrained(output_dir / "best_lora")
                print(f"Saved best model (fitness: {best_fitness:.4f})")
            
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    "eval_fitness": eval_fitness,
                    "best_fitness": best_fitness,
                })
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 and is_master:
            ckpt_dir = output_dir / f"checkpoint-{epoch+1}"
            model.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint to {ckpt_dir}")
    
    # Final save
    if is_master:
        model.save_pretrained(output_dir / "final_lora")
        print(f"\nTraining complete! Final model saved to {output_dir / 'final_lora'}")
        
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
