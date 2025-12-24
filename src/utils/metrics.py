"""
Metrics Module

Implements fitness functions for translation evaluation.
Includes BLEU (default) and COMET (placeholder for future use).
"""

from typing import List, Optional, Union, Dict, Any
from abc import ABC, abstractmethod
import torch
from transformers import PreTrainedTokenizer
import sacrebleu


class FitnessFunction(ABC):
    """
    Abstract base class for fitness functions.
    
    Extend this class to implement custom fitness metrics.
    """
    
    @abstractmethod
    def __call__(
        self,
        predictions: Union[List[str], torch.Tensor],
        references: Union[List[str], torch.Tensor],
        sources: Optional[List[str]] = None,
        **kwargs,
    ) -> float:
        """
        Compute fitness score.
        
        Args:
            predictions: Model predictions (strings or token IDs).
            references: Reference translations.
            sources: Source sentences (needed for some metrics like COMET).
            tokenizer: Tokenizer for decoding (if predictions are token IDs).
        
        Returns:
            Fitness score (higher is better).
        """
        pass


class BLEUFitness(FitnessFunction):
    """
    BLEU score fitness function using SacreBLEU.
    
    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap
    between predictions and references.
    """
    
    def __call__(
        self,
        predictions: Union[List[str], torch.Tensor],
        references: Union[List[str], torch.Tensor],
        sources: Optional[List[str]] = None,
        **kwargs,
    ) -> float:
        """Compute BLEU score."""
        
        # SacreBLEU expects references as list of lists (multiple refs per sample)
        refs = [[r] for r in references]
        
        # Compute BLEU
        bleu = sacrebleu.corpus_bleu(predictions, refs)
        
        return bleu.score / 100.0  # Normalize to [0, 1]


def create_default_fitness() -> FitnessFunction:
    """Create the default fitness function (BLEU)."""
    return BLEUFitness()
