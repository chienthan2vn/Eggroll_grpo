"""
Metrics Module

Implements fitness functions for translation evaluation.
Includes BLEU (default) and COMET (placeholder for future use).
"""

from typing import List, Optional, Union, Dict, Any
from abc import ABC, abstractmethod
import torch
from transformers import PreTrainedTokenizer

# Try to import metrics libraries
try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not installed. BLEU metrics unavailable.")

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: unbabel-comet not installed. COMET metrics unavailable.")


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
        tokenizer: Optional[PreTrainedTokenizer] = None,
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
    
    def decode_if_needed(
        self,
        texts: Union[List[str], torch.Tensor],
        tokenizer: Optional[PreTrainedTokenizer],
    ) -> List[str]:
        """Helper to decode token IDs to strings if needed."""
        if isinstance(texts, torch.Tensor):
            if tokenizer is None:
                raise ValueError("Tokenizer required for decoding tensor predictions")
            return tokenizer.batch_decode(texts, skip_special_tokens=True)
        return texts


class BLEUFitness(FitnessFunction):
    """
    BLEU score fitness function using SacreBLEU.
    
    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap
    between predictions and references.
    """
    
    def __init__(self, lowercase: bool = False):
        """
        Initialize BLEU fitness.
        
        Args:
            lowercase: Whether to lowercase texts before computing.
        """
        if not SACREBLEU_AVAILABLE:
            raise ImportError("sacrebleu is required for BLEUFitness")
        self.lowercase = lowercase
    
    def __call__(
        self,
        predictions: Union[List[str], torch.Tensor],
        references: Union[List[str], torch.Tensor],
        sources: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ) -> float:
        """Compute BLEU score."""
        # Decode if needed
        pred_texts = self.decode_if_needed(predictions, tokenizer)
        ref_texts = self.decode_if_needed(references, tokenizer)
        
        # SacreBLEU expects references as list of lists (multiple refs per sample)
        refs = [[r] for r in ref_texts]
        
        # Compute BLEU
        bleu = sacrebleu.corpus_bleu(
            pred_texts,
            [[r[0] for r in refs]],  # Single reference per sample
            lowercase=self.lowercase,
        )
        
        return bleu.score / 100.0  # Normalize to [0, 1]


class COMETFitness(FitnessFunction):
    """
    COMET score fitness function.
    
    COMET (Crosslingual Optimized Metric for Evaluation of Translation)
    is a neural metric that considers source, hypothesis, and reference.
    
    NOTE: This is a placeholder. COMET requires downloading a model
    which can be several GB.
    """
    
    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da"):
        """
        Initialize COMET fitness.
        
        Args:
            model_name: Name of COMET model to use.
        """
        if not COMET_AVAILABLE:
            raise ImportError("unbabel-comet is required for COMETFitness")
        
        self.model_name = model_name
        self._model = None  # Lazy loading
    
    def _load_model(self):
        """Lazy load COMET model."""
        if self._model is None:
            model_path = download_model(self.model_name)
            self._model = load_from_checkpoint(model_path)
    
    def __call__(
        self,
        predictions: Union[List[str], torch.Tensor],
        references: Union[List[str], torch.Tensor],
        sources: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ) -> float:
        """Compute COMET score."""
        self._load_model()
        
        # Decode if needed
        pred_texts = self.decode_if_needed(predictions, tokenizer)
        ref_texts = self.decode_if_needed(references, tokenizer)
        
        if sources is None:
            raise ValueError("COMET requires source texts")
        
        # Prepare data for COMET
        data = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(sources, pred_texts, ref_texts)
        ]
        
        # Compute COMET
        output = self._model.predict(data, batch_size=8, gpus=1)
        
        return output.system_score


class CombinedFitness(FitnessFunction):
    """
    Combine multiple fitness functions with weights.
    
    Example:
        fitness = CombinedFitness([
            (BLEUFitness(), 0.7),
            (COMETFitness(), 0.3),
        ])
    """
    
    def __init__(self, fitness_functions: List[tuple]):
        """
        Initialize combined fitness.
        
        Args:
            fitness_functions: List of (FitnessFunction, weight) tuples.
        """
        self.fitness_functions = fitness_functions
        
        # Normalize weights
        total_weight = sum(w for _, w in fitness_functions)
        self.fitness_functions = [
            (fn, w / total_weight) for fn, w in fitness_functions
        ]
    
    def __call__(
        self,
        predictions: Union[List[str], torch.Tensor],
        references: Union[List[str], torch.Tensor],
        sources: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ) -> float:
        """Compute weighted sum of fitness scores."""
        total_score = 0.0
        
        for fn, weight in self.fitness_functions:
            score = fn(
                predictions=predictions,
                references=references,
                sources=sources,
                tokenizer=tokenizer,
                **kwargs,
            )
            total_score += weight * score
        
        return total_score


# Convenience functions

def compute_bleu(
    predictions: Union[List[str], torch.Tensor],
    references: Union[List[str], torch.Tensor],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    **kwargs,
) -> float:
    """
    Compute BLEU score (convenience function).
    
    Args:
        predictions: Predicted translations.
        references: Reference translations.
        tokenizer: Tokenizer for decoding (if needed).
    
    Returns:
        BLEU score in [0, 1].
    """
    fitness = BLEUFitness()
    return fitness(predictions, references, tokenizer=tokenizer, **kwargs)


def compute_comet(
    predictions: Union[List[str], torch.Tensor],
    references: Union[List[str], torch.Tensor],
    sources: List[str],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    **kwargs,
) -> float:
    """
    Compute COMET score (convenience function).
    
    Args:
        predictions: Predicted translations.
        references: Reference translations.
        sources: Source sentences.
        tokenizer: Tokenizer for decoding (if needed).
    
    Returns:
        COMET score.
    """
    fitness = COMETFitness()
    return fitness(predictions, references, sources=sources, tokenizer=tokenizer, **kwargs)


# Default fitness function for training
def create_default_fitness() -> FitnessFunction:
    """Create the default fitness function (BLEU)."""
    return BLEUFitness()
