"""
Base Interface for Objective Functions.

This abstract class defines the interface that all objective functions must implement,
allowing easy swapping of different fitness evaluation strategies in the genetic algorithm.
"""

from abc import ABC, abstractmethod
from typing import List
from Bio.SeqRecord import SeqRecord


class BaseObjectiveFunction(ABC):
    """
    Abstract base class for objective functions used in the genetic algorithm.
    """
    
    def __init__(self, initial_sequences: List[SeqRecord]):
        """
        Initialize the objective function with the initial sequences.
        
        Args:
            initial_sequences: List of initial sequences (SeqRecord objects)
        """
        self.initial_sequences = initial_sequences
        self.sequence_map = {seq.id: str(seq.seq) for seq in initial_sequences}
        self.sequence_ids = [seq.id for seq in initial_sequences]
    
    @abstractmethod
    def compute_fitness(self, aligned_sequences: List[str], **kwargs) -> float:
        """
        Compute the fitness score for a given alignment.
        
        This method must be implemented by all subclasses.
        
        Args:
            aligned_sequences: List of aligned sequences (strings with gaps)
            **kwargs: Additional parameters specific to each objective function
            
        Returns:
            Fitness score (float) - higher values indicate better alignments
        """
        pass
    
    def get_sequence_ids(self) -> List[str]:
        """
        Get the sequence IDs.
        
        Returns:
            List of sequence IDs
        """
        return self.sequence_ids
    
    def get_sequence_map(self) -> dict:
        """
        Get the mapping of sequence IDs to original sequences.
        
        Returns:
            Dictionary mapping sequence IDs to sequences
        """
        return self.sequence_map
