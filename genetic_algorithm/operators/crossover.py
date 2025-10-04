"""
Crossover (Recombination) Operator for Multiple Sequence Alignment Genetic Algorithm.

Based on the research paper that describes a new approach in mating technique,
where a specific section of the first chromosome is exchanged with the corresponding
section of the second chromosome.

Optimal crossover probability: 0.8
"""

import random
import copy
from typing import Tuple
from genetic_algorithm.alignment import Alignment, AlignedSegment


class CrossoverOperator:
    def __init__(self, crossover_probability: float = 0.8):
        """
        Initialize the crossover operator.
        
        Args:
            crossover_probability: Crossover probability (default: 0.8 based on the paper)
        """
        self.crossover_probability = crossover_probability
    
    def crossover(self, parent1: Alignment, parent2: Alignment) -> Tuple[Alignment, Alignment]:
        """
        Perform crossover between two parents to produce two offspring.
        
        The applied technique involves selecting a specific section in the first chromosome
        and exchanging the corresponding section with the second chromosome.
        
        Args:
            parent1: First parent (Alignment)
            parent2: Second parent (Alignment)
            
        Returns:
            Tuple containing two offspring (offspring1, offspring2)
        """
        # Check if crossover should occur based on probability
        if random.random() > self.crossover_probability:
            # Return copies of parents without modification
            return parent1.copy_alignment(), parent2.copy_alignment()
        
        # Check if parents have the same number of sequences
        if len(parent1.aligned_segments) != len(parent2.aligned_segments):
            raise ValueError("Parents must have the same number of sequences for crossover")
        
        # Create copies of parents to modify
        offspring1 = parent1.copy_alignment()
        offspring2 = parent2.copy_alignment()
        
        # Select crossover points
        num_sequences = len(parent1.aligned_segments)
        
        # Method 1: Crossover by entire sequence
        # Randomly select which sequences to swap
        sequences_to_swap = random.sample(range(num_sequences), random.randint(1, num_sequences // 2))
        
        for seq_index in sequences_to_swap:
            # Swap corresponding sequences between parents
            offspring1.aligned_segments[seq_index] = copy.deepcopy(parent2.aligned_segments[seq_index])
            offspring2.aligned_segments[seq_index] = copy.deepcopy(parent1.aligned_segments[seq_index])
        
        return offspring1, offspring2
    
    def single_point_crossover(self, parent1: Alignment, parent2: Alignment) -> Tuple[Alignment, Alignment]:
        """
        Perform single-point crossover for all sequences.
        
        This method implements the approach described in the paper where a specific section
        is selected and exchanged between chromosomes.
        
        Args:
            parent1: First parent (Alignment)
            parent2: Second parent (Alignment)
            
        Returns:
            Tuple containing two offspring (offspring1, offspring2)
        """
        # Check if crossover should occur
        if random.random() > self.crossover_probability:
            return parent1.copy_alignment(), parent2.copy_alignment()
        
        # Check parent compatibility
        if len(parent1.aligned_segments) != len(parent2.aligned_segments):
            raise ValueError("Parents must have the same number of sequences for crossover")
        
        # If lengths are different, use default crossover method
        if parent1.alignment_length != parent2.alignment_length:
            return self.crossover(parent1, parent2)
        
        # Create copies of parents
        offspring1 = parent1.copy_alignment()
        offspring2 = parent2.copy_alignment()
        
        # Select a random crossover point
        crossover_point = random.randint(1, parent1.alignment_length - 1)
        
        # Apply crossover to all sequences at the same point
        for i in range(len(parent1.aligned_segments)):
            # Get sequences from parents
            seq1 = parent1.aligned_segments[i].sequence
            seq2 = parent2.aligned_segments[i].sequence
            
            # Perform crossover
            new_seq1 = seq1[:crossover_point] + seq2[crossover_point:]
            new_seq2 = seq2[:crossover_point] + seq1[crossover_point:]
            
            # Update offspring
            offspring1.aligned_segments[i].sequence = new_seq1
            offspring2.aligned_segments[i].sequence = new_seq2
        
        return offspring1, offspring2
    
    def uniform_crossover(self, parent1: Alignment, parent2: Alignment) -> Tuple[Alignment, Alignment]:
        """
        Perform uniform crossover where each position is exchanged with a probability.
        
        Args:
            parent1: First parent (Alignment)
            parent2: Second parent (Alignment)
            
        Returns:
            Tuple containing two offspring (offspring1, offspring2)
        """
        # Check if crossover should occur
        if random.random() > self.crossover_probability:
            return parent1.copy_alignment(), parent2.copy_alignment()
        
        # Check parent compatibility
        if len(parent1.aligned_segments) != len(parent2.aligned_segments):
            raise ValueError("Parents must have the same number of sequences for crossover")
        
        # If lengths are different, use default crossover method
        if parent1.alignment_length != parent2.alignment_length:
            return self.crossover(parent1, parent2)
        
        # Create copies of parents
        offspring1 = parent1.copy_alignment()
        offspring2 = parent2.copy_alignment()
        
        # Apply uniform crossover
        for i in range(len(parent1.aligned_segments)):
            seq1 = list(parent1.aligned_segments[i].sequence)
            seq2 = list(parent2.aligned_segments[i].sequence)
            
            new_seq1 = []
            new_seq2 = []
            
            for j in range(len(seq1)):
                if random.random() < 0.5:
                    # Keep original characters
                    new_seq1.append(seq1[j])
                    new_seq2.append(seq2[j])
                else:
                    # Exchange characters
                    new_seq1.append(seq2[j])
                    new_seq2.append(seq1[j])
            
            # Update offspring
            offspring1.aligned_segments[i].sequence = ''.join(new_seq1)
            offspring2.aligned_segments[i].sequence = ''.join(new_seq2)
        
        return offspring1, offspring2
    
    def set_crossover_probability(self, probability: float):
        """
        Set the crossover probability.
        
        Args:
            probability: New crossover probability (0.0 to 1.0)
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0")
        self.crossover_probability = probability