"""
Mutation Operator for Multiple Sequence Alignment Genetic Algorithm.

Based on the research paper that describes the OP-mutation technique (optimal probability),
where two positions are randomly selected per chromosome and substituted one for the other.

Optimal mutation probability: 0.5
"""

import random
import copy
from typing import List
from genetic_algorithm.alignment import Alignment


class MutationOperator:
    def __init__(self, mutation_probability: float = 0.5):
        """
        Initialize the mutation operator.
        
        Args:
            mutation_probability: Mutation probability (default: 0.5 based on the paper)
        """
        self.mutation_probability = mutation_probability
        
        # Valid characters for biological sequences
        self.dna_chars = ['A', 'T', 'G', 'C', '-']
        self.protein_chars = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                             'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
    
    def mutate(self, individual: Alignment, sequence_type: str = "protein") -> Alignment:
        """
        Apply mutation to the individual.
        
        The mutation technique is based on random selection of two positions per chromosome
        and substitution of one for the other.
        
        Args:
            individual: Individual to be mutated (Alignment)
            sequence_type: Sequence type ("dna" or "protein")
            
        Returns:
            Mutated individual
        """
        # Check if mutation should occur
        if random.random() > self.mutation_probability:
            return individual.copy_alignment()
        
        # Create a copy of the individual to mutate
        mutated_individual = individual.copy_alignment()
        
        # Apply mutation to each sequence (chromosome)
        for segment in mutated_individual.aligned_segments:
            self._mutate_sequence(segment, sequence_type)
        
        return mutated_individual
    
    def _mutate_sequence(self, segment, sequence_type: str):
        """
        Apply mutation to a specific sequence.
        
        Implements the technique described in the paper: random selection of two positions
        per chromosome and substitution of one for the other.
        
        Args:
            segment: Aligned segment to be mutated
            sequence_type: Sequence type ("dna" or "protein")
        """
        sequence = list(segment.sequence)
        seq_length = len(sequence)
        
        if seq_length < 2:
            return  # Cannot mutate very short sequences
        
        # Method 1: Position swapping (as described in the paper)
        self._swap_positions_mutation(sequence)
        
        # Method 2: Point mutation (random alteration of one gene for another)
        self._point_mutation(sequence, sequence_type)
        
        # Update sequence in segment
        segment.sequence = ''.join(sequence)
    
    def _swap_positions_mutation(self, sequence: List[str]):
        """
        Implement position swapping mutation as described in the paper.
        
        Randomly selects two positions per chromosome and substitutes one for the other.
        
        Args:
            sequence: List of sequence characters
        """
        if len(sequence) < 2:
            return
        
        # Select two different random positions
        pos1, pos2 = random.sample(range(len(sequence)), 2)
        
        # Swap characters at both positions
        sequence[pos1], sequence[pos2] = sequence[pos2], sequence[pos1]
    
    def _point_mutation(self, sequence: List[str], sequence_type: str):
        """
        Implement point mutation by randomly altering one gene for another.
        
        Args:
            sequence: List of sequence characters
            sequence_type: Sequence type ("dna" or "protein")
        """
        if not sequence:
            return
        
        # Determine valid characters based on sequence type
        valid_chars = self.dna_chars if sequence_type.lower() == "dna" else self.protein_chars
        
        # Select a random position for mutation
        mutation_pos = random.randint(0, len(sequence) - 1)
        
        # Select a new character different from current
        current_char = sequence[mutation_pos]
        available_chars = [c for c in valid_chars if c != current_char]
        
        if available_chars:
            sequence[mutation_pos] = random.choice(available_chars)
    
    def gap_shift_mutation(self, individual: Alignment) -> Alignment:
        """
        Implement alignment-specific mutation: gap shifting.
        
        This type of mutation moves gaps to different positions, maintaining
        sequence length but altering the alignment.
        
        Args:
            individual: Individual to be mutated
            
        Returns:
            Mutated individual
        """
        # Check if mutation should occur
        if random.random() > self.mutation_probability:
            return individual.copy_alignment()
        
        mutated_individual = individual.copy_alignment()
        
        for segment in mutated_individual.aligned_segments:
            self._gap_shift_sequence(segment)
        
        return mutated_individual
    
    def _gap_shift_sequence(self, segment):
        """
        Apply gap shifting to a sequence.
        
        Args:
            segment: Aligned segment to be mutated
        """
        sequence = list(segment.sequence)
        
        # Find gap and non-gap positions
        gap_positions = [i for i, char in enumerate(sequence) if char == '-']
        non_gap_positions = [i for i, char in enumerate(sequence) if char != '-']
        
        if len(gap_positions) < 1 or len(non_gap_positions) < 1:
            return  # Not enough gaps or characters to shift
        
        # Randomly select a gap to move
        gap_pos = random.choice(gap_positions)
        
        # Select a new position for the gap
        new_pos = random.randint(0, len(sequence) - 1)
        
        if gap_pos != new_pos:
            # Move the gap
            sequence[gap_pos], sequence[new_pos] = sequence[new_pos], sequence[gap_pos]
        
        segment.sequence = ''.join(sequence)
    
    def insertion_deletion_mutation(self, individual: Alignment) -> Alignment:
        """
        Implement insertion/deletion gap mutation.
        
        Adds or removes gaps at random positions, adjusting alignment
        length as necessary.
        
        Args:
            individual: Individual to be mutated
            
        Returns:
            Mutated individual
        """
        # Check if mutation should occur
        if random.random() > self.mutation_probability:
            return individual.copy_alignment()
        
        mutated_individual = individual.copy_alignment()
        
        # Decide between insertion or deletion
        if random.random() < 0.5:
            self._insert_gap_column(mutated_individual)
        else:
            self._delete_gap_column(mutated_individual)
        
        return mutated_individual
    
    def _insert_gap_column(self, individual: Alignment):
        """
        Insert a gap column at a random position.
        
        Args:
            individual: Individual to be modified
        """
        if individual.alignment_length == 0:
            return
        
        # Select position for insertion
        insert_pos = random.randint(0, individual.alignment_length)
        
        # Insert gap in all sequences
        for segment in individual.aligned_segments:
            sequence = segment.sequence
            new_sequence = sequence[:insert_pos] + '-' + sequence[insert_pos:]
            segment.sequence = new_sequence
        
        # Update alignment length
        individual.alignment_length += 1
    
    def _delete_gap_column(self, individual: Alignment):
        """
        Remove a gap column if it exists.
        
        Args:
            individual: Individual to be modified
        """
        if individual.alignment_length <= 1:
            return
        
        # Search for gap-only columns
        gap_columns = []
        for pos in range(individual.alignment_length):
            if all(segment.sequence[pos] == '-' for segment in individual.aligned_segments):
                gap_columns.append(pos)
        
        if gap_columns:
            # Remove a random gap column
            del_pos = random.choice(gap_columns)
            
            for segment in individual.aligned_segments:
                sequence = segment.sequence
                new_sequence = sequence[:del_pos] + sequence[del_pos + 1:]
                segment.sequence = new_sequence
            
            # Update alignment length
            individual.alignment_length -= 1
    
    def set_mutation_probability(self, probability: float):
        """
        Set the mutation probability.
        
        Args:
            probability: New mutation probability (0.0 to 1.0)
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0")
        self.mutation_probability = probability