"""
Mutation Operator for Multiple Sequence Alignment Genetic Algorithm.

Based on the research paper that describes the OP-mutation technique (optimal probability),
where two positions are randomly selected per chromosome and substituted one for the other.

Additionally implements SAGA mutation operators:
- Gap Insertion with phylogenetic groups
- Block Shuffling (16 variants)
- Block Searching
- Local Optimal Rearrangement

Optimal mutation probability: 0.5
"""

import random
from typing import List
from genetic_algorithm.alignment import Alignment
from genetic_algorithm.utils.phylogenetic_tree import PhylogeneticTreeHelper
from genetic_algorithm.operators.saga_operators import (
    SAGABlockOperators, SAGABlockSearching, SAGALocalRearrangement
)


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
        
        # Initialize SAGA operators
        self.saga_block_ops = SAGABlockOperators(mutation_probability)
        self.saga_block_search = SAGABlockSearching(mutation_probability)
        self.saga_local_rearrange = SAGALocalRearrangement(mutation_probability)
    
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
    
    # ========== SAGA OPERATORS ==========
    
    def saga_gap_insertion(self, individual: Alignment, objective_function=None,
                          mode: str = 'stochastic', max_gap_length: int = 5) -> Alignment:
        """
        SAGA Gap Insertion Operator.
        
        Divides sequences into two groups (G1 and G2) based on phylogenetic tree,
        then inserts gaps of random length at different positions.
        
        Args:
            individual: Individual to be mutated
            objective_function: Objective function for hill-climbing mode
            mode: 'stochastic' or 'hill_climbing'
            max_gap_length: Maximum length of gap to insert
            
        Returns:
            Mutated individual
        """
        # Check if mutation should occur
        if random.random() > self.mutation_probability:
            return individual.copy_alignment()
        
        mutated_individual = individual.copy_alignment()
        
        # Build phylogenetic tree and split sequences into two groups
        tree_helper = PhylogeneticTreeHelper()
        G1_indices, G2_indices = tree_helper.split_sequences_by_tree(
            mutated_individual.aligned_segments, 
            split_method='subtree'
        )
        
        # Generate random gap length
        gap_length = random.randint(1, max_gap_length)
        gap_string = '-' * gap_length
        
        # Select position P1 for G1
        if mode == 'hill_climbing' and objective_function is not None:
            P1 = self._find_best_gap_position(mutated_individual, G1_indices, 
                                             gap_length, objective_function)
        else:
            P1 = random.randint(0, mutated_individual.alignment_length)
        
        # Insert gaps in G1 at position P1
        for idx in G1_indices:
            segment = mutated_individual.aligned_segments[idx]
            segment.sequence = segment.sequence[:P1] + gap_string + segment.sequence[P1:]
        
        # Select position P2 for G2 (at maximum distance from P1)
        # Maximum distance means near the opposite end of the alignment
        alignment_half = mutated_individual.alignment_length // 2
        if P1 < alignment_half:
            # P1 is in first half, put P2 in second half
            min_pos = alignment_half
            max_pos = mutated_individual.alignment_length + gap_length
        else:
            # P1 is in second half, put P2 in first half
            min_pos = 0
            max_pos = alignment_half
        
        if mode == 'hill_climbing' and objective_function is not None:
            P2 = self._find_best_gap_position_in_range(
                mutated_individual, G2_indices, gap_length, 
                objective_function, min_pos, max_pos
            )
        else:
            P2 = random.randint(min_pos, max_pos)
        
        # Insert gaps in G2 at position P2
        for idx in G2_indices:
            segment = mutated_individual.aligned_segments[idx]
            segment.sequence = segment.sequence[:P2] + gap_string + segment.sequence[P2:]
        
        # Update alignment length
        mutated_individual.alignment_length += gap_length * 2
        
        return mutated_individual
    
    def _find_best_gap_position(self, individual: Alignment, group_indices: List[int],
                                gap_length: int, objective_function) -> int:
        """
        Find the best position to insert gaps using hill-climbing.
        
        Args:
            individual: Current alignment
            group_indices: Indices of sequences in the group
            gap_length: Length of gap to insert
            objective_function: Objective function to evaluate fitness
            
        Returns:
            Best position to insert gap
        """
        best_position = 0
        best_score = float('-inf')
        gap_string = '-' * gap_length
        
        # Try a sample of positions (not all, for efficiency)
        sample_size = min(20, individual.alignment_length + 1)
        positions_to_test = random.sample(
            range(individual.alignment_length + 1), 
            sample_size
        )
        
        for pos in positions_to_test:
            # Create temporary alignment with gap inserted
            temp_individual = individual.copy_alignment()
            for idx in group_indices:
                segment = temp_individual.aligned_segments[idx]
                segment.sequence = segment.sequence[:pos] + gap_string + segment.sequence[pos:]
            
            # Update alignment length
            temp_individual.alignment_length = len(temp_individual.aligned_segments[0].sequence)
            
            # Evaluate fitness
            temp_individual.calculate_fitness(objective_function)
            score = temp_individual.fitness_score
            
            if score > best_score:
                best_score = score
                best_position = pos
        
        return best_position
    
    def _find_best_gap_position_in_range(self, individual: Alignment, group_indices: List[int],
                                         gap_length: int, objective_function,
                                         min_pos: int, max_pos: int) -> int:
        """
        Find the best position to insert gaps within a range using hill-climbing.
        
        Args:
            individual: Current alignment
            group_indices: Indices of sequences in the group
            gap_length: Length of gap to insert
            objective_function: Objective function to evaluate fitness
            min_pos: Minimum position (inclusive)
            max_pos: Maximum position (exclusive)
            
        Returns:
            Best position to insert gap within range
        """
        if min_pos >= max_pos:
            return min_pos
        
        best_position = min_pos
        best_score = float('-inf')
        gap_string = '-' * gap_length
        
        # Try a sample of positions in the range
        range_size = max_pos - min_pos
        sample_size = min(10, range_size)
        
        if range_size > 0:
            positions_to_test = random.sample(range(min_pos, max_pos), sample_size)
        else:
            positions_to_test = [min_pos]
        
        for pos in positions_to_test:
            # Create temporary alignment with gap inserted
            temp_individual = individual.copy_alignment()
            for idx in group_indices:
                segment = temp_individual.aligned_segments[idx]
                segment.sequence = segment.sequence[:pos] + gap_string + segment.sequence[pos:]
            
            # Update alignment length
            temp_individual.alignment_length = len(temp_individual.aligned_segments[0].sequence)
            
            # Evaluate fitness
            temp_individual.calculate_fitness(objective_function)
            score = temp_individual.fitness_score
            
            if score > best_score:
                best_score = score
                best_position = pos
        
        return best_position
    # ========== SAGA BLOCK OPERATORS (delegated to saga_operators.py) ==========
    
    def saga_block_shuffling(self, individual: Alignment, objective_function=None,
                            block_type: str = 'random', movement_type: str = 'random',
                            mode: str = 'stochastic') -> Alignment:
        """
        SAGA Block Shuffling Operator (16 variants).
        
        Delegates to SAGABlockOperators. See saga_operators.py for details.
        """
        return self.saga_block_ops.block_shuffling(
            individual, objective_function, block_type, movement_type, mode
        )
    
    def saga_block_searching(self, individual: Alignment, min_block_size: int = 3,
                            max_block_size: int = 10) -> Alignment:
        """
        SAGA Block Searching Operator.
        
        Delegates to SAGABlockSearching. See saga_operators.py for details.
        """
        return self.saga_block_search.block_searching(
            individual, min_block_size, max_block_size
        )
    
    def saga_local_rearrangement(self, individual: Alignment, objective_function,
                                block_size: int = 10, exhaustive_threshold: int = 1000) -> Alignment:
        """
        SAGA Local Optimal Rearrangement Operator.
        
        Delegates to SAGALocalRearrangement. See saga_operators.py for details.
        """
        return self.saga_local_rearrange.local_rearrangement(
            individual, objective_function, block_size, exhaustive_threshold
        )
