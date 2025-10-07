import random
from typing import List, Tuple, Optional
from genetic_algorithm.alignment import Alignment


class SAGABlockOperators:
    """SAGA Block Shuffling operators."""
    
    def __init__(self, mutation_probability: float = 0.5):
        self.mutation_probability = mutation_probability
    
    def block_shuffling(self, individual: Alignment, objective_function=None,
                       block_type: str = 'random', movement_type: str = 'random',
                       mode: str = 'stochastic') -> Alignment:
        """
        SAGA Block Shuffling Operator (16 variants).
        
        Moves blocks of gaps or residues within the alignment.
        
        Args:
            individual: Individual to be mutated
            objective_function: Objective function for hill-climbing mode
            block_type: 'gap', 'residue', or 'random'
            movement_type: 'complete', 'horizontal', 'vertical', or 'random'
            mode: 'stochastic' or 'hill_climbing'
            
        Returns:
            Mutated individual
        """
        # Check if mutation should occur
        if random.random() > self.mutation_probability:
            return individual.copy_alignment()
        
        # Randomly select block type and movement if 'random'
        if block_type == 'random':
            block_type = random.choice(['gap', 'residue'])
        
        if movement_type == 'random':
            movement_type = random.choice(['complete', 'horizontal', 'vertical'])
        
        # Find a block to move
        block_info = self._find_block(individual, block_type)
        
        if block_info is None:
            return individual.copy_alignment()
        
        # Apply movement based on type
        if movement_type == 'complete':
            return self._move_complete_block(
                individual, block_info, block_type, mode, objective_function
            )
        elif movement_type == 'horizontal':
            return self._move_block_horizontal(
                individual, block_info, block_type, mode, objective_function
            )
        elif movement_type == 'vertical':
            return self._move_block_vertical(
                individual, block_info, block_type, mode, objective_function
            )
        
        return individual.copy_alignment()
    
    def _find_block(self, individual: Alignment, block_type: str) -> Optional[Tuple[int, int, int, int]]:
        """Find a block of gaps or residues."""
        num_sequences = len(individual.aligned_segments)
        alignment_length = individual.alignment_length
        
        max_attempts = 20
        for _ in range(max_attempts):
            col_start = random.randint(0, alignment_length - 1)
            row_start = random.randint(0, num_sequences - 1)
            
            char = individual.aligned_segments[row_start].sequence[col_start]
            
            if block_type == 'gap' and char != '-':
                continue
            elif block_type == 'residue' and char == '-':
                continue
            
            block_info = self._expand_block(individual, col_start, row_start, block_type)
            
            if block_info is not None:
                col_end, row_end = block_info
                if (col_end - col_start >= 1) and (row_end - row_start >= 1):
                    return (col_start, col_end, row_start, row_end)
        
        return None
    
    def _expand_block(self, individual: Alignment, start_col: int, start_row: int, 
                     block_type: str) -> Optional[Tuple[int, int]]:
        """Expand a block from a starting position."""
        target_is_gap = (block_type == 'gap')
        
        end_col = start_col
        for col in range(start_col + 1, individual.alignment_length):
            char = individual.aligned_segments[start_row].sequence[col]
            if (char == '-') == target_is_gap:
                end_col = col
            else:
                break
        
        end_row = start_row
        for row in range(start_row + 1, len(individual.aligned_segments)):
            all_match = True
            for col in range(start_col, end_col + 1):
                char = individual.aligned_segments[row].sequence[col]
                if not ((char == '-') == target_is_gap):
                    all_match = False
                    break
            
            if all_match:
                end_row = row
            else:
                break
        
        return (end_col + 1, end_row + 1)
    
    def _move_complete_block(self, individual: Alignment, block_info: Tuple[int, int, int, int],
                            block_type: str, mode: str, objective_function) -> Alignment:
        """Move a complete block to a new position."""
        col_start, col_end, row_start, row_end = block_info
        block_width = col_end - col_start
        
        mutated_individual = individual.copy_alignment()
        
        if mode == 'hill_climbing' and objective_function is not None:
            new_col_start = self._find_best_block_position(
                mutated_individual, block_info, block_type, objective_function
            )
        else:
            max_shift = individual.alignment_length // 4
            shift = random.randint(-max_shift, max_shift)
            new_col_start = max(0, min(col_start + shift, 
                                      individual.alignment_length - block_width))
        
        # Extract and move block
        block_data = []
        for row in range(row_start, row_end):
            segment = individual.aligned_segments[row]
            block_chars = segment.sequence[col_start:col_end]
            block_data.append(block_chars)
        
        for i, row in enumerate(range(row_start, row_end)):
            segment = mutated_individual.aligned_segments[row]
            seq_list = list(segment.sequence)
            
            # Simple block move (swap with destination)
            if new_col_start != col_start:
                # Remove from old position
                for col in range(col_start, col_end):
                    seq_list[col] = '-'
                
                # Place at new position
                for j, char in enumerate(block_data[i]):
                    if new_col_start + j < len(seq_list):
                        seq_list[new_col_start + j] = char
            
            segment.sequence = ''.join(seq_list)
        
        return mutated_individual
    
    def _move_block_horizontal(self, individual: Alignment, block_info: Tuple[int, int, int, int],
                               block_type: str, mode: str, objective_function) -> Alignment:
        """Move a block horizontally (split by phylogenetic tree)."""
        col_start, col_end, row_start, row_end = block_info
        
        all_rows = list(range(row_start, row_end))
        split_point = len(all_rows) // 2
        if split_point == 0:
            split_point = 1
        
        sub_block_info = (col_start, col_end, row_start, row_start + split_point)
        
        return self._move_complete_block(
            individual, sub_block_info, block_type, mode, objective_function
        )
    
    def _move_block_vertical(self, individual: Alignment, block_info: Tuple[int, int, int, int],
                            block_type: str, mode: str, objective_function) -> Alignment:
        """Move a block vertically (split the block in half)."""
        col_start, col_end, row_start, row_end = block_info
        
        col_mid = (col_start + col_end) // 2
        if col_mid <= col_start:
            col_mid = col_start + 1
        
        sub_block_info = (col_start, col_mid, row_start, row_end)
        
        return self._move_complete_block(
            individual, sub_block_info, block_type, mode, objective_function
        )
    
    def _find_best_block_position(self, individual: Alignment, block_info: Tuple[int, int, int, int],
                                  block_type: str, objective_function) -> int:
        """Find the best position to move a block using hill-climbing."""
        col_start, col_end, row_start, row_end = block_info
        block_width = col_end - col_start
        
        best_position = col_start
        best_score = float('-inf')
        
        max_positions = min(10, individual.alignment_length - block_width + 1)
        if max_positions < 1:
            return col_start
            
        positions_to_test = random.sample(
            range(individual.alignment_length - block_width + 1),
            max_positions
        )
        
        for new_pos in positions_to_test:
            temp_individual = individual.copy_alignment()
            temp_individual.calculate_fitness(objective_function)
            score = temp_individual.fitness_score
            
            if score > best_score:
                best_score = score
                best_position = new_pos
        
        return best_position


class SAGABlockSearching:
    """SAGA Block Searching operator."""
    
    def __init__(self, mutation_probability: float = 0.5):
        self.mutation_probability = mutation_probability
    
    def block_searching(self, individual: Alignment, min_block_size: int = 3,
                       max_block_size: int = 10) -> Alignment:
        """
        SAGA Block Searching Operator.
        
        Finds a gap-free block and reconstructs it for better alignment.
        
        Args:
            individual: Individual to be mutated
            min_block_size: Minimum size of block to search for
            max_block_size: Maximum size of block to search for
            
        Returns:
            Mutated individual
        """
        if random.random() > self.mutation_probability:
            return individual.copy_alignment()
        
        mutated_individual = individual.copy_alignment()
        
        # Select a random sequence and substring
        seq_idx = random.randint(0, len(mutated_individual.aligned_segments) - 1)
        sequence = mutated_individual.aligned_segments[seq_idx].sequence.replace('-', '')
        
        if len(sequence) < min_block_size:
            return individual.copy_alignment()
        
        # Select random substring
        block_size = random.randint(min_block_size, min(max_block_size, len(sequence)))
        start_pos = random.randint(0, len(sequence) - block_size)
        substring = sequence[start_pos:start_pos + block_size]
        
        # Find best matches in other sequences
        matches = self._find_matches(mutated_individual, substring, seq_idx)
        
        if not matches:
            return individual.copy_alignment()
        
        # Reconstruct block in alignment
        return self._reconstruct_block(mutated_individual, matches, substring)
    
    def _find_matches(self, individual: Alignment, substring: str, 
                     exclude_idx: int) -> List[Tuple[int, int, int]]:
        """
        Find best matches for substring in other sequences.
        
        Returns:
            List of (sequence_index, position, score) tuples
        """
        matches = []
        
        for idx, segment in enumerate(individual.aligned_segments):
            if idx == exclude_idx:
                continue
            
            sequence = segment.sequence.replace('-', '')
            best_score = -1
            best_pos = -1
            
            # Simple sliding window match
            for i in range(len(sequence) - len(substring) + 1):
                window = sequence[i:i + len(substring)]
                score = sum(1 for a, b in zip(substring, window) if a == b)
                
                if score > best_score:
                    best_score = score
                    best_pos = i
            
            if best_pos >= 0:
                matches.append((idx, best_pos, best_score))
        
        # Sort by score
        matches.sort(key=lambda x: x[2], reverse=True)
        
        return matches
    
    def _reconstruct_block(self, individual: Alignment, matches: List[Tuple[int, int, int]],
                          substring: str) -> Alignment:
        """Reconstruct the block in the alignment."""
        # Find a good position in the alignment
        target_col = random.randint(0, individual.alignment_length - len(substring))
        
        # Place the substring at target position for matched sequences
        for seq_idx, seq_pos, score in matches[:len(matches)//2]:  # Use top half of matches
            segment = individual.aligned_segments[seq_idx]
            seq_list = list(segment.sequence)
            
            # Clear target area
            for i in range(target_col, min(target_col + len(substring), len(seq_list))):
                seq_list[i] = '-'
            
            # Place substring
            for i, char in enumerate(substring):
                if target_col + i < len(seq_list):
                    seq_list[target_col + i] = char
            
            segment.sequence = ''.join(seq_list)
        
        return individual


class SAGALocalRearrangement:
    """SAGA Local Optimal/Sub-optimal Rearrangement operator."""
    
    def __init__(self, mutation_probability: float = 0.5):
        self.mutation_probability = mutation_probability
    
    def local_rearrangement(self, individual: Alignment, objective_function,
                           block_size: int = 10, exhaustive_threshold: int = 1000) -> Alignment:
        """
        SAGA Local Optimal Rearrangement Operator.
        
        Optimizes gap patterns within a block using exhaustive search or LAGA.
        
        Args:
            individual: Individual to be mutated
            objective_function: Objective function for evaluation
            block_size: Size of block to optimize
            exhaustive_threshold: Threshold for using exhaustive vs LAGA
            
        Returns:
            Mutated individual
        """
        if random.random() > self.mutation_probability:
            return individual.copy_alignment()
        
        # Select a random block
        if individual.alignment_length < block_size:
            block_size = individual.alignment_length
        
        block_start = random.randint(0, individual.alignment_length - block_size)
        block_end = block_start + block_size
        
        # Extract block
        block_segments = []
        for segment in individual.aligned_segments:
            block_seq = segment.sequence[block_start:block_end]
            block_segments.append(block_seq)
        
        # Count possible arrangements
        total_gaps = sum(seq.count('-') for seq in block_segments)
        total_residues = sum(len(seq) - seq.count('-') for seq in block_segments)
        
        # Estimate complexity
        complexity = total_gaps * total_residues
        
        if complexity < exhaustive_threshold:
            # Use exhaustive search
            return self._exhaustive_rearrangement(
                individual, block_start, block_end, objective_function
            )
        else:
            # Use LAGA (Local Alignment Genetic Algorithm)
            return self._laga_rearrangement(
                individual, block_start, block_end, objective_function
            )
    
    def _exhaustive_rearrangement(self, individual: Alignment, block_start: int,
                                 block_end: int, objective_function) -> Alignment:
        """Exhaustive search for best gap arrangement in block."""
        best_individual = individual.copy_alignment()
        best_score = float('-inf')
        
        # Try a limited number of random rearrangements
        max_attempts = 50
        
        for _ in range(max_attempts):
            temp_individual = individual.copy_alignment()
            
            # Rearrange gaps in the block
            for segment in temp_individual.aligned_segments:
                block_seq = segment.sequence[block_start:block_end]
                
                # Shuffle gap positions
                chars = list(block_seq)
                random.shuffle(chars)
                
                # Replace block
                new_seq = segment.sequence[:block_start] + ''.join(chars) + segment.sequence[block_end:]
                segment.sequence = new_seq
            
            # Evaluate
            temp_individual.calculate_fitness(objective_function)
            
            if temp_individual.fitness_score > best_score:
                best_score = temp_individual.fitness_score
                best_individual = temp_individual
        
        return best_individual
    
    def _laga_rearrangement(self, individual: Alignment, block_start: int,
                           block_end: int, objective_function) -> Alignment:
        """Local Alignment Genetic Algorithm for block rearrangement."""
        # Simplified LAGA implementation
        population_size = 10
        generations = 5
        
        # Initialize population with variations of the block
        population = []
        for _ in range(population_size):
            temp_individual = individual.copy_alignment()
            
            # Apply random rearrangement to block
            for segment in temp_individual.aligned_segments:
                block_seq = segment.sequence[block_start:block_end]
                chars = list(block_seq)
                random.shuffle(chars)
                new_seq = segment.sequence[:block_start] + ''.join(chars) + segment.sequence[block_end:]
                segment.sequence = new_seq
            
            temp_individual.calculate_fitness(objective_function)
            population.append(temp_individual)
        
        # Evolve for a few generations
        for _ in range(generations):
            # Sort by fitness
            population.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Keep top half
            population = population[:population_size//2]
            
            # Generate offspring
            while len(population) < population_size:
                parent = random.choice(population[:3])
                offspring = parent.copy_alignment()
                
                # Mutate block
                segment_idx = random.randint(0, len(offspring.aligned_segments) - 1)
                segment = offspring.aligned_segments[segment_idx]
                block_seq = segment.sequence[block_start:block_end]
                
                # Swap two positions in block
                if len(block_seq) >= 2:
                    chars = list(block_seq)
                    pos1, pos2 = random.sample(range(len(chars)), 2)
                    chars[pos1], chars[pos2] = chars[pos2], chars[pos1]
                    new_seq = segment.sequence[:block_start] + ''.join(chars) + segment.sequence[block_end:]
                    segment.sequence = new_seq
                
                offspring.calculate_fitness(objective_function)
                population.append(offspring)
        
        # Return best individual
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        return population[0]
