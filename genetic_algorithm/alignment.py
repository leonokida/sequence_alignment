import random
import copy

class Alignment:
    def __init__(self, initial_sequences: list[str]):
        self.sequences, self.alignment_length = self.random_initialize(initial_sequences)

        # Other attributes
        self.fitness_score = 0

    def __eq__(self, other):
        return self.sequences == other.sequences

    def random_initialize(self, initial_sequences: list[str]) -> tuple[list[str], int]:
        """
        Initializes alignment for the first epoch with random offsets and a single sequence alignment length.
        """

        # 1. Gets maximum sequence length
        sequence_lengths = [len(seq) for seq in initial_sequences]
        max_seq_len = max(sequence_lengths)
        
        # Finds a maximum offset value (25% of the maximum sequence length)
        max_offset = max(1, max_seq_len // 4)
        
        # 2. Computes a random offset for each sequence
        offsets = [random.randint(0, max_offset) for _ in initial_sequences]
        
        # 3. Computes the length of the alignment (L)
        # L = Longest sequence's length + its offset
        max_aligned_length = 0
        for length, offset in zip(sequence_lengths, offsets):
            max_aligned_length = max(max_aligned_length, length + offset)
        L = max_aligned_length
        
        # 4. Aligns sequences with offsets
        aligned_sequences = []
        for i, seq in enumerate(initial_sequences):
            offset = offsets[i]
            # Adds the leading gaps to sequence
            leading_gaps = '-' * offset
            aligned_seq = leading_gaps + seq
            
            # Adds trailing gaps so that the sequence has length L
            trailing_gaps = '-' * (L - len(aligned_seq))
            
            final_aligned_seq = aligned_seq + trailing_gaps
            aligned_sequences.append(final_aligned_seq)
            
        return aligned_sequences, L
    
    def calculate_fitness(self, objective_function: function) -> None:
        """
        Calculates the fitness score of the alignment based on the objective function.
        """
        self.fitness_score = objective_function(self.sequences, self.alignment_length)
    
    def copy_alignment(self) -> 'Alignment':
        return copy.deepcopy(self)