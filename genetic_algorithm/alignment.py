import random
import copy
from Bio.SeqRecord import SeqRecord

class AlignedSegment:
    def __init__(self, seq_id: str, sequence: str):
        self.id = seq_id
        self.sequence = sequence

    def __eq__(self, other):
        return self.id == other.id and self.sequence == other.sequence

class Alignment:
    def __init__(self, initial_records: list[SeqRecord]):
        self.aligned_segments, self.alignment_length = self.random_initialize(initial_records)

        # Other attributes
        self.fitness_score = 0

    def __eq__(self, other) -> bool:
        if not isinstance(other, Alignment):
            return NotImplemented
        return self.aligned_segments == other.aligned_segments

    def get_sequences_and_ids(self) -> tuple[list[str], list[str]]:
        """Returns sequences and ids of the records."""
        sequences = [seg.sequence for seg in self.aligned_segments]
        ids = [seg.id for seg in self.aligned_segments]
        return sequences, ids
    
    def random_initialize(self, initial_records: list[SeqRecord]) -> tuple[list[AlignedSegment], int]:
        """
        Initializes the alignment with random offsets, preserving the IDs.
        """
        # 1. Obtains length of each sequence
        data = [(rec.id, str(rec.seq)) for rec in initial_records]
        sequence_lengths = [len(seq_str) for _, seq_str in data]
        max_seq_len = max(sequence_lengths)
        
        # Finds a maximum offset value (25% of the maximum sequence length)
        max_offset = max(1, max_seq_len // 4)
        
        # 2. Computes a random offset for each sequence
        offsets = [random.randint(0, max_offset) for _ in data]
        
        # 3. Computes the length of the alignment (L)
        max_aligned_length = 0
        for length, offset in zip(sequence_lengths, offsets):
            max_aligned_length = max(max_aligned_length, length + offset)
        L = max_aligned_length
        
        # 4. Aligns sequences with offsets and packages with IDs
        aligned_segments: list[AlignedSegment] = []
        for i, (seq_id, seq) in enumerate(data):
            offset = offsets[i]
            
            # Applies leading and trailing gaps
            leading_gaps = '-' * offset
            aligned_seq_str = leading_gaps + seq
            trailing_gaps = '-' * (L - len(aligned_seq_str))
            
            final_aligned_seq = aligned_seq_str + trailing_gaps
            
            # Keeps the new sequence while preseving the ID
            aligned_segments.append(AlignedSegment(seq_id, final_aligned_seq))
            
        return aligned_segments, L
    
    def calculate_fitness(self, objective_function) -> None:
        """
        Computes the fitness score based on the objective function.
        """
        sequences, sequence_ids = self.get_sequences_and_ids()
        
        self.fitness_score = objective_function.compute_fitness(
            aligned_sequences=sequences, 
            sequence_ids=sequence_ids, 
            alignment_length=self.alignment_length
        )
    
    def copy_alignment(self) -> 'Alignment':
        """
        Deep copy of the alignment
        """
        return copy.deepcopy(self)