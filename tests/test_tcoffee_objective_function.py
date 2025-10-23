import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from genetic_algorithm.objective_function.tcoffee_objective_function import TCoffeeObjectiveFunction

@pytest.fixture
def tcoffee_obj():
    sequences = [
        SeqRecord(Seq("ACD-"), id="seq1"),
        SeqRecord(Seq("AC-E"), id="seq2"),
        SeqRecord(Seq("A-DG"), id="seq3")
    ]
    obj = TCoffeeObjectiveFunction(sequences)
    return obj, sequences

def test_initialization(tcoffee_obj):
    obj, sequences = tcoffee_obj
    
    # Test sequence mapping
    assert len(obj.sequence_map) == 3
    assert set(obj.sequence_ids) == {"seq1", "seq2", "seq3"}
    
    # Test sequence weights normalized
    assert abs(sum(obj.weights.values()) - 1.0) < 1e-5
    
    # Test pairwise library is non-empty
    assert len(obj.pairwise_weights) > 0

def test_pairwise_library_counts(tcoffee_obj):
    obj, _ = tcoffee_obj
    # ('A','A') occurs in positions 0 of all sequences: 3 pairs (seq1-seq2, seq1-seq3, seq2-seq3) (two updates per occurrence)
    assert obj.pairwise_weights[('A','A')] == 6

def test_compute_fitness_nonnegative(tcoffee_obj):
    obj, _ = tcoffee_obj
    aligned_sequences = ["ACD-", "AC-E", "A-DG"]
    fitness = obj.compute_fitness(aligned_sequences)
    assert fitness >= 0

def test_consistency_scores_increase_with_alignment(tcoffee_obj):
    obj, _ = tcoffee_obj
    aligned_sequences = ["ACD-", "AC-E", "A-DG"]
    better_aligned = ["ACD-", "ACD-", "ACDG"]
    
    fitness_original = obj.compute_fitness(aligned_sequences)
    fitness_better = obj.compute_fitness(better_aligned)
    
    # More consistent alignment should have equal or higher score
    assert fitness_better >= fitness_original
