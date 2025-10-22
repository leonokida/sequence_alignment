import pytest
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from genetic_algorithm.objective_function.objective_function import ObjectiveFunction


def test_objective_function_basic():
    """Verifica se o fitness roda e gera valor numérico válido."""
    seqs = [
        SeqRecord(Seq("ACDE"), id="seq1"),
        SeqRecord(Seq("ACDF"), id="seq2"),
        SeqRecord(Seq("ACDG"), id="seq3"),
    ]

    obj_func = ObjectiveFunction(seqs)

    total_weight = sum(obj_func.weights.values())
    assert pytest.approx(total_weight, 0.001) == 1.0

    aligned = ["ACD-E", "ACDFE", "ACDGE"]
    ids = ["seq1", "seq2", "seq3"]
    alignment_length = len(aligned[0])  # ✅ correção

    fitness = obj_func.compute_fitness(aligned, ids, alignment_length)

    # Deve ser um float válido e finito (não NaN)
    assert isinstance(fitness, float)
    assert not (fitness is None or fitness != fitness)


def test_pair_score_affine_with_gaps():
    """Confere se a penalidade de gap reduz o score."""
    seqs = [
        SeqRecord(Seq("ACD"), id="seq1"),
        SeqRecord(Seq("ACD"), id="seq2")
    ]
    obj_func = ObjectiveFunction(seqs)

    score_match = obj_func._calculate_pair_score_affine("ACD", "ACD")
    score_gap = obj_func._calculate_pair_score_affine("ACD-", "ACDE")

    # Score com gap deve ser menor
    assert score_gap < score_match


def test_weight_normalization_diff_sequences():
    """Verifica se pesos são normalizados corretamente em casos distintos."""
    seqs_similares = [
        SeqRecord(Seq("ACDE"), id="seq1"),
        SeqRecord(Seq("ACDF"), id="seq2"),
    ]
    seqs_diferentes = [
        SeqRecord(Seq("AAAA"), id="s1"),
        SeqRecord(Seq("WWWW"), id="s2"),
    ]

    obj_sim = ObjectiveFunction(seqs_similares)
    obj_diff = ObjectiveFunction(seqs_diferentes)

    # Pesos normalizados devem somar 1
    assert pytest.approx(sum(obj_sim.weights.values()), 0.001) == 1.0
    assert pytest.approx(sum(obj_diff.weights.values()), 0.001) == 1.0

    # Valores dentro do intervalo esperado
    for weights in [obj_sim.weights, obj_diff.weights]:
        for w in weights.values():
            assert 0.0 < w <= 1.0
