"""
Example usage of genetic operators for Multiple Sequence Alignment.

This file demonstrates how to use the implemented operators based on the research paper,
with optimal probabilities defined (crossover: 0.8, mutation: 0.5).
"""

import random
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from genetic_algorithm.alignment import Alignment
from genetic_algorithm.operators import CrossoverOperator, MutationOperator, SelectionOperator
from genetic_algorithm.objective_function.saga_objective_function import SAGAObjectiveFunction


def create_sample_sequences():
    """
    Create sample sequences for testing.
    """
    sequences = [
        SeqRecord(Seq("ACGTACGTACGT"), id="seq1"),
        SeqRecord(Seq("ACGTACGT"), id="seq2"),
        SeqRecord(Seq("CGTACGTACGTACGT"), id="seq3"),
        SeqRecord(Seq("ACGTACGTACGTAC"), id="seq4")
    ]
    return sequences


def create_initial_population(sequences, population_size=10):
    """
    Create an initial population of alignments.
    """
    population = []
    for i in range(population_size):
        alignment = Alignment(sequences)
        population.append(alignment)
    return population


def evaluate_population(population, objective_function):
    """
    Evaluate the fitness of the entire population.
    """
    for individual in population:
        individual.calculate_fitness(objective_function)


def demonstrate_crossover():
    """
    Demonstrate the use of the crossover operator.
    """
    print("=== Crossover Operator Demonstration ===")
    
    # Create example sequences
    sequences = create_sample_sequences()
    
    # Create two parents
    parent1 = Alignment(sequences)
    parent2 = Alignment(sequences)
    
    print(f"Parent 1 - Sequence 1: {parent1.aligned_segments[0].sequence}")
    print(f"Parent 2 - Sequence 1: {parent2.aligned_segments[0].sequence}")
    
    # Initialize crossover operator with optimal probability
    crossover_op = CrossoverOperator(crossover_probability=0.8)
    
    # Apply crossover
    offspring1, offspring2 = crossover_op.crossover(parent1, parent2)
    
    print(f"Offspring 1 - Sequence 1: {offspring1.aligned_segments[0].sequence}")
    print(f"Offspring 2 - Sequence 1: {offspring2.aligned_segments[0].sequence}")
    
    # Demonstrate single-point crossover
    print("\n--- Single-Point Crossover ---")
    offspring3, offspring4 = crossover_op.single_point_crossover(parent1, parent2)
    print(f"Offspring 3 - Sequence 1: {offspring3.aligned_segments[0].sequence}")
    print(f"Offspring 4 - Sequence 1: {offspring4.aligned_segments[0].sequence}")


def demonstrate_mutation():
    """
    Demonstrate the use of the mutation operator.
    """
    print("\n=== Mutation Operator Demonstration ===")
    
    # Create example sequences
    sequences = create_sample_sequences()
    individual = Alignment(sequences)
    
    print(f"Original - Sequence 1: {individual.aligned_segments[0].sequence}")
    
    # Initialize mutation operator with optimal probability
    mutation_op = MutationOperator(mutation_probability=0.5)
    
    # Apply default mutation
    mutated1 = mutation_op.mutate(individual, sequence_type="dna")
    print(f"Mutated 1 - Sequence 1: {mutated1.aligned_segments[0].sequence}")
    
    # Apply gap shift mutation
    mutated2 = mutation_op.gap_shift_mutation(individual)
    print(f"Mutated 2 (gap shift) - Sequence 1: {mutated2.aligned_segments[0].sequence}")
    
    # Apply insertion/deletion mutation
    mutated3 = mutation_op.insertion_deletion_mutation(individual)
    print(f"Mutated 3 (indel) - Sequence 1: {mutated3.aligned_segments[0].sequence}")
    print(f"Original length: {individual.alignment_length}, Mutated length: {mutated3.alignment_length}")


def demonstrate_selection():
    """
    Demonstrate the use of the selection operator.
    """
    print("\n=== Selection Operator Demonstration ===")
    
    # Create population and objective function
    sequences = create_sample_sequences()
    population = create_initial_population(sequences, population_size=5)
    objective_function = SAGAObjectiveFunction(sequences)
    
    # Evaluate the population
    evaluate_population(population, objective_function)
    
    # Show population fitness
    print("Population fitness:")
    for i, ind in enumerate(population):
        print(f"Individual {i}: {ind.fitness_score:.4f}")
    
    # Initialize selection operator
    selection_op = SelectionOperator()
    
    # Demonstrate tournament selection
    print("\n--- Tournament Selection ---")
    selected = selection_op.tournament_selection(population, tournament_size=3)
    print(f"Selected individual - Fitness: {selected.fitness_score:.4f}")
    
    # Demonstrate parent selection
    print("\n--- Parent Selection ---")
    parent1, parent2 = selection_op.select_parents(population, selection_method="tournament")
    print(f"Parent 1 - Fitness: {parent1.fitness_score:.4f}")
    print(f"Parent 2 - Fitness: {parent2.fitness_score:.4f}")
    
    # Demonstrate elitist selection
    print("\n--- Elitist Selection ---")
    elites = selection_op.elitist_selection(population, num_elites=2)
    print("Selected elites:")
    for i, elite in enumerate(elites):
        print(f"Elite {i}: {elite.fitness_score:.4f}")


def demonstrate_complete_generation():
    """
    Demonstrate a complete genetic algorithm generation cycle.
    """
    print("\n=== Complete Generation Cycle Demonstration ===")
    
    # Configuration
    sequences = create_sample_sequences()
    population_size = 8
    
    # Create initial population
    population = create_initial_population(sequences, population_size)
    objective_function = SAGAObjectiveFunction(sequences)
    
    # Evaluate initial population
    evaluate_population(population, objective_function)
    print(f"Initial average fitness: {sum(ind.fitness_score for ind in population) / len(population):.4f}")
    
    # Initialize operators with optimal probabilities from the paper
    crossover_op = CrossoverOperator(crossover_probability=0.8)
    mutation_op = MutationOperator(mutation_probability=0.5)
    selection_op = SelectionOperator()
    
    # Generate new generation
    offspring = []
    
    # Create offspring until reaching population size
    while len(offspring) < population_size:
        # Parent selection
        parent1, parent2 = selection_op.select_parents(population, selection_method="tournament")
        
        # Crossover
        child1, child2 = crossover_op.crossover(parent1, parent2)
        
        # Mutation
        child1 = mutation_op.mutate(child1, sequence_type="dna")
        child2 = mutation_op.mutate(child2, sequence_type="dna")
        
        offspring.extend([child1, child2])
    
    # Limit to population size
    offspring = offspring[:population_size]
    
    # Evaluate offspring
    evaluate_population(offspring, objective_function)
    
    # Survivor selection
    new_population = selection_op.survivor_selection(
        population, offspring, population_size, selection_method="elitist"
    )
    
    print(f"New generation average fitness: {sum(ind.fitness_score for ind in new_population) / len(new_population):.4f}")
    print(f"Best fitness: {max(ind.fitness_score for ind in new_population):.4f}")


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    print("Genetic Operators Demonstration for MSA")
    print("Based on the paper with optimal probabilities: Crossover=0.8, Mutation=0.5")
    print("=" * 60)
    
    try:
        demonstrate_crossover()
        demonstrate_mutation()
        demonstrate_selection()
        demonstrate_complete_generation()
        
        print("\n" + "=" * 60)
        print("Demonstration completed successfully!")
        print("The operators are implemented and working according to the paper.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()