"""
Example usage of the Genetic Algorithm orchestrator for Multiple Sequence Alignment.

This script demonstrates how to:
1. Load sequences from a file
2. Choose an objective function (SAGA or WSP)
3. Configure and run the genetic algorithm
4. Save and analyze results
"""

from genetic_algorithm.utils.read_sequences_file import read_fasta_file
from genetic_algorithm.objective_function.saga_objective_function import SAGAObjectiveFunction
from genetic_algorithm.objective_function.wsp_objective_function import WSPObjectiveFunction
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm


def run_saga_example():
    """
    Example using SAGA objective function (PAM250 with affine gap penalties).
    """
    print("=" * 80)
    print("EXAMPLE: Genetic Algorithm with SAGA Objective Function")
    print("=" * 80)
    
    # 1. Load sequences
    sequences_file = "sequences/seqdump_1.txt"
    sequences = read_fasta_file(sequences_file)
    print(f"\nLoaded {len(sequences)} sequences from {sequences_file}")
    for seq in sequences:
        print(f"  - {seq.id}: {len(seq.seq)} residues")
    
    # 2. Initialize objective function
    objective_function = SAGAObjectiveFunction(sequences)
    print("\nObjective Function: SAGA (PAM250 + Affine Gap Penalties)")
    print(f"  Gap Open Penalty: {objective_function.GAP_OPEN_PENALTY}")
    print(f"  Gap Extend Penalty: {objective_function.GAP_EXTEND_PENALTY}")
    
    # 3. Configure genetic algorithm
    ga = GeneticAlgorithm(
        initial_sequences=sequences,
        objective_function=objective_function,
        population_size=50,          # Size of population
        num_generations=100,         # Number of generations
        crossover_probability=0.8,   # Crossover probability (paper recommendation)
        mutation_probability=0.5,    # Mutation probability (paper recommendation)
        elitism_count=2,             # Number of elite individuals to preserve
        selection_method="tournament", # Selection method
        tournament_size=3,           # Tournament size
        crossover_method="single_point", # Crossover method
        mutation_method="standard",  # Mutation method
        verbose=True                 # Print progress
    )
    
    print("\n" + "=" * 80)
    print("Starting Genetic Algorithm...")
    print("=" * 80)
    
    # 4. Run the algorithm
    results = ga.run(save_results=True, output_dir="results")
    
    # 5. Get and display best alignment
    best_alignment = ga.get_best_alignment()
    print("\n" + "=" * 80)
    print("BEST ALIGNMENT FOUND")
    print("=" * 80)
    print(f"Fitness Score: {best_alignment.fitness_score:.4f}")
    print(f"Alignment Length: {best_alignment.alignment_length}")
    print(f"\nAligned Sequences:")
    for segment in best_alignment.aligned_segments:
        print(f">{segment.id}")
        print(f"{segment.sequence}")
    
    return results


def run_wsp_example():
    """
    Example using WSP (Weighted Sum of Pairs) objective function.
    """
    print("=" * 80)
    print("EXAMPLE: Genetic Algorithm with WSP Objective Function")
    print("=" * 80)
    
    # 1. Load sequences
    sequences_file = "sequences/seqdump_1.txt"
    sequences = read_fasta_file(sequences_file)
    print(f"\nLoaded {len(sequences)} sequences from {sequences_file}")
    for seq in sequences:
        print(f"  - {seq.id}: {len(seq.seq)} residues")
    
    # 2. Initialize objective function
    objective_function = WSPObjectiveFunction(sequences)
    print("\nObjective Function: WSP (Weighted Sum of Pairs)")
    print(f"  Pairwise library size: {len(objective_function.pairwise_weights)}")
    
    # 3. Configure genetic algorithm
    ga = GeneticAlgorithm(
        initial_sequences=sequences,
        objective_function=objective_function,
        population_size=50,
        num_generations=100,
        crossover_probability=0.8,
        mutation_probability=0.5,
        elitism_count=2,
        selection_method="tournament",
        tournament_size=3,
        crossover_method="single_point",
        mutation_method="standard",
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("Starting Genetic Algorithm...")
    print("=" * 80)
    
    # 4. Run the algorithm
    results = ga.run(save_results=True, output_dir="results")
    
    # 5. Get and display best alignment
    best_alignment = ga.get_best_alignment()
    print("\n" + "=" * 80)
    print("BEST ALIGNMENT FOUND")
    print("=" * 80)
    print(f"Fitness Score: {best_alignment.fitness_score:.4f}")
    print(f"Alignment Length: {best_alignment.alignment_length}")
    print(f"\nAligned Sequences:")
    for segment in best_alignment.aligned_segments:
        print(f">{segment.id}")
        print(f"{segment.sequence}")
    
    return results


def compare_objective_functions():
    """
    Compare SAGA and WSP (Weighted Sum of Pairs) objective functions on the same sequences.
    """
    print("=" * 80)
    print("COMPARISON: SAGA vs WSP Objective Functions")
    print("=" * 80)
    
    sequences_file = "sequences/seqdump_1.txt"
    sequences = read_fasta_file(sequences_file)
    
    # Run with SAGA
    print("\n### Running with SAGA Objective Function ###\n")
    saga_objective = SAGAObjectiveFunction(sequences)
    saga_ga = GeneticAlgorithm(
        initial_sequences=sequences,
        objective_function=saga_objective,
        population_size=30,
        num_generations=50,
        verbose=False
    )
    saga_ga.run(save_results=True, output_dir="results")
    saga_best_fitness = saga_ga.get_best_alignment().fitness_score
    
    # Run with WSP
    print("\n### Running with WSP Objective Function ###\n")
    wsp_objective = WSPObjectiveFunction(sequences)
    wsp_ga = GeneticAlgorithm(
        initial_sequences=sequences,
        objective_function=wsp_objective,
        population_size=30,
        num_generations=50,
        verbose=False
    )
    wsp_ga.run(save_results=True, output_dir="results")
    wsp_best_fitness = wsp_ga.get_best_alignment().fitness_score
    
    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"SAGA Best Fitness: {saga_best_fitness:.4f}")
    print(f"WSP Best Fitness: {wsp_best_fitness:.4f}")
    print("\nNote: Fitness scores are not directly comparable between different")
    print("objective functions as they use different scoring systems.")


def custom_configuration_example():
    """
    Example with custom configuration and different operators.
    """
    print("=" * 80)
    print("EXAMPLE: Custom Configuration with Different Operators")
    print("=" * 80)
    
    sequences_file = "sequences/seqdump_1.txt"
    sequences = read_fasta_file(sequences_file)
    
    objective_function = SAGAObjectiveFunction(sequences)
    
    # Custom configuration
    ga = GeneticAlgorithm(
        initial_sequences=sequences,
        objective_function=objective_function,
        population_size=100,          # Larger population
        num_generations=200,          # More generations
        crossover_probability=0.9,    # Higher crossover rate
        mutation_probability=0.3,     # Lower mutation rate
        elitism_count=5,              # More elites
        selection_method="rank",      # Rank-based selection
        tournament_size=5,            # Larger tournament
        crossover_method="uniform",   # Uniform crossover
        mutation_method="gap_shift",  # Gap shift mutation
        verbose=True
    )
    
    results = ga.run(save_results=True, output_dir="results")
    
    best_alignment = ga.get_best_alignment()
    print(f"\nFinal Best Fitness: {best_alignment.fitness_score:.4f}")
    
    return results


if __name__ == "__main__":
    # Choose which example to run
    print("Select an example to run:")
    print("1. SAGA Objective Function")
    print("2. WSP (Weighted Sum of Pairs) Objective Function")
    print("3. Compare Both Objective Functions")
    print("4. Custom Configuration")
    
    choice = input("\nEnter choice (1-4) or press Enter for default (1): ").strip()
    
    if choice == "2":
        run_wsp_example()
    elif choice == "3":
        compare_objective_functions()
    elif choice == "4":
        custom_configuration_example()
    else:
        run_saga_example()
