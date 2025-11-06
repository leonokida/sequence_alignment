"""
Example using SAGA operators in the Genetic Algorithm.

This script demonstrates how to use the advanced SAGA mutation operators:
- SAGA Gap Insertion
- SAGA Block Shuffling
- SAGA Block Searching
- SAGA Local Rearrangement
- SAGA Mixed (random selection)
"""

from genetic_algorithm import GeneticAlgorithm, SAGAObjectiveFunction
from genetic_algorithm.utils.read_sequences_file import read_fasta_file


def compare_saga_operators():
    """
    Compare different SAGA operators on the same sequences.
    """
    print("=" * 80)
    print("COMPARISON: Different SAGA Mutation Operators")
    print("=" * 80)
    
    # Load sequences
    sequences = read_fasta_file("sequences/seqdump_1.txt")
    print(f"\nLoaded {len(sequences)} sequences")
    
    # Common configuration
    common_config = {
        'initial_sequences': sequences,
        'objective_function': SAGAObjectiveFunction(sequences),
        'population_size': 30,
        'num_generations': 20,
        'crossover_probability': 0.8,
        'mutation_probability': 0.5,
        'elitism_count': 2,
        'selection_method': "tournament",
        'crossover_method': "single_point",
        'verbose': False
    }
    
    # Test different SAGA operators
    saga_operators = [
        ('Standard Mutation', 'standard'),
        ('SAGA Gap Insertion', 'saga_gap_insertion'),
        ('SAGA Block Shuffling', 'saga_block_shuffling'),
        ('SAGA Block Searching', 'saga_block_searching'),
        ('SAGA Mixed', 'saga_mixed'),
    ]
    
    results_summary = []
    
    for name, mutation_method in saga_operators:
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
        
        ga = GeneticAlgorithm(
            **common_config,
            mutation_method=mutation_method
        )
        
        results = ga.run(save_results=True, output_dir="results")
        best = ga.get_best_alignment()
        
        initial_fitness = results.generation_history[0]['best_fitness']
        final_fitness = best.fitness_score
        improvement = final_fitness - initial_fitness
        
        results_summary.append({
            'name': name,
            'initial': initial_fitness,
            'final': final_fitness,
            'improvement': improvement,
            'run_id': results.run_id
        })
        
        print(f"Initial Fitness: {initial_fitness:.4f}")
        print(f"Final Fitness: {final_fitness:.4f}")
        print(f"Improvement: {improvement:.4f}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Operator':<30} {'Initial':<15} {'Final':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for result in results_summary:
        print(f"{result['name']:<30} "
              f"{result['initial']:<15.4f} "
              f"{result['final']:<15.4f} "
              f"{result['improvement']:<15.4f}")
    
    # Find best operator
    best_result = max(results_summary, key=lambda x: x['improvement'])
    print("\n" + "=" * 80)
    print(f"Best Operator: {best_result['name']}")
    print(f"Best Improvement: {best_result['improvement']:.4f}")
    print(f"Results saved in: results/{best_result['run_id']}")
    print("=" * 80)


def saga_gap_insertion_example():
    """
    Example using SAGA Gap Insertion operator specifically.
    """
    print("=" * 80)
    print("EXAMPLE: SAGA Gap Insertion Operator")
    print("=" * 80)
    
    sequences = read_fasta_file("sequences/seqdump_1.txt")
    print(f"\nLoaded {len(sequences)} sequences")
    
    objective_function = SAGAObjectiveFunction(sequences)
    print("\nObjective Function: SAGA")
    print("Mutation Method: SAGA Gap Insertion")
    print("  - Divides sequences into phylogenetic groups")
    print("  - Inserts gaps at different positions in each group")
    print("  - Uses hill-climbing to find optimal gap positions")
    
    ga = GeneticAlgorithm(
        initial_sequences=sequences,
        objective_function=objective_function,
        population_size=50,
        num_generations=50,
        mutation_method="saga_gap_insertion",
        verbose=True
    )
    
    results = ga.run(save_results=True, output_dir="results")
    
    best = ga.get_best_alignment()
    print(f"\nFinal Best Fitness: {best.fitness_score:.4f}")
    
    return results


def saga_block_shuffling_example():
    """
    Example using SAGA Block Shuffling operator.
    """
    print("=" * 80)
    print("EXAMPLE: SAGA Block Shuffling Operator")
    print("=" * 80)
    
    sequences = read_fasta_file("sequences/seqdump_1.txt")
    print(f"\nLoaded {len(sequences)} sequences")
    
    objective_function = SAGAObjectiveFunction(sequences)
    print("\nObjective Function: SAGA")
    print("Mutation Method: SAGA Block Shuffling")
    print("  - Selects blocks of sequences (contiguous or non-contiguous)")
    print("  - Moves blocks left or right")
    print("  - 16 different variants available")
    
    ga = GeneticAlgorithm(
        initial_sequences=sequences,
        objective_function=objective_function,
        population_size=50,
        num_generations=50,
        mutation_method="saga_block_shuffling",
        verbose=True
    )
    
    results = ga.run(save_results=True, output_dir="results")
    
    best = ga.get_best_alignment()
    print(f"\nFinal Best Fitness: {best.fitness_score:.4f}")
    
    return results


def saga_mixed_example():
    """
    Example using mixed SAGA operators (random selection).
    """
    print("=" * 80)
    print("EXAMPLE: SAGA Mixed Operators")
    print("=" * 80)
    
    sequences = read_fasta_file("sequences/seqdump_1.txt")
    print(f"\nLoaded {len(sequences)} sequences")
    
    objective_function = SAGAObjectiveFunction(sequences)
    print("\nObjective Function: SAGA")
    print("Mutation Method: SAGA Mixed")
    print("  - Randomly selects from multiple SAGA operators")
    print("  - Includes: Gap Insertion, Block Shuffling, Block Searching")
    print("  - Maintains diversity through operator variation")
    
    ga = GeneticAlgorithm(
        initial_sequences=sequences,
        objective_function=objective_function,
        population_size=50,
        num_generations=100,
        mutation_method="saga_mixed",
        verbose=True
    )
    
    results = ga.run(save_results=True, output_dir="results")
    
    best = ga.get_best_alignment()
    print(f"\nFinal Best Fitness: {best.fitness_score:.4f}")
    
    return results


def quick_saga_test():
    """
    Quick test with SAGA operators for fast verification.
    """
    print("=" * 80)
    print("QUICK TEST - SAGA Operators")
    print("=" * 80)
    
    sequences = read_fasta_file("sequences/seqdump_1.txt")
    print(f"\nLoaded {len(sequences)} sequences")
    
    ga = GeneticAlgorithm(
        initial_sequences=sequences,
        objective_function=SAGAObjectiveFunction(sequences),
        population_size=20,
        num_generations=10,
        mutation_method="saga_mixed",  # Use mixed SAGA operators
        verbose=True
    )
    
    results = ga.run(save_results=True, output_dir="results")
    
    best = ga.get_best_alignment()
    initial_fitness = results.generation_history[0]['best_fitness']
    improvement = best.fitness_score - initial_fitness
    improvement_pct = (improvement / abs(initial_fitness)) * 100 if initial_fitness != 0 else 0
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Initial Best Fitness: {initial_fitness:.4f}")
    print(f"Final Best Fitness: {best.fitness_score:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
    print("Mutation Method: saga_mixed")


if __name__ == "__main__":
    print("Select an example to run:")
    print("1. Quick Test (SAGA Mixed)")
    print("2. Compare All SAGA Operators")
    print("3. SAGA Gap Insertion")
    print("4. SAGA Block Shuffling")
    print("5. SAGA Mixed (Full Run)")
    
    choice = input("\nEnter choice (1-5) or press Enter for default (1): ").strip()
    
    if choice == "2":
        compare_saga_operators()
    elif choice == "3":
        saga_gap_insertion_example()
    elif choice == "4":
        saga_block_shuffling_example()
    elif choice == "5":
        saga_mixed_example()
    else:
        quick_saga_test()
