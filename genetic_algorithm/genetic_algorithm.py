"""
Genetic Algorithm Orchestrator for Multiple Sequence Alignment.

This module implements the main genetic algorithm loop, managing:
- Population initialization
- Fitness evaluation
- Selection of best individuals
- Crossover and mutation operations
- Result tracking and storage
"""

import random
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from Bio.SeqRecord import SeqRecord

from genetic_algorithm.alignment import Alignment
from genetic_algorithm.objective_function.base_objective_function import BaseObjectiveFunction
from genetic_algorithm.operators.selection import SelectionOperator
from genetic_algorithm.operators.crossover import CrossoverOperator
from genetic_algorithm.operators.mutation import MutationOperator


class GeneticAlgorithmResults:
    """
    Class to store and manage results from genetic algorithm execution.
    """
    
    def __init__(self, run_id: str = None):
        """
        Initialize results storage.
        
        Args:
            run_id: Unique identifier for this run (auto-generated if not provided)
        """
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.generation_history: List[Dict[str, Any]] = []
        self.best_individual_history: List[Alignment] = []
        self.config: Dict[str, Any] = {}
        self.final_best_individual: Optional[Alignment] = None
        
    def add_generation(self, generation: int, population: List[Alignment], 
                      best_individual_overall: Alignment, 
                      best_individual_generation: Alignment,
                      avg_fitness: float,
                      diversity: float = 0.0):
        """
        Record statistics for a generation.
        
        Args:
            generation: Current generation number
            population: Current population
            best_individual_overall: Best individual observed so far (across all generations)
            best_individual_generation: Best individual in this specific generation
            avg_fitness: Average fitness of the population
            diversity: Population diversity metric
        """
        fitness_scores = [ind.fitness_score for ind in population]
        
        generation_data = {
            'generation': generation,
            'best_fitness_overall': best_individual_overall.fitness_score,  # Melhor fitness observado até agora
            'best_fitness_generation': best_individual_generation.fitness_score,  # Melhor fitness desta geração
            'avg_fitness': avg_fitness,  # Fitness médio da geração
            'min_fitness': min(fitness_scores),
            'max_fitness': max(fitness_scores),
            'diversity': diversity,
            'timestamp': datetime.now().isoformat()
        }
        
        self.generation_history.append(generation_data)
        self.best_individual_history.append(best_individual_overall.copy_alignment())
    
    def set_config(self, config: Dict[str, Any]):
        """
        Store configuration parameters.
        
        Args:
            config: Dictionary with algorithm configuration
        """
        self.config = config
    
    def set_final_best(self, best_individual: Alignment):
        """
        Store the final best individual.
        
        Args:
            best_individual: Best individual from all generations
        """
        self.final_best_individual = best_individual.copy_alignment()
    
    def save_to_directory(self, output_dir: str = "results"):
        """
        Save all results to a directory.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir) / self.run_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_file = output_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save generation history as CSV
        history_file = output_path / "generation_history.csv"
        if self.generation_history:
            with open(history_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.generation_history[0].keys())
                writer.writeheader()
                writer.writerows(self.generation_history)
        
        # Save final best alignment
        if self.final_best_individual:
            best_alignment_file = output_path / "best_alignment.fasta"
            with open(best_alignment_file, 'w') as f:
                for segment in self.final_best_individual.aligned_segments:
                    f.write(f">{segment.id}\n")
                    f.write(f"{segment.sequence}\n")
            
            # Save best alignment info
            best_info_file = output_path / "best_alignment_info.json"
            best_info = {
                'fitness_score': self.final_best_individual.fitness_score,
                'alignment_length': self.final_best_individual.alignment_length,
                'num_sequences': len(self.final_best_individual.aligned_segments),
                'sequence_ids': [seg.id for seg in self.final_best_individual.aligned_segments]
            }
            with open(best_info_file, 'w') as f:
                json.dump(best_info, f, indent=2)
        
        # Save summary
        summary_file = output_path / "summary.json"
        
        # Calculate improvement metrics
        initial_best = self.generation_history[0]['best_fitness_overall'] if self.generation_history else None
        final_best = self.final_best_individual.fitness_score if self.final_best_individual else None
        improvement = (final_best - initial_best) if (final_best is not None and initial_best is not None) else None
        improvement_percent = (improvement / abs(initial_best) * 100) if (improvement is not None and initial_best != 0) else None
        
        summary = {
            'run_id': self.run_id,
            'total_generations': len(self.generation_history),
            'initial_best_fitness': initial_best,
            'final_best_fitness': final_best,
            'improvement_absolute': improvement,
            'improvement_percent': improvement_percent,
            'initial_avg_fitness': self.generation_history[0]['avg_fitness'] if self.generation_history else None,
            'final_avg_fitness': self.generation_history[-1]['avg_fitness'] if self.generation_history else None,
            'convergence_generation': self._find_convergence_generation(),
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        return str(output_path)
    
    def _find_convergence_generation(self) -> Optional[int]:
        """
        Find the generation where convergence occurred (no improvement for last 10% of generations).
        
        Returns:
            Generation number where convergence started, or None
        """
        if not self.generation_history or len(self.generation_history) < 10:
            return None
        
        # Look for the generation where best fitness stopped improving
        window_size = max(10, len(self.generation_history) // 10)
        
        for i in range(len(self.generation_history) - window_size):
            current_best = self.generation_history[i]['best_fitness_overall']
            future_best = self.generation_history[i + window_size]['best_fitness_overall']
            
            # If no improvement in the next window_size generations
            if abs(future_best - current_best) < 1e-6:
                return self.generation_history[i]['generation']
        
        return None


class GeneticAlgorithm:
    """
    Main Genetic Algorithm orchestrator for Multiple Sequence Alignment.
    """
    
    def __init__(self, 
                 initial_sequences: List[SeqRecord],
                 objective_function: BaseObjectiveFunction,
                 population_size: int = 100,
                 num_generations: int = 100,
                 crossover_probability: float = 0.8,
                 mutation_probability: float = 0.5,
                 elitism_count: int = 2,
                 selection_method: str = "tournament",
                 tournament_size: int = 3,
                 crossover_method: str = "single_point",
                 mutation_method: str = "standard",
                 verbose: bool = True):
        """
        Initialize the genetic algorithm.
        
        Args:
            initial_sequences: List of sequences to align (SeqRecord objects)
            objective_function: Objective function to evaluate fitness
            population_size: Number of individuals in the population (must be even)
            num_generations: Number of generations to evolve
            crossover_probability: Probability of crossover
            mutation_probability: Probability of mutation
            elitism_count: Number of best individuals to preserve (elitism)
            selection_method: Selection method ("tournament", "roulette", "rank")
            tournament_size: Tournament size for tournament selection
            crossover_method: Crossover method ("single_point", "uniform", "default")
            mutation_method: Mutation method 
                - "standard": Swap + point mutation
                - "gap_shift": Gap shifting
                - "insertion_deletion": Insert/delete gaps
                - "saga_gap_insertion": SAGA gap insertion with phylogenetic groups
                - "saga_block_shuffling": SAGA block shuffling (16 variants)
                - "saga_block_searching": SAGA block searching
                - "saga_local_rearrangement": SAGA local optimal rearrangement
                - "saga_mixed": Random mix of SAGA operators
            verbose: Print progress information
        """
        self.initial_sequences = initial_sequences
        self.objective_function = objective_function
        self.population_size = population_size if population_size % 2 == 0 else population_size + 1
        self.num_generations = num_generations
        self.elitism_count = elitism_count
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.verbose = verbose
        
        # Initialize operators
        self.selection_operator = SelectionOperator()
        self.crossover_operator = CrossoverOperator(crossover_probability)
        self.mutation_operator = MutationOperator(mutation_probability)
        
        # Initialize results tracking
        self.results = GeneticAlgorithmResults()
        
        # Store configuration
        self.results.set_config({
            'population_size': self.population_size,
            'num_generations': self.num_generations,
            'crossover_probability': crossover_probability,
            'mutation_probability': mutation_probability,
            'elitism_count': self.elitism_count,
            'selection_method': self.selection_method,
            'tournament_size': self.tournament_size,
            'crossover_method': self.crossover_method,
            'mutation_method': self.mutation_method,
            'objective_function': type(objective_function).__name__,
            'num_sequences': len(initial_sequences),
            'sequence_ids': [seq.id for seq in initial_sequences]
        })
    
    def initialize_population(self) -> List[Alignment]:
        """
        Initialize the population with random individuals.
        
        Returns:
            Initial population
        """
        if self.verbose:
            print(f"Initializing population of {self.population_size} individuals...")
        
        population = []
        for i in range(self.population_size):
            individual = Alignment(self.initial_sequences)
            population.append(individual)
            
            if self.verbose and (i + 1) % 20 == 0:
                print(f"  Created {i + 1}/{self.population_size} individuals")
        
        return population
    
    def evaluate_population(self, population: List[Alignment]) -> None:
        """
        Evaluate fitness for all individuals in the population.
        
        Args:
            population: Population to evaluate
        """
        for individual in population:
            individual.calculate_fitness(self.objective_function)
    
    def calculate_diversity(self, population: List[Alignment]) -> float:
        """
        Calculate population diversity as average pairwise differences.
        
        Args:
            population: Current population
            
        Returns:
            Diversity metric (0 to 1)
        """
        if len(population) < 2:
            return 0.0
        
        # Sample a subset for efficiency
        sample_size = min(10, len(population))
        sample = random.sample(population, sample_size)
        
        total_differences = 0
        comparisons = 0
        
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                # Compare alignments
                differences = sum(
                    1 for k in range(len(sample[i].aligned_segments))
                    if sample[i].aligned_segments[k].sequence != sample[j].aligned_segments[k].sequence
                )
                total_differences += differences
                comparisons += 1
        
        if comparisons == 0:
            return 0.0
        
        avg_differences = total_differences / comparisons
        max_possible_differences = len(sample[0].aligned_segments)
        
        return avg_differences / max_possible_differences if max_possible_differences > 0 else 0.0
    
    def evolve_generation(self, population: List[Alignment]) -> List[Alignment]:
        """
        Evolve one generation using selection, crossover, and mutation.
        
        Args:
            population: Current population
            
        Returns:
            New population for next generation
        """
        # Sort population by fitness (descending)
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Keep elite individuals
        new_population = [ind.copy_alignment() for ind in population[:self.elitism_count]]
        
        # Keep the better half (after elites)
        survivors = population[self.elitism_count:self.population_size // 2]
        
        # Generate offspring to replace eliminated individuals
        num_offspring_needed = self.population_size - len(new_population) - len(survivors)
        offspring = []
        
        while len(offspring) < num_offspring_needed:
            # Select parents from the entire population (gives better individuals higher chance)
            parent1, parent2 = self.selection_operator.select_parents(
                population,
                selection_method=self.selection_method,
                tournament_size=self.tournament_size
            )
            
            # Apply crossover
            if self.crossover_method == "single_point":
                child1, child2 = self.crossover_operator.single_point_crossover(parent1, parent2)
            elif self.crossover_method == "uniform":
                child1, child2 = self.crossover_operator.uniform_crossover(parent1, parent2)
            else:
                child1, child2 = self.crossover_operator.crossover(parent1, parent2)
            
            # Apply mutation
            if self.mutation_method == "gap_shift":
                child1 = self.mutation_operator.gap_shift_mutation(child1)
                child2 = self.mutation_operator.gap_shift_mutation(child2)
            elif self.mutation_method == "insertion_deletion":
                child1 = self.mutation_operator.insertion_deletion_mutation(child1)
                child2 = self.mutation_operator.insertion_deletion_mutation(child2)
            elif self.mutation_method == "saga_gap_insertion":
                child1 = self.mutation_operator.saga_gap_insertion(child1, self.objective_function)
                child2 = self.mutation_operator.saga_gap_insertion(child2, self.objective_function)
            elif self.mutation_method == "saga_block_shuffling":
                child1 = self.mutation_operator.saga_block_shuffling(child1, self.objective_function)
                child2 = self.mutation_operator.saga_block_shuffling(child2, self.objective_function)
            elif self.mutation_method == "saga_block_searching":
                child1 = self.mutation_operator.saga_block_searching(child1)
                child2 = self.mutation_operator.saga_block_searching(child2)
            elif self.mutation_method == "saga_local_rearrangement":
                child1 = self.mutation_operator.saga_local_rearrangement(child1, self.objective_function)
                child2 = self.mutation_operator.saga_local_rearrangement(child2, self.objective_function)
            elif self.mutation_method == "saga_mixed":
                # Apply a random SAGA operator
                import random
                saga_ops = [
                    lambda ind: self.mutation_operator.saga_gap_insertion(ind, self.objective_function),
                    lambda ind: self.mutation_operator.saga_block_shuffling(ind, self.objective_function),
                    lambda ind: self.mutation_operator.saga_block_searching(ind),
                ]
                child1 = random.choice(saga_ops)(child1)
                child2 = random.choice(saga_ops)(child2)
            else:
                child1 = self.mutation_operator.mutate(child1)
                child2 = self.mutation_operator.mutate(child2)
            
            # Evaluate offspring
            child1.calculate_fitness(self.objective_function)
            child2.calculate_fitness(self.objective_function)
            
            offspring.extend([child1, child2])
        
        # Combine elites, survivors, and offspring
        new_population.extend(survivors)
        new_population.extend(offspring[:num_offspring_needed])
        
        return new_population
    
    def run(self, save_results: bool = True, output_dir: str = "results") -> GeneticAlgorithmResults:
        """
        Run the genetic algorithm for the specified number of generations.
        
        Args:
            save_results: Whether to save results to disk
            output_dir: Directory to save results
            
        Returns:
            Results object containing all execution data
        """
        if self.verbose:
            print("=" * 60)
            print("GENETIC ALGORITHM FOR MULTIPLE SEQUENCE ALIGNMENT")
            print("=" * 60)
            print(f"Population size: {self.population_size}")
            print(f"Number of generations: {self.num_generations}")
            print(f"Objective function: {type(self.objective_function).__name__}")
            print("=" * 60)
        
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial population
        if self.verbose:
            print("\nEvaluating initial population...")
        self.evaluate_population(population)
        
        # Track best individual
        best_individual = max(population, key=lambda x: x.fitness_score).copy_alignment()
        
        # Record initial generation
        avg_fitness = sum(ind.fitness_score for ind in population) / len(population)
        diversity = self.calculate_diversity(population)
        current_best = max(population, key=lambda x: x.fitness_score)
        self.results.add_generation(0, population, best_individual, current_best, avg_fitness, diversity)
        
        if self.verbose:
            print("\nGeneration 0:")
            print(f"  Best fitness (overall): {best_individual.fitness_score:.4f}")
            print(f"  Best fitness (generation): {current_best.fitness_score:.4f}")
            print(f"  Avg fitness: {avg_fitness:.4f}")
            print(f"  Diversity: {diversity:.4f}")
        
        # Evolution loop
        for generation in range(1, self.num_generations + 1):
            # Evolve population
            population = self.evolve_generation(population)
            
            # Find best individual in this generation
            current_best = max(population, key=lambda x: x.fitness_score)
            
            # Update global best if improved
            if current_best.fitness_score > best_individual.fitness_score:
                best_individual = current_best.copy_alignment()
            
            # Calculate statistics
            avg_fitness = sum(ind.fitness_score for ind in population) / len(population)
            diversity = self.calculate_diversity(population)
            
            # Record generation (best overall vs best of this generation)
            self.results.add_generation(generation, population, best_individual, current_best, avg_fitness, diversity)
            
            # Print progress
            if self.verbose and (generation % 10 == 0 or generation == self.num_generations):
                print(f"\nGeneration {generation}:")
                print(f"  Best fitness (overall): {best_individual.fitness_score:.4f}")
                print(f"  Best fitness (generation): {current_best.fitness_score:.4f}")
                print(f"  Avg fitness: {avg_fitness:.4f}")
                print(f"  Diversity: {diversity:.4f}")
        
        # Store final best individual
        self.results.set_final_best(best_individual)
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("EVOLUTION COMPLETED")
            print("=" * 60)
            print(f"Final best fitness: {best_individual.fitness_score:.4f}")
            print(f"Improvement: {best_individual.fitness_score - self.results.generation_history[0]['best_fitness']:.4f}")
        
        # Save results if requested
        if save_results:
            output_path = self.results.save_to_directory(output_dir)
            if self.verbose:
                print(f"Results saved to: {output_path}")
        
        return self.results
    
    def get_best_alignment(self) -> Alignment:
        """
        Get the best alignment found by the algorithm.
        
        Returns:
            Best alignment (Alignment object)
        """
        if self.results.final_best_individual is None:
            raise ValueError("No results available. Run the algorithm first.")
        
        return self.results.final_best_individual
