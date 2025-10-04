"""
Selection Operator for Multiple Sequence Alignment Genetic Algorithm.

Implements different selection methods based on fitness function, ensuring that
the best individual has a higher probability of reproduction and mating.

Based on the research paper that describes selection as the process of choosing the best
individual from the generation, depending on the fitness function.
"""

import random
import copy
from typing import List, Tuple
from genetic_algorithm.alignment import Alignment


class SelectionOperator:
    def __init__(self):
        """
        Initialize the selection operator.
        """
        pass
    
    def tournament_selection(self, population: List[Alignment], tournament_size: int = 3) -> Alignment:
        """
        Implement tournament selection.
        
        Randomly selects a group of individuals and returns the best among them.
        This method ensures that individuals with better fitness have higher probability
        of being selected.
        
        Args:
            population: Population of individuals
            tournament_size: Tournament size
            
        Returns:
            Selected individual
        """
        if len(population) == 0:
            raise ValueError("Population cannot be empty")
        
        # Select random candidates for tournament
        tournament_size = min(tournament_size, len(population))
        tournament_candidates = random.sample(population, tournament_size)
        
        # Return the best candidate (highest fitness)
        best_candidate = max(tournament_candidates, key=lambda x: x.fitness_score)
        return best_candidate.copy_alignment()
    
    def roulette_wheel_selection(self, population: List[Alignment]) -> Alignment:
        """
        Implement roulette wheel selection.
        
        Selection probability is proportional to individual fitness.
        Individuals with higher fitness have greater chance of being selected.
        
        Args:
            population: Population of individuals
            
        Returns:
            Selected individual
        """
        if len(population) == 0:
            raise ValueError("Population cannot be empty")
        
        # Calculate total fitness and adjust for positive values if necessary
        fitness_scores = [ind.fitness_score for ind in population]
        min_fitness = min(fitness_scores)
        
        # Adjust fitness to positive values if there are negative values
        if min_fitness < 0:
            adjusted_fitness = [score - min_fitness + 1 for score in fitness_scores]
        else:
            adjusted_fitness = fitness_scores
        
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness == 0:
            # If all fitness are equal, select randomly
            return random.choice(population).copy_alignment()
        
        # Generate a random number between 0 and total_fitness
        spin = random.uniform(0, total_fitness)
        
        # Find corresponding individual
        current_sum = 0
        for i, fitness in enumerate(adjusted_fitness):
            current_sum += fitness
            if current_sum >= spin:
                return population[i].copy_alignment()
        
        # Fallback: return last individual
        return population[-1].copy_alignment()
    
    def rank_selection(self, population: List[Alignment]) -> Alignment:
        """
        Implement rank selection.
        
        Individuals are sorted by fitness and selection probability
        is based on rank, not absolute fitness value.
        
        Args:
            population: Population of individuals
            
        Returns:
            Selected individual
        """
        if len(population) == 0:
            raise ValueError("Population cannot be empty")
        
        # Sort population by fitness (lowest to highest)
        sorted_population = sorted(population, key=lambda x: x.fitness_score)
        n = len(sorted_population)
        
        # Calculate probabilities based on rank (higher rank = higher probability)
        rank_sum = n * (n + 1) // 2
        
        # Generate random number
        spin = random.uniform(0, rank_sum)
        
        # Find individual based on rank
        current_sum = 0
        for i in range(n):
            current_sum += (i + 1)  # Rank starts at 1
            if current_sum >= spin:
                return sorted_population[i].copy_alignment()
        
        # Fallback: return best individual
        return sorted_population[-1].copy_alignment()
    
    def elitist_selection(self, population: List[Alignment], num_elites: int = 1) -> List[Alignment]:
        """
        Implement elitist selection.
        
        Select the best individuals from population based on fitness.
        Ensures that the best individuals are preserved.
        
        Args:
            population: Population of individuals
            num_elites: Number of elites to select
            
        Returns:
            List of best individuals
        """
        if len(population) == 0:
            raise ValueError("Population cannot be empty")
        
        # Sort by fitness (highest to lowest)
        sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        
        # Select the best
        num_elites = min(num_elites, len(sorted_population))
        elites = [ind.copy_alignment() for ind in sorted_population[:num_elites]]
        
        return elites
    
    def select_parents(self, population: List[Alignment], 
                      selection_method: str = "tournament", 
                      **kwargs) -> Tuple[Alignment, Alignment]:
        """
        Select two parents from the population for reproduction.
        
        Args:
            population: Population of individuals
            selection_method: Selection method ("tournament", "roulette", "rank")
            **kwargs: Additional arguments for the selection method
            
        Returns:
            Tuple containing two selected parents
        """
        if len(population) < 2:
            raise ValueError("Population must have at least 2 individuals for parent selection")
        
        # Select the first parent
        if selection_method == "tournament":
            tournament_size = kwargs.get("tournament_size", 3)
            parent1 = self.tournament_selection(population, tournament_size)
        elif selection_method == "roulette":
            parent1 = self.roulette_wheel_selection(population)
        elif selection_method == "rank":
            parent1 = self.rank_selection(population)
        else:
            raise ValueError(f"Selection method '{selection_method}' not recognized")
        
        # Select the second parent (different from the first)
        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            if selection_method == "tournament":
                parent2 = self.tournament_selection(population, tournament_size)
            elif selection_method == "roulette":
                parent2 = self.roulette_wheel_selection(population)
            elif selection_method == "rank":
                parent2 = self.rank_selection(population)
            
            # Check if parents are different (optional, depending on implementation)
            if parent2.aligned_segments != parent1.aligned_segments:
                break
            attempts += 1
        else:
            # If cannot find a different parent, use random selection
            available_parents = [ind for ind in population 
                               if ind.aligned_segments != parent1.aligned_segments]
            if available_parents:
                parent2 = random.choice(available_parents).copy_alignment()
            else:
                parent2 = random.choice(population).copy_alignment()
        
        return parent1, parent2
    
    def survivor_selection(self, population: List[Alignment], 
                          offspring: List[Alignment], 
                          population_size: int,
                          selection_method: str = "elitist") -> List[Alignment]:
        """
        Select survivors for the next generation.
        
        Combines current population and offspring and selects the best
        to form the new population.
        
        Args:
            population: Current population
            offspring: Generated offspring
            population_size: Size of the new population
            selection_method: Method for survivor selection
            
        Returns:
            New selected population
        """
        # Combine current population and offspring
        combined_population = population + offspring
        
        if selection_method == "elitist":
            # Select the best based on fitness
            sorted_population = sorted(combined_population, 
                                     key=lambda x: x.fitness_score, 
                                     reverse=True)
            return sorted_population[:population_size]
        
        elif selection_method == "generational":
            # Completely replace old population with offspring
            if len(offspring) >= population_size:
                sorted_offspring = sorted(offspring, 
                                        key=lambda x: x.fitness_score, 
                                        reverse=True)
                return sorted_offspring[:population_size]
            else:
                # If there are not enough offspring, complete with best from current population
                best_parents = self.elitist_selection(population, 
                                                    population_size - len(offspring))
                return offspring + best_parents
        
        else:
            raise ValueError(f"Survivor selection method '{selection_method}' not recognized")