"""
Genetic Algorithm for Multiple Sequence Alignment

This package provides a complete implementation of a genetic algorithm
for solving the multiple sequence alignment problem.
"""

from genetic_algorithm.alignment import Alignment, AlignedSegment
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmResults
from genetic_algorithm.objective_function.base_objective_function import BaseObjectiveFunction
from genetic_algorithm.objective_function.saga_objective_function import SAGAObjectiveFunction
from genetic_algorithm.objective_function.tcoffee_objective_function import TCoffeeObjectiveFunction
from genetic_algorithm.operators.selection import SelectionOperator
from genetic_algorithm.operators.crossover import CrossoverOperator
from genetic_algorithm.operators.mutation import MutationOperator

__all__ = [
    'Alignment',
    'AlignedSegment',
    'GeneticAlgorithm',
    'GeneticAlgorithmResults',
    'BaseObjectiveFunction',
    'SAGAObjectiveFunction',
    'TCoffeeObjectiveFunction',
    'SelectionOperator',
    'CrossoverOperator',
    'MutationOperator',
]

__version__ = '1.0.0'
