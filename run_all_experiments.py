"""
Comprehensive Experiment Runner for Multiple Sequence Alignment using Genetic Algorithm.

This script runs multiple experiments with different configurations and consolidates results:
- Different objective functions (SAGA, T-Coffee)
- Different mutation operators (standard, SAGA operators)
- Different selection methods
- Different crossover methods
- Multiple runs for statistical significance

Results are saved and analyzed automatically.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any
import time

from genetic_algorithm import (
    GeneticAlgorithm, 
    SAGAObjectiveFunction, 
    TCoffeeObjectiveFunction
)
from genetic_algorithm.utils.read_sequences_file import read_fasta_file


class ExperimentRunner:
    """
    Orchestrates multiple experiments and consolidates results.
    """
    
    def __init__(self, 
                 sequences_file: str,
                 output_dir: str = "experiments",
                 max_sequences: int = None):
        """
        Initialize experiment runner.
        
        Args:
            sequences_file: Path to sequences file
            output_dir: Directory to save all experiment results
            max_sequences: Maximum number of sequences to use (None = all)
        """
        self.sequences_file = sequences_file
        self.output_dir = output_dir
        self.max_sequences = max_sequences
        
        # Load sequences
        all_sequences = read_fasta_file(sequences_file)
        if max_sequences:
            self.sequences = all_sequences[:max_sequences]
            print(f"Using first {max_sequences} of {len(all_sequences)} sequences")
        else:
            self.sequences = all_sequences
            print(f"Using all {len(self.sequences)} sequences")
        
        # Create output directory
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(output_dir, self.experiment_id)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Store all experiment results
        self.all_results = []
        
        print(f"Experiment directory: {self.experiment_dir}")
    
    def run_single_experiment(self, 
                             config: Dict[str, Any],
                             experiment_name: str,
                             num_runs: int = 1) -> List[Dict]:
        """
        Run a single experiment configuration multiple times.
        
        Args:
            config: Configuration dictionary for GeneticAlgorithm
            experiment_name: Name for this experiment
            num_runs: Number of repetitions for statistical significance
            
        Returns:
            List of results for each run
        """
        print(f"\n{'='*80}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*80}")
        
        run_results = []
        
        for run in range(1, num_runs + 1):
            print(f"\nRun {run}/{num_runs}")
            print("-" * 40)
            
            start_time = time.time()
            
            # Create GA instance
            ga = GeneticAlgorithm(
                initial_sequences=self.sequences,
                **config,
                verbose=False  # Disable verbose for cleaner output
            )
            
            # Run algorithm
            results = ga.run(
                save_results=True, 
                output_dir=os.path.join(self.experiment_dir, experiment_name, f"run_{run}")
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Get final results
            best = ga.get_best_alignment()
            
            # Use the new metric names
            initial_best_fitness = results.generation_history[0]['best_fitness_overall']
            initial_avg_fitness = results.generation_history[0]['avg_fitness']
            final_best_fitness = best.fitness_score
            final_avg_fitness = results.generation_history[-1]['avg_fitness']
            improvement = final_best_fitness - initial_best_fitness
            
            # Store results
            run_result = {
                'experiment_name': experiment_name,
                'run': run,
                'config': config.copy(),
                'initial_best_fitness': initial_best_fitness,
                'initial_avg_fitness': initial_avg_fitness,
                'final_best_fitness': final_best_fitness,
                'final_avg_fitness': final_avg_fitness,
                'improvement': improvement,
                'improvement_percent': (improvement / abs(initial_best_fitness) * 100) if initial_best_fitness != 0 else 0,
                'num_generations': len(results.generation_history),
                'elapsed_time_seconds': elapsed_time,
                'run_id': results.run_id,
                'num_sequences': len(self.sequences),
                'convergence_generation': results.generation_history[-1].get('generation') if results.generation_history else None
            }
            
            run_results.append(run_result)
            
            print(f"Initial Best Fitness: {initial_best_fitness:.4f}")
            print(f"Final Best Fitness: {final_best_fitness:.4f}")
            print(f"Improvement: {improvement:.4f} ({run_result['improvement_percent']:.2f}%)")
            print(f"Time: {elapsed_time:.2f}s")
        
        return run_results
    
    def run_objective_function_comparison(self, num_runs: int = 3):
        """
        Compare SAGA vs T-Coffee objective functions.
        """
        print("\n" + "="*80)
        print("COMPARISON 1: Objective Functions (SAGA vs T-Coffee)")
        print("="*80)
        
        base_config = {
            'population_size': 30,
            'num_generations': 50,
            'crossover_probability': 0.8,
            'mutation_probability': 0.5,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'crossover_method': 'single_point',
            'mutation_method': 'standard'
        }
        
        # SAGA objective function
        saga_config = base_config.copy()
        saga_config['objective_function'] = SAGAObjectiveFunction(self.sequences)
        saga_results = self.run_single_experiment(
            saga_config, 
            "obj_saga", 
            num_runs
        )
        self.all_results.extend(saga_results)
        
        # T-Coffee objective function
        tcoffee_config = base_config.copy()
        tcoffee_config['objective_function'] = TCoffeeObjectiveFunction(self.sequences)
        tcoffee_results = self.run_single_experiment(
            tcoffee_config, 
            "obj_tcoffee", 
            num_runs
        )
        self.all_results.extend(tcoffee_results)
    
    def run_mutation_operator_comparison(self, num_runs: int = 3):
        """
        Compare different mutation operators.
        """
        print("\n" + "="*80)
        print("COMPARISON 2: Mutation Operators")
        print("="*80)
        
        base_config = {
            'objective_function': SAGAObjectiveFunction(self.sequences),
            'population_size': 30,
            'num_generations': 50,
            'crossover_probability': 0.8,
            'mutation_probability': 0.5,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'crossover_method': 'single_point'
        }
        
        mutation_methods = [
            ('standard', 'mut_standard'),
            ('gap_shift', 'mut_gap_shift'),
            ('insertion_deletion', 'mut_insertion_deletion'),
            ('saga_gap_insertion', 'mut_saga_gap'),
            ('saga_block_shuffling', 'mut_saga_block'),
            ('saga_mixed', 'mut_saga_mixed')
        ]
        
        for mutation_method, exp_name in mutation_methods:
            config = base_config.copy()
            config['mutation_method'] = mutation_method
            
            results = self.run_single_experiment(
                config,
                exp_name,
                num_runs
            )
            self.all_results.extend(results)
    
    def run_selection_method_comparison(self, num_runs: int = 3):
        """
        Compare different selection methods.
        """
        print("\n" + "="*80)
        print("COMPARISON 3: Selection Methods")
        print("="*80)
        
        base_config = {
            'objective_function': SAGAObjectiveFunction(self.sequences),
            'population_size': 30,
            'num_generations': 50,
            'crossover_probability': 0.8,
            'mutation_probability': 0.5,
            'elitism_count': 2,
            'crossover_method': 'single_point',
            'mutation_method': 'standard'
        }
        
        selection_methods = [
            ('tournament', 3, 'sel_tournament'),
            ('roulette', None, 'sel_roulette'),
            ('rank', None, 'sel_rank')
        ]
        
        for selection_method, tournament_size, exp_name in selection_methods:
            config = base_config.copy()
            config['selection_method'] = selection_method
            if tournament_size:
                config['tournament_size'] = tournament_size
            
            results = self.run_single_experiment(
                config,
                exp_name,
                num_runs
            )
            self.all_results.extend(results)
    
    def run_crossover_method_comparison(self, num_runs: int = 3):
        """
        Compare different crossover methods.
        """
        print("\n" + "="*80)
        print("COMPARISON 4: Crossover Methods")
        print("="*80)
        
        base_config = {
            'objective_function': SAGAObjectiveFunction(self.sequences),
            'population_size': 30,
            'num_generations': 50,
            'crossover_probability': 0.8,
            'mutation_probability': 0.5,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'mutation_method': 'standard'
        }
        
        crossover_methods = [
            ('default', 'cross_default'),
            ('single_point', 'cross_single_point'),
            ('uniform', 'cross_uniform')
        ]
        
        for crossover_method, exp_name in crossover_methods:
            config = base_config.copy()
            config['crossover_method'] = crossover_method
            
            results = self.run_single_experiment(
                config,
                exp_name,
                num_runs
            )
            self.all_results.extend(results)
    
    def run_parameter_sensitivity_analysis(self, num_runs: int = 3):
        """
        Analyze sensitivity to population size and mutation/crossover probabilities.
        """
        print("\n" + "="*80)
        print("COMPARISON 5: Parameter Sensitivity")
        print("="*80)
        
        base_config = {
            'objective_function': SAGAObjectiveFunction(self.sequences),
            'num_generations': 50,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'crossover_method': 'single_point',
            'mutation_method': 'standard'
        }
        
        # Population sizes
        for pop_size in [20, 30, 50]:
            config = base_config.copy()
            config['population_size'] = pop_size
            config['crossover_probability'] = 0.8
            config['mutation_probability'] = 0.5
            
            results = self.run_single_experiment(
                config,
                f"param_pop_{pop_size}",
                num_runs
            )
            self.all_results.extend(results)
        
        # Mutation probabilities
        for mut_prob in [0.3, 0.5, 0.7]:
            config = base_config.copy()
            config['population_size'] = 30
            config['crossover_probability'] = 0.8
            config['mutation_probability'] = mut_prob
            
            results = self.run_single_experiment(
                config,
                f"param_mut_{int(mut_prob*100)}",
                num_runs
            )
            self.all_results.extend(results)
        
        # Crossover probabilities
        for cross_prob in [0.6, 0.8, 0.9]:
            config = base_config.copy()
            config['population_size'] = 30
            config['crossover_probability'] = cross_prob
            config['mutation_probability'] = 0.5
            
            results = self.run_single_experiment(
                config,
                f"param_cross_{int(cross_prob*100)}",
                num_runs
            )
            self.all_results.extend(results)
    
    def save_consolidated_results(self):
        """
        Save all results to CSV and JSON files.
        """
        print("\n" + "="*80)
        print("Saving Consolidated Results")
        print("="*80)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_results)
        
        # Remove complex objects for CSV
        df_csv = df.drop(columns=['config', 'objective_function'], errors='ignore')
        
        # Save CSV
        csv_path = os.path.join(self.experiment_dir, "consolidated_results.csv")
        df_csv.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # Save JSON (with config details)
        json_path = os.path.join(self.experiment_dir, "consolidated_results.json")
        # Convert objective functions to strings for JSON
        results_for_json = []
        for result in self.all_results:
            result_copy = result.copy()
            if 'config' in result_copy and 'objective_function' in result_copy['config']:
                result_copy['config']['objective_function'] = type(result_copy['config']['objective_function']).__name__
            results_for_json.append(result_copy)
        
        with open(json_path, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        print(f"Saved JSON: {json_path}")
        
        return df_csv
    
    def generate_summary_report(self, df: pd.DataFrame):
        """
        Generate a summary report with statistics and visualizations.
        """
        print("\n" + "="*80)
        print("Generating Summary Report")
        print("="*80)
        
        report_path = os.path.join(self.experiment_dir, "SUMMARY_REPORT.md")
        
        with open(report_path, 'w') as f:
            f.write("# Experiment Summary Report\n\n")
            f.write(f"**Experiment ID**: {self.experiment_id}\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Number of Sequences**: {self.all_results[0]['num_sequences']}\n\n")
            f.write(f"**Total Experiments**: {len(self.all_results)}\n\n")
            f.write("---\n\n")
            
            # Group by experiment type
            grouped = df.groupby('experiment_name')
            
            f.write("## Results by Experiment\n\n")
            
            summary_stats = grouped.agg({
                'initial_best_fitness': ['mean', 'std'],
                'final_best_fitness': ['mean', 'std'],
                'improvement': ['mean', 'std'],
                'improvement_percent': ['mean', 'std'],
                'elapsed_time_seconds': ['mean', 'std']
            }).round(4)
            
            f.write("### Statistical Summary\n\n")
            f.write("```\n")
            f.write(summary_stats.to_string())
            f.write("\n```\n\n")
            
            # Best configurations
            f.write("## Best Configurations\n\n")
            
            f.write("### Top 10 by Final Fitness\n\n")
            top_fitness = df.nlargest(10, 'final_best_fitness')[['experiment_name', 'run', 'final_best_fitness', 'improvement_percent']]
            f.write(top_fitness.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("### Top 10 by Improvement\n\n")
            top_improvement = df.nlargest(10, 'improvement')[['experiment_name', 'run', 'improvement', 'improvement_percent']]
            f.write(top_improvement.to_markdown(index=False))
            f.write("\n\n")
            
            # Comparison by category
            f.write("## Comparisons\n\n")
            
            # Objective functions
            obj_results = df[df['experiment_name'].str.startswith('obj_')]
            if not obj_results.empty:
                f.write("### Objective Functions\n\n")
                obj_summary = obj_results.groupby('experiment_name').agg({
                    'final_best_fitness': ['mean', 'std'],
                    'improvement_percent': ['mean', 'std']
                }).round(4)
                f.write(obj_summary.to_markdown())
                f.write("\n\n")
            
            # Mutation operators
            mut_results = df[df['experiment_name'].str.startswith('mut_')]
            if not mut_results.empty:
                f.write("### Mutation Operators\n\n")
                mut_summary = mut_results.groupby('experiment_name').agg({
                    'final_best_fitness': ['mean', 'std'],
                    'improvement_percent': ['mean', 'std']
                }).round(4)
                f.write(mut_summary.to_markdown())
                f.write("\n\n")
            
            # Selection methods
            sel_results = df[df['experiment_name'].str.startswith('sel_')]
            if not sel_results.empty:
                f.write("### Selection Methods\n\n")
                sel_summary = sel_results.groupby('experiment_name').agg({
                    'final_best_fitness': ['mean', 'std'],
                    'improvement_percent': ['mean', 'std']
                }).round(4)
                f.write(sel_summary.to_markdown())
                f.write("\n\n")
        
        print(f"Summary report saved: {report_path}")
        
        return report_path
    
    def generate_visualizations(self, df: pd.DataFrame):
        """
        Generate visualization plots.
        """
        print("\n" + "="*80)
        print("Generating Visualizations")
        print("="*80)
        
        viz_dir = os.path.join(self.experiment_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Box plot by experiment
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Final fitness by experiment
        ax1 = axes[0, 0]
        df.boxplot(column='final_best_fitness', by='experiment_name', ax=ax1, rot=45)
        ax1.set_title('Final Fitness by Experiment')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Final Fitness')
        
        # Improvement percentage by experiment
        ax2 = axes[0, 1]
        df.boxplot(column='improvement_percent', by='experiment_name', ax=ax2, rot=45)
        ax2.set_title('Improvement % by Experiment')
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Improvement %')
        
        # Execution time by experiment
        ax3 = axes[1, 0]
        df.boxplot(column='elapsed_time_seconds', by='experiment_name', ax=ax3, rot=45)
        ax3.set_title('Execution Time by Experiment')
        ax3.set_xlabel('Experiment')
        ax3.set_ylabel('Time (seconds)')
        
        # Scatter: Time vs Improvement
        ax4 = axes[1, 1]
        ax4.scatter(df['elapsed_time_seconds'], df['improvement_percent'], alpha=0.6)
        ax4.set_xlabel('Execution Time (seconds)')
        ax4.set_ylabel('Improvement %')
        ax4.set_title('Time vs Improvement Trade-off')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('')  # Remove auto title
        plt.tight_layout()
        
        fig_path = os.path.join(viz_dir, "overview.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        plt.close()
        
        # 2. Comparison plots by category
        self._plot_category_comparison(df, 'obj_', 'Objective Functions', viz_dir)
        self._plot_category_comparison(df, 'mut_', 'Mutation Operators', viz_dir)
        self._plot_category_comparison(df, 'sel_', 'Selection Methods', viz_dir)
        self._plot_category_comparison(df, 'cross_', 'Crossover Methods', viz_dir)
        self._plot_category_comparison(df, 'param_', 'Parameter Sensitivity', viz_dir)
    
    def _plot_category_comparison(self, df: pd.DataFrame, prefix: str, title: str, viz_dir: str):
        """Helper to plot comparisons within a category."""
        category_df = df[df['experiment_name'].str.startswith(prefix)]
        
        if category_df.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Mean final fitness with error bars
        ax1 = axes[0]
        grouped = category_df.groupby('experiment_name')
        means = grouped['final_best_fitness'].mean()
        stds = grouped['final_best_fitness'].std()
        means.plot(kind='bar', ax=ax1, yerr=stds, capsize=5, color='steelblue')
        ax1.set_title(f'{title} - Final Fitness')
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Final Fitness')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        
        # Mean improvement with error bars
        ax2 = axes[1]
        means = grouped['improvement_percent'].mean()
        stds = grouped['improvement_percent'].std()
        means.plot(kind='bar', ax=ax2, yerr=stds, capsize=5, color='coral')
        ax2.set_title(f'{title} - Improvement %')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Improvement %')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        filename = f"comparison_{prefix.rstrip('_')}.png"
        fig_path = os.path.join(viz_dir, filename)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        plt.close()
    
    def run_all_experiments(self, num_runs: int = 3):
        """
        Run all experiment categories.
        
        Args:
            num_runs: Number of repetitions per configuration
        """
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE EXPERIMENT SUITE")
        print("="*80)
        print(f"Sequences: {len(self.sequences)}")
        print(f"Runs per configuration: {num_runs}")
        print(f"Output directory: {self.experiment_dir}")
        print("="*80)
        
        start_time = time.time()
        
        # Run all comparisons
        self.run_objective_function_comparison(num_runs)
        self.run_mutation_operator_comparison(num_runs)
        self.run_selection_method_comparison(num_runs)
        self.run_crossover_method_comparison(num_runs)
        self.run_parameter_sensitivity_analysis(num_runs)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*80)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Total experiments: {len(self.all_results)}")
        
        # Save and analyze results
        df = self.save_consolidated_results()
        self.generate_summary_report(df)
        self.generate_visualizations(df)
        
        print("\n" + "="*80)
        print(f"All results saved to: {self.experiment_dir}")
        print("="*80)
        
        return df


def main():
    """
    Main function to run comprehensive experiments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive GA experiments')
    parser.add_argument('--sequences', default='sequences/seqdump_1.txt',
                       help='Path to sequences file')
    parser.add_argument('--max-sequences', type=int, default=10,
                       help='Maximum number of sequences to use (for speed)')
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of runs per configuration')
    parser.add_argument('--output-dir', default='experiments',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPREHENSIVE GENETIC ALGORITHM EXPERIMENT SUITE")
    print("="*80)
    print(f"Sequences file: {args.sequences}")
    print(f"Max sequences: {args.max_sequences}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Create and run experiments
    runner = ExperimentRunner(
        sequences_file=args.sequences,
        output_dir=args.output_dir,
        max_sequences=args.max_sequences
    )
    
    runner.run_all_experiments(num_runs=args.num_runs)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUITE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults directory: {runner.experiment_dir}")
    print("\nGenerated files:")
    print("  - consolidated_results.csv")
    print("  - consolidated_results.json")
    print("  - SUMMARY_REPORT.md")
    print("  - visualizations/")
    print("\nYou can now analyze the results!")
    
    return runner


if __name__ == "__main__":
    main()
