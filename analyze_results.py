"""
Script for analyzing results from genetic algorithm runs.

This script provides tools to load and analyze saved results,
including visualization of convergence and comparison of runs.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from Bio import SeqIO


class ResultsAnalyzer:
    """
    Analyzer for genetic algorithm results.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Path to the results directory
        """
        self.results_path = Path(results_dir)
        
        if not self.results_path.exists():
            raise ValueError(f"Results directory not found: {results_dir}")
        
        # Load data
        self.config = self._load_config()
        self.history = self._load_history()
        self.summary = self._load_summary()
        self.best_alignment = self._load_best_alignment()
    
    def _load_config(self) -> Dict:
        """Load configuration file."""
        config_file = self.results_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_history(self) -> pd.DataFrame:
        """Load generation history."""
        history_file = self.results_path / "generation_history.csv"
        if history_file.exists():
            return pd.read_csv(history_file)
        return pd.DataFrame()
    
    def _load_summary(self) -> Dict:
        """Load summary file."""
        summary_file = self.results_path / "summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_best_alignment(self) -> List:
        """Load best alignment sequences."""
        alignment_file = self.results_path / "best_alignment.fasta"
        if alignment_file.exists():
            return list(SeqIO.parse(alignment_file, "fasta"))
        return []
    
    def print_summary(self):
        """Print a summary of the results."""
        print("=" * 80)
        print(f"RESULTS ANALYSIS: {self.results_path.name}")
        print("=" * 80)
        
        print("\n### Configuration ###")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        print("\n### Summary ###")
        for key, value in self.summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n### Best Alignment ###")
        if self.best_alignment:
            print(f"  Number of sequences: {len(self.best_alignment)}")
            print(f"  Alignment length: {len(self.best_alignment[0].seq)}")
            print("  Sequences:")
            for record in self.best_alignment[:5]:  # Show first 5
                print(f"    - {record.id}: {str(record.seq)[:50]}...")
            if len(self.best_alignment) > 5:
                print(f"    ... and {len(self.best_alignment) - 5} more")
    
    def plot_convergence(self, save_fig: bool = False):
        """
        Plot the convergence of the algorithm.
        
        Args:
            save_fig: Whether to save the figure
        """
        if self.history.empty:
            print("No history data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Fitness over generations
        ax1 = axes[0]
        ax1.plot(self.history['generation'], self.history['best_fitness'], 
                label='Best Fitness', linewidth=2, color='green')
        ax1.plot(self.history['generation'], self.history['avg_fitness'], 
                label='Average Fitness', linewidth=2, color='blue', alpha=0.7)
        ax1.fill_between(self.history['generation'], 
                         self.history['min_fitness'], 
                         self.history['max_fitness'],
                         alpha=0.2, color='gray', label='Min-Max Range')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.set_title('Fitness Evolution Over Generations', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Diversity over generations
        ax2 = axes[1]
        ax2.plot(self.history['generation'], self.history['diversity'], 
                linewidth=2, color='orange')
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Diversity', fontsize=12)
        ax2.set_title('Population Diversity Over Generations', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = self.results_path / "convergence_plot.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {fig_path}")
        
        plt.show()
    
    def plot_fitness_statistics(self, save_fig: bool = False):
        """
        Plot detailed fitness statistics.
        
        Args:
            save_fig: Whether to save the figure
        """
        if self.history.empty:
            print("No history data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Best fitness progression
        ax1 = axes[0, 0]
        ax1.plot(self.history['generation'], self.history['best_fitness'], 
                linewidth=2, color='green', marker='o', markersize=4)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title('Best Fitness Progression')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement rate
        ax2 = axes[0, 1]
        if len(self.history) > 1:
            improvement = self.history['best_fitness'].diff()
            ax2.plot(self.history['generation'][1:], improvement[1:], 
                    linewidth=2, color='blue')
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Fitness Improvement')
            ax2.set_title('Improvement per Generation')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fitness spread
        ax3 = axes[1, 0]
        spread = self.history['max_fitness'] - self.history['min_fitness']
        ax3.plot(self.history['generation'], spread, 
                linewidth=2, color='purple')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Fitness Spread (Max - Min)')
        ax3.set_title('Population Fitness Spread')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Correlation between diversity and improvement
        ax4 = axes[1, 1]
        if len(self.history) > 1:
            improvement = self.history['best_fitness'].diff()
            ax4.scatter(self.history['diversity'][1:], improvement[1:], 
                       alpha=0.6, s=50, color='orange')
            ax4.set_xlabel('Diversity')
            ax4.set_ylabel('Fitness Improvement')
            ax4.set_title('Diversity vs Improvement')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = self.results_path / "fitness_statistics.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {fig_path}")
        
        plt.show()
    
    def get_statistics(self) -> Dict:
        """
        Get detailed statistics about the run.
        
        Returns:
            Dictionary with statistics
        """
        if self.history.empty:
            return {}
        
        stats = {
            'total_generations': len(self.history),
            'initial_best_fitness': self.history.iloc[0]['best_fitness'],
            'final_best_fitness': self.history.iloc[-1]['best_fitness'],
            'total_improvement': self.history.iloc[-1]['best_fitness'] - self.history.iloc[0]['best_fitness'],
            'avg_improvement_per_generation': (self.history.iloc[-1]['best_fitness'] - self.history.iloc[0]['best_fitness']) / len(self.history),
            'max_fitness_ever': self.history['best_fitness'].max(),
            'avg_fitness_final': self.history.iloc[-1]['avg_fitness'],
            'diversity_final': self.history.iloc[-1]['diversity'],
            'diversity_initial': self.history.iloc[0]['diversity'],
        }
        
        return stats


def compare_runs(run_dirs: List[str]):
    """
    Compare multiple runs.
    
    Args:
        run_dirs: List of result directory paths
    """
    analyzers = [ResultsAnalyzer(run_dir) for run_dir in run_dirs]
    
    print("=" * 80)
    print("COMPARISON OF MULTIPLE RUNS")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, analyzer in enumerate(analyzers):
        if not analyzer.history.empty:
            label = f"Run {i+1}: {analyzer.results_path.name}"
            ax.plot(analyzer.history['generation'], 
                   analyzer.history['best_fitness'],
                   linewidth=2, label=label, marker='o', markersize=3)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title('Comparison of Best Fitness Across Runs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print("\n### Summary Comparison ###")
    print(f"{'Run':<20} {'Final Fitness':<20} {'Improvement':<20} {'Generations':<15}")
    print("-" * 80)
    
    for i, analyzer in enumerate(analyzers):
        if not analyzer.history.empty:
            final_fitness = analyzer.history.iloc[-1]['best_fitness']
            initial_fitness = analyzer.history.iloc[0]['best_fitness']
            improvement = final_fitness - initial_fitness
            generations = len(analyzer.history)
            
            print(f"{analyzer.results_path.name:<20} {final_fitness:<20.4f} {improvement:<20.4f} {generations:<15}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_directory>")
        print("Example: python analyze_results.py results/20231103_143052")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    try:
        analyzer = ResultsAnalyzer(results_dir)
        analyzer.print_summary()
        
        print("\n" + "=" * 80)
        print("Generating plots...")
        print("=" * 80)
        
        analyzer.plot_convergence(save_fig=True)
        analyzer.plot_fitness_statistics(save_fig=True)
        
        stats = analyzer.get_statistics()
        print("\n### Detailed Statistics ###")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
