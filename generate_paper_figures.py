import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def generate_convergence_plot():
    # Load consolidated results
    with open('experiments_full/20251115_095647/consolidated_results.json', 'r') as f:
        data = json.load(f)
    
    # Filter for obj_saga and obj_wsp
    saga_runs = [run for run in data if run['experiment_name'] == 'obj_saga']
    wsp_runs = [run for run in data if run['experiment_name'] == 'obj_wsp']
    
    # We need generation history. Since it's not in the consolidated json, 
    # we'll read the CSVs for the first run of each to show a representative curve
    # or average them if possible. Let's try to average the first 5 runs.
    
    def get_avg_history(runs, exp_name):
        all_histories = []
        min_len = 1000
        
        for run in runs[:5]: # Take first 5 runs
            run_id = run['run_id']
            run_num = run['run']
            # Path construction: experiments_full/TIMESTAMP/EXP_NAME/run_X/RUN_ID/generation_history.csv
            # Note: The timestamp in path is the main one, but inside run_X there is another timestamp folder
            
            base_path = f"experiments_full/20251115_095647/{exp_name}/run_{run_num}/{run_id}/generation_history.csv"
            
            if os.path.exists(base_path):
                df = pd.read_csv(base_path)
                # Use 'best_fitness_overall' or 'best_fitness_generation'
                # Based on the CSV content, 'best_fitness_overall' is monotonic, 'best_fitness_generation' is per gen.
                # Usually convergence plots show the best fitness found so far.
                all_histories.append(df['best_fitness_overall'].values)
                if len(df) < min_len:
                    min_len = len(df)
        
        if not all_histories:
            return None
            
        # Trim to min length and average
        trimmed = [h[:min_len] for h in all_histories]
        avg_history = np.mean(trimmed, axis=0)
        return avg_history

    saga_history = get_avg_history(saga_runs, 'obj_saga')
    wsp_history = get_avg_history(wsp_runs, 'obj_wsp')
    
    plt.figure(figsize=(10, 6))
    
    if saga_history is not None:
        plt.plot(saga_history, label='SAGA Fitness', color='blue', linewidth=2)
        
    # WSP values are much larger, so we might need dual axis or normalized values
    # Let's check the scale. SAGA is around -200 to 100. WSP is around 300,000.
    # Dual axis is better.
    
    ax1 = plt.gca()
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('SAGA Fitness Score', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    if wsp_history is not None:
        ax2 = ax1.twinx()
        ax2.plot(wsp_history, label='WSP Fitness', color='green', linewidth=2, linestyle='--')
        ax2.set_ylabel('WSP Fitness Score', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
    
    plt.title('Convergence Comparison: SAGA vs WSP Fitness Functions')
    plt.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if wsp_history is not None else ([], [])
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('artigo/figures/convergence_comparison.png', dpi=300)
    print("Generated convergence_comparison.png")

def generate_simple_score_plot():
    # Data from the previous analysis
    data = {
        'SAGA (Baseline)': -49448.50,
        'WSP (Proposed)': -50717.00,
        'SAGA Block Mut.': -36375.40,
        'Standard Mut.': -50312.10
    }
    
    experiments = list(data.keys())
    scores = list(data.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(experiments, scores, color=['blue', 'green', 'red', 'gray'])
    
    plt.title('Alignment Quality Comparison (Simple Score: +1 Match, -1 Mismatch)')
    plt.ylabel('Simple Score (Higher is Better)')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig('artigo/figures/simple_score_comparison.png', dpi=300)
    print("Generated simple_score_comparison.png")

if __name__ == "__main__":
    generate_convergence_plot()
    generate_simple_score_plot()
