import os
import glob
import json
import numpy as np
from Bio import SeqIO

def calculate_simple_score_from_strings(seqs):
    """
    Calculates Sum of Pairs score with:
    Match: +1
    Mismatch: -1
    Gap-Gap: 0
    """
    score = 0
    num_seqs = len(seqs)
    
    for i in range(num_seqs):
        seq1 = seqs[i]
        for j in range(i + 1, num_seqs):
            seq2 = seqs[j]
            
            # We can vectorize this comparison
            s1_arr = np.array(list(seq1))
            s2_arr = np.array(list(seq2))
            
            # Matches (excluding gap-gap)
            matches = (s1_arr == s2_arr) & (s1_arr != '-') & (s2_arr != '-')
            
            # Gap-Gap (ignored usually, or 0)
            gap_gap = (s1_arr == '-') & (s2_arr == '-')
            
            # Mismatches (including gap-residue)
            # Everything else is a mismatch
            mismatches = ~matches & ~gap_gap
            
            score += np.sum(matches) * 1
            score += np.sum(mismatches) * -1
            
    return score

def main():
    base_dir = "experiments_full"
    results = {}

    # Find all best_alignment.fasta files
    search_pattern = os.path.join(base_dir, "**", "best_alignment.fasta")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} alignment files.")
    
    for file_path in files:
        # Extract experiment info from path
        parts = file_path.split(os.sep)
        
        try:
            # Let's look for 'run_' in path to anchor
            run_idx = -1
            for k, part in enumerate(parts):
                if part.startswith('run_'):
                    run_idx = k
                    break
            
            if run_idx != -1:
                exp_type = parts[run_idx - 1]
            else:
                continue

            # Read alignment using SeqIO to handle potential length discrepancies
            records = list(SeqIO.parse(file_path, "fasta"))
            
            if not records:
                continue
                
            # Normalize lengths (pad with gaps if necessary)
            max_len = max(len(r.seq) for r in records)
            
            # Create a list of sequence strings, padded
            seqs = []
            for r in records:
                s = str(r.seq)
                if len(s) < max_len:
                    s += "-" * (max_len - len(s))
                seqs.append(s)
            
            score = calculate_simple_score_from_strings(seqs)
            
            if exp_type not in results:
                results[exp_type] = []
            
            results[exp_type].append(score)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Print Summary
    print("\n--- Simple Score Evaluation (+1/-1) ---\n")
    print(f"{'Experiment':<25} | {'Mean Score':<15} | {'Std Dev':<15} | {'Min':<10} | {'Max':<10} | {'Count':<5}")
    print("-" * 90)
    
    for exp_type, scores in sorted(results.items()):
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        count = len(scores)
        
        print(f"{exp_type:<25} | {mean_score:<15.2f} | {std_score:<15.2f} | {min_score:<10} | {max_score:<10} | {count:<5}")

if __name__ == "__main__":
    main()

