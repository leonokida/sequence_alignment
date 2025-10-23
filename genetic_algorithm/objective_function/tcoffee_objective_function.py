from collections import defaultdict
from typing import List
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

class TCoffeeObjectiveFunction:
    def __init__(self, initial_sequences: List[SeqRecord]):
        self.sequence_map = {seq.id: str(seq.seq) for seq in initial_sequences}
        self.sequence_ids = [seq.id for seq in initial_sequences]
        
        # Computes weights
        self.weights = self._calculate_weights(initial_sequences)
        
        # Build pairwise residue library
        self.pairwise_weights: dict[tuple[str, str], float] = defaultdict(float)
        self.build_pairwise_library(initial_sequences)
    
    def _calculate_weights(self, initial_sequences: List[SeqRecord]) -> dict[str, float]:
        """
        Calculates sequence weights (Clustal W-like) using Neighbor-Joining on pairwise sequence identity.
        """
        
        # Pad sequences to same length for MSA
        max_len = max(len(seq.seq) for seq in initial_sequences)
        padded_sequences = []
        for seq in initial_sequences:
            padded_seq = SeqRecord(
                seq.seq + '-' * (max_len - len(seq.seq)),
                id=seq.id,
                name=seq.name,
                description=seq.description
            )
            padded_sequences.append(padded_seq)
        
        # Convert to MultipleSeqAlignment
        msa = MultipleSeqAlignment(padded_sequences)
        
        # 1. Calculate Pairwise Distances based on literal identity (1 - % identity)
        calculator = DistanceCalculator('identity') 
        calculator.get_distance(msa)

        # 2. Construct the Guide Tree using Neighbor-Joining
        constructor = DistanceTreeConstructor(calculator, 'nj')
        guide_tree = constructor.build_tree(msa)

        weights: dict[str, float] = {}
        
        # 3. Derive Weights from Branch Lengths
        for terminal in guide_tree.get_terminals():
            # Sum the branch lengths from the root to the leaf (terminal)
            total_branch_length = guide_tree.distance(terminal)
            weights[terminal.name] = total_branch_length

        # 4. Normalize the Weights
        total_weight_sum = sum(weights.values())
        
        if total_weight_sum == 0:
            # Handle case where all sequences are identical (distance 0)
            num_seq = len(initial_sequences)
            return {seq.id: 1.0 / num_seq for seq in initial_sequences}

        normalized_weights = {
            seq_id: weight / total_weight_sum for seq_id, weight in weights.items()
        }
        
        return normalized_weights
    
    def build_pairwise_library(self, sequences: List[SeqRecord]):
        """
        Builds the data-driven consistency matrix from all pairwise alignments.
        Only considers residues aligned at the same positions.
        """
        # Convert sequences to strings
        seq_strs = [str(seq.seq).upper() for seq in sequences]
        num_seq = len(seq_strs)
        
        for i in range(num_seq):
            seq_i = seq_strs[i]
            for j in range(i + 1, num_seq):
                seq_j = seq_strs[j]
                if len(seq_i) != len(seq_j):
                    raise ValueError("All sequences must be pre-aligned (same length)")
                
                for k in range(len(seq_i)):
                    res_i, res_j = seq_i[k], seq_j[k]
                    if res_i != '-' and res_j != '-':
                        # Add weight 1 for each aligned residue pair
                        self.pairwise_weights[(res_i, res_j)] += 1.0
                        self.pairwise_weights[(res_j, res_i)] += 1.0  # symmetric
    
    def _calculate_pair_consistency_score(self, seq_i: str, seq_j: str) -> float:
        """
        Computes the consistency score for a single pair of sequences.
        """
        score = 0.0
        for k in range(len(seq_i)):
            res_i, res_j = seq_i[k].upper(), seq_j[k].upper()
            if res_i != '-' and res_j != '-':
                score += self.pairwise_weights.get((res_i, res_j), 0.0)
        return score
    
    def compute_fitness(self, aligned_sequences: List[str]) -> float:
        """
        Computes the total fitness score (weighted sum-of-pairs) for a candidate MSA.
        """
        total_score = 0.0
        num_seq = len(aligned_sequences)
        
        for i in range(num_seq):
            for j in range(i + 1, num_seq):
                seq_i = aligned_sequences[i]
                seq_j = aligned_sequences[j]
                
                # Sequence weights
                id_i, id_j = self.sequence_ids[i], self.sequence_ids[j]
                weight_i = self.weights.get(id_i, 1.0)
                weight_j = self.weights.get(id_j, 1.0)
                pair_weight = weight_i * weight_j
                
                # Pairwise consistency score
                pair_score = self._calculate_pair_consistency_score(seq_i, seq_j)
                
                # Weighted sum
                total_score += pair_score * pair_weight
        
        return total_score
