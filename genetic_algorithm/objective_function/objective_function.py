from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Align import MultipleSeqAlignment
from typing import List
from Bio.SeqRecord import SeqRecord # Usado para tipagem e inicialização
from genetic_algorithm.objective_function.pam250 import PAM_250

class ObjectiveFunction:
    def __init__(self, initial_sequences: List[SeqRecord]):
        self.cost_matrix = PAM_250
        
        # Gap penalties
        self.GAP_OPEN_PENALTY = -12
        self.GAP_EXTEND_PENALTY = -2
        
        # Computes weights
        self.weights = self._calculate_weights(initial_sequences)
        
        # Mapping of sequences and IDs
        self.sequence_map = {seq.id: str(seq.seq) for seq in initial_sequences}
        self.sequence_ids = [seq.id for seq in initial_sequences]
        
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
    
    def compute_fitness(self, 
            aligned_sequences: List[str], 
            sequence_ids: List[str], 
            alignment_length: int) -> float:
        """
        Calculates the fitness score using the Weighted Sum-of-Pairs (WSP) method with affine gap penalties.
        """
        num_sequences = len(aligned_sequences)
        total_score: float = 0.0

        # Iterate over all unique pairs of sequences (i, j)
        for i in range(num_sequences):
            for j in range(i + 1, num_sequences):
                
                seq_i = aligned_sequences[i]
                seq_j = aligned_sequences[j]
                
                id_i = sequence_ids[i]
                id_j = sequence_ids[j]

                # 1. Retrieve and Calculate Pair Weight (Product is common)
                weight_i = self.weights.get(id_i, 1.0)
                weight_j = self.weights.get(id_j, 1.0)
                pair_weight = weight_i * weight_j
                
                # 2. Calculate the Pair Score with Affine Penalties
                pair_score = self._calculate_pair_score_affine(seq_i, seq_j)
                
                # 3. Apply Weight and Accumulate Total Score
                total_score += pair_score * pair_weight

        # The overall score is the fitness (to be maximized)
        return total_score

    def _calculate_pair_score_affine(self, seq_i: str, seq_j: str) -> float:
        """
        Calculates the score for a single pair of aligned sequences using
        the substitution matrix and affine gap penalties.
        """
        
        score: float = 0
        in_gap_i: bool = False  # Tracks if the previous residue in seq_i was a gap
        in_gap_j: bool = False  # Tracks if the previous residue in seq_j was a gap
        
        for k in range(len(seq_i)):
            res_i = seq_i[k].upper()
            res_j = seq_j[k].upper()
            
            is_gap_i = (res_i == '-')
            is_gap_j = (res_j == '-')

            # --- Affine Gap Penalty Logic ---
            if is_gap_i or is_gap_j:
                
                # 1. Gap in seq_i aligned with a residue in seq_j
                if is_gap_i and not is_gap_j:
                    if not in_gap_i: # Gap opening
                        score += self.GAP_OPEN_PENALTY
                    else: # Gap extension
                        score += self.GAP_EXTEND_PENALTY

                # 2. Gap in seq_j aligned with a residue in seq_i
                if is_gap_j and not is_gap_i:
                    if not in_gap_j: # Gap opening
                        score += self.GAP_OPEN_PENALTY
                    else: # Gap extension
                        score += self.GAP_EXTEND_PENALTY
                
                # Note: Gap-Gap alignment ('-' vs '-') scores 0, so no explicit penalty is needed.

            # --- Substitution Logic (Match/Mismatch) ---
            else: # Neither is a gap
                try:
                    score += self.cost_matrix[res_i][res_j]
                except KeyError:
                    # Handle unknown residues (e.g., 'X', 'Z', or case issues)
                    score += -10
            
            # Update gap tracking status for the next column
            in_gap_i = is_gap_i
            in_gap_j = is_gap_j

        return score