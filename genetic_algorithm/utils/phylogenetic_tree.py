import random
from typing import List, Tuple
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Phylo.BaseTree import Tree
from Bio.Align import MultipleSeqAlignment


class PhylogeneticTreeHelper:
    """Helper class for phylogenetic tree operations in SAGA."""
    
    def __init__(self):
        self.tree = None
        self.sequence_ids = []
    
    def build_tree_from_alignment(self, aligned_segments: List) -> Tree:
        """
        Build a phylogenetic tree from aligned sequences using Neighbor-Joining.
        
        Args:
            aligned_segments: List of AlignedSegment objects
            
        Returns:
            Bio.Phylo Tree object
        """
        # Convert aligned segments to SeqRecord objects for tree construction
        seq_records = []
        for segment in aligned_segments:
            seq_record = SeqRecord(
                Seq(segment.sequence),
                id=segment.id,
                name=segment.id,
                description=""
            )
            seq_records.append(seq_record)
        
        self.sequence_ids = [seg.id for seg in aligned_segments]
        
        # Convert to MultipleSeqAlignment for distance calculation
        msa = MultipleSeqAlignment(seq_records)
        
        # Calculate pairwise distances
        calculator = DistanceCalculator('identity')
        calculator.get_distance(msa)
        
        # Build tree using Neighbor-Joining
        constructor = DistanceTreeConstructor(calculator, 'nj')
        self.tree = constructor.build_tree(msa)
        
        return self.tree
    
    def split_sequences_by_tree(self, aligned_segments: List, 
                                 split_method: str = 'random') -> Tuple[List[int], List[int]]:
        """
        Split sequences into two groups (G1 and G2) based on phylogenetic tree structure.
        
        Args:
            aligned_segments: List of AlignedSegment objects
            split_method: Method to split ('random', 'balanced', 'subtree')
            
        Returns:
            Tuple of (G1_indices, G2_indices) where indices refer to positions in aligned_segments
        """
        if self.tree is None:
            self.build_tree_from_alignment(aligned_segments)
        
        num_sequences = len(aligned_segments)
        
        if split_method == 'random':
            return self._random_split(num_sequences)
        elif split_method == 'balanced':
            return self._balanced_split(num_sequences)
        elif split_method == 'subtree':
            return self._subtree_split(aligned_segments)
        else:
            raise ValueError(f"Unknown split method: {split_method}")
    
    def _random_split(self, num_sequences: int) -> Tuple[List[int], List[int]]:
        """
        Randomly split sequences into two groups.
        
        Args:
            num_sequences: Total number of sequences
            
        Returns:
            Tuple of (G1_indices, G2_indices)
        """
        indices = list(range(num_sequences))
        random.shuffle(indices)
        
        # Split roughly in half
        split_point = num_sequences // 2
        if split_point == 0:
            split_point = 1
        
        G1 = indices[:split_point]
        G2 = indices[split_point:]
        
        return G1, G2
    
    def _balanced_split(self, num_sequences: int) -> Tuple[List[int], List[int]]:
        """
        Split sequences into two balanced groups based on tree distance.
        
        Args:
            num_sequences: Total number of sequences
            
        Returns:
            Tuple of (G1_indices, G2_indices)
        """
        # For balanced split, use random but ensure groups are roughly equal
        indices = list(range(num_sequences))
        random.shuffle(indices)
        
        mid = num_sequences // 2
        G1 = indices[:mid]
        G2 = indices[mid:]
        
        return G1, G2
    
    def _subtree_split(self, aligned_segments: List) -> Tuple[List[int], List[int]]:
        """
        Split sequences based on phylogenetic tree structure (subtree selection).
        This is the most biologically meaningful split method.
        
        Args:
            aligned_segments: List of AlignedSegment objects
            
        Returns:
            Tuple of (G1_indices, G2_indices)
        """
        if self.tree is None or not hasattr(self.tree, 'clade'):
            # Fallback to random split if tree is not available
            return self._random_split(len(aligned_segments))
        
        # Get all internal nodes (clades)
        internal_clades = [c for c in self.tree.find_clades() if not c.is_terminal()]
        
        if not internal_clades:
            # No internal nodes, use random split
            return self._random_split(len(aligned_segments))
        
        # Select a random internal clade as split point
        split_clade = random.choice(internal_clades)
        
        # Get terminal nodes (leaves) under this clade
        subtree_terminals = split_clade.get_terminals()
        subtree_names = {t.name for t in subtree_terminals}
        
        # Create mapping from sequence ID to index
        id_to_index = {seg.id: idx for idx, seg in enumerate(aligned_segments)}
        
        # Split into two groups
        G1 = [id_to_index[name] for name in subtree_names if name in id_to_index]
        G2 = [idx for idx in range(len(aligned_segments)) if idx not in G1]
        
        # Ensure both groups are non-empty
        if len(G1) == 0 or len(G2) == 0:
            return self._random_split(len(aligned_segments))
        
        return G1, G2
    
    def get_tree_distance(self, seq_id1: str, seq_id2: str) -> float:
        """
        Get phylogenetic distance between two sequences in the tree.
        
        Args:
            seq_id1: ID of first sequence
            seq_id2: ID of second sequence
            
        Returns:
            Distance between sequences in the tree
        """
        if self.tree is None:
            return 0.0
        
        try:
            terminal1 = next(self.tree.find_clades(seq_id1))
            terminal2 = next(self.tree.find_clades(seq_id2))
            return self.tree.distance(terminal1, terminal2)
        except Exception:
            return 0.0
    
    def get_most_distant_sequences(self, aligned_segments: List) -> Tuple[int, int]:
        """
        Find the two most phylogenetically distant sequences.
        
        Args:
            aligned_segments: List of AlignedSegment objects
            
        Returns:
            Tuple of (index1, index2) of most distant sequences
        """
        if self.tree is None:
            self.build_tree_from_alignment(aligned_segments)
        
        max_distance = -1
        max_pair = (0, 1)
        
        for i in range(len(aligned_segments)):
            for j in range(i + 1, len(aligned_segments)):
                distance = self.get_tree_distance(
                    aligned_segments[i].id,
                    aligned_segments[j].id
                )
                if distance > max_distance:
                    max_distance = distance
                    max_pair = (i, j)
        
        return max_pair


def create_phylogenetic_groups(aligned_segments: List, 
                               method: str = 'subtree') -> Tuple[List[int], List[int]]:
    """
    Convenience function to create two phylogenetic groups from aligned sequences.
    
    Args:
        aligned_segments: List of AlignedSegment objects
        method: Split method ('random', 'balanced', 'subtree')
        
    Returns:
        Tuple of (G1_indices, G2_indices)
    """
    helper = PhylogeneticTreeHelper()
    return helper.split_sequences_by_tree(aligned_segments, split_method=method)
