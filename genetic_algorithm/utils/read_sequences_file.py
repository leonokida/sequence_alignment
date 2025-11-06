from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from typing import List

def get_sequences_from_fasta(path: str) -> List[SeqRecord]:
    """
    Reads sequences from FASTA file
    
    Args:
        path: Path to the FASTA file
        
    Returns:
        List of SeqRecord objects with sequences
    """
    
    try:
        records = list(SeqIO.parse(path, "fasta", alphabet=None)) 
        
        for record in records:
            # Remove gaps and convert to uppercase
            seq_str = str(record.seq).upper().replace('-', '')
            record.seq = record.seq.__class__(seq_str)
            
        return records

    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado no caminho: {path}")
        return []


# Alias for compatibility
def read_fasta_file(path: str) -> List[SeqRecord]:
    """
    Reads sequences from FASTA file (alias for get_sequences_from_fasta)
    
    Args:
        path: Path to the FASTA file
        
    Returns:
        List of SeqRecord objects with sequences
    """
    return get_sequences_from_fasta(path)