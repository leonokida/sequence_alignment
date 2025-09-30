from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def get_sequences_from_fasta(path: str) -> list[SeqRecord]:
    """
    Reads sequences from FASTA file
    """
    
    try:
        records = list(SeqIO.parse(path, "fasta", alphabet=None)) 
        
        for record in records:
            record.seq = record.seq.upper().ungap('-') 
            
        return records

    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado no caminho: {path}")
        return []