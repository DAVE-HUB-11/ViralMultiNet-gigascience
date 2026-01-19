from Bio import SeqIO
import os

def extract_sequences(fasta_file, output_file):
    with open(fasta_file, 'r') as infile, open(output_file, 'w') as outfile:
        for record in SeqIO.parse(infile, 'fasta'):
            sequence = str(record.seq).upper()
            outfile.write(sequence + '\n')

# 示例用法
extract_sequences('GCF_000002775.4_Pop_tri_v3.fna', 'Pop_tri_v3.txt')
