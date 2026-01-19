# Data Directory

This directory contains the processed datasets, structural information, and pre-computed embeddings used in ViralMultiNet.

---

## 📊 Dataset Files

### Main Datasets (Head100 Samples)

These files contain the first 100 samples from each split for quick testing and demonstration:

| File | Samples | Description |
|------|---------|-------------|
| `dataset_balanced_train_head100.xlsx` | 100 | Training set samples with ORF sequences and labels |
| `dataset_balanced_val_head100.xlsx` | 100 | Validation set samples |
| `dataset_balanced_test_head100.xlsx` | 100 | Test set samples |

**Columns**:
- `orf_id`: Unique identifier for each ORF
- `sequence`: Nucleotide sequence (300-900 bp)
- `label`: Functional category (Enzymatic/Structural/Transport/Other)
- `annotation`: Functional description from UniProt/GenBank
- `source`: Lineage information (Alpha/Beta/Delta/Omicron/Reference)

### Complete Dataset Access

The full dataset (66,011 samples) is available from:
- **Raw data**: NCBI SRA BioProject [PRJNA1251232](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1251232)
- **Processed data**: [Contact authors for full dataset]

**Class distribution** (full dataset after balancing):
- Enzymatic: 16,503 samples (~25%)
- Structural: 16,502 samples (~25%)
- Transport: 16,503 samples (~25%)
- Other: 16,503 samples (~25%)

---

## 🧬 Annotation Embeddings

Pre-computed BERT embeddings (768-dimensional) for functional annotations:

| File | Shape | Description |
|------|-------|-------------|
| `anno_embs_test.npy` | (100, 768) | Test set annotation embeddings |
| `anno_embs_val.npy` | (100, 768) | Validation set annotation embeddings |

**Generation**:
```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Embed annotation text
inputs = tokenizer(annotation_text, return_tensors='pt', 
                   padding=True, truncation=True, max_length=512)
outputs = model(**inputs)
embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token (768-dim)
```

---

## 🧪 3D Protein Structures

### PDB Files

| File | PDB ID | Description | Resolution |
|------|--------|-------------|------------|
| `6vxx.pdb1` | 6VXX | SARS-CoV-2 Spike protein (closed state) | 2.80 Å |
| `7bv2.pdb1` | 7BV2 | SARS-CoV-2 Spike protein (alternative conformation) | 3.20 Å |

**Source**: [RCSB Protein Data Bank](https://www.rcsb.org/)

**Key regions in 6VXX**:
- **Chain A**: Used for structure-aware attention
- **RBD (Receptor-Binding Domain)**: Residues 319-541
- **S1/S2 Cleavage Site**: Residues 681-685
- **Fusion Peptide**: Residues 816-835

### Structural Feature Extraction

**Extracted features** (for each residue):
1. **3D Coordinates**: Cα atom positions (x, y, z) in Ångströms
2. **Secondary Structure**: DSSP annotations (α-helix, β-sheet, coil)
3. **Solvent Accessibility**: Relative solvent-accessible surface area (rSASA)
4. **Functional Domains**: Binary indicators for key regions

**Usage**:
```python
from Bio.PDB import PDBParser, DSSP

parser = PDBParser(QUIET=True)
structure = parser.get_structure('6vxx', 'data/6vxx.pdb1')
model = structure[0]
chain = model['A']

# Extract Cα coordinates
for residue in chain:
    if 'CA' in residue:
        ca_atom = residue['CA']
        coords = ca_atom.get_coord()  # (x, y, z)
```

---

## 🗺️ K-mer to Residue Mapping

### Structural Alignment Tables

| File | Format | Description |
|------|--------|-------------|
| `kmer2res_search.per_kmer_hits` | TSV | K-mer → residue position mapping |
| `kmer2res_search.per_residue` | TSV | Residue-level structural annotations |

#### `kmer2res_search.per_kmer_hits`

Maps k-mer sequences to their corresponding residue positions in the Spike protein:

**Columns**:
- `kmer`: K-mer sequence (4-7 nucleotides)
- `kmer_length`: Length of k-mer (4/5/6/7)
- `residue_start`: Starting residue position in Spike protein
- `residue_end`: Ending residue position
- `alignment_score`: BLAST alignment score
- `identity`: Sequence identity percentage

**Example**:
```
kmer          kmer_length  residue_start  residue_end  alignment_score  identity
ATGGTT        6           319            320          28.5             100.0
TGGTTC        6           320            321          28.5             100.0
```

#### `kmer2res_search.per_residue`

Residue-level structural features for structure-aware attention:

**Columns**:
- `residue_id`: Residue position (1-1273 for Spike)
- `residue_type`: Amino acid (single-letter code)
- `secondary_structure`: H (helix), E (sheet), C (coil)
- `rsa`: Relative solvent accessibility (0-1)
- `ca_x`, `ca_y`, `ca_z`: Cα atom 3D coordinates (Å)
- `functional_domain`: RBD/S1S2/FP/None

**Example**:
```
residue_id  residue_type  secondary_structure  rsa   ca_x    ca_y    ca_z   functional_domain
319         N             C                    0.42  45.123  67.891  23.456  RBD
320         L             H                    0.18  46.789  68.234  24.123  RBD
```

---

## 🎯 High-Attention K-mer Regions

Post-training analysis results showing k-mers that received high attention weights:

| File | K-mer Size | Sequences | Description |
|------|-----------|-----------|-------------|
| `high_attention_kmers_branch0_clean.fasta` | 4-mer | ~500 | High-attention 4-mers |
| `high_attention_kmers_branch1_clean.fasta` | 5-mer | ~450 | High-attention 5-mers |
| `high_attention_kmers_branch2_clean.fasta` | 6-mer | ~420 | High-attention 6-mers |
| `high_attention_kmers_branch3_clean.fasta` | 7-mer | ~380 | High-attention 7-mers |

**Selection criteria**: Top 5% k-mers by aggregated attention score across all test samples.

**Format** (FASTA):
```
>4mer_001|attention_score=0.945|residue_range=319-320|domain=RBD
ATGG
>4mer_002|attention_score=0.932|residue_range=681-682|domain=S1S2_cleavage
TGAC
```

**Biological validation**:
- High-attention k-mers consistently align with known functional regions:
  - **RBD enrichment**: 68% of top k-mers map to residues 319-541
  - **S1/S2 cleavage enrichment**: 15% map to residues 681-685
  - **Fusion peptide enrichment**: 12% map to residues 816-835

---

## 🔄 K-mer Tokenizers

Pre-built tokenizers for each k-mer scale:

```
../code/tokenizer_4mer/   # Vocabulary: 256 tokens (4^4)
../code/tokenizer_5mer/   # Vocabulary: 1,024 tokens (4^5)
../code/tokenizer_6mer/   # Vocabulary: 4,096 tokens (4^6)
../code/tokenizer_7mer/   # Vocabulary: 16,384 tokens (4^7)
```

**Usage**:
```python
from transformers import PreTrainedTokenizerFast

tokenizer_4mer = PreTrainedTokenizerFast.from_pretrained('../code/tokenizer_4mer')
tokens = tokenizer_4mer.encode("ATGGTTCGACTT")
# tokens = [12, 56, 78, ...]  # 4-mer token IDs
```

---

## 📝 Data Processing Pipeline

To reproduce the data processing from raw NCBI sequences:

### Step 1: Download Raw Data
```bash
# Install SRA Toolkit
conda install -c bioconda sra-tools

# Download from NCBI
prefetch SRX28474964
fasterq-dump SRX28474964 --split-files
```

### Step 2: Quality Control & Assembly
```bash
# Quality control
fastqc SRX28474964_1.fastq SRX28474964_2.fastq
trimmomatic PE SRX28474964_1.fastq SRX28474964_2.fastq \
    SRX28474964_1_clean.fastq SRX28474964_1_unpaired.fastq \
    SRX28474964_2_clean.fastq SRX28474964_2_unpaired.fastq \
    ILLUMINACLIP:TruSeq3-PE.fa:2:30:10 MINLEN:50

# Hybrid assembly
bwa mem -t 8 NC_045512.2.fasta \
    SRX28474964_1_clean.fastq SRX28474964_2_clean.fastq | \
    samtools view -bS - | samtools sort -o aligned.bam
bcftools mpileup -Ou -f NC_045512.2.fasta aligned.bam | \
    bcftools call -mv -Oz -o variants.vcf.gz
bcftools consensus -f NC_045512.2.fasta variants.vcf.gz > consensus.fasta

# De novo assembly (for novel variants)
megahit -1 SRX28474964_1_clean.fastq -2 SRX28474964_2_clean.fastq \
    -o megahit_output --k-list 21,41,61,81,99
```

### Step 3: ORF Prediction & Validation
```bash
# Predict ORFs
prodigal -i consensus.fasta -o orfs.gff -a orfs.faa -p meta

# Validate against reference
blastn -query orfs.fasta -db NC_045512.2.fasta \
    -evalue 1e-10 -outfmt 6 -out orfs_blast.txt

# Filter by identity ≥85%, coverage ≥80%
awk '$3 >= 85 && $4/$13*100 >= 80' orfs_blast.txt > validated_orfs.txt
```

### Step 4: Dereplication
```bash
# Cluster at 95% identity
vsearch --cluster_fast orfs.fasta --id 0.95 \
    --centroids orfs_clustered_95.fasta \
    --uc clusters.uc

# Select representative sequences
cd-hit-est -i orfs_clustered_95.fasta -o orfs_derep.fasta \
    -c 0.95 -n 10 -M 16000 -T 8
```

### Step 5: Functional Annotation
```bash
# Align to UniProt/Swiss-Prot
diamond blastx --query orfs_derep.fasta \
    --db uniprot_sprot.dmnd --evalue 1e-5 \
    --outfmt 6 qseqid sseqid pident qcovs evalue stitle \
    --out orfs_uniprot.txt --sensitive

# Extract annotations and assign labels
python ../code/train_viralmultinet/data_processing/assign_labels.py \
    --blast orfs_uniprot.txt \
    --output dataset_labeled.csv
```

### Step 6: Generate Embeddings & Structural Features
```bash
# Generate BERT embeddings for annotations
python ../code/train_viralmultinet/data_processing/annotation_embedding.py \
    --input dataset_labeled.csv \
    --output anno_embs.npy

# Extract structural features from PDB
python ../code/train_viralmultinet/data_processing/structure_alignment.py \
    --pdb 6vxx.pdb1 \
    --sequences dataset_labeled.csv \
    --output kmer2res_search
```

---

## 📊 Data Statistics

### Dataset Composition

**By viral lineage** (full dataset):
- Reference (Wuhan-Hu-1): 42,075 ORFs (63.8%)
- Alpha (B.1.1.7): 6,601 ORFs (10.0%)
- Beta (B.1.351): 5,281 ORFs (8.0%)
- Delta (B.1.617.2): 6,601 ORFs (10.0%)
- Omicron (B.1.1.529): 5,453 ORFs (8.3%)

**ORF length distribution**:
- Mean: 587 ± 142 bp
- Median: 561 bp
- Range: 300-900 bp

**Annotation quality**:
- UniProt/Swiss-Prot matches: 62,689/66,011 (95.0%)
- High-confidence (E-value < 1e-10): 58,234 (88.2%)
- Expert-reviewed labels: 500 ORFs (κ = 0.92)

---

## ⚠️ Important Notes

1. **Head100 files**: Provided samples are for quick testing only. For full experiments, use complete dataset.

2. **Structural features**: Only ORFs with ≥70% identity to Spike protein have structure-based features. Others use learned embeddings.

3. **Data licensing**: 
   - PDB structures: Public domain
   - NCBI sequences: Public domain
   - UniProt annotations: CC BY 4.0 license

4. **File formats**:
   - `.xlsx`: Excel format for easy inspection
   - `.npy`: NumPy binary format for efficient loading
   - `.pdb1`: PDB format with chain A highlighted

---

## 📧 Contact

For questions about the dataset or to request the full dataset:
- **Email**: xwx@guet.edu.cn
- **GitHub Issues**: [Report data issues](https://github.com/DAVE-HUB-11/ViralMultiNet-gigascience/issues)

---

**Last Updated**: January 19, 2025
