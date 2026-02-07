# ViralMultiNet: ViralMultiNet: A structure-aware multimodal framework for viral protein function prediction in wastewater surveillance
# ViralMultiNet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18301000.svg)](https://doi.org/10.5281/zenodo.18301000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of "ViralMultiNet: A Structure-Aware Multimodal 
Framework for Viral Genome Classification"
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"ViralMultiNet: A Structure-Aware Multimodal Framework for Viral Genome Classification"**, addressing reviewer comments on reproducibility, code structure, and data access.

---

## 📋 Overview

ViralMultiNet is a multi-scale Transformer-based framework that integrates:

- **Multi-scale k-mer sequence encodings** (4-7-mers) for hierarchical feature extraction
- **Semantic embeddings** derived from functional annotations via BERT
- **Structure-aware attention mechanisms** constrained by 3D protein structures (PDB: 6VXX)
- **Triple knowledge distillation** (response-based + feature-based + attention-transfer) for efficiency
- **LoRA + Flash Attention** optimizations for ~40.4% training speedup

### Key Results

| Metric | Performance |
|--------|-------------|
| **Macro F1** | 0.921 ± 0.004 |
| **Accuracy** | 0.928 ± 0.003 |
| **AUC** | 0.983 ± 0.007 |
| **Training Time Reduction** | ~40.4% (94.3 → 56.2 min/epoch) |

---

## 🗂️ Repository Structure

```
ViralMultiNet-gigascience/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── environment.txt                     # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── code/                               # Source code
│   ├── train_viralmultinet/           # Main training pipeline
│   │   └── (integrated LoRA, training, and distillation code)
│   │
│   ├── viralbert/                     # Model architecture components
│   │   ├── teacher.py                 # Teacher model (Structure-Aware Attention)
│   │   ├── student.py                 # Student model (Cross-Modal + Flash Attention)
│   │   └── attention mechanisms       
│   │
│   ├── tokenizer/                     # K-mer tokenization utilities
│   │   └── tokenizer.py               # Multi-scale k-mer tokenizer
│   │
│   ├── text_only_ablation/            # Ablation study: annotation-only baseline
│   ├── text/                          # Text embedding utilities
│   ├── split_for_pretrain_and_finetune/ # Data splitting scripts
│   ├── parse_annotations.py           # Annotation parsing from UniProt/GenBank
│   ├── label/                         # Label assignment scripts
│   ├── document_processor/            # Document processing utilities
│   ├── dnabert/                       # DNA-BERT baseline implementation
│   ├── create_DataLoader/             # Data loading utilities
│   ├── corpus/                        # Corpus management
│   ├── Constructing_Knowledge_Graphs/ # Knowledge graph construction (optional)
│   ├── biobert_model/                 # BioBERT baseline implementation
│   └── annotation/                    # Annotation embedding generation
│
├── data/                               # Data files
│   ├── README.md                      # Data description (see data/README.md)
│   │
│   ├── dataset_balanced_train_head100.xlsx  # Training set (head100 samples)
│   ├── dataset_balanced_val_head100.xlsx    # Validation set
│   ├── dataset_balanced_test_head100.xlsx   # Test set
│   │
│   ├── anno_embs_test.npy             # Pre-computed annotation embeddings
│   ├── anno_embs_val.npy
│   │
│   ├── 6vxx.pdb1                      # SARS-CoV-2 Spike structure (PDB: 6VXX)
│   ├── 7bv2.pdb1                      # Alternative Spike structure (PDB: 7BV2)
│   │
│   ├── kmer2res_search.per_kmer_hits  # K-mer → residue mapping
│   ├── kmer2res_search.per_residue    # Residue-level structural features
│   │
│   └── high_attention_kmers_branch*.fasta  # High-attention regions (4-7-mer)
│
└── figures/                            # Figures from manuscript
    ├── Figure1_workflow.png
    ├── Figure2_architecture.png
    ├── Figure3_ablation.png
    └── Figure4_attention_spike.png
```

---

## 🔧 Installation

### Prerequisites

- Python >= 3.8
- CUDA >= 11.0 (for GPU acceleration)
- 16GB+ RAM recommended
- NVIDIA GPU with 12GB+ VRAM (for training)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/DAVE-HUB-11/ViralMultiNet-gigascience.git
cd ViralMultiNet-gigascience

# Create conda environment
conda create -n viralmultinet python=3.8
conda activate viralmultinet

# Install dependencies
pip install -r environment.txt
```

### Key Dependencies

- `torch >= 1.10.0` (with CUDA support)
- `transformers >= 4.20.0` (for BERT embeddings)
- `flash-attn >= 2.0.0` (for Flash Attention)
- `biopython >= 1.79` (for PDB parsing)
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`

---

## 📊 Data

### Dataset Overview

The SARS-CoV-2 ORF-level dataset consists of:

- **Total samples**: 66,011 (after augmentation)
- **Training set**: 47,091 samples
- **Validation set**: 9,417 samples
- **Test set**: 9,503 samples

**Classes**: 
- Enzymatic (25%)
- Structural (25%)
- Transport (25%)
- Other (25%)

### Data Sources

1. **Raw sequencing data**: 
   - NCBI SRA BioProject: [PRJNA1251232](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1251232)
   - Accession: SRX28474964

2. **Protein structures**:
   - [6VXX](https://www.rcsb.org/structure/6VXX): SARS-CoV-2 Spike protein (closed state)
   - [7BV2](https://www.rcsb.org/structure/7BV2): Alternative Spike conformation

3. **Functional annotations**:
   - UniProt/Swiss-Prot (release 2024_01)
   - GenBank metadata

For detailed data description, see [`data/README.md`](data/README.md).

---

## 🚀 Quick Start

### 1. Data Preparation

```bash
# Parse annotations from UniProt/GenBank
cd code
python parse_annotations.py \
    --input ../data/dataset_balanced_train_head100.xlsx \
    --output ../data/parsed_annotations.csv

# Generate annotation embeddings using BioBERT
python biobert_model/generate_embeddings.py \
    --annotations ../data/parsed_annotations.csv \
    --output ../data/anno_embs_train.npy
```

### 2. Training the Model

```bash
# Train ViralMultiNet with integrated LoRA and distillation
cd code/train_viralmultinet
python train.py \
    --train_data ../../data/dataset_balanced_train_head100.xlsx \
    --val_data ../../data/dataset_balanced_val_head100.xlsx \
    --anno_embs_train ../../data/anno_embs_train.npy \
    --anno_embs_val ../../data/anno_embs_val.npy \
    --structure_mapping ../../data/kmer2res_search.per_kmer_hits \
    --pdb_file ../../data/6vxx.pdb1 \
    --output_dir ./checkpoints \
    --epochs 50 \
    --batch_size 32
```

**Hyperparameters** (optimized via grid search):
- `--alpha 0.3`: Response-based distillation weight
- `--beta 0.5`: Feature-based distillation weight
- `--gamma 0.2`: Attention-transfer distillation weight
- `--use_lora`: Enable LoRA fine-tuning
- `--use_flash_attn`: Enable Flash Attention

### 3. Evaluation

```bash
# Evaluate on test set
python evaluate.py \
    --checkpoint ./checkpoints/best_model.pth \
    --test_data ../../data/dataset_balanced_test_head100.xlsx \
    --anno_embs_test ../../data/anno_embs_test.npy \
    --output_dir ./results
```

### 4. Ablation Studies

```bash
# Text-only baseline
cd code/text_only_ablation
python train_text_only.py --train_data ../../data/dataset_balanced_train_head100.xlsx

# DNA-BERT baseline
cd code/dnabert
python train_dnabert.py --train_data ../../data/dataset_balanced_train_head100.xlsx
```

### 5. Interpretability Analysis

```bash
# Generate attention maps for Spike protein
python interpretability.py \
    --checkpoint ./checkpoints/best_model.pth \
    --pdb_file ../../data/6vxx.pdb1 \
    --output_dir ./attention_maps
```

---

## 🧪 Reproducing Paper Results

### Full Experimental Pipeline

```bash
# Step 1: Prepare data splits (stratified, leakage-aware)
cd code/split_for_pretrain_and_finetune
python split_data.py \
    --input ../../data/full_dataset.csv \
    --output_dir ../../data/splits \
    --train_ratio 0.714 \
    --val_ratio 0.143 \
    --test_ratio 0.144 \
    --similarity_threshold 0.95

# Step 2: Build k-mer tokenizers (4-7-mer)
cd ../tokenizer
python build_tokenizers.py \
    --sequences ../../data/splits/train.csv \
    --output_dir ./

# Step 3: Generate annotation embeddings
cd ../annotation
python generate_embeddings.py \
    --annotations ../../data/splits/train.csv \
    --model biobert \
    --output ../../data/anno_embs_train.npy

# Step 4: Extract structural features from PDB
cd ../document_processor
python extract_structural_features.py \
    --pdb ../../data/6vxx.pdb1 \
    --sequences ../../data/splits/train.csv \
    --output ../../data/kmer2res_search

# Step 5: Train teacher model (Structure-Aware Attention)
cd ../viralbert
python train_teacher.py \
    --train_data ../../data/splits/train.csv \
    --val_data ../../data/splits/val.csv \
    --structure_features ../../data/kmer2res_search.per_residue \
    --output_dir ./teacher_checkpoints

# Step 6: Train student model with distillation + LoRA + Flash Attention
cd ../train_viralmultinet
python train.py \
    --teacher_checkpoint ../viralbert/teacher_checkpoints/best_teacher.pth \
    --train_data ../../data/splits/train.csv \
    --val_data ../../data/splits/val.csv \
    --use_distillation \
    --use_lora \
    --use_flash_attn \
    --alpha 0.3 --beta 0.5 --gamma 0.2 --delta 0.15 \
    --output_dir ./student_checkpoints

# Step 7: 5-fold cross-validation
python cross_validation.py \
    --data ../../data/full_dataset.csv \
    --n_folds 5 \
    --output_dir ./cv_results

# Step 8: Generate all figures and tables
python generate_paper_results.py \
    --cv_results ./cv_results \
    --figures_dir ../../figures
```

### Expected Results

After 5-fold cross-validation:

| Model | Macro F1 | Accuracy | AUC | Training Time/Epoch |
|-------|----------|----------|-----|---------------------|
| **Full Model** (Student + LoRA + Flash) | 0.921±0.004 | 0.928±0.003 | 0.983±0.007 | 56.2±2.1 min |
| Full Model (w/o LoRA) | 0.916±0.005 | 0.922±0.004 | 0.982±0.008 | 94.3±4.5 min |
| Teacher Model | 0.924±0.009 | 0.925±0.007 | 0.985±0.009 | ~120 min |
| Sequence Only | 0.872±0.008 | 0.881±0.009 | 0.983±0.009 | - |
| Annotation Only | 0.886±0.008 | 0.897±0.006 | 0.978±0.007 | - |
| DNABERT + CLIP | 0.904±0.008 | 0.908±0.011 | 0.986±0.009 | - |

---

## 📐 Model Architecture Details

### Multi-scale K-mer Encoding

The tokenizer module generates hierarchical representations:

- **4-mer**: 256 tokens → Local motifs and binding signals
- **5-mer**: 1,024 tokens → Secondary structure-related segments
- **6-mer**: 4,096 tokens → Intermediate functional fragments
- **7-mer**: 16,384 tokens → Regional context and evolutionary constraints

```python
# Example usage
from code.tokenizer import MultiScaleTokenizer

tokenizer = MultiScaleTokenizer(kmer_sizes=[4, 5, 6, 7])
tokens = tokenizer.encode("ATGGTTCGACTTAAG...")
# Returns: {4: [12, 56, 78, ...], 5: [234, 567, ...], 6: [...], 7: [...]}
```

### Attention Mechanisms

**Teacher Model** (`code/viralbert/teacher.py`):
```python
# Structure-Aware Attention
attention_scores = f(Q, S)
# Q: query from sequence
# S: structural features from PDB 6VXX (3D coords, secondary structure, rSASA)
```

**Student Model** (`code/viralbert/student.py`):
```python
# Cross-Modal Attention with Flash Attention
from flash_attn import flash_attn_func

Q = sequence_features  # From multi-scale k-mer encoder
K, V = annotation_embeddings  # From BioBERT
attention = flash_attn_func(Q, K, V)  # Accelerated computation
```

### Knowledge Distillation

Implemented in `code/train_viralmultinet/`:

```python
L_KD = α·L_resp + β·L_feat + γ·L_attn

# Components:
# L_resp: Response-based (soft targets, temperature T=3.0)
# L_feat: Feature-based (intermediate hidden states alignment)
# L_attn: Attention-transfer (structure-aware attention maps)

# Optimized hyperparameters (via grid search on validation set):
# α = 0.3, β = 0.5, γ = 0.2, δ = 0.15
```

---

## 🔬 Code Modules Explained

### Core Components

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `train_viralmultinet/` | Main training pipeline with LoRA + distillation | `train.py`, `evaluate.py` |
| `viralbert/` | Model architectures | `teacher.py`, `student.py` |
| `tokenizer/` | Multi-scale k-mer tokenization | `tokenizer.py` |
| `annotation/` | Generate BERT embeddings from text | `generate_embeddings.py` |
| `document_processor/` | Extract structural features from PDB | `extract_structural_features.py` |
| `parse_annotations.py` | Parse UniProt/GenBank annotations | Script |

### Baseline Implementations

| Module | Baseline | Description |
|--------|----------|-------------|
| `text_only_ablation/` | Annotation-only | Train with text embeddings only |
| `dnabert/` | DNA-BERT | Sequence-only baseline using DNA-BERT |
| `biobert_model/` | BioBERT | Text embedding baseline |

### Utilities

| Module | Purpose |
|--------|---------|
| `split_for_pretrain_and_finetune/` | Data splitting with leakage prevention |
| `label/` | Hierarchical label assignment (Enzymatic > Structural > Transport > Other) |
| `corpus/` | Corpus management and statistics |
| `create_DataLoader/` | PyTorch DataLoader creation |
| `Constructing_Knowledge_Graphs/` | (Optional) Knowledge graph integration |

---

## 📊 Interpretability Analysis

### Attention Mapping to 3D Structure

```bash
cd code/train_viralmultinet
python interpretability.py \
    --checkpoint ./checkpoints/best_model.pth \
    --pdb ../data/6vxx.pdb1 \
    --kmer_residue_mapping ../data/kmer2res_search.per_kmer_hits \
    --output_dir ./attention_maps
```

**Outputs**:
- `attention_spike_protein.png`: Attention overlay on 3D structure
- `attention_heatmap.csv`: Residue-level importance scores
- `high_attention_regions.fasta`: Top 5% high-attention k-mers

**Key validated regions** (from paper Figure 4):
- **RBD (319-541)**: 68% of high-attention k-mers
- **S1/S2 cleavage (681-685)**: 15% of high-attention k-mers
- **Fusion peptide (816-835)**: 12% of high-attention k-mers

---

## 📄 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{li2025viralmultinet,
  title={ViralMultiNet: A Structure-Aware Multimodal Framework for Viral Genome Classification},
  author={Li, GuoDong and Lai, TingLian and Liu, FuGuo and Xu, WenXia},
  journal={GigaScience},
  year={2025},
  note={In press}
}
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to:

- Report bugs via [GitHub Issues](https://github.com/DAVE-HUB-11/ViralMultiNet-gigascience/issues)
- Submit feature requests
- Create pull requests for improvements

---

## 📧 Contact

- **Corresponding Author**: WenXia Xu
  - Email: xwx@guet.edu.cn
  - Affiliation: School of Mathematics and Computational Science, Guilin University of Electronic Technology

- **GitHub Issues**: For technical questions or bug reports, please open an issue.

---

## 🙏 Acknowledgments

This work was supported by [funding information].

### Data Sources

- **NCBI Sequence Read Archive**: Raw metagenomic sequencing data (SRX28474964)
- **UniProt/Swiss-Prot**: Functional annotations (release 2024_01)
- **Protein Data Bank**: 3D structures (6VXX, 7BV2)

### Tools & Libraries

- PyTorch for deep learning framework
- Hugging Face Transformers for BERT embeddings
- Flash Attention for efficient attention computation
- BioPython for biological data processing

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Known Issues & Limitations

1. **Current dataset scope**: Primarily focused on SARS-CoV-2 ORFs
   - Future work will extend to broader viral families

2. **Annotation dependency**: Labels derived from UniProt annotations
   - Not direct experimental ground truth
   - See Discussion section in paper for mitigation strategies

3. **Computational requirements**: 
   - Training requires GPU with 12GB+ VRAM
   - Consider using gradient checkpointing for limited memory

---

## 🔮 Future Directions

- Extend to broader viral families (influenza, HIV, etc.)
- Incorporate richer multimodal metadata (host information, geographic data)
- Integrate with real-time wastewater surveillance pipelines
- Fine-grained functional ontologies beyond 4-class classification
- Orthogonal biological validation (conservation signals, mutation impact)

---

**Last Updated**: January 19, 2025
