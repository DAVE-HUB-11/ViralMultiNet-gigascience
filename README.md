# ViralMultiNet-gigascience

\# ViralMultiNet: A Structure-Aware Multimodal Framework for Viral Genome Classification



\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

\[!\[Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)



Official implementation of \*\*"ViralMultiNet: A Structure-Aware Multimodal Framework for Viral Genome Classification"\*\*, addressing reviewer comments on reproducibility, code structure, and data access.



---



\## 📋 Overview



ViralMultiNet is a multi-scale Transformer-based framework that integrates:



\- \*\*Multi-scale k-mer sequence encodings\*\* (4-7-mers) for hierarchical feature extraction

\- \*\*Semantic embeddings\*\* derived from functional annotations via BERT

\- \*\*Structure-aware attention mechanisms\*\* constrained by 3D protein structures (PDB: 6VXX)

\- \*\*Triple knowledge distillation\*\* (response-based + feature-based + attention-transfer) for efficiency

\- \*\*LoRA + Flash Attention\*\* optimizations for ~40.4% training speedup



\### Key Results



| Metric | Performance |

|--------|-------------|

| \*\*Macro F1\*\* | 0.921 ± 0.004 |

| \*\*Accuracy\*\* | 0.928 ± 0.003 |

| \*\*AUC\*\* | 0.983 ± 0.007 |

| \*\*Training Time Reduction\*\* | ~40.4% (94.3 → 56.2 min/epoch) |



---



\## 🗂️ Repository Structure



```

ViralMultiNet-gigascience/

├── README.md                           # This file

├── LICENSE                             # MIT License

├── environment.txt                     # Python dependencies

├── .gitignore                          # Git ignore rules

│

├── code/                               # Source code

│   ├── train\_viralmultinet/           # Main training pipeline

│   │   ├── models/                    # Model architectures

│   │   │   ├── teacher.py            # Teacher model (Structure-Aware Attention)

│   │   │   ├── student.py            # Student model (Cross-Modal + Flash Attention)

│   │   │   ├── attention.py          # Attention mechanisms

│   │   │   ├── distillation.py       # Triple knowledge distillation

│   │   │   └── lora\_integration.py   # LoRA implementation

│   │   ├── data\_processing/           # Data preprocessing

│   │   │   ├── kmer\_encoding.py      # Multi-scale k-mer tokenization

│   │   │   ├── annotation\_embedding.py # BERT-based text embeddings

│   │   │   └── structure\_alignment.py # PDB structure feature extraction

│   │   ├── training/                  # Training scripts

│   │   │   ├── train.py              # Main training loop

│   │   │   ├── config.yaml           # Hyperparameter configuration

│   │   │   └── utils.py              # Training utilities

│   │   └── evaluation/                # Evaluation scripts

│   │       ├── evaluate.py           # Model evaluation

│   │       ├── metrics.py            # Performance metrics

│   │       └── interpretability.py   # Attention visualization

│   └── tokenizer\_4mer/                # K-mer tokenizers

│       tokenizer\_5mer/

│       tokenizer\_6mer/

│       tokenizer\_7mer/

│

├── data/                               # Data files

│   ├── README.md                      # Data description

│   │

│   ├── dataset\_balanced\_train\_head100.xlsx  # Training set (head100 samples)

│   ├── dataset\_balanced\_val\_head100.xlsx    # Validation set (head100 samples)

│   ├── dataset\_balanced\_test\_head100.xlsx   # Test set (head100 samples)

│   │

│   ├── anno\_embs\_test.npy             # Annotation embeddings (test)

│   ├── anno\_embs\_val.npy              # Annotation embeddings (validation)

│   │

│   ├── 6vxx.pdb1                      # SARS-CoV-2 Spike protein structure (PDB: 6VXX)

│   ├── 7bv2.pdb1                      # Alternative Spike structure (PDB: 7BV2)

│   │

│   ├── kmer2res\_search.per\_kmer\_hits  # K-mer to residue mapping table

│   ├── kmer2res\_search.per\_residue    # Residue-level structure annotation

│   │

│   └── high\_attention\_kmers\_branch0\_clean.fasta  # High-attention k-mers (4-mer)

│       high\_attention\_kmers\_branch1\_clean.fasta  # High-attention k-mers (5-mer)

│       high\_attention\_kmers\_branch2\_clean.fasta  # High-attention k-mers (6-mer)

│       high\_attention\_kmers\_branch3\_clean.fasta  # High-attention k-mers (7-mer)

│

└── figures/                            # Figures from manuscript

&nbsp;   ├── Figure1\_workflow.png           # Overall framework

&nbsp;   ├── Figure2\_architecture.png       # Model architecture

&nbsp;   ├── Figure3\_ablation.png           # Ablation study results

&nbsp;   ├── Figure4\_attention\_spike.png    # Attention on Spike protein

&nbsp;   └── ...

```



---



\## 🔧 Installation



\### Prerequisites



\- Python >= 3.8

\- CUDA >= 11.0 (for GPU acceleration)

\- 16GB+ RAM recommended

\- NVIDIA GPU with 12GB+ VRAM (for training)



\### Setup Environment



```bash

\# Clone the repository

git clone https://github.com/DAVE-HUB-11/ViralMultiNet-gigascience.git

cd ViralMultiNet-gigascience



\# Create conda environment

conda create -n viralmultinet python=3.8

conda activate viralmultinet



\# Install dependencies

pip install -r requirements.txt



\# Or use the provided environment file

pip install -r environment.txt

```



\### Key Dependencies



\- `torch >= 1.10.0` (with CUDA support)

\- `transformers >= 4.20.0` (for BERT embeddings)

\- `flash-attn >= 2.0.0` (for Flash Attention)

\- `biopython >= 1.79` (for PDB parsing)

\- `numpy >= 1.21.0`

\- `pandas >= 1.3.0`

\- `scikit-learn >= 1.0.0`



---



\## 📊 Data



\### Dataset Overview



The SARS-CoV-2 ORF-level dataset consists of:



\- \*\*Total samples\*\*: 66,011 (after augmentation)

\- \*\*Training set\*\*: 47,091 samples

\- \*\*Validation set\*\*: 9,417 samples

\- \*\*Test set\*\*: 9,503 samples



\*\*Classes\*\*: 

\- Enzymatic (25%)

\- Structural (25%)

\- Transport (25%)

\- Other (25%)



\### Data Sources



1\. \*\*Raw sequencing data\*\*: 

&nbsp;  - NCBI SRA BioProject: \[PRJNA1251232](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1251232)

&nbsp;  - Accession: SRX28474964



2\. \*\*Protein structures\*\*:

&nbsp;  - \[6VXX](https://www.rcsb.org/structure/6VXX): SARS-CoV-2 Spike protein (closed state)

&nbsp;  - \[7BV2](https://www.rcsb.org/structure/7BV2): Alternative Spike conformation



3\. \*\*Functional annotations\*\*:

&nbsp;  - UniProt/Swiss-Prot (release 2024\_01)

&nbsp;  - GenBank metadata



\### Data Files Description



| File | Description | Usage |

|------|-------------|-------|

| `dataset\_balanced\_\*.xlsx` | Balanced ORF sequences with labels | Training/validation/testing |

| `anno\_embs\_\*.npy` | Pre-computed BERT embeddings (768-dim) | Text modality input |

| `6vxx.pdb1`, `7bv2.pdb1` | 3D protein structures | Structure-aware attention |

| `kmer2res\_search.per\_kmer\_hits` | K-mer → residue mapping | Structural alignment |

| `kmer2res\_search.per\_residue` | Residue-level structural features | 3D coordinate extraction |

| `high\_attention\_kmers\_branch\*.fasta` | Post-training high-attention regions | Interpretability analysis |



---



\## 🚀 Quick Start



\### 1. Training the Model



```bash

\# Train with default configuration

cd code/train\_viralmultinet

python training/train.py --config training/config.yaml



\# Train with custom hyperparameters

python training/train.py \\

&nbsp;   --alpha 0.3 \\

&nbsp;   --beta 0.5 \\

&nbsp;   --gamma 0.2 \\

&nbsp;   --delta 0.15 \\

&nbsp;   --batch\_size 32 \\

&nbsp;   --epochs 50

```



\### 2. Evaluation



```bash

\# Evaluate on test set

python evaluation/evaluate.py \\

&nbsp;   --checkpoint checkpoints/best\_model.pth \\

&nbsp;   --test\_data ../../data/dataset\_balanced\_test\_head100.xlsx



\# Generate attention maps for interpretability

python evaluation/interpretability.py \\

&nbsp;   --checkpoint checkpoints/best\_model.pth \\

&nbsp;   --output\_dir attention\_maps/

```



\### 3. Inference on New Sequences



```python

from train\_viralmultinet.models.student import ViralMultiNetStudent

import torch



\# Load trained model

model = ViralMultiNetStudent.load\_from\_checkpoint('checkpoints/best\_model.pth')

model.eval()



\# Prepare input sequence

sequence = "ATGGTTCGACTT..."  # Your viral ORF sequence

annotation = "Spike glycoprotein, membrane fusion..."  # Functional description



\# Predict

with torch.no\_grad():

&nbsp;   prediction = model.predict(sequence, annotation)

&nbsp;   

print(f"Predicted class: {prediction\['class']}")

print(f"Confidence: {prediction\['confidence']:.3f}")

```



---



\## 🧪 Reproducing Paper Results



\### Full Training Pipeline



```bash

\# Step 1: Preprocess data (if using raw NCBI data)

python code/train\_viralmultinet/data\_processing/preprocess\_raw\_data.py



\# Step 2: Generate k-mer tokenizers

python code/train\_viralmultinet/data\_processing/build\_tokenizers.py



\# Step 3: Extract structural features from PDB

python code/train\_viralmultinet/data\_processing/structure\_alignment.py \\

&nbsp;   --pdb data/6vxx.pdb1 \\

&nbsp;   --sequences data/dataset\_balanced\_train\_head100.xlsx



\# Step 4: Train teacher model (Structure-Aware Attention)

python code/train\_viralmultinet/training/train\_teacher.py



\# Step 5: Train student model (with knowledge distillation)

python code/train\_viralmultinet/training/train.py \\

&nbsp;   --teacher\_checkpoint checkpoints/teacher\_best.pth \\

&nbsp;   --use\_distillation



\# Step 6: Evaluate and generate figures

python code/train\_viralmultinet/evaluation/evaluate.py

python code/train\_viralmultinet/evaluation/generate\_figures.py

```



\### Expected Results



After 5-fold cross-validation, you should observe:



| Model | Macro F1 | Accuracy | AUC | Training Time/Epoch |

|-------|----------|----------|-----|---------------------|

| Full Model (Student + LoRA + Flash) | 0.921±0.004 | 0.928±0.003 | 0.983±0.007 | 56.2±2.1 min |

| Full Model (w/o LoRA) | 0.916±0.005 | 0.922±0.004 | 0.982±0.008 | 94.3±4.5 min |

| Teacher Model | 0.924±0.009 | 0.925±0.007 | 0.985±0.009 | ~120 min |

| Sequence Only | 0.872±0.008 | 0.881±0.009 | 0.983±0.009 | - |



---



\## 📐 Model Architecture Details



\### Multi-scale K-mer Encoding



\- \*\*4-mer\*\*: 256 tokens → Local motifs and binding signals

\- \*\*5-mer\*\*: 1,024 tokens → Secondary structure-related segments

\- \*\*6-mer\*\*: 4,096 tokens → Intermediate functional fragments

\- \*\*7-mer\*\*: 16,384 tokens → Regional context and evolutionary constraints



\### Attention Mechanisms



\*\*Teacher Model (Structure-Aware Attention)\*\*:

```python

attention\_scores = f(Q, S)  # Q: query, S: structural features from PDB 6VXX

\# S includes: 3D coordinates, secondary structure (α-helix, β-sheet), solvent accessibility

```



\*\*Student Model (Cross-Modal + Flash Attention)\*\*:

```python

\# Cross-modal attention

Q = sequence\_features

K, V = annotation\_embeddings

attention = FlashAttention(Q, K, V)  # Accelerated with Flash Attention

```



\### Knowledge Distillation



Triple distillation loss:

```

L\_KD = α·L\_resp + β·L\_feat + γ·L\_attn



where:

\- L\_resp: Response-based (soft targets, T=3.0)

\- L\_feat: Feature-based (intermediate hidden states)

\- L\_attn: Attention-transfer (structure-aware maps)



Optimized hyperparameters (via grid search):

α = 0.3, β = 0.5, γ = 0.2, δ = 0.15

```



---



\## 🔬 Interpretability Analysis



\### Attention Mapping to 3D Structure



The framework maps learned attention weights onto the SARS-CoV-2 Spike protein structure:



```bash

\# Generate attention heatmaps

python code/train\_viralmultinet/evaluation/interpretability.py \\

&nbsp;   --checkpoint checkpoints/best\_model.pth \\

&nbsp;   --pdb data/6vxx.pdb1 \\

&nbsp;   --output attention\_maps/



\# Outputs:

\# - attention\_spike\_protein.png: Attention overlay on 3D structure

\# - high\_attention\_regions.csv: Residue-level importance scores

```



\*\*Key validated regions\*\*:

\- \*\*Receptor-Binding Domain (RBD)\*\*: Residues 319-541

\- \*\*S1/S2 Cleavage Site\*\*: Residues 681-685

\- \*\*Fusion Peptide (FP)\*\*: Residues 816-835



See `data/high\_attention\_kmers\_branch\*.fasta` for post-training identified regions.



---



\## 📄 Citation



If you use this code or data in your research, please cite:



```bibtex

@article{li2025viralmultinet,

&nbsp; title={ViralMultiNet: A Structure-Aware Multimodal Framework for Viral Genome Classification},

&nbsp; author={Li, GuoDong and Lai, TingLian and Liu, FuGuo and Xu, WenXia},

&nbsp; journal={GigaScience},

&nbsp; year={2025},

&nbsp; note={In press}

}

```



---



\## 🤝 Contributing



We welcome contributions! Please feel free to:



\- Report bugs via \[GitHub Issues](https://github.com/DAVE-HUB-11/ViralMultiNet-gigascience/issues)

\- Submit feature requests

\- Create pull requests for improvements



\### Development Guidelines



1\. Fork the repository

2\. Create a feature branch (`git checkout -b feature/YourFeature`)

3\. Commit your changes (`git commit -m 'Add YourFeature'`)

4\. Push to the branch (`git push origin feature/YourFeature`)

5\. Open a Pull Request



---



\## 📧 Contact



\- \*\*Corresponding Author\*\*: WenXia Xu

&nbsp; - Email: xwx@guet.edu.cn

&nbsp; - Affiliation: School of Mathematics and Computational Science, Guilin University of Electronic Technology



\- \*\*GitHub Issues\*\*: For technical questions or bug reports, please open an issue on this repository.



---



\## 🙏 Acknowledgments



This work was supported by \[funding information].



\### Data Sources



\- \*\*NCBI Sequence Read Archive\*\*: Raw metagenomic sequencing data (SRX28474964)

\- \*\*UniProt/Swiss-Prot\*\*: Functional annotations (release 2024\_01)

\- \*\*Protein Data Bank\*\*: 3D structures (6VXX, 7BV2)



\### Tools \& Libraries



\- PyTorch for deep learning framework

\- Hugging Face Transformers for BERT embeddings

\- Flash Attention for efficient attention computation

\- BioPython for biological data processing



---



\## 📜 License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



---



\## 📝 Changelog



\### Version 1.0.0 (January 2025)

\- Initial release

\- Complete implementation of teacher-student framework

\- LoRA + Flash Attention optimizations

\- Triple knowledge distillation

\- Structure-aware attention mechanisms

\- Comprehensive evaluation and interpretability tools



---



\## ⚠️ Known Issues \& Limitations



1\. \*\*Current dataset scope\*\*: Primarily focused on SARS-CoV-2 ORFs

&nbsp;  - Future work will extend to broader viral families



2\. \*\*Annotation dependency\*\*: Labels derived from UniProt annotations

&nbsp;  - Not direct experimental ground truth

&nbsp;  - See Discussion section in paper for mitigation strategies



3\. \*\*Computational requirements\*\*: 

&nbsp;  - Training requires GPU with 12GB+ VRAM

&nbsp;  - Consider using gradient checkpointing for limited memory



---



\## 🔮 Future Directions



\- Extend to broader viral families (influenza, HIV, etc.)

\- Incorporate richer multimodal metadata (host information, geographic data)

\- Integrate with real-time wastewater surveillance pipelines

\- Fine-grained functional ontologies beyond 4-class classification

\- Orthogonal biological validation (conservation signals, mutation impact)



---



\*\*Last Updated\*\*: January 19, 2025



