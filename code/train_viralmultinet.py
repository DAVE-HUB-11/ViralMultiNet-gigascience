import torch
if not torch.cuda.is_available():
    print("WARNING: No CUDA GPU detected. Running on CPU, performance may be slow.")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaConfig, PreTrainedTokenizerFast, RobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report, average_precision_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import numpy as np
from tqdm.auto import tqdm
import argparse
import random
from collections import Counter
import optuna
import itertools
from sklearn.model_selection import StratifiedKFold
import sys
from sklearn.model_selection import train_test_split
import gc
import ast  # Make sure to add this import
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ================== Config Parameters ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "csv_path": r"C:\Users\Raytrack\Desktop\data\dataset_balanced_with_domains.csv",
    "kmer_scales": [4, 5, 6, 7],
    "max_length": 256,
    "batch_size": 64,         # Try 128 first
    "num_workers": 5,          # 
    "num_epochs": 5,           # 
    "learning_rate": 3e-5,
    "teacher_path": "./teacher.pt",
    "student_path": "./student.pt",
    "temperature": 3.0,
    "alpha": 0.35,
    "kd_loss_weight": 1.5,
    "feat_loss_weight": 0.05,
    "attn_loss_weight": 2.0,
    "num_classes": 4,
    "dropout": 0.1,
    "teacher": {"num_layers": 6, "hidden_size": 512, "out_dim": 256, "num_attention_heads": 8},
    "student": {"num_layers": 4, "hidden_size": 224, "out_dim": 224, "num_attention_heads": 6},
    "fp16": True,              # Mixed precision
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "enable_resample": True,   # Enable resampling or not
    "enable_eda": True,        # Enable EDA data augmentation or not
    "attn_distill_weight": 1.0,  # New, attention distillation loss weight
    "enable_structure_attention": True,  # Enable structure-aware attention or not
    "enable_adversarial_defense": True,  # Enable adversarial defense or not
    "enable_pgd_adversarial": True,      # Enable PGD adversarial defense or not
    "adversarial_epsilon": 0.005,          # PGD perturbation strength
    "use_dytnorm": True,                # Use DyTNorm layer, False for native LayerNorm
    "adversarial_scale_weights": {        # Adversarial perturbation weights for different kmer scales
        4: 1.0,    # Keep unchanged
        5: 0.8,    # Slightly decay weight
        6: 0.6,    # Faster decay
        7: 0.4     # Smaller perturbation
    },
    "hidden_size": 256,          # Hidden layer dimension of student model
    "intermediate_size": 1024,   # Intermediate layer dimension of student model
    "teacher_hidden_size": 768,  # Hidden layer dimension of teacher model
    "teacher_intermediate_size": 3072,  # Intermediate layer dimension of teacher model
    "domain_loss_weight": 0.5,  # You can adjust this weight
    "use_multimodal": True,  # Multimodal switch
    "use_cross_modal_attention": True,   # Use cross-modal attention or not
    # LoRA settings
    "enable_lora": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "lora_adapter_dir": "./lora_adapters",
    "lora_load_dir": None,
    # RAG settings
    "enable_rag": False,
    "rag_index_path": "./rag_index",
    "rag_corpus_path": None,  # Optional text file, one document per line
    "rag_model_name": "sentence-transformers/all-mpnet-base-v2",
    "rag_topk": 5,
    "rag_embed_dim": 768,
    # Prompt settings
    "enable_prompt_embed": False,
    "prompt_model_name": "sentence-transformers/all-mpnet-base-v2",  # for embedding prompted summary
    "enable_soft_prompt": False,
    "soft_prompt_len": 8,
}

# After CONFIG, add global cache
TOKENIZER_CACHE = {}

# ========== Utility Functions ==========
def plot_attention_heatmap(attn_matrix, tokens, title="Mean Attention Heatmap"):
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention weight')
    plt.title(title)
    plt.xlabel('Key position')
    plt.ylabel('Query position')
    plt.xticks(ticks=np.arange(len(tokens)), labels=tokens, rotation=90, fontsize=6)
    plt.yticks(ticks=np.arange(len(tokens)), labels=tokens, fontsize=6)
    plt.tight_layout()
    plt.show()

def extract_high_attention_kmers(mean_attn, tokens, top_n=5, window=1):
    """
    Extract top_n k-mer fragments with the highest attention
    window: fragment length (usually 1, or k-mer length)
    """
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(mean_attn):
        attn_sum = mean_attn.sum(dim=1).cpu().numpy()
    else:
        attn_sum = mean_attn.sum(axis=1)
    
    # Get indices of top_n highest attention scores
    top_idx = np.argsort(-attn_sum)[:top_n]  # Using negative to sort in descending order
    
    high_kmers = []
    for idx in top_idx:
        # Fix here: ensure idx is a scalar
        if hasattr(idx, 'item'):
            idx = idx.item()  # If it's a PyTorch tensor
        elif isinstance(idx, np.ndarray):
            if idx.size > 1:  # If it's a multi-element array, take the first element
                idx = idx[0]
            else:
                idx = idx.item()  # If it's a single-element array
        else:
            idx = int(idx)  # Otherwise keep the original conversion
            
        # Ensure idx is within valid range
        if 0 <= idx < len(tokens):
            kmer = tokens[idx:idx+window]
            high_kmers.append((''.join(kmer), idx, float(attn_sum[idx])))
    
    return high_kmers

def get_mean_attention(attentions):
    try:
        import torch
        if isinstance(attentions, torch.Tensor):
            return attentions.mean().item()
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(attentions, np.ndarray):
            return float(attentions.mean())
    except ImportError:
        pass

    # Python list
    def flatten(l):
        for el in l:
            if isinstance(el, (list, tuple)):
                yield from flatten(el)
            else:
                yield el
    flat = list(flatten(attentions))
    return sum(flat) / len(flat)

def write_fasta(kmers, filename="high_attention_kmers.fasta"):
    with open(filename, "w") as f:
        for i, (kmer, idx, score) in enumerate(kmers):
            f.write(f">highattn_{i}_pos{idx}_score{score:.3f}\n{kmer}\n")

# if sys.platform == "win32":
#     CONFIG["num_workers"] = 0

def get_vocab_size(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def reset_classifier(model):
    """
    Reset the classifier parameters of the model (usually used to reinitialize the classification head after transfer learning).
    """
    if hasattr(model, "classifier"):
        for layer in model.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

# =============== Focal Loss ===============
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Tensor or None
        self.reduction = reduction
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# =============== EDA ===============
def eda_dna(sequence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=2):
    """Simple EDA augmentation for DNA sequences, returns a list of augmented sequences"""
    def get_kmers(seq, k=3):
        return [seq[i:i+k] for i in range(len(seq)-k+1)]
    def to_seq(kmers):
        return ''.join(kmers)
    tokens = get_kmers(sequence)
    n = len(tokens)
    augmented = []
    for _ in range(num_aug):
        new_tokens = tokens.copy()
        # Random swap
        for _ in range(int(alpha_rs * n)):
            if len(new_tokens) > 1:
                idx1, idx2 = random.sample(range(len(new_tokens)), 2)
                new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]
        # Random deletion
        new_tokens = [t for t in new_tokens if random.random() > alpha_rd]
        if not new_tokens:
            new_tokens = tokens.copy()
        augmented.append(to_seq(new_tokens))
    return augmented

# ================== Config Parameters ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== Multi-scale Dataset ==================
class MultiScaleDNADataset(Dataset):
    def __init__(self, df, kmer_scales=None, anno_embs=None):
        self.kmer_scales = kmer_scales or CONFIG["kmer_scales"]
        self.encodings = []
        self.tokenizers = []
        for scale in self.kmer_scales:
            if scale in TOKENIZER_CACHE:
                tokenizer = TOKENIZER_CACHE[scale]
                print(f"[Dataset] Using cached tokenizer for {scale}mer")
            else:
                try:
                    tokenizer_path = os.path.join(os.path.dirname(CONFIG["csv_path"]), f"tokenizer_{scale}mer")
                    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
                    # Check if tokenizer is loaded correctly
                    vocab_size = len(tokenizer.get_vocab())
                    print(f"[Dataset] Loaded tokenizer for {scale}mer, vocab_size={vocab_size}")
                    if tokenizer.pad_token is None:
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                        print(f"[Dataset] Added [PAD] token to {scale}mer tokenizer")
                    TOKENIZER_CACHE[scale] = tokenizer
                except Exception as e:
                    print(f"[Dataset] ERROR loading tokenizer for {scale}mer: {e}")
                    print(f"[Dataset] Falling back to roberta-base tokenizer")
                    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.tokenizers.append(tokenizer)
            texts = df[f"text_{scale}mer"].tolist()
            enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=CONFIG["max_length"],
                return_tensors="pt"
            )
            # Check if generated input_ids are within vocab range
            max_id = enc['input_ids'].max().item()
            min_id = enc['input_ids'].min().item()
            vocab_size = len(tokenizer.get_vocab())
            print(f"[Dataset] {scale}mer - input_ids range: min={min_id}, max={max_id}, vocab_size={vocab_size}")
            # Ensure input_ids do not exceed vocab_size
            if max_id >= vocab_size:
                print(f"[WARNING] {scale}mer - input_ids.max()={max_id} >= vocab_size={vocab_size}!!!")
            self.encodings.append(enc)
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.anno_embs = anno_embs if CONFIG["use_multimodal"] else None
        assert all(0 <= l < CONFIG["num_classes"] for l in self.labels.tolist()), f"Label out of range: {set(self.labels.tolist())}"
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {}
        for i, scale in enumerate(self.kmer_scales):
            item[f'input_ids_{scale}'] = self.encodings[i]['input_ids'][idx]
            item[f'attention_mask_{scale}'] = self.encodings[i]['attention_mask'][idx]
        item['labels'] = self.labels[idx]
        if self.anno_embs is not None and CONFIG["use_multimodal"]:
            item['anno_emb'] = torch.tensor(self.anno_embs[idx], dtype=torch.float)
        return item

# ================== Roberta model ==================
class DyTNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.alpha = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.ones(normalized_shape))
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.delta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(((x - mean) ** 2).mean(dim=-1, keepdim=True) + self.eps)
        x_hat = (x - mean) / std
        y = self.alpha * torch.tanh(self.beta * x_hat) + self.gamma * x_hat + self.delta
        return y

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 为两个模态分别创建投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # 多头注意力投影
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_modal, key_value_modal, attention_mask=None):
        """
        Cross-modal attention: query_modal focuses on key_value_modal
        query_modal: [B, L_q, C]
        key_value_modal: [B, L_kv, C]
        attention_mask: [B, L_q, L_kv] Optional
        """
        B, L_q, C = query_modal.shape
        _, L_kv, _ = key_value_modal.shape
        
        # Multi-head projection
        q = self.q_proj(query_modal).reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, D]
        k = self.k_proj(key_value_modal).reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D]
        v = self.v_proj(key_value_modal).reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D]
        
        # Attention calculation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_kv]
        
        # Apply attention mask (if any)
        if attention_mask is not None:
            # Expand mask to fit multi-head dimensions
            expanded_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(expanded_mask == 0, float('-inf'))
        
        # Attention distribution
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Weighted sum
        context = torch.matmul(attn_probs, v)  # [B, H, L_q, D]
        context = context.transpose(1, 2).reshape(B, L_q, C)  # [B, L_q, C]
        
        # Output projection
        output = self.out_proj(context)
        
        return output, attn_probs

class StructuralAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Structure-aware attention projection layers
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Structure information fusion module
        self.structure_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Output projection layer
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_modal, key_value_modal, attention_mask=None):
        """
        Structure-aware attention: integrates local structure information of the sequence
        query_modal: [B, L_q, C]
        key_value_modal: [B, L_kv, C] - Can be query_modal itself or other context
        attention_mask: [B, L_q, L_kv] Optional
        """
        B, L_q, C = query_modal.shape
        _, L_kv, _ = key_value_modal.shape
        
        # Multi-head projection
        q = self.q_proj(query_modal).reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, D]
        k = self.k_proj(key_value_modal).reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D]
        v = self.v_proj(key_value_modal).reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D]
        
        # Structure-aware attention calculation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_kv]
        
        # Apply attention mask (if any)
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(expanded_mask == 0, float('-inf'))
        
        # Attention distribution
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Weighted sum
        context = torch.matmul(attn_probs, v)  # [B, H, L_q, D]
        context = context.transpose(1, 2).reshape(B, L_q, C)  # [B, L_q, C]
        
        # Structure gate fusion - Enhance local sequence structure understanding
        if query_modal is key_value_modal or key_value_modal is None:
            # Self-attention mode: Use sequence itself structure
            gate = self.structure_gate(torch.cat([query_modal, context], dim=-1))
            context = gate * context + (1 - gate) * query_modal
        
        # Output projection
        output = self.out_proj(context)
        
        return output, attn_probs

class CustomEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = CrossModalAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross_modal_context=None, mask=None):
        # If there's no cross-modal context, use self-attention
        context_to_use = cross_modal_context if cross_modal_context is not None else x
        attn_out, attn_weights = self.self_attn(x, context_to_use, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights

class CustomBertBackbone(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, dropout=0.1, vocab_size=30522):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            CustomEncoderLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        print(f"[CustomBertBackbone] Initialized with vocab_size={vocab_size}")

    def forward(self, input_ids, structure_info=None, mask=None):
        # Check if input_ids are out of range
        max_id = input_ids.max().item()
        min_id = input_ids.min().item()
        vocab_size = self.embedding.num_embeddings
        
        # Safe handling: If out of range, clip input_ids
        if max_id >= vocab_size:
            print(f"[WARNING] input_ids.max()={max_id} >= vocab_size={vocab_size}, clipping to vocab_size-1")
            # Create a clipped input_ids copy to avoid modifying original data
            safe_input_ids = input_ids.clone()
            # Modify all IDs exceeding vocab_size to <UNK> or last valid ID
            safe_input_ids = torch.clamp(safe_input_ids, 0, vocab_size-1)
            input_ids = safe_input_ids
            # Update max for debugging
            max_id = input_ids.max().item()
        
        assert max_id < vocab_size, (
            f"After clipping, input_ids.max()={max_id} still >= vocab_size={vocab_size}, min={min_id}, shape={input_ids.shape}. "
            f"Please check your tokenizer and embedding configuration."
        )
        
        x = self.embedding(input_ids)
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, structure_info, mask)
            all_attn_weights.append(attn_weights)
        return x, all_attn_weights  # Return attention weights for each layer

class MultiScaleModel(nn.Module):
    def __init__(self, kmer_scales, num_layers, hidden_size, out_dim, num_attention_heads, class_weights=None, focal_loss=None, anno_dim=768, tokenizers=None, is_teacher=False):
        super().__init__()
        self.kmer_scales = kmer_scales
        self.is_teacher = is_teacher  # New flag, used to distinguish between teacher and student models
        self.backbones = nn.ModuleList()
        for i, scale in enumerate(kmer_scales):
            tokenizer = tokenizers[i] if tokenizers is not None else None
            if tokenizer is not None:
                vocab_size = len(tokenizer.get_vocab())
            else:
                vocab_size = 30522
            # DEBUG: Print vocab_size and input_ids max/min
            try:
                # Sample a batch of input_ids
                sample_input_ids = None
                if hasattr(tokenizer, 'encode'):
                    # Try to encode a typical kmer with the tokenizer
                    example_kmer = 'A' * scale
                    sample_input_ids = tokenizer.encode(example_kmer, add_special_tokens=False)
                print(f"[DEBUG] scale={scale}, vocab_size={vocab_size}, sample_input_ids={sample_input_ids}")
            except Exception as e:
                print(f"[DEBUG] scale={scale}, vocab_size={vocab_size}, sample_input_ids=ERROR: {e}")
            self.backbones.append(
                CustomBertBackbone(
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    dropout=CONFIG["dropout"],
                    vocab_size=vocab_size
                )
            )
        
        if CONFIG.get("use_dytnorm", True):
            self.norms = nn.ModuleList([DyTNorm(out_dim) for _ in kmer_scales])
            self.final_norm = DyTNorm(out_dim * len(kmer_scales))
        else:
            self.norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in kmer_scales])
            self.final_norm = nn.LayerNorm(out_dim * len(kmer_scales))
        
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, out_dim) for _ in kmer_scales])
        
        # Feature interaction layer
        self.interaction = nn.Sequential(
            nn.Linear(out_dim * len(kmer_scales), out_dim * len(kmer_scales)),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(out_dim * len(kmer_scales), out_dim * len(kmer_scales))
        )
        
        # Classifier
        classifier_in_dim = out_dim * len(kmer_scales)
        if CONFIG["use_multimodal"] and anno_dim > 0:
            classifier_in_dim += anno_dim
        if CONFIG.get("enable_rag", False):
            classifier_in_dim += CONFIG.get("rag_embed_dim", 768)
        if CONFIG.get("enable_prompt_embed", False):
            classifier_in_dim += anno_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, out_dim * len(kmer_scales) // 2),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(out_dim * len(kmer_scales) // 2, CONFIG["num_classes"])
        )
        
        # Fusion layer (MLP+residual) and gate mechanism
        if CONFIG["use_multimodal"]:
            self.fusion_layer = nn.Sequential(
                nn.Linear(classifier_in_dim, classifier_in_dim),
                nn.ReLU(),
                nn.Dropout(CONFIG["dropout"])
            )
            self.gate_layer = nn.Sequential(
                nn.Linear(classifier_in_dim, classifier_in_dim),
                nn.Sigmoid()
            )
        
        self.class_weights = class_weights
        self.focal_loss = focal_loss
        self.use_structure_attention = CONFIG.get("enable_structure_attention", False)
        # Soft prompt parameter for annotation branch
        if CONFIG.get("enable_soft_prompt", False) and CONFIG.get("use_multimodal", False):
            self.soft_prompt = nn.Parameter(torch.randn(CONFIG.get("soft_prompt_len", 8), anno_dim))
        
        if self.use_structure_attention:
            if self.is_teacher:
                # Teacher model uses structure-aware attention
                print("[INFO] Teacher model using StructuralAttention")
                self.structure_attentions = nn.ModuleList([
                    StructuralAttention(hidden_size, num_heads=4, dropout=CONFIG["dropout"])
                    for _ in kmer_scales
                ])
            else:
                # Student model uses cross-modal attention
                print("[INFO] Student model using CrossModalAttention")
                self.structure_attentions = nn.ModuleList([
                    CrossModalAttention(hidden_size, num_heads=4, dropout=CONFIG["dropout"])
                    for _ in kmer_scales
                ])

    def forward(self, batch, return_attentions=False):
        features = []
        all_attentions = []
        hidden_states = []
        
        # First collect initial representations for all scales
        initial_states = []
        for i, scale in enumerate(self.kmer_scales):
            input_ids = batch[f'input_ids_{scale}']
            embedded = self.backbones[i].embedding(input_ids)
            initial_states.append(embedded)
        
        # Then apply cross-modal attention to each scale
        for i, scale in enumerate(self.kmer_scales):
            # Get current scale embedding
            current_state = initial_states[i]
            
            # If cross-modal attention is enabled, integrate information from all other scales into current scale
            if self.use_structure_attention:
                # Create cross-modal context, can be average or concatenation of other scales
                other_states = [s for j, s in enumerate(initial_states) if j != i]
                if other_states:
                    # Here we use simple average, but more complex methods can be used
                    cross_modal_context = torch.stack(other_states).mean(0)
                    # Apply cross-modal attention
                    hidden_state, attn_weights = self.backbones[i](input_ids, cross_modal_context)
                else:
                    hidden_state, attn_weights = self.backbones[i](input_ids)
            else:
                hidden_state, attn_weights = self.backbones[i](input_ids)
            
            hidden_states.append(hidden_state)
            transformed_features = self.fcs[i](hidden_state)
            x = transformed_features[:, 0]
            x = self.norms[i](x)
            features.append(x)
            
            if return_attentions:
                all_attentions.append(attn_weights)
        
        # Feature fusion
        weights = F.softmax(torch.ones(len(self.kmer_scales)), dim=0)
        concat = torch.cat([w * f for w, f in zip(weights, features)], dim=-1)
        
        # Apply final feature processing
        x = self.final_norm(concat)
        x = self.interaction(x)
        if CONFIG["use_multimodal"] and 'anno_emb' in batch:
            anno_emb = batch['anno_emb'].to(x.device)
            if anno_emb.dim() == 3 and anno_emb.shape[1] == 1:
                anno_emb = anno_emb.squeeze(1)
            assert x.shape[0] == anno_emb.shape[0], f"Batch size mismatch: {x.shape[0]} vs {anno_emb.shape[0]}"
            # soft prompt (additive)
            if CONFIG.get("enable_soft_prompt", False) and hasattr(self, 'soft_prompt'):
                prompt = self.soft_prompt.unsqueeze(0).expand(anno_emb.shape[0], -1, -1).mean(dim=1)
                anno_emb = anno_emb + prompt
            x = torch.cat([x, anno_emb], dim=-1)
            fusion_out = self.fusion_layer(x)
            gate = self.gate_layer(x)
            x = gate * x + (1 - gate) * fusion_out
        # RAG retrieved embeddings (mean of top-k)
        if CONFIG.get("enable_rag", False) and 'retrieved_emb' in batch:
            retrieved = batch['retrieved_emb'].to(x.device)
            if retrieved.dim() == 3:
                retrieved = retrieved.mean(dim=1)
            assert x.shape[0] == retrieved.shape[0], f"Batch size mismatch: {x.shape[0]} vs {retrieved.shape[0]}"
            x = torch.cat([x, retrieved], dim=-1)
        # Prompt-based embedding concatenation
        if CONFIG.get("enable_prompt_embed", False) and 'prompted_emb' in batch:
            prompted = batch['prompted_emb'].to(x.device)
            if prompted.dim() == 3 and prompted.shape[1] == 1:
                prompted = prompted.squeeze(1)
            assert x.shape[0] == prompted.shape[0], f"Batch size mismatch: {x.shape[0]} vs {prompted.shape[0]}"
            x = torch.cat([x, prompted], dim=-1)
        
        logits = self.classifier(x)
        loss = None
        if 'labels' in batch and batch['labels'] is not None:
            if self.focal_loss is not None:
                loss = self.focal_loss(logits, batch['labels'])
            else:
                weight = self.class_weights if self.class_weights is not None else None
                loss = F.cross_entropy(logits, batch['labels'], weight=weight)
        
        out = {
            "loss": loss,
            "logits": logits,
            "features": features,
            "hidden_states": hidden_states
        }
        
        if return_attentions:
            out["attentions"] = all_attentions
        
        return out

    def forward_with_embeddings(self, batch, embeddings_dict, return_attentions=False):
        features = []
        all_attentions = []
        hidden_states = []

        for i, scale in enumerate(self.kmer_scales):
            embeddings = embeddings_dict[f'embeddings_{scale}']
            hidden_state, attn_weights = self.backbones[i].layers[0](embeddings)
            for layer in self.backbones[i].layers[1:]:
                hidden_state, attn_weights = layer(hidden_state)
            hidden_states.append(hidden_state)
            if self.use_structure_attention:
                # Since StructuralAttention and CrossModalAttention interfaces are consistent, we can directly call
                hidden_state, _ = self.structure_attentions[i](hidden_state, hidden_state)
            transformed_features = self.fcs[i](hidden_state)
            x = transformed_features[:, 0]
            x = self.norms[i](x)
            features.append(x)
            if return_attentions:
                all_attentions.append(attn_weights)

        weights = F.softmax(torch.ones(len(self.kmer_scales)), dim=0)
        concat = torch.cat([w * f for w, f in zip(weights, features)], dim=-1)
        x = self.final_norm(concat)
        x = self.interaction(x)
        if CONFIG["use_multimodal"] and 'anno_emb' in batch:
            anno_emb = batch['anno_emb'].to(x.device)
            if anno_emb.dim() == 3 and anno_emb.shape[1] == 1:
                anno_emb = anno_emb.squeeze(1)
            assert x.shape[0] == anno_emb.shape[0], f"Batch size mismatch: {x.shape[0]} vs {anno_emb.shape[0]}"
            x = torch.cat([x, anno_emb], dim=-1)
            fusion_out = self.fusion_layer(x)
            gate = self.gate_layer(x)
            x = gate * x + (1 - gate) * fusion_out
        if CONFIG.get("enable_rag", False) and 'retrieved_emb' in batch:
            retrieved = batch['retrieved_emb'].to(x.device)
            if retrieved.dim() == 3:
                retrieved = retrieved.mean(dim=1)
            assert x.shape[0] == retrieved.shape[0], f"Batch size mismatch: {x.shape[0]} vs {retrieved.shape[0]}"
            x = torch.cat([x, retrieved], dim=-1)
        logits = self.classifier(x)

        loss = None
        if 'labels' in batch and batch['labels'] is not None:
            if self.focal_loss is not None:
                loss = self.focal_loss(logits, batch['labels'])
            else:
                weight = self.class_weights if self.class_weights is not None else None
                loss = F.cross_entropy(logits, batch['labels'], weight=weight)

        out = {
            "loss": loss,
            "logits": logits,
            "features": features,
            "hidden_states": hidden_states
        }

        if return_attentions:
            out["attentions"] = all_attentions

        return out

def pgd_attack(embeddings, epsilon, batch, model, scale_idx, alpha=0.01, num_iter=1):
    perturbed = embeddings.detach().clone()
    perturbed.requires_grad = True
    for _ in range(num_iter):
        embeddings_all = []
        for j, s in enumerate(CONFIG["kmer_scales"]):
            embedding_layer = model.backbones[j].embedding
            embeddings_all.append(embedding_layer(batch[f'input_ids_{s}']))
        embeddings_dict = {}
        for j, s in enumerate(CONFIG["kmer_scales"]):
            if j == scale_idx:
                embeddings_dict[f'embeddings_{s}'] = perturbed
            else:
                embeddings_dict[f'embeddings_{s}'] = embeddings_all[j].detach()
        outputs_adv = model.forward_with_embeddings(batch, embeddings_dict)
        loss_adv = outputs_adv["loss"]
        model.zero_grad()
        if perturbed.grad is not None:
            perturbed.grad.zero_()
        loss_adv.backward()
        grad = perturbed.grad
        perturbed = perturbed + alpha * grad.sign()
        perturbation = torch.clamp(perturbed - embeddings, min=-epsilon, max=epsilon)
        perturbed = embeddings + perturbation
        perturbed = perturbed.detach().clone()
        perturbed.requires_grad = True
    return perturbed

# ================== Training and distillation main loop ==================
class DistillTrainer:
    def __init__(self, stage="teacher", train_df=None, val_df=None):
        self.stage = stage
        self.train_df = train_df
        self.val_df = val_df

        # Distributed attributes
        try:
            self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        except Exception:
            self.local_rank = -1
        try:
            self.rank = int(os.environ.get('RANK', 0))
        except Exception:
            self.rank = 0
        try:
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        except Exception:
            self.world_size = 1

        # Unify calculation of class weights and focal loss
        class_counts = Counter(self.train_df['label'])
        total = sum(class_counts.values())
        num_classes = CONFIG["num_classes"]
        self.class_weights = torch.tensor(
            [total / (num_classes * class_counts[i]) for i in range(num_classes)],
            dtype=torch.float
        ).to(DEVICE)
        self.focal_loss = FocalLoss(gamma=2.5, alpha=self.class_weights)

        # First load dataset to get tokenizers
        print("Initializing training and validation dataset...")
        self.train_dataset = MultiScaleDNADataset(self.train_df, CONFIG["kmer_scales"])
        self.val_dataset = MultiScaleDNADataset(self.val_df, CONFIG["kmer_scales"])
        
        # Print tokenizer vocab_size, confirm correct loading
        tokenizers = self.train_dataset.tokenizers
        for i, scale in enumerate(CONFIG["kmer_scales"]):
            if i < len(tokenizers):
                tok = tokenizers[i]
                vocab_size = len(tok.get_vocab()) if hasattr(tok, 'get_vocab') else -1
                print(f"[Trainer] {scale}mer tokenizer vocab_size = {vocab_size}")
        
        # Configure data loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            sampler=DistributedSampler(self.train_dataset) if dist.is_initialized() else None,
            shuffle=not dist.is_initialized()
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            sampler=DistributedSampler(self.val_dataset, shuffle=False) if dist.is_initialized() else None,
            shuffle=False
        )
        
        print(f"Initializing {stage} model...")
        # Optional RAG runner
        self.rag = None
        if CONFIG.get("enable_rag", False):
            try:
                from rag_utils import RagRunner
                self.rag = RagRunner(model_name=CONFIG.get("rag_model_name", "sentence-transformers/all-mpnet-base-v2"),
                                     index_dir=CONFIG.get("rag_index_path", "./rag_index"))
                if not self.rag.index_exists():
                    corpus_path = CONFIG.get("rag_corpus_path")
                    if corpus_path and os.path.exists(corpus_path):
                        with open(corpus_path, "r", encoding="utf-8") as f:
                            corpus_texts = [line.strip() for line in f if line.strip()]
                        print(f"[RAG] Building index from corpus ({len(corpus_texts)} docs)...")
                        self.rag.build_and_save(corpus_texts)
                    else:
                        print("[RAG] No index/corpus found. RAG will be disabled for this run.")
                        self.rag = None
            except Exception as e:
                print(f"[RAG] Initialization failed: {e}")
                self.rag = None
        # Optional Prompt-based embedder
        self.prompter = None
        if CONFIG.get("enable_prompt_embed", False):
            try:
                from prompt_utils import PromptEmbedder
                self.prompter = PromptEmbedder(model_name=CONFIG.get("prompt_model_name", "sentence-transformers/all-mpnet-base-v2"))
            except Exception as e:
                print(f"[PROMPT] Initialization failed: {e}")
                self.prompter = None
        if stage == "teacher":
            self.model = MultiScaleModel(
                kmer_scales=[4,5,6,7],
                num_layers=4,  # Or according to your configuration
                hidden_size=384,
                out_dim=192,   # Unify to 192
                num_attention_heads=4,
                anno_dim=0,    # teacher stage does not concatenate anno_emb
                class_weights=None,  # Optional
                focal_loss=None,     # Optional
                tokenizers=tokenizers,
                is_teacher=True      # Specify as teacher model
            ).to(DEVICE)
            # Load self-supervised pre-training weights
            pretrained_path = "pretrained_multiscale_bert.pt"
            if os.path.exists(pretrained_path):
                print(f"Loading pretrained weights from {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)  # Increase weights_only=True to avoid warning
                def is_compatible_key(k):
                    # Only retain backbones related parameters
                    return k.startswith("backbones")
                filtered_state_dict = {k: v for k, v in state_dict.items() if is_compatible_key(k)}
                self.model.load_state_dict(filtered_state_dict, strict=False)
                reset_classifier(self.model)
            else:
                print(f"Pretrained weights not found at {pretrained_path}, using random init.")
        else:
            # 1. First load teacher model
            self.teacher = MultiScaleModel(
                kmer_scales=[4,5,6,7],
                num_layers=4,  # Or according to your configuration
                hidden_size=384,
                out_dim=192,   # Unify to 192
                num_attention_heads=4,
                anno_dim=768,  # BERT embedding dimension
                class_weights=None,  # Optional
                focal_loss=None,     # Optional
                tokenizers=tokenizers,
                is_teacher=True      # Specify as teacher model
            ).to(DEVICE)
            
            # 2. Load teacher model weights
            state_dict = torch.load(CONFIG["teacher_path"], map_location=DEVICE, weights_only=True)  # Increase weights_only=True to avoid warning
            def is_compatible_key(k):
                # Only retain backbones related parameters
                return k.startswith("backbones")
            filtered_state_dict = {k: v for k, v in state_dict.items() if is_compatible_key(k)}
            self.teacher.load_state_dict(filtered_state_dict, strict=False)
            self.teacher.eval()  # Set to evaluation mode
            
            # 3. Create student model (using random initialization)
            self.model = MultiScaleModel(
                kmer_scales=[4,5,6,7],
                num_layers=2,  # Or according to your configuration
                hidden_size=256,
                out_dim=192,   # Unify to 192
                num_attention_heads=4,
                anno_dim=768,  # BERT embedding dimension
                class_weights=self.class_weights,
                focal_loss=self.focal_loss,
                tokenizers=tokenizers,
                is_teacher=False     # Specify as student model
            ).to(DEVICE)
            
            # Inject LoRA adapters into student backbones (only for student stage)
            if CONFIG.get("enable_lora", False) and self.stage != "teacher":
                lora_cfg = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=CONFIG["lora_r"],
                    lora_alpha=CONFIG["lora_alpha"],
                    lora_dropout=CONFIG["lora_dropout"],
                    target_modules=CONFIG.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"])
                )
                for i in range(len(self.model.backbones)):
                    self.model.backbones[i] = get_peft_model(self.model.backbones[i], lora_cfg)
                # Optionally load existing adapters
                if CONFIG.get("lora_load_dir"):
                    load_dir = CONFIG["lora_load_dir"]
                    for i, scale in enumerate(self.model.kmer_scales):
                        adapter_dir = os.path.join(load_dir, f"scale{scale}")
                        if os.path.isdir(adapter_dir):
                            try:
                                self.model.backbones[i].load_adapter(adapter_dir)
                            except Exception as e:
                                print(f"[WARN] Failed to load LoRA adapter for scale {scale}: {e}")

            # 4. Create hidden layer adapter
            self.hidden_adapter = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(192, 192),  # student_out_dim, teacher_out_dim
                    nn.ReLU(),
                    nn.Linear(192, 192)
                ).to(DEVICE) for _ in [4,5,6,7]
            ])
            
            # No need to load weights, use default random initialization
            
        # Wrap model with DDP if distributed is initialized
        if dist.is_initialized():
            if DEVICE.type == 'cuda' and self.local_rank != -1:
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                self.model = DDP(self.model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=CONFIG["num_epochs"] * len(self.train_loader)
        )

        # Add in DistillTrainer.__init__
        self.f1_history = []
        self.precision_history = []
        self.epoch_times = []

        print("Initialization complete, starting training...")

    def calculate_distillation_loss(self, student_logits, teacher_logits):
        """
        Calculate distillation loss
        Args:
            student_logits: Output logits of student model
            teacher_logits: Output logits of teacher model
        Returns:
            Distillation loss
        """
        T = CONFIG["temperature"]  # Temperature parameter
        alpha = CONFIG["alpha"]    # Balance parameter
        
        # Soft target loss
        soft_targets = F.softmax(teacher_logits / T, dim=-1)
        soft_prob = F.log_softmax(student_logits / T, dim=-1)
        soft_targets = soft_targets.detach()  # Stop gradient
        distill_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T * T)
        
        return distill_loss

    def train_teacher(self):
        for epoch in range(CONFIG["num_epochs"]):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Teacher Epoch {epoch+1}")
            for batch in pbar:
                batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = self.model(batch)
                
                # Main task loss
                loss = outputs["loss"]
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Adversarial training
                if CONFIG.get("enable_adversarial_defense", False):
                    if CONFIG.get("enable_pgd_adversarial", False):
                        epsilon = CONFIG.get("adversarial_epsilon", 0.1)
                        alpha = 0.01  # Note: alpha variable needs to be defined
                        # Get current scale corresponding weight
                        for i, scale in enumerate(CONFIG["kmer_scales"]):
                            current_scale = CONFIG["kmer_scales"][i]
                            scale_weight = CONFIG.get("adversarial_scale_weights", {}).get(current_scale, 1.0)
                            adjusted_epsilon = epsilon * scale_weight
                            adjusted_alpha = alpha * scale_weight
                            # 1. Get embedding
                            input_ids = batch[f'input_ids_{scale}']
                            attention_mask = batch[f'attention_mask_{scale}']
                            embedding_layer = self.model.backbones[i].embedding
                            embeddings = embedding_layer(input_ids)
                            embeddings = embeddings.detach().clone().requires_grad_(True)
                            # 2. PGD perturbation
                            perturbed_embeddings = pgd_attack(
                                embeddings, adjusted_epsilon, batch, self.model, i, adjusted_alpha, num_iter=1
                            )
                            embeddings_dict = {}
                            for j, s in enumerate(CONFIG["kmer_scales"]):
                                if j == i:
                                    embeddings_dict[f'embeddings_{s}'] = perturbed_embeddings
                                else:
                                    embedding_layer = self.model.backbones[j].embedding
                                    embeddings_dict[f'embeddings_{s}'] = embedding_layer(batch[f'input_ids_{s}']).detach()
                            # 3. Again forward propagation
                            outputs_adv2 = self.model.forward_with_embeddings(batch, embeddings_dict)
                            loss_adv2 = outputs_adv2["loss"]
                            self.optimizer.zero_grad()
                            loss_adv2.backward()
                            self.optimizer.step()
                
                # Update progress bar information
                pbar.set_postfix({
                    'loss': loss.item()
                })
            
            self.scheduler.step()
            self.evaluate(epoch)
            torch.save(self.model.state_dict(), CONFIG["teacher_path"])

    def train_student(self):
        print("\nStarting student model training with knowledge distillation...")
        best_val_f1 = 0
        patience = 5
        no_improve = 0
        
        for epoch in range(CONFIG["num_epochs"]):
            start_time = time.time()
            self.model.train()
            self.teacher.eval()
            total_loss = 0
            total_kd_loss = 0
            total_ce_loss = 0
            total_feat_loss = 0
            total_attn_loss = 0
            
            pbar = tqdm(self.train_loader, desc=f"Student Epoch {epoch+1}")
            for batch in pbar:
                batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
                # RAG retrieval per batch (optional)
                if self.rag is not None and CONFIG.get("enable_rag", False):
                    try:
                        query_texts = self._make_query_texts(batch)
                        if query_texts:
                            retrieved_embs = self.rag.retrieve_embeddings(query_texts, topk=CONFIG.get("rag_topk", 5))
                            batch['retrieved_emb'] = torch.tensor(retrieved_embs, dtype=torch.float, device=DEVICE).mean(dim=1)
                    except Exception as e:
                        print(f"[RAG] Batch retrieval skipped due to error: {e}")
                # Prompt-based embedding per batch (optional)
                if self.prompter is not None and CONFIG.get("enable_prompt_embed", False) and CONFIG.get("use_multimodal", False):
                    try:
                        query_texts = self._make_query_texts(batch)
                        if query_texts:
                            prompted_embs = self.prompter.embed_summaries(query_texts)
                            batch['prompted_emb'] = torch.tensor(prompted_embs, dtype=torch.float, device=DEVICE)
                    except Exception as e:
                        print(f"[PROMPT] Batch prompting skipped due to error: {e}")
                
                # Get teacher model output (no gradient)
                with torch.no_grad():
                    teacher_outputs = self.teacher(batch, return_attentions=True)
                    teacher_logits = teacher_outputs["logits"]
                    teacher_attentions = teacher_outputs["attentions"]
                
                # Get student model output
                student_outputs = self.model(batch, return_attentions=True)
                student_logits = student_outputs["logits"]
                student_attentions = student_outputs["attentions"]
                
                # 1. Hard label loss (cross entropy)
                if self.focal_loss is not None:
                    ce_loss = self.focal_loss(student_logits, batch['labels'])
                else:
                    ce_loss = F.cross_entropy(student_logits, batch['labels'], weight=self.class_weights)
                
                # 2. Knowledge distillation loss (soft label)
                kd_loss = self.calculate_distillation_loss(student_logits, teacher_logits)
                
                # 3. Feature distillation loss
                feat_loss = 0
                for i, (s_feat, t_feat) in enumerate(zip(student_outputs.get("features", []), teacher_outputs.get("features", []))):
                    s_feat_adapted = self.hidden_adapter[i](s_feat)
                    feat_loss += self.calculate_distillation_loss(s_feat_adapted, t_feat)
                
                # 4. Attention distillation loss
                attn_loss = 0
                if student_attentions and teacher_attentions:
                    for s_attn, t_attn in zip(student_attentions, teacher_attentions):
                        attn_loss += self.calculate_distillation_loss(s_attn[-1], t_attn[-1])  # Use last layer attention
                
                # Total loss
                loss = (CONFIG["alpha"] * ce_loss + 
                       (1 - CONFIG["alpha"]) * CONFIG["kd_loss_weight"] * kd_loss +
                       CONFIG["feat_loss_weight"] * feat_loss +
                       CONFIG["attn_loss_weight"] * attn_loss)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                # Record loss
                total_loss += loss.item()
                total_kd_loss += kd_loss.item()
                total_ce_loss += ce_loss.item()
                total_feat_loss += feat_loss.item()
                total_attn_loss += attn_loss.item()
                
                # Update progress bar
                avg_loss = total_loss / (pbar.n + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'kd_loss': f'{total_kd_loss/(pbar.n + 1):.4f}',
                    'ce_loss': f'{total_ce_loss/(pbar.n + 1):.4f}'
                })
            
            # Evaluate
            metrics = self.evaluate(epoch)
            current_f1 = metrics['macro_f1']
            current_precision = metrics['macro_precision']
            self.f1_history.append(current_f1)
            self.precision_history.append(current_precision)
            
            # Learning rate adjustment
            self.scheduler.step()
            
            # Early stopping
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                print(f"New best F1: {best_val_f1:.4f}")
                torch.save(self.model.state_dict(), CONFIG["student_path"])
                abs_path = os.path.abspath(CONFIG["student_path"])
                print(f"[Model saved] student.pt path: {abs_path}")
                # Save LoRA adapters if enabled
                if CONFIG.get("enable_lora", False):
                    base_dir = CONFIG.get("lora_adapter_dir", "./lora_adapters")
                    os.makedirs(base_dir, exist_ok=True)
                    for i, scale in enumerate(self.model.kmer_scales):
                        adapter_dir = os.path.join(base_dir, f"scale{scale}")
                        try:
                            if hasattr(self.model.backbones[i], "save_pretrained"):
                                self.model.backbones[i].save_pretrained(adapter_dir)
                        except Exception as e:
                            print(f"[WARN] Failed to save LoRA adapter for scale {scale}: {e}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            
            # Print current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")
            epoch_time = time.time() - start_time
            self.epoch_times.append(epoch_time)

    def train_student_nodistill(self):
        print("\nTraining student model WITHOUT distillation (only hard labels)...")
        best_val_f1 = 0
        patience = 5
        no_improve = 0

        for epoch in range(CONFIG["num_epochs"]):
            self.model.train()
            total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Student(no distill) Epoch {epoch+1}")
            for batch in pbar:
                batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
                if self.rag is not None and CONFIG.get("enable_rag", False):
                    try:
                        query_texts = self._make_query_texts(batch)
                        if query_texts:
                            retrieved_embs = self.rag.retrieve_embeddings(query_texts, topk=CONFIG.get("rag_topk", 5))
                            batch['retrieved_emb'] = torch.tensor(retrieved_embs, dtype=torch.float, device=DEVICE).mean(dim=1)
                    except Exception as e:
                        print(f"[RAG] Batch retrieval skipped due to error: {e}")
                if self.prompter is not None and CONFIG.get("enable_prompt_embed", False) and CONFIG.get("use_multimodal", False):
                    try:
                        query_texts = self._make_query_texts(batch)
                        if query_texts:
                            prompted_embs = self.prompter.embed_summaries(query_texts)
                            batch['prompted_emb'] = torch.tensor(prompted_embs, dtype=torch.float, device=DEVICE)
                    except Exception as e:
                        print(f"[PROMPT] Batch prompting skipped due to error: {e}")
                outputs = self.model(batch)
                loss = outputs["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / (pbar.n + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            metrics = self.evaluate(epoch)
            current_f1 = metrics['macro_f1']
            self.scheduler.step()
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                print(f"New best F1: {best_val_f1:.4f}")
                torch.save(self.model.state_dict(), CONFIG["student_path"])
                abs_path = os.path.abspath(CONFIG["student_path"])
                print(f"[Model saved] student.pt path: {abs_path}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")

    def evaluate(self, epoch, silent=False):
        if self.rank != 0:
            dist.barrier()
            return {}
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
                if self.rag is not None and CONFIG.get("enable_rag", False):
                    try:
                        query_texts = self._make_query_texts(batch)
                        if query_texts:
                            retrieved_embs = self.rag.retrieve_embeddings(query_texts, topk=CONFIG.get("rag_topk", 5))
                            batch['retrieved_emb'] = torch.tensor(retrieved_embs, dtype=torch.float, device=DEVICE).mean(dim=1)
                    except Exception as e:
                        if not silent:
                            print(f"[RAG] Eval retrieval skipped due to error: {e}")
                if self.prompter is not None and CONFIG.get("enable_prompt_embed", False) and CONFIG.get("use_multimodal", False):
                    try:
                        query_texts = self._make_query_texts(batch)
                        if query_texts:
                            prompted_embs = self.prompter.embed_summaries(query_texts)
                            batch['prompted_emb'] = torch.tensor(prompted_embs, dtype=torch.float, device=DEVICE)
                    except Exception as e:
                        if not silent:
                            print(f"[PROMPT] Eval prompting skipped due to error: {e}")
                outputs = self.model(batch)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                all_labels.extend(batch["labels"].cpu().numpy())
                all_preds.extend(preds)
                all_probs.append(probs)
        all_probs = np.vstack(all_probs)
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        macro_precision = precision_score(all_labels, all_preds, average='macro')
        macro_recall = recall_score(all_labels, all_preds, average='macro')
        try:
            auc = roc_auc_score(label_binarize(all_labels, classes=np.arange(CONFIG["num_classes"])), all_probs, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.5
        try:
            auc_pr = average_precision_score(label_binarize(all_labels, classes=np.arange(CONFIG["num_classes"])), all_probs, average='macro')
        except Exception:
            auc_pr = 0.0
        if not silent:
            print("Overall evaluation results:")
            print(f"ROC AUC (macro): {auc:.4f}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Macro Recall: {macro_recall:.4f}")
            print(f"Macro Precision: {macro_precision:.4f}")
            labels_unique, labels_counts = np.unique(all_labels, return_counts=True)
            print(f"Label distribution: {(labels_unique, labels_counts)}")
            print(f"Predicted value range: [{all_probs.min():.3f}, {all_probs.max():.3f}]")
            print("Detailed classification report:")
            print(classification_report(all_labels, all_preds, digits=4))
        minority_class_index = np.argmin(labels_counts)
        minority_recall = recall_score(all_labels, all_preds, average=None)[minority_class_index]  # Assume minority_class_index known
        mcc = matthews_corrcoef(all_labels, all_preds)
        if self.rank == 0:
            print(f"ROC AUC (macro): {auc:.4f}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Macro Recall: {macro_recall:.4f}")
            print(f"Macro Precision: {macro_precision:.4f}")
            print(f"Label distribution: {(labels_unique, labels_counts)}")
            print(f"Predicted value range: [{all_probs.min():.3f}, {all_probs.max():.3f}]")
            print("Detailed classification report:")
            print(classification_report(all_labels, all_preds, digits=4))
        return {
            "acc": acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "auc": auc,
            "auc_pr": auc_pr,
            "minority_recall": minority_recall,
            "mcc": mcc
        }

    def _make_query_texts(self, batch):
        try:
            # Prefer explicit annotation texts if provided in batch
            if 'anno_texts' in batch and batch['anno_texts'] is not None:
                return batch['anno_texts']
            # Fallback: decode first-scale tokens to text snippets
            if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'tokenizers') and len(self.train_dataset.tokenizers) > 0:
                tok = self.train_dataset.tokenizers[0]
                input_ids_1 = batch.get('input_ids_4') or next((batch[k] for k in batch if k.startswith('input_ids_')), None)
                if input_ids_1 is None:
                    return None
                ids_cpu = input_ids_1.detach().cpu().tolist()
                texts = []
                for ids in ids_cpu:
                    tokens = tok.convert_ids_to_tokens(ids)
                    # remove special tokens
                    tokens = [t for t in tokens if t not in (tok.pad_token, tok.cls_token if hasattr(tok, 'cls_token') else None, tok.sep_token if hasattr(tok, 'sep_token') else None)]
                    texts.append(" ".join(tokens[:256]))
                return texts
        except Exception:
            return None
        return None

    def run(self):
        if self.stage == "teacher":
            self.train_teacher()
        elif self.stage == "student_nodistill":
            self.train_student_nodistill()
        else:
            self.train_student()

# ========== Move attention_distillation_loss and objective here ==========
def attention_distillation_loss(student_attn, teacher_attn, T=1.0):
    # student_attn, teacher_attn: [batch, heads, seq, seq]
    loss = F.kl_div(
        F.log_softmax(student_attn / T, dim=-1),
        F.softmax(teacher_attn / T, dim=-1),
        reduction='batchmean'
    ) * (T * T)
    return loss

def objective(trial):
    # Searchable hyperparameters
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    kd_loss_weight = trial.suggest_float("kd_loss_weight", 0.5, 2.0)
    feat_loss_weight = trial.suggest_float("feat_loss_weight", 0.01, 0.2)
    attn_loss_weight = trial.suggest_float("attn_loss_weight", 0.5, 3.0)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Modify CONFIG
    CONFIG["alpha"] = alpha
    CONFIG["kd_loss_weight"] = kd_loss_weight
    CONFIG["feat_loss_weight"] = feat_loss_weight
    CONFIG["attn_loss_weight"] = attn_loss_weight
    CONFIG["learning_rate"] = learning_rate
    CONFIG["batch_size"] = batch_size

    # Ensure teacher weights exist
    CONFIG["teacher_path"] = "./teacher.pt"  # Or your teacher path
    if not os.path.exists(CONFIG["teacher_path"]):
        raise FileNotFoundError(f"Teacher model not found at {CONFIG['teacher_path']}. Please train the teacher model first.")

    # Reload data
    train_df = pd.read_csv(r"C:\Users\Raytrack\Desktop\data\dataset_balanced_train.csv")
    val_df = pd.read_csv(r"C:\Users\Raytrack\Desktop\data\dataset_balanced_val.csv")
    anno_embs_train = np.load(r"C:\Users\Raytrack\Desktop\data\anno_embs_train.npy")
    anno_embs_val = np.load(r"C:\Users\Raytrack\Desktop\data\anno_embs_val.npy")

    train_dataset = MultiScaleDNADataset(train_df, kmer_scales=CONFIG["kmer_scales"], anno_embs=anno_embs_train)
    val_dataset = MultiScaleDNADataset(val_df, kmer_scales=CONFIG["kmer_scales"], anno_embs=anno_embs_val)

    trainer = DistillTrainer(
        stage="student",
        train_df=train_df,
        val_df=val_df
    )
    trainer.train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    trainer.val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # Train only 1 epoch to speed up search
    CONFIG["num_epochs"] = 1

    trainer.train_student()
    metrics = trainer.evaluate(epoch=-1, silent=True)
    del trainer, train_dataset, val_dataset, train_df, val_df
    gc.collect()
    torch.cuda.empty_cache()
    return metrics["macro_f1"]

# ================== main entry point ==================
if __name__ == "__main__":
    # Add this before parser
    import os
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Explicitly set MASTER_ADDR and MASTER_PORT for Windows compatibility
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='gloo' if sys.platform == 'win32' else 'nccl',
            init_method='tcp://127.0.0.1:29501',
            world_size=world_size,
            rank=rank
        )
        dist.barrier()  # Synchronize after init
        if rank == 0:
            print('Distributed init successful')

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["teacher", "student", "student_nodistill"], required=True)
    parser.add_argument("--n_trials", type=int, default=5, help="Optuna trials per fold")
    parser.add_argument("--optuna", action="store_true", help="Use Optuna for hyperparameter search")
    parser.add_argument("--multimodal", action="store_true", help="Enable multimodal text annotations")
    parser.add_argument("--enable_lora", action="store_true", help="Enable LoRA adapters for student stage")
    parser.add_argument("--enable_rag", action="store_true", help="Enable RAG retrieval for multimodal")
    parser.add_argument("--rag_index_path", type=str, default=None, help="Path to RAG index directory")
    parser.add_argument("--rag_corpus_path", type=str, default=None, help="Text corpus file for RAG index (one doc per line)")
    parser.add_argument("--enable_prompt_embed", action="store_true", help="Enable prompt-based annotation embeddings")
    parser.add_argument("--enable_soft_prompt", action="store_true", help="Enable learnable soft prompt for annotation")
    parser.add_argument("--rag_topk", type=int, default=None, help="Top-k retrieved docs")
    parser.add_argument("--no_annotation", action="store_true", help="Disable annotations (sequence only)")
    parser.add_argument("--no_sequence", action="store_true", help="Disable sequence (annotation only)")
    parser.add_argument("--no_distillation", action="store_true", help="Disable distillation for student")
    parser.add_argument("--no_pgd", action="store_true", help="Disable PGD adversarial training")
    parser.add_argument("--single_scale", action="store_true", help="Use only 4-mer scale")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--head_only", action="store_true", help="Fine-tune only head")
    parser.add_argument("--few_shot_percent", type=float, default=0, help="Few-shot percentage (0-100)")
    parser.add_argument("--cross_virus", action="store_true", help="Train on SARS-CoV-2, test on others")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of runs")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 2024, 1001], help="Random seeds")
    args = parser.parse_args()

    # Update CONFIG based on args
    if args.no_annotation:
        CONFIG["use_multimodal"] = False
    if args.no_sequence:
        CONFIG["use_multimodal"] = True  # But disable sequence parts - need logic
        # Add custom logic in model to ignore sequence
    if args.no_distillation and args.stage == "student":
        args.stage = "student_nodistill"
    if args.no_pgd:
        CONFIG["enable_pgd_adversarial"] = False
    if args.single_scale:
        CONFIG["kmer_scales"] = [4]
    CONFIG["lora_r"] = args.lora_r
    if args.head_only:
        # Add logic to freeze all but classifier
        pass # Placeholder for head-only training logic
    if args.few_shot_percent > 0:
        # Subsample train_df to percent
        pass # Placeholder for few-shot training logic
    if args.cross_virus:
        # Load specific datasets
        pass # Placeholder for cross-virus training logic

    CONFIG["use_multimodal"] = args.multimodal
    CONFIG["enable_lora"] = args.enable_lora
    if args.enable_rag:
        CONFIG["enable_rag"] = True
    if args.enable_prompt_embed:
        CONFIG["enable_prompt_embed"] = True
    if args.enable_soft_prompt:
        CONFIG["enable_soft_prompt"] = True
    if args.rag_index_path:
        CONFIG["rag_index_path"] = args.rag_index_path
    if args.rag_corpus_path:
        CONFIG["rag_corpus_path"] = args.rag_corpus_path
    if args.rag_topk is not None:
        CONFIG["rag_topk"] = args.rag_topk
    train_df = pd.read_csv(r"C:\Users\Raytrack\Desktop\data\dataset_balanced_train.csv")
    val_df = pd.read_csv(r"C:\Users\Raytrack\Desktop\data\dataset_balanced_val.csv")
    test_df = pd.read_csv(r"C:\Users\Raytrack\Desktop\data\dataset_balanced_test.csv")
    if CONFIG["use_multimodal"]:
        anno_embs_train = np.load(r"C:\Users\Raytrack\Desktop\data\anno_embs_train.npy")
        anno_embs_val = np.load(r"C:\Users\Raytrack\Desktop\data\anno_embs_val.npy")
        anno_embs_test = np.load(r"C:\Users\Raytrack\Desktop\data\anno_embs_test.npy")
    else:
        anno_embs_train = None
        anno_embs_val = None
        anno_embs_test = None
    if args.stage == "teacher":
        train_dataset = MultiScaleDNADataset(train_df, kmer_scales=CONFIG["kmer_scales"], anno_embs=anno_embs_train)
        val_dataset = MultiScaleDNADataset(val_df, kmer_scales=CONFIG["kmer_scales"], anno_embs=anno_embs_val)
        test_dataset = MultiScaleDNADataset(test_df, kmer_scales=CONFIG["kmer_scales"], anno_embs=anno_embs_test)
    else:
        train_dataset = MultiScaleDNADataset(train_df, kmer_scales=CONFIG["kmer_scales"], anno_embs=anno_embs_train)
        val_dataset = MultiScaleDNADataset(val_df, kmer_scales=CONFIG["kmer_scales"], anno_embs=anno_embs_val)
        test_dataset = MultiScaleDNADataset(test_df, kmer_scales=CONFIG["kmer_scales"], anno_embs=anno_embs_test)
    trainer = DistillTrainer(
        stage=args.stage,
        train_df=train_df,
        val_df=val_df
    )
    trainer.train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    trainer.val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # DDP wrapping is handled inside DistillTrainer; no action needed here

    # For multiple runs
    results = []
    for i in range(args.repetitions):
        seed = args.seeds[i % len(args.seeds)]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Run trainer
        trainer = DistillTrainer(
            stage=args.stage,
            train_df=train_df,
            val_df=val_df
        )
        trainer.train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
        trainer.val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
        trainer.run()
        metrics = trainer.evaluate(epoch=-1, silent=True)
        # Add efficiency metrics
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics['gpu_mem'] = mem_info.used / 1024**2
        # Trainable params
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        metrics['trainable_params'] = trainable_params
        # Time tracking - add in trainer
        results.append(metrics)

    # Aggregate mean and std
    # Save to file or print
    print("\n--- Aggregated Results ---")
    for i, result in enumerate(results):
        print(f"\nRun {i+1}:")
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print(f"Macro Precision: {result['macro_precision']:.4f}")
        print(f"Macro Recall: {result['macro_recall']:.4f}")
        print(f"ROC AUC (macro): {result['auc']:.4f}")
        print(f"AUC PR (macro): {result['auc_pr']:.4f}")
        print(f"GPU Memory: {result['gpu_mem']:.2f} MB")
        print(f"Trainable Parameters: {result['trainable_params']}")

    # Example of how to save results to a file
    import json
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # After training
    if args.stage in ["student", "student_nodistill"]:
        if args.enable_vllm and (rank == 0 or rank == -1):
            print("\nPreparing model for VLLM inference...")
            save_for_vllm(trainer.model.module if isinstance(trainer.model, DDP) else trainer.model, "vllm_student_model")
            
            # Load with VLLM (assuming compatible architecture)
            vllm_model = vllm.LLM(
                model="vllm_student_model",
                tokenizer="roberta-base",  # Adjust to your tokenizer
                enable_lora=True if CONFIG["enable_lora"] else False
            )
            
            # Sample inference
            sample_sequence = "Your sample DNA sequence here"  # Replace with actual sample
            sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=CONFIG["max_length"])
            outputs = vllm_model.generate([sample_sequence], sampling_params)
            print("VLLM Inference Output:", outputs)
        else:
            print("\nVLLM inference disabled.")

    if local_rank != -1:
        dist.destroy_process_group()

def save_for_vllm(model, path="vllm_model"):
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(path)
    else:
        torch.save(model.state_dict(), os.path.join(path, "pytorch_model.bin"))
