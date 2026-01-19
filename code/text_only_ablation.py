import torch
if not torch.cuda.is_available():
    raise RuntimeError("本程序只支持GPU运行，请在有CUDA GPU的环境下运行！")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaConfig, PreTrainedTokenizerFast, RobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report, average_precision_score
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
import ast  # 确保添加这个导入
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CONFIG = {
    "csv_path": r"C:/Users/Raytrack/Desktop/dataset_balanced_with_domains.csv",
    "kmer_scales": [4, 5, 6, 7],
    "max_length": 256,
    "batch_size": 128,         # 先试128，显存足够可继续增大
    "num_workers": 0,          # 或8
    "num_epochs": 20,           # 只训练1个epoch
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
    "student": {"num_layers": 6, "hidden_size": 448, "out_dim": 336, "num_attention_heads": 8},
    "fp16": True,              # 混合精度
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "enable_resample": True,   # 是否启用重采样
    "enable_eda": True,        # 是否启用EDA数据增强
    "attn_distill_weight": 1.0,  # 新增，注意力蒸馏损失权重
    "enable_structure_attention": True,  # 是否启用结构感知注意力
    "enable_adversarial_defense": True,  # 是否启用对抗防御
    "enable_pgd_adversarial": True,      # 是否启用PGD对抗防御
    "adversarial_epsilon": 0.005,          # PGD扰动强度
    "use_dytnorm": True,                # 是否使用DyTNorm层，False则用原生LayerNorm
    "adversarial_scale_weights": {        # 不同kmer scale的对抗扰动权重
        4: 1.0,    # 保持不变
        5: 0.8,    # 稍微调整权重衰减
        6: 0.6,    # 更快的衰减
        7: 0.4     # 更小的扰动
    },
    "hidden_size": 256,          # 学生模型的隐藏层维度
    "intermediate_size": 1024,   # 学生模型的中间层维度
    "teacher_hidden_size": 768,  # 教师模型的隐藏层维度
    "teacher_intermediate_size": 3072,  # 教师模型的中间层维度
    "domain_loss_weight": 0.5,  # 可以调整此权重
    "use_multimodal": False,  # 多模态开关
    "use_cross_modal_attention": True,   # 是否使用跨模态注意力
}

# ========== 工具函数区 ==========
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
    提取attention和最高的top_n个k-mer片段
    window: 片段长度（通常为1，或k-mer长度）
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
        # 修复这里：确保idx是标量
        if hasattr(idx, 'item'):
            idx = idx.item()  # 如果是PyTorch张量
        elif isinstance(idx, np.ndarray):
            if idx.size > 1:  # 如果是多元素数组，只取第一个元素
                idx = idx[0]
            else:
                idx = idx.item()  # 如果是单元素数组
        else:
            idx = int(idx)  # 其他情况下保持原有转换
            
        # 确保idx在有效范围内
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

    # Python 列表
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
    重置模型的分类器参数（通常用于迁移学习后重新初始化分类头）。
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
    """对DNA序列进行简单EDA增强，返回增强后的序列列表"""
    def get_kmers(seq, k=3):
        return [seq[i:i+k] for i in range(len(seq)-k+1)]
    def to_seq(kmers):
        return ''.join(kmers)
    tokens = get_kmers(sequence)
    n = len(tokens)
    augmented = []
    for _ in range(num_aug):
        new_tokens = tokens.copy()
        # 随机交换
        for _ in range(int(alpha_rs * n)):
            if len(new_tokens) > 1:
                idx1, idx2 = random.sample(range(len(new_tokens)), 2)
                new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]
        # 随机删除
        new_tokens = [t for t in new_tokens if random.random() > alpha_rd]
        if not new_tokens:
            new_tokens = tokens.copy()
        augmented.append(to_seq(new_tokens))
    return augmented

# ================== 配置参数 ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 多尺度数据集 ==================
class MultiScaleDNADataset(Dataset):
    def __init__(self, df, kmer_scales=None, anno_embs=None):
        self.kmer_scales = kmer_scales or CONFIG["kmer_scales"]
        self.encodings = []
        self.tokenizers = []
        for scale in self.kmer_scales:
            try:
                tokenizer_path = os.path.join(os.path.dirname(CONFIG["csv_path"]), f"tokenizer_{scale}mer")
                tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
                # 检查tokenizer是否正确加载
                vocab_size = len(tokenizer.get_vocab())
                print(f"[Dataset] Loaded tokenizer for {scale}mer, vocab_size={vocab_size}")
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    print(f"[Dataset] Added [PAD] token to {scale}mer tokenizer")
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
            # 检查生成的input_ids是否在vocab范围内
            max_id = enc['input_ids'].max().item()
            min_id = enc['input_ids'].min().item()
            vocab_size = len(tokenizer.get_vocab())
            print(f"[Dataset] {scale}mer - input_ids range: min={min_id}, max={max_id}, vocab_size={vocab_size}")
            # 确保input_ids不超出vocab_size
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

# ================== Roberta模型 ==================
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
        跨模态注意力：query_modal关注key_value_modal
        query_modal: [B, L_q, C]
        key_value_modal: [B, L_kv, C]
        attention_mask: [B, L_q, L_kv] 可选
        """
        B, L_q, C = query_modal.shape
        _, L_kv, _ = key_value_modal.shape
        
        # 多头投影
        q = self.q_proj(query_modal).reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, D]
        k = self.k_proj(key_value_modal).reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D]
        v = self.v_proj(key_value_modal).reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D]
        
        # 注意力计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_kv]
        
        # 应用注意力掩码（如果有）
        if attention_mask is not None:
            # 拓展掩码以适配多头维度
            expanded_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(expanded_mask == 0, float('-inf'))
        
        # 注意力分布
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 加权求和
        context = torch.matmul(attn_probs, v)  # [B, H, L_q, D]
        context = context.transpose(1, 2).reshape(B, L_q, C)  # [B, L_q, C]
        
        # 输出投影
        output = self.out_proj(context)
        
        return output, attn_probs

class StructuralAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 结构感知注意力的投影层
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # 结构信息融合模块
        self.structure_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # 输出投影层
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_modal, key_value_modal, attention_mask=None):
        """
        结构感知注意力：融合序列的局部结构信息
        query_modal: [B, L_q, C]
        key_value_modal: [B, L_kv, C] - 可以是query_modal本身或其他上下文
        attention_mask: [B, L_q, L_kv] 可选
        """
        B, L_q, C = query_modal.shape
        _, L_kv, _ = key_value_modal.shape
        
        # 多头投影
        q = self.q_proj(query_modal).reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, D]
        k = self.k_proj(key_value_modal).reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D]
        v = self.v_proj(key_value_modal).reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D]
        
        # 结构感知注意力计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_kv]
        
        # 应用注意力掩码（如果有）
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(expanded_mask == 0, float('-inf'))
        
        # 注意力分布
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 加权求和
        context = torch.matmul(attn_probs, v)  # [B, H, L_q, D]
        context = context.transpose(1, 2).reshape(B, L_q, C)  # [B, L_q, C]
        
        # 结构门控融合 - 增强局部序列结构理解
        if query_modal is key_value_modal or key_value_modal is None:
            # 自注意力模式：使用序列自身结构
            gate = self.structure_gate(torch.cat([query_modal, context], dim=-1))
            context = gate * context + (1 - gate) * query_modal
        
        # 输出投影
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
        # 如果没有跨模态上下文，就使用自注意力
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
        # 检查 input_ids 是否越界
        max_id = input_ids.max().item()
        min_id = input_ids.min().item()
        vocab_size = self.embedding.num_embeddings
        
        # 安全处理：如果发现越界，裁剪input_ids
        if max_id >= vocab_size:
            print(f"[WARNING] input_ids.max()={max_id} >= vocab_size={vocab_size}, clipping to vocab_size-1")
            # 创建裁剪后的input_ids副本，避免修改原始数据
            safe_input_ids = input_ids.clone()
            # 将所有超出vocab_size的ID修改为<UNK>或最后一个有效ID
            safe_input_ids = torch.clamp(safe_input_ids, 0, vocab_size-1)
            input_ids = safe_input_ids
            # 更新最大值供调试
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
        return x, all_attn_weights  # 返回每层的结构感知注意力权重

class MultiScaleModel(nn.Module):
    def __init__(self, kmer_scales, num_layers, hidden_size, out_dim, num_attention_heads, class_weights=None, focal_loss=None, anno_dim=768, tokenizers=None, is_teacher=False):
        super().__init__()
        self.kmer_scales = kmer_scales
        self.is_teacher = is_teacher  # 新增标志，用于区分教师模型和学生模型
        self.backbones = nn.ModuleList()
        for i, scale in enumerate(kmer_scales):
            tokenizer = tokenizers[i] if tokenizers is not None else None
            if tokenizer is not None:
                vocab_size = len(tokenizer.get_vocab())
            else:
                vocab_size = 30522
            # DEBUG: 打印vocab_size和input_ids的最大最小值
            try:
                # 取样一个batch的input_ids
                sample_input_ids = None
                if hasattr(tokenizer, 'encode'):
                    # 尝试用tokenizer编码一个典型kmer
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
        
        # 特征交互层
        self.interaction = nn.Sequential(
            nn.Linear(out_dim * len(kmer_scales), out_dim * len(kmer_scales)),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(out_dim * len(kmer_scales), out_dim * len(kmer_scales))
        )
        
        # 分类器
        classifier_in_dim = out_dim * len(kmer_scales)
        if CONFIG["use_multimodal"] and anno_dim > 0:
            classifier_in_dim += anno_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, out_dim * len(kmer_scales) // 2),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(out_dim * len(kmer_scales) // 2, CONFIG["num_classes"])
        )
        
        # 融合层（MLP+残差）和门控机制
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
        
        if self.use_structure_attention:
            if self.is_teacher:
                # 教师模型使用结构感知注意力
                print("[INFO] Teacher model using StructuralAttention")
                self.structure_attentions = nn.ModuleList([
                    StructuralAttention(hidden_size, num_heads=4, dropout=CONFIG["dropout"])
                    for _ in kmer_scales
                ])
            else:
                # 学生模型使用跨模态注意力
                print("[INFO] Student model using CrossModalAttention")
                self.structure_attentions = nn.ModuleList([
                    CrossModalAttention(hidden_size, num_heads=4, dropout=CONFIG["dropout"])
                    for _ in kmer_scales
                ])

    def forward(self, batch, return_attentions=False):
        features = []
        all_attentions = []
        hidden_states = []
        
        # 首先收集所有尺度的初始表示
        initial_states = []
        for i, scale in enumerate(self.kmer_scales):
            input_ids = batch[f'input_ids_{scale}']
            embedded = self.backbones[i].embedding(input_ids)
            initial_states.append(embedded)
        
        # 然后对每个尺度应用跨模态注意力
        for i, scale in enumerate(self.kmer_scales):
            # 获取当前尺度的嵌入
            current_state = initial_states[i]
            
            # 如果启用了跨模态注意力，将所有其他尺度的信息融合到当前尺度
            if self.use_structure_attention:
                # 创建跨模态上下文，可以是其他尺度的平均或拼接
                other_states = [s for j, s in enumerate(initial_states) if j != i]
                if other_states:
                    # 这里使用简单平均，也可以用更复杂的方法
                    cross_modal_context = torch.stack(other_states).mean(0)
                    # 应用跨模态注意力
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
        
        # 特征融合
        weights = F.softmax(torch.ones(len(self.kmer_scales)), dim=0)
        concat = torch.cat([w * f for w, f in zip(weights, features)], dim=-1)
        
        # 应用最终的特征处理
        x = self.final_norm(concat)
        x = self.interaction(x)
        if CONFIG["use_multimodal"] and 'anno_emb' in batch:
            anno_emb = batch['anno_emb'].to(x.device)
            if anno_emb.dim() == 3 and anno_emb.shape[1] == 1:
                anno_emb = anno_emb.squeeze(1)
            assert x.shape[0] == anno_emb.shape[0], f"Batch size mismatch: {x.shape[0]} vs {anno_emb.shape[0]}"
            if x.shape[1] + anno_emb.shape[1] != self.classifier[0].in_features:
                raise ValueError(f"拼接后维度({x.shape[1]}+{anno_emb.shape[1]})与分类器输入({self.classifier[0].in_features})不符")
            x = torch.cat([x, anno_emb], dim=-1)
            fusion_out = self.fusion_layer(x)
            gate = self.gate_layer(x)
            x = gate * x + (1 - gate) * fusion_out
        
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
                # 由于StructuralAttention和CrossModalAttention接口一致，可以直接调用
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
            if x.shape[1] + anno_emb.shape[1] != self.classifier[0].in_features:
                raise ValueError(f"拼接后维度({x.shape[1]}+{anno_emb.shape[1]})与分类器输入({self.classifier[0].in_features})不符")
            x = torch.cat([x, anno_emb], dim=-1)
            fusion_out = self.fusion_layer(x)
            gate = self.gate_layer(x)
            x = gate * x + (1 - gate) * fusion_out
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

# ================== 训练与蒸馏主循环 ==================
class DistillTrainer:
    def __init__(self, stage="teacher", train_df=None, val_df=None):
        self.stage = stage
        self.train_df = train_df
        self.val_df = val_df

        # 统一计算类别权重和 focal loss
        class_counts = Counter(self.train_df['label'])
        total = sum(class_counts.values())
        num_classes = CONFIG["num_classes"]
        self.class_weights = torch.tensor(
            [total / (num_classes * class_counts[i]) for i in range(num_classes)],
            dtype=torch.float
        ).to(DEVICE)
        self.focal_loss = FocalLoss(gamma=2.5, alpha=self.class_weights)

        # 先加载数据集获取tokenizers
        print("初始化训练和验证数据集...")
        self.train_dataset = MultiScaleDNADataset(self.train_df, CONFIG["kmer_scales"])
        self.val_dataset = MultiScaleDNADataset(self.val_df, CONFIG["kmer_scales"])
        
        # 打印tokenizer的vocab_size，确认正确加载
        tokenizers = self.train_dataset.tokenizers
        for i, scale in enumerate(CONFIG["kmer_scales"]):
            if i < len(tokenizers):
                tok = tokenizers[i]
                vocab_size = len(tok.get_vocab()) if hasattr(tok, 'get_vocab') else -1
                print(f"[Trainer] {scale}mer tokenizer vocab_size = {vocab_size}")
        
        # 配置数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            shuffle=False
        )
        
        print(f"初始化{stage}模型...")
        if stage == "teacher":
            self.model = MultiScaleModel(
                kmer_scales=[4,5,6,7],
                num_layers=4,  # 或根据你的配置
                hidden_size=384,
                out_dim=192,   # 统一为192
                num_attention_heads=4,
                anno_dim=0,    # teacher 阶段不拼接 anno_emb
                class_weights=None,  # 可选
                focal_loss=None,     # 可选
                tokenizers=tokenizers,
                is_teacher=True      # 指定是教师模型
            ).to(DEVICE)
            # 加载自监督预训练权重
            pretrained_path = "pretrained_multiscale_bert.pt"
            if os.path.exists(pretrained_path):
                print(f"Loading pretrained weights from {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)  # 增加weights_only=True避免警告
                def is_compatible_key(k):
                    # 只保留backbones相关参数
                    return k.startswith("backbones")
                filtered_state_dict = {k: v for k, v in state_dict.items() if is_compatible_key(k)}
                self.model.load_state_dict(filtered_state_dict, strict=False)
                reset_classifier(self.model)
            else:
                print(f"Pretrained weights not found at {pretrained_path}, using random init.")
        else:
            # 1. 首先加载教师模型
            self.teacher = MultiScaleModel(
                kmer_scales=[4,5,6,7],
                num_layers=4,  # 或根据你的配置
                hidden_size=384,
                out_dim=192,   # 统一为192
                num_attention_heads=4,
                anno_dim=768,  # BERT embedding 维度
                class_weights=None,  # 可选
                focal_loss=None,     # 可选
                tokenizers=tokenizers,
                is_teacher=True      # 指定是教师模型
            ).to(DEVICE)
            
            # 2. 加载教师模型的权重
            state_dict = torch.load(CONFIG["teacher_path"], map_location=DEVICE, weights_only=True)  # 增加weights_only=True避免警告
            def is_compatible_key(k):
                # 只保留backbones相关参数
                return k.startswith("backbones")
            filtered_state_dict = {k: v for k, v in state_dict.items() if is_compatible_key(k)}
            self.teacher.load_state_dict(filtered_state_dict, strict=False)
            self.teacher.eval()  # 设置为评估模式
            
            # 3. 创建学生模型（使用随机初始化）
            self.model = MultiScaleModel(
                kmer_scales=[4,5,6,7],
                num_layers=2,  # 或根据你的配置
                hidden_size=256,
                out_dim=192,   # 统一为192
                num_attention_heads=4,
                anno_dim=768,  # BERT embedding 维度
                class_weights=self.class_weights,
                focal_loss=self.focal_loss,
                tokenizers=tokenizers,
                is_teacher=False     # 指定是学生模型
            ).to(DEVICE)
            
            # 4. 创建隐藏层适配器
            self.hidden_adapter = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(192, 192),  # student_out_dim, teacher_out_dim
                    nn.ReLU(),
                    nn.Linear(192, 192)
                ).to(DEVICE) for _ in [4,5,6,7]
            ])
            
            # 不需要加载权重，使用默认的随机初始化
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=CONFIG["num_epochs"] * len(self.train_loader)
        )

        # 在DistillTrainer.__init__里加
        self.f1_history = []
        self.precision_history = []

    def calculate_distillation_loss(self, student_logits, teacher_logits):
        """
        计算蒸馏损失
        Args:
            student_logits: 学生模型的输出logits
            teacher_logits: 教师模型的输出logits
        Returns:
            蒸馏损失
        """
        T = CONFIG["temperature"]  # 温度参数
        alpha = CONFIG["alpha"]    # 平衡参数
        
        # 软目标损失
        soft_targets = F.softmax(teacher_logits / T, dim=-1)
        soft_prob = F.log_softmax(student_logits / T, dim=-1)
        soft_targets = soft_targets.detach()  # 停止梯度
        distill_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T * T)
        
        return distill_loss

    def train_teacher(self):
        for epoch in range(CONFIG["num_epochs"]):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Teacher Epoch {epoch+1}")
            for batch in pbar:
                batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = self.model(batch)
                
                # 主任务损失
                loss = outputs["loss"]
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 对抗训练
                if CONFIG.get("enable_adversarial_defense", False):
                    if CONFIG.get("enable_pgd_adversarial", False):
                        epsilon = CONFIG.get("adversarial_epsilon", 0.1)
                        alpha = 0.01  # 注意：alpha 变量需要定义
                        # 获取当前scale对应的权重
                        for i, scale in enumerate(CONFIG["kmer_scales"]):
                            current_scale = CONFIG["kmer_scales"][i]
                            scale_weight = CONFIG.get("adversarial_scale_weights", {}).get(current_scale, 1.0)
                            adjusted_epsilon = epsilon * scale_weight
                            adjusted_alpha = alpha * scale_weight
                            # 1. 获取embedding
                            input_ids = batch[f'input_ids_{scale}']
                            attention_mask = batch[f'attention_mask_{scale}']
                            embedding_layer = self.model.backbones[i].embedding
                            embeddings = embedding_layer(input_ids)
                            embeddings = embeddings.detach().clone().requires_grad_(True)
                            # 2. PGD扰动
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
                            # 3. 再次前向传播
                            outputs_adv2 = self.model.forward_with_embeddings(batch, embeddings_dict)
                            loss_adv2 = outputs_adv2["loss"]
                            self.optimizer.zero_grad()
                            loss_adv2.backward()
                            self.optimizer.step()
                
                # 更新进度条信息
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
                
                # 获取教师模型输出（不需要梯度）
                with torch.no_grad():
                    teacher_outputs = self.teacher(batch, return_attentions=True)
                    teacher_logits = teacher_outputs["logits"]
                    teacher_attentions = teacher_outputs["attentions"]
                
                # 获取学生模型输出
                student_outputs = self.model(batch, return_attentions=True)
                student_logits = student_outputs["logits"]
                student_attentions = student_outputs["attentions"]
                
                # 1. 硬标签损失（交叉熵）
                if self.focal_loss is not None:
                    ce_loss = self.focal_loss(student_logits, batch['labels'])
                else:
                    ce_loss = F.cross_entropy(student_logits, batch['labels'], weight=self.class_weights)
                
                # 2. 知识蒸馏损失（soft label）
                kd_loss = self.calculate_distillation_loss(student_logits, teacher_logits)
                
                # 3. 特征蒸馏损失
                feat_loss = 0
                for i, (s_feat, t_feat) in enumerate(zip(student_outputs.get("features", []), teacher_outputs.get("features", []))):
                    s_feat_adapted = self.hidden_adapter[i](s_feat)
                    feat_loss += self.calculate_distillation_loss(s_feat_adapted, t_feat)
                
                # 4. 注意力蒸馏损失
                attn_loss = 0
                if student_attentions and teacher_attentions:
                    for s_attn, t_attn in zip(student_attentions, teacher_attentions):
                        attn_loss += self.calculate_distillation_loss(s_attn[-1], t_attn[-1])  # 使用最后一层的注意力
                
                # 总损失
                loss = (CONFIG["alpha"] * ce_loss + 
                       (1 - CONFIG["alpha"]) * CONFIG["kd_loss_weight"] * kd_loss +
                       CONFIG["feat_loss_weight"] * feat_loss +
                       CONFIG["attn_loss_weight"] * attn_loss)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                # 记录损失
                total_loss += loss.item()
                total_kd_loss += kd_loss.item()
                total_ce_loss += ce_loss.item()
                total_feat_loss += feat_loss.item()
                total_attn_loss += attn_loss.item()
                
                # 更新进度条
                avg_loss = total_loss / (pbar.n + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'kd_loss': f'{total_kd_loss/(pbar.n + 1):.4f}',
                    'ce_loss': f'{total_ce_loss/(pbar.n + 1):.4f}'
                })
            
            # 评估
            metrics = self.evaluate(epoch)
            current_f1 = metrics['macro_f1']
            current_precision = metrics['macro_precision']
            self.f1_history.append(current_f1)
            self.precision_history.append(current_precision)
            
            # 学习率调整
            self.scheduler.step()
            
            # 早停
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                print(f"New best F1: {best_val_f1:.4f}")
                torch.save(self.model.state_dict(), CONFIG["student_path"])
                abs_path = os.path.abspath(CONFIG["student_path"])
                print(f"[模型已保存] student.pt 路径: {abs_path}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            
            # 打印当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")

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
                print(f"[模型已保存] student.pt 路径: {abs_path}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")

    def evaluate(self, epoch, silent=False):
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
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
            print("整体评估结果:")
            print(f"ROC AUC (macro): {auc:.4f}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Macro Recall: {macro_recall:.4f}")
            print(f"Macro Precision: {macro_precision:.4f}")
            labels_unique, labels_counts = np.unique(all_labels, return_counts=True)
            print(f"标签分布: {(labels_unique, labels_counts)}")
            print(f"预测值范围: [{all_probs.min():.3f}, {all_probs.max():.3f}]")
            print("分类详细报告：")
            print(classification_report(all_labels, all_preds, digits=4))
        return {
            "acc": acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "auc": auc,
            "auc_pr": auc_pr
        }

    def run(self):
        if self.stage == "teacher":
            self.train_teacher()
        elif self.stage == "student_nodistill":
            self.train_student_nodistill()
        else:
            self.train_student()

# ========== 将 attention_distillation_loss 和 objective 移到这里 ==========
def attention_distillation_loss(student_attn, teacher_attn, T=1.0):
    # student_attn, teacher_attn: [batch, heads, seq, seq]
    loss = F.kl_div(
        F.log_softmax(student_attn / T, dim=-1),
        F.softmax(teacher_attn / T, dim=-1),
        reduction='batchmean'
    ) * (T * T)
    return loss

def objective(trial):
    # 搜索的超参数
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    kd_loss_weight = trial.suggest_float("kd_loss_weight", 0.5, 2.0)
    feat_loss_weight = trial.suggest_float("feat_loss_weight", 0.01, 0.2)
    attn_loss_weight = trial.suggest_float("attn_loss_weight", 0.5, 3.0)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # 修改CONFIG
    CONFIG["alpha"] = alpha
    CONFIG["kd_loss_weight"] = kd_loss_weight
    CONFIG["feat_loss_weight"] = feat_loss_weight
    CONFIG["attn_loss_weight"] = attn_loss_weight
    CONFIG["learning_rate"] = learning_rate
    CONFIG["batch_size"] = batch_size

    # 确保 teacher 权重存在
    CONFIG["teacher_path"] = "./teacher.pt"  # 或你的 teacher 路径
    if not os.path.exists(CONFIG["teacher_path"]):
        raise FileNotFoundError(f"Teacher model not found at {CONFIG['teacher_path']}. Please train the teacher model first.")

    # 重新加载数据
    train_df = pd.read_csv("C:/Users/Raytrack/Desktop/dataset_balanced_train.csv")
    val_df = pd.read_csv("C:/Users/Raytrack/Desktop/dataset_balanced_val.csv")
    anno_embs_train = np.load("C:/Users/Raytrack/Desktop/anno_embs_train.npy")
    anno_embs_val = np.load("C:/Users/Raytrack/Desktop/anno_embs_val.npy")

    train_dataset = MultiScaleDNADataset(train_df, kmer_scales=CONFIG["kmer_scales"], anno_embs=anno_embs_train)
    val_dataset = MultiScaleDNADataset(val_df, kmer_scales=CONFIG["kmer_scales"], anno_embs=anno_embs_val)

    trainer = DistillTrainer(
        stage="student",
        train_df=train_df,
        val_df=val_df
    )
    trainer.train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    trainer.val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # 只训练1个epoch加快搜索
    CONFIG["num_epochs"] = 1

    trainer.train_student()
    metrics = trainer.evaluate(epoch=-1, silent=True)
    del trainer, train_dataset, val_dataset, train_df, val_df
    gc.collect()
    torch.cuda.empty_cache()
    return metrics["macro_f1"]

# ================== main入口 ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["teacher", "student", "student_nodistill"], required=True)
    parser.add_argument("--n_trials", type=int, default=5, help="Optuna trials per fold")
    parser.add_argument("--optuna", action="store_true", help="Use Optuna for hyperparameter search")
    parser.add_argument("--multimodal", action="store_true", help="Enable multimodal text annotations")
    args = parser.parse_args()
    CONFIG["use_multimodal"] = args.multimodal
    train_df = pd.read_csv("C:/Users/Raytrack/Desktop/dataset_balanced_train.csv")
    val_df = pd.read_csv("C:/Users/Raytrack/Desktop/dataset_balanced_val.csv")
    test_df = pd.read_csv("C:/Users/Raytrack/Desktop/dataset_balanced_test.csv")
    if CONFIG["use_multimodal"]:
        anno_embs_train = np.load("C:/Users/Raytrack/Desktop/anno_embs_train.npy")
        anno_embs_val = np.load("C:/Users/Raytrack/Desktop/anno_embs_val.npy")
        anno_embs_test = np.load("C:/Users/Raytrack/Desktop/anno_embs_test.npy")
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
    trainer.run()
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    trainer.model.eval()
    all_labels, all_preds = [], []
