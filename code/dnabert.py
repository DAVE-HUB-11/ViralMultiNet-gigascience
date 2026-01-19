import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertModel,
    BertConfig,
    get_cosine_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from tqdm.auto import tqdm
import gc
import psutil
import argparse
from pathlib import Path
import math
import json
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext
from imblearn.combine import SMOTETomek
from collections import Counter
import random
import re
from sklearn.model_selection import StratifiedShuffleSplit, KFold

# ================== 硬件配置 ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# Helper function for autocast
def get_autocast():
    return torch.amp.autocast(
        device_type='cuda' if torch.cuda.is_available() else 'cpu',
        enabled=CONFIG["fp16"]
    )


# ================== 全局参数 ==================
CONFIG = {
    "train_csv": r"C:\\Users\\10785\\Desktop\\dataset_balanced_train.csv",
    "val_csv": r"C:\\Users\\10785\\Desktop\\dataset_balanced_val.csv",
    "test_csv": r"C:\\Users\\10785\\Desktop\\dataset_balanced_test.csv",
    "kmer_scales": [4, 5, 6],
    "max_length": 256,
    "batch_size": 32,
    "num_workers": 0,
    "num_epochs": 8,
    "learning_rate": 5e-5,
    "temperature": 2.0,
    "alpha": 0.7,
    "teacher_path": "./teacher.pt",
    "student_path": "./student.pt",
    "best_student_path": "./best_student.pt",
    "contrastive_temp": 0.07,
    "adv_epsilon": 0.01,
    "defense": {
        "enable_adversarial": True,
        "fgsm_epsilon": 0.1,
        "pgd_epsilon": 0.1,
        "pgd_steps": 3,
        "pgd_alpha": 0.01,
        "mixup_alpha": 0.2,
        "enable_mixup": True,
        "label_smoothing": 0.1,
    },
    "gradient_accumulation_steps": 4,
    "fp16": True,
    "memory_efficient": False,
    "chunk_size": 128,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "dropout": 0.2,
    "hidden_size": {
        "teacher": 768,    # 修改为768，这样可以被12整除
        "student": 384     # 修改为384，这样可以被8整除
    },
    "attention_window": 128,
    "gradient_checkpointing": False,
    "num_classes": 4,
    "structure_aware": {
        "num_heads": 12,
        "structure_dim": 96,
        "dropout": 0.1,
        "max_relative_position": 32,
        "use_structure_attention": True
    },
    "loss_type": "focal",
    "focal": {"gamma": 2.5, "alpha": 0.3},
    "attn_distill_weight": 1.0,
    "eda": {
        "alpha_sr": 0.15,    # 提高同义词替换概率
        "alpha_ri": 0.15,    # 提高随机插入概率
        "alpha_rs": 0.15,    # 提高随机交换概率
        "alpha_rd": 0.15,    # 提高随机删除概率
        "num_aug": 6         # 增加每个少数类样本的增强数量
    },
    "class_weights": True,  # 是否使用类别权重
    "focal_loss": {
        "gamma": 2.0,
        "alpha": None  # 将根据类别分布自动计算
    },
    "dnabert4_model": "C:/Users/Raytrack/Desktop/bert4",
    "dnabert4_tokenizer": "C:/Users/Raytrack/Desktop/bert4",
    "dnabert5_model": "C:/Users/Raytrack/Desktop/bert5",
    "dnabert5_tokenizer": "C:/Users/Raytrack/Desktop/bert5",
    "dnabert6_model": "C:/Users/Raytrack/Desktop/bert6",
    "dnabert6_tokenizer": "C:/Users/Raytrack/Desktop/bert6",
}

# 创建必要的目录
os.makedirs("runs", exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["teacher_path"]), exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["student_path"]), exist_ok=True)


def check_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


class MultiScaleDNADataset(Dataset):
    """多尺度DNA BERT数据集"""
    def __init__(self, df):
        self.kmer_scales = CONFIG["kmer_scales"]
        self.encodings = []
        self.tokenizers = []
        for scale in self.kmer_scales:
            tokenizer = BertTokenizer.from_pretrained(CONFIG[f"dnabert{scale}_tokenizer"], local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            texts = df[f"text_{scale}mer"].tolist()
            enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=CONFIG["max_length"],
                return_tensors="pt"
            )
            self.encodings.append(enc)
            self.tokenizers.append(tokenizer)
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {}
        for i, scale in enumerate(self.kmer_scales):
            item[f'input_ids_{scale}'] = self.encodings[i]['input_ids'][idx]
            item[f'attention_mask_{scale}'] = self.encodings[i]['attention_mask'][idx]
        item['labels'] = self.labels[idx]
        return item


class TanhNorm(nn.Module):
    """优化的TanhNorm层，用于替代LayerNorm"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self._initialized = False

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        self._initialized = True

    def forward(self, x):
        if not self._initialized:
            self.reset_parameters()

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (std + self.eps)
        return torch.tanh(x_norm) * self.weight + self.bias


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification"""

    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DNABERTMultiScaleClassifier(nn.Module):
    """多尺度DNA BERT分类模型"""
    def __init__(self):
        super().__init__()
        self.kmer_scales = CONFIG["kmer_scales"]
        self.berts = nn.ModuleList()
        out_dims = []
        for scale in self.kmer_scales:
            bert = BertModel.from_pretrained(CONFIG[f"dnabert{scale}_model"], local_files_only=True)
            self.berts.append(bert)
            out_dims.append(bert.config.hidden_size)
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.classifier = nn.Linear(sum(out_dims), CONFIG["num_classes"])
    def forward(self, batch):
        features = []
        for i, scale in enumerate(self.kmer_scales):
            input_ids = batch[f'input_ids_{scale}']
            attention_mask = batch[f'attention_mask_{scale}']
            outputs = self.berts[i](input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]
            features.append(pooled)
        concat = torch.cat(features, dim=-1)
        x = self.dropout(concat)
        logits = self.classifier(x)
        loss = None
        if 'labels' in batch and batch['labels'] is not None:
            if CONFIG.get("loss_type", "ce") == "focal":
                loss_fn = FocalLoss(CONFIG["focal"]["gamma"], CONFIG["focal"]["alpha"])
                loss = loss_fn(logits, batch['labels'])
            else:
                loss = F.cross_entropy(logits, batch['labels'])
        return {"loss": loss, "logits": logits} if loss is not None else logits


class TrainingPipeline:
    def __init__(self):
        self.writer = SummaryWriter(f"runs/dnabert4")
        self.checkpoint_dir = os.path.join("checkpoints", "dnabert4")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._init_components()
    def _init_components(self):
        check_memory()
        # 直接读取 train/val/test 三个文件
        self.train_df = pd.read_csv(CONFIG["train_csv"])
        self.val_df = pd.read_csv(CONFIG["val_csv"])
        self.test_df = pd.read_csv(CONFIG["test_csv"])
        print("\n训练集分布:")
        print("-" * 50)
        print(Counter(self.train_df['label']))
        print("-" * 50)
        print("验证集分布:")
        print("-" * 50)
        print(Counter(self.val_df['label']))
        print("-" * 50)
        print("测试集分布:")
        print("-" * 50)
        print(Counter(self.test_df['label']))
        print("-" * 50)
        # 计算类别权重（基于训练集）
        if CONFIG["class_weights"]:
            class_counts = Counter(self.train_df['label'])
            total = sum(class_counts.values())
            class_weights = {label: total / (len(class_counts) * count) 
                           for label, count in class_counts.items()}
            CONFIG["focal_loss"]["alpha"] = class_weights
            print("\n类别权重:")
            print("-" * 50)
            for label, weight in sorted(class_weights.items()):
                print(f"类别 {label}: {weight:.4f}")
            print("-" * 50)
        gc.collect()
        self.model = DNABERTMultiScaleClassifier().to(DEVICE)
        self.criterion = FocalLoss(
            gamma=CONFIG["focal"]["gamma"],
            alpha=CONFIG["focal"]["alpha"]
        ).to(DEVICE)
        if CONFIG["focal_loss"]["alpha"] is not None:
            self.criterion.alpha = CONFIG["focal_loss"]["alpha"]
        total_steps = (len(self.train_df) // CONFIG["batch_size"]) * CONFIG["num_epochs"]
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=CONFIG["warmup_steps"],
            num_training_steps=total_steps
        )
        self.scaler = torch.cuda.amp.GradScaler()
    def _create_dataloader(self, df, shuffle=True):
        dataset = MultiScaleDNADataset(df)
        return DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            pin_memory=False,
            shuffle=shuffle
        )
    def _train_epoch(self, epoch):
        self.model.train()
        accumulated_loss = 0
        optimizer_steps = 0
        train_loader = self._create_dataloader(self.train_df)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            # 确保所有tensor都在DEVICE上
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)
            is_optimizer_step = (batch_idx + 1) % CONFIG["gradient_accumulation_steps"] == 0
            with get_autocast():
                outputs = self.model(batch)
                loss = outputs["loss"] / CONFIG["gradient_accumulation_steps"]
            self.scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            if is_optimizer_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                current_lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=accumulated_loss / optimizer_steps,
                    lr=f"{current_lr:.2e}"
                )
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        check_memory()
        torch.cuda.empty_cache()
        gc.collect()
        return accumulated_loss / optimizer_steps
    def _evaluate(self):
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        val_loader = self._create_dataloader(self.val_df, shuffle=False)
        with torch.no_grad():
            for batch in val_loader:
                # 确保所有tensor都在DEVICE上
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(DEVICE)
                inputs = {
                    'input_ids_4': batch['input_ids_4'],
                    'attention_mask_4': batch['attention_mask_4'],
                    'input_ids_5': batch['input_ids_5'],
                    'attention_mask_5': batch['attention_mask_5'],
                    'input_ids_6': batch['input_ids_6'],
                    'attention_mask_6': batch['attention_mask_6']
                }
                labels = batch['labels'].cpu().numpy()
                with get_autocast():
                    outputs = self.model(batch)
                    logits = outputs["logits"]
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_probs.append(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels)
                torch.cuda.empty_cache()
        all_probs = np.vstack(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
            for i in range(CONFIG["num_classes"]):
                try:
                    binary_true = (all_labels == i).astype(int)
                    binary_pred = (all_preds == i).astype(int)
                    binary_prob = all_probs[:, i]
                    class_auc = roc_auc_score(binary_true, binary_prob)
                    class_precision = precision_score(binary_true, binary_pred)
                    class_recall = recall_score(binary_true, binary_pred)
                    class_f1 = f1_score(binary_true, binary_pred)
                    print(f"\n类别 {i} 的评估指标:")
                    print(f"ROC AUC: {class_auc:.4f}")
                    print(f"Precision: {class_precision:.4f}")
                    print(f"Recall: {class_recall:.4f}")
                    print(f"F1 Score: {class_f1:.4f}")
                except Exception as e:
                    print(f"类别 {i} 的指标计算失败: {str(e)}")
        except Exception as e:
            print(f"警告: 多分类评估指标计算失败 - {str(e)}")
            print(f"标签分布: {np.unique(all_labels, return_counts=True)}")
            print(f"预测值范围: [{all_probs.min():.3f}, {all_probs.max():.3f}]")
            auc = 0.5
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        macro_recall = recall_score(all_labels, all_preds, average='macro')
        macro_precision = precision_score(all_labels, all_preds, average='macro')
        print(f"\n整体评估结果:")
        print(f"ROC AUC (macro): {auc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"标签分布: {np.unique(all_labels, return_counts=True)}")
        print(f"预测值范围: [{all_probs.min():.3f}, {all_probs.max():.3f}]")
        self.writer.add_scalar("Val/F1", macro_f1, self.current_epoch)
        self.writer.add_scalar("Val/Recall", macro_recall, self.current_epoch)
        self.writer.add_scalar("Val/Precision", macro_precision, self.current_epoch)
        return auc, acc, macro_f1, macro_recall, macro_precision
    def _save_checkpoint(self, epoch, auc, acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'auc': auc,
            'acc': acc
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    def _test_evaluate(self):
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        test_loader = self._create_dataloader(self.test_df, shuffle=False)
        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(DEVICE)
                with get_autocast():
                    outputs = self.model(batch)
                    logits = outputs["logits"]
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_probs.append(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                torch.cuda.empty_cache()
        all_probs = np.vstack(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
            for i in range(CONFIG["num_classes"]):
                try:
                    binary_true = (all_labels == i).astype(int)
                    binary_pred = (all_preds == i).astype(int)
                    binary_prob = all_probs[:, i]
                    class_auc = roc_auc_score(binary_true, binary_prob)
                    class_precision = precision_score(binary_true, binary_pred)
                    class_recall = recall_score(binary_true, binary_pred)
                    class_f1 = f1_score(binary_true, binary_pred)
                    print(f"\n[TEST] 类别 {i} 的评估指标:")
                    print(f"ROC AUC: {class_auc:.4f}")
                    print(f"Precision: {class_precision:.4f}")
                    print(f"Recall: {class_recall:.4f}")
                    print(f"F1 Score: {class_f1:.4f}")
                except Exception as e:
                    print(f"[TEST] 类别 {i} 的指标计算失败: {str(e)}")
        except Exception as e:
            print(f"[TEST] 警告: 多分类评估指标计算失败 - {str(e)}")
            print(f"标签分布: {np.unique(all_labels, return_counts=True)}")
            print(f"预测值范围: [{all_probs.min():.3f}, {all_probs.max():.3f}]")
            auc = 0.5
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        macro_recall = recall_score(all_labels, all_preds, average='macro')
        macro_precision = precision_score(all_labels, all_preds, average='macro')
        print(f"\n[TEST] 整体评估结果:")
        print(f"ROC AUC (macro): {auc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"标签分布: {np.unique(all_labels, return_counts=True)}")
        print(f"预测值范围: [{all_probs.min():.3f}, {all_probs.max():.3f}]")
        return auc, acc, macro_f1, macro_recall, macro_precision
    def run(self):
        best_auc = 0
        best_f1 = 0
        self.current_epoch = 0
        try:
            for epoch in range(CONFIG["num_epochs"]):
                self.current_epoch = epoch
                avg_loss = self._train_epoch(epoch)
                auc, acc, f1, recall, precision = self._evaluate()
                self._save_checkpoint(epoch, auc, acc)
                if f1 > best_f1 or (f1 == best_f1 and auc > best_auc):
                    best_f1 = f1
                    best_auc = auc
                    torch.save(self.model.state_dict(), CONFIG["teacher_path"])
                print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")
                self.writer.add_scalar("Train/Loss", avg_loss, epoch)
                self.writer.add_scalar("Val/AUC", auc, epoch)
                self.writer.add_scalar("Val/Acc", acc, epoch)
                self.writer.add_scalar("Train/LR", self.scheduler.get_last_lr()[0], epoch)
        except RuntimeError as e:
            print(f"训练中断: {str(e)}")
            if "out of memory" in str(e):
                print("内存不足，建议进一步减小batch_size或模型大小")
        finally:
            self.writer.close()
        print("\n================ 测试集评估 ================")
        self._test_evaluate()
        return best_auc, best_f1


if __name__ == "__main__":
    try:
        if not os.path.exists(CONFIG["train_csv"]) or not os.path.exists(CONFIG["val_csv"]) or not os.path.exists(CONFIG["test_csv"]):
            raise FileNotFoundError(f"找不到数据文件: {CONFIG['train_csv']}, {CONFIG['val_csv']}, {CONFIG['test_csv']}")
        print("当前配置:")
        for k, v in CONFIG.items():
            print(f"{k}: {v}")
        check_memory()
        pipeline = TrainingPipeline()
        pipeline.run()
    except Exception as e:
        print(f"运行出错: {str(e)}")
        raise