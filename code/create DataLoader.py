#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataloader.py

构建 PyTorch Dataset + DataLoader：
- 从 dnabert_dataset_multiscale.csv 读取多尺度分词列（默认用 6-mer）
- 加载对应的 Tokenizer（如 tokenizer_6mer）
- 返回 batch {"input_ids","attention_mask","labels"}
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, DataCollatorWithPadding

# ———— 配置 ————
CSV_FILE     = r"C:\Users\10785\Desktop\dnabert_dataset_multiscale.csv"  # 更新为你的文件路径
K            = 6  # 选择 k-mer 尺度，默认使用 6-mer
COL_TEXT     = f"text_{K}mer"
COL_LABEL    = "label"
TOKENIZER_DIR= rf"C:\Users\10785\Desktop\tokenizer_{K}mer"  # 更新为你的 tokenizer 文件路径
BATCH_SIZE   = 32
MAX_LENGTH   = 512  # 跟微调时保持一致
SHUFFLE      = True
NUM_WORKERS  = 4
# ———————— #

class DNABertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self.texts[idx]
        lbl = int(self.labels[idx])
        # 分词
        encoded = self.tokenizer(
            seq,
            truncation=True,
            padding=False,  # 让 DataCollator 统一 padding
            max_length=self.max_length,
            return_attention_mask=True,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(lbl, dtype=torch.long),
        }

def collate_fn(batch):
    """
    使用 HuggingFace 的 DataCollatorWithPadding 简化 padding
    """
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    return collator(batch)

if __name__ == "__main__":
    # 1. 读取多尺度 CSV
    df = pd.read_csv(CSV_FILE, usecols=[COL_TEXT, COL_LABEL])
    print(f"📊 共加载 {len(df):,} 条样本，示例：")
    print(df.head())

    # 2. 加载对应 tokenizer
    if not os.path.isdir(TOKENIZER_DIR):
        raise FileNotFoundError(f"找不到 Tokenizer 目录：{TOKENIZER_DIR}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    print(f"🛠 已加载 Tokenizer：{TOKENIZER_DIR}")

    # 3. 构建 Dataset
    dataset = DNABertDataset(
        texts=df[COL_TEXT].tolist(),
        labels=df[COL_LABEL].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    # 4. 构建 DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        collate_fn=data_collator
    )

    # 5. 简单测试一个 batch
    batch = next(iter(loader))
    print("\n▶ 测试一个 batch 输出：")
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)

    # 6. 暴露 loader 和 dataset 供后续使用
    #    你可以在训练脚本里这样做：
    #    from build_dataloader import loader as train_loader
