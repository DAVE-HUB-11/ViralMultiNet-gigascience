#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tokenizers.py

根据多尺度 k-mer 词表（如 vocab_6mer.txt）构造 DNABERT-2 风格的
Tokenizer（WordLevel + Whitespace 分词 + BERT 后处理），并导出为
HuggingFace PreTrainedTokenizerFast 可加载的 json 格式。
"""

import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

# —— 配置区 —— #
K = 6
VOCAB_TXT = rf"C:\Users\10785\Desktop\vocab_{K}mer.txt"
OUT_DIR   = rf"C:\Users\10785\Desktop\tokenizer_{K}mer"
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
# ———————— #

# 1. 读入 vocab_{k}mer.txt
with open(VOCAB_TXT, "r", encoding="utf-8") as f:
    tokens = [line.strip() for line in f if line.strip()]

# 2. 构建 id->token 和 token->id 映射
#    我们将保留前 5 个 id 给 special tokens
vocab = {}
# special tokens
for i, sp in enumerate(SPECIAL_TOKENS):
    vocab[sp] = i
# k-mer tokens
for idx, tk in enumerate(tokens, start=len(SPECIAL_TOKENS)):
    vocab[tk] = idx

# 3. 创建 WordLevel 模型并封装在 Tokenizer
tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 4. 设置 BERT-style 后处理：添加 [CLS]、[SEP]
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair  ="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[
        ("[CLS]", vocab["[CLS]"]),
        ("[SEP]", vocab["[SEP]"]),
    ],
)

# 5. 保存原生 tokenizer
tokenizer_json = f"{OUT_DIR}/tokenizer.json"
tokenizer.save(tokenizer_json)
print(f"✅ 原生 tokenizer 保存到：{tokenizer_json}")

# 6. 用 Transformers 的 PreTrainedTokenizerFast 包装，方便后续直接 from_pretrained 加载
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tokenizer_json,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# 7. 保存为 HuggingFace 格式（词表和配置）
hf_tokenizer.save_pretrained(OUT_DIR)
print(f"✅ HuggingFace Tokenizer 配置保存到：{OUT_DIR}")
