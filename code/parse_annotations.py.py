#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 DNABERT-2 微调数据集（优化版）：
- 新增k-mer编码、序列截断、链特异性处理
- 增强数据清洗和错误处理
"""

import pandas as pd
from Bio import SeqIO
import re
from tqdm import tqdm  # 进度条支持

# ———— 配置参数 ————
K_MER_SIZE = 6  # 适配DNABERT-2的k-mer大小
MAX_SEQ_LENGTH = 512  # 最大序列长度（token数）
STRAND_AWARE = False  # 是否考虑链特异性（True时负链取反向互补）

# ———— 文件路径 ————
MATCHES_CSV = r"C:\Users\10785\Desktop\matches.csv"
ANNOT_XLSX = r"C:\Users\10785\Desktop\annotations.csv.xlsx"
FNA_FILE = r"C:\Users\10785\Desktop\sra数据\genes.fna"
OUTPUT_CSV = r"C:\Users\10785\Desktop\dnabert_dataset.csv"


def dna_to_kmers(sequence: str, k: int = K_MER_SIZE) -> str:
    """将DNA序列转换为k-mer分词字符串"""
    # 清洗序列：去除非ATCGN字符并转大写
    seq_clean = re.sub(r'[^ATCGN]', '', sequence.upper())
    # 生成k-mer列表
    kmers = [seq_clean[i:i + k] for i in range(len(seq_clean) - k + 1)]
    return ' '.join(kmers)


def process_sequence(raw_seq: str) -> str:
    """序列处理流水线：清洗->k-mer->截断"""
    # 链特异性处理（需要GFF信息时可扩展）
    if STRAND_AWARE and is_negative_strand(orf_id):  # 需实现is_negative_strand
        raw_seq = str(Seq(raw_seq).reverse_complement())

    # k-mer编码
    kmer_seq = dna_to_kmers(raw_seq)

    # 长度截断
    tokens = kmer_seq.split()
    if len(tokens) > MAX_SEQ_LENGTH:
        tokens = tokens[:MAX_SEQ_LENGTH]
    return ' '.join(tokens)


# ———— Step 1: 读取 matches.csv ————
print("🔄 读取 matches.csv ...")
matches_df = pd.read_csv(
    MATCHES_CSV,
    header=0,
    usecols=[0, 1],
    dtype={"subject_id": str}
).rename(columns={"query_id": "orf_id"})
print(f"✅ 已读取 {len(matches_df):,} 条 ORF→UniProt 映射")

# ———— Step 2: 读取注释表并打标签 ————
print("\n🔄 读取注释表 ...")
df = pd.read_excel(ANNOT_XLSX, engine="openpyxl")
print(df.columns)
anno_df = pd.read_excel(ANNOT_XLSX, engine="openpyxl")[["ids", "function"]]
anno_df = anno_df.dropna(subset=["ids", "function"]).astype({"ids": str})
anno_df = anno_df.rename(columns={"ids": "subject_id", "function": "annotation"})

# 增强标签映射（支持中英文混合标注）
LABEL_MAP = {
    0: ['酶', 'enzyme'],
    1: ['结构蛋白', 'structural', 'capsid', 'spike'],
    2: ['转运蛋白', 'transporter'],
    3: ['其他']  # 默认类别
}


def map_annotation_to_label(text: str) -> int:
    text = str(text).lower()
    for label, keywords in LABEL_MAP.items():
        if any(kw in text for kw in keywords):
            return label
    return 3  # 默认类别


anno_df["label"] = anno_df["annotation"].apply(map_annotation_to_label)
print(f"✅ 注释表共包含 {len(anno_df):,} 条有效注释")

# ———— Step 3: 合并 ORF→label 映射 ————
print("\n🔄 合并 ORF_ID 和标签 ...")
orf2label = matches_df.merge(
    anno_df,
    on="subject_id",
    how="inner"
)[["orf_id", "subject_id", "annotation", "label"]]
print(f"✅ 合并后得到 {len(orf2label):,} 条有标签 ORF (去重前)")

# 按ORF_ID去重（保留最后出现的标签）
orf2label = orf2label.drop_duplicates(subset=["orf_id"], keep='last')
print(f"✅ 去重后有效 ORF 数量: {len(orf2label):,}")

# ———— Step 4: 加载并预处理基因序列 ————
print(f"\n🔄 加载并预处理基因序列 (k={K_MER_SIZE})...")
fna_dict = {}
seen_ids = set()
skipped = 0

for record in tqdm(SeqIO.parse(FNA_FILE, "fasta"), desc="Processing Genes"):
    seqid = record.id
    if seqid in seen_ids:
        skipped += 1
        continue
    seen_ids.add(seqid)

    # 序列预处理
    raw_seq = str(record.seq)
    processed_seq = process_sequence(raw_seq)

    # 过滤无效序列
    if len(processed_seq.replace(' ', '')) < K_MER_SIZE:  # 短于k-mer的序列
        skipped += 1
        continue

    fna_dict[seqid] = processed_seq

print(f"✅ 成功加载 {len(fna_dict):,} 条唯一序列 | 跳过 {skipped} 条无效序列")

# ———— Step 5: 构建最终数据集 ————
print("\n🔄 构建数据集...")
sequences, labels, orf_ids, subject_ids, annotations = [], [], [], [], []
missing_count = 0

for _, row in tqdm(orf2label.iterrows(), total=len(orf2label), desc="Matching Sequences"):
    oid, sid, anno, lbl = row["orf_id"], row["subject_id"], row["annotation"], row["label"]
    if oid not in fna_dict:
        missing_count += 1
        continue
    sequences.append(fna_dict[oid])
    labels.append(lbl)
    orf_ids.append(oid)
    subject_ids.append(sid)
    annotations.append(anno)

print(f"⚠️  {missing_count} 条 ORF 缺少对应序列")
print(f"✅ 最终数据集包含 {len(sequences):,} 条样本")

# ———— Step 6: 保存数据集 ————
print("\n💾 写入文件...")
out_df = pd.DataFrame({
    "orf_id": orf_ids,
    "subject_id": subject_ids,
    "annotation": annotations,
    "text": sequences,
    "label": labels
})

# 添加统计信息
out_df['seq_length'] = out_df['text'].apply(lambda x: len(x.split()))
length_stats = out_df['seq_length'].describe()

out_df.to_csv(OUTPUT_CSV, index=False)
print(f"🎉 完成！文件已保存到：{OUTPUT_CSV}")
print("\n📊 序列长度统计：")
print(f"平均长度: {length_stats['mean']:.1f} ± {length_stats['std']:.1f} tokens")
print(f"最小值: {length_stats['min']} | 中位数: {length_stats['50%']} | 最大值: {length_stats['max']}")