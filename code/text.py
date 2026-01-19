import torch.nn as nn
from transformers import AutoModel
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
)
from collections import Counter
from pathlib import Path

class TanhNorm(nn.Module):
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

class MultiKmerBERTClassifier(nn.Module):
    def __init__(self, num_classes=4, model_names=None, dropout=0.2):
        super().__init__()
        assert model_names is not None and len(model_names) == 3
        self.bert_4 = AutoModel.from_pretrained(Path(model_names[0]), local_files_only=True)
        self.bert_5 = AutoModel.from_pretrained(Path(model_names[1]), local_files_only=True)
        self.bert_6 = AutoModel.from_pretrained(Path(model_names[2]), local_files_only=True)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.bert_4.config.hidden_size
        self.classifier = nn.Linear(hidden_size * 3, num_classes)

    def forward(self, input_ids_list, attention_mask_list, labels=None):
        pooled_outputs = []
        for bert, input_ids, attention_mask in zip(
            [self.bert_4, self.bert_5, self.bert_6],
            input_ids_list, attention_mask_list
        ):
            outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            pooled_outputs.append(pooled)
        concat = torch.cat(pooled_outputs, dim=1)
        logits = self.classifier(concat)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits} if loss is not None else logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_names = [
    r"C:\Users\10785\Desktop\bert_4mer",
    r"C:\Users\10785\Desktop\bert_5mer",
    r"C:\Users\10785\Desktop\bert_6mer"
]
tokenizer_4 = AutoTokenizer.from_pretrained(Path(r"C:\Users\10785\Desktop\bert_4mer"), local_files_only=True)
tokenizer_5 = AutoTokenizer.from_pretrained(model_names[1], local_files_only=True)
tokenizer_6 = AutoTokenizer.from_pretrained(model_names[2], local_files_only=True)

model = MultiKmerBERTClassifier(num_classes=4, model_names=model_names).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids_list = [x.to(device) for x in batch['input_ids_list']]
        attention_mask_list = [x.to(device) for x in batch['attention_mask_list']]
        labels = batch['labels'].to(device)
        outputs = model(input_ids_list, attention_mask_list, labels)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
# 保存教师模型
torch.save(model.state_dict(), "biobert_teacher_multi_kmer.pth")

student_model = MultiKmerBERTClassifier(num_classes=4).to(device)
student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=2e-5)
T = 2.0
alpha = 0.7

model.eval()
student_model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        student_optimizer.zero_grad()
        input_ids_list = [x.to(device) for x in batch['input_ids_list']]
        attention_mask_list = [x.to(device) for x in batch['attention_mask_list']]
        labels = batch['labels'].to(device)

        with torch.no_grad():
            teacher_logits = model(input_ids_list, attention_mask_list)['logits']

        student_outputs = student_model(input_ids_list, attention_mask_list)
        student_logits = student_outputs['logits']

        loss = distillation_loss(student_logits, teacher_logits, labels, T=T, alpha=alpha)
        loss.backward()
        student_optimizer.step()
# 保存学生模型
torch.save(student_model.state_dict(), "biobert_student_multi_kmer.pth")

df = pd.read_csv(r"C:\Users\10785\Desktop\dnabert_dataset_multiscale.csv")

class MultiKmerDataset(Dataset):
    def __init__(self, df, max_length=256, tokenizer_4=None, tokenizer_5=None, tokenizer_6=None):
        self.texts_4 = df["text_4mer"].tolist()
        self.texts_5 = df["text_5mer"].tolist()
        self.texts_6 = df["text_6mer"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer_4 = tokenizer_4
        self.tokenizer_5 = tokenizer_5
        self.tokenizer_6 = tokenizer_6
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encodings = []
        for text, tokenizer in zip(
            [self.texts_4[idx], self.texts_5[idx], self.texts_6[idx]],
            [self.tokenizer_4, self.tokenizer_5, self.tokenizer_6]
        ):
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            encodings.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            })
        return {
            'input_ids_list': [e['input_ids'] for e in encodings],
            'attention_mask_list': [e['attention_mask'] for e in encodings],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.7):
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss

def build_model(use_tanhnorm=False):
    model = MultiKmerBERTClassifier(num_classes=4, model_names=model_names).to(device)
    if use_tanhnorm:
        replace_layernorm_with_tanhnorm(model.bert_4)
        replace_layernorm_with_tanhnorm(model.bert_5)
        replace_layernorm_with_tanhnorm(model.bert_6)
    return model

model = build_model(use_tanhnorm=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids_list = [x.to(device) for x in batch['input_ids_list']]
        attention_mask_list = [x.to(device) for x in batch['attention_mask_list']]
        labels = batch['labels'].to(device)
        outputs = model(input_ids_list, attention_mask_list, labels)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
    # 可加验证集评估
torch.save(model.state_dict(), "multi_kmer_bert_model.pth")

model = build_model(use_tanhnorm=True)
model.load_state_dict(torch.load("multi_kmer_bert_model.pth"))
model.eval()

def evaluate_model(model, dataloader, device, num_classes=4):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_list = [x.to(device) for x in batch['input_ids_list']]
            attention_mask_list = [x.to(device) for x in batch['attention_mask_list']]
            labels = batch['labels'].cpu().numpy()
            outputs = model(input_ids_list, attention_mask_list)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 1. 每个类别的二分类AUC、Precision、Recall、F1
    print("=== Per-class metrics ===")
    for i in range(num_classes):
        binary_true = (all_labels == i).astype(int)
        binary_pred = (all_preds == i).astype(int)
        binary_prob = all_probs[:, i]
        try:
            class_auc = roc_auc_score(binary_true, binary_prob)
        except:
            class_auc = np.nan
        class_precision = precision_score(binary_true, binary_pred, zero_division=0)
        class_recall = recall_score(binary_true, binary_pred, zero_division=0)
        class_f1 = f1_score(binary_true, binary_pred, zero_division=0)
        print(f"Class {i}: AUC={class_auc:.4f}, Precision={class_precision:.4f}, Recall={class_recall:.4f}, F1={class_f1:.4f}")

    # 2. 整体多分类指标
    try:
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        macro_auc = np.nan
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\n=== Overall metrics ===")
    print(f"ROC AUC (macro): {macro_auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")

    # 3. 其他信息
    print("\n=== Other info ===")
    print("Label distribution:", Counter(all_labels))
    print(f"Predicted probability range: [{all_probs.min():.3f}, {all_probs.max():.3f}]")

    # 返回主要指标
    return {
        "macro_auc": macro_auc,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision
    }

def replace_layernorm_with_tanhnorm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            tanh_norm = TanhNorm(module.normalized_shape, module.eps)
            setattr(model, name, tanh_norm)
        else:
            replace_layernorm_with_tanhnorm(module)

# 加载模型
bert_4 = AutoModel.from_pretrained(r"C:\Users\10785\Desktop\bert_4mer", local_files_only=True)

# 加载tokenizer
tokenizer_4 = AutoTokenizer.from_pretrained(Path(r"C:\Users\10785\Desktop\bert_4mer"), local_files_only=True)

# 下载5-mer
bert_5 = AutoModel.from_pretrained(r"C:\Users\10785\Desktop\bert_5mer")

# 下载6-mer
bert_6 = AutoModel.from_pretrained(r"C:\Users\10785\Desktop\bert_6mer")

train_dataset = MultiKmerDataset(
    df_train,
    tokenizer_4=tokenizer_4,
    tokenizer_5=tokenizer_5,
    tokenizer_6=tokenizer_6
)