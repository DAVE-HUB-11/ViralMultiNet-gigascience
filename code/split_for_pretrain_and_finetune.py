import pandas as pd
from sklearn.model_selection import train_test_split

# 配置
csv_path = r"C:/Users/10785/Desktop/dnabert_dataset_multiscale.csv"
pretrain_path = r"C:/Users/10785/Desktop/pretrain.csv"
finetune_train_path = r"C:/Users/10785/Desktop/finetune_train.csv"
finetune_val_path = r"C:/Users/10785/Desktop/finetune_val.csv"
finetune_test_path = r"C:/Users/10785/Desktop/finetune_test.csv"

# 读取原始数据
all_df = pd.read_csv(csv_path)

# 1. 生成预训练集（无标签，保留所有文本列）
pretrain_df = all_df.copy()
if 'label' in pretrain_df.columns:
    pretrain_df = pretrain_df.drop(columns=['label'])
pretrain_df.to_csv(pretrain_path, index=False)
print(f"预训练数据已保存到: {pretrain_path}")

# 2. 生成微调用训练/验证/测试集（分层采样，保留标签）
train_df, temp_df = train_test_split(all_df, test_size=0.2, stratify=all_df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
train_df.to_csv(finetune_train_path, index=False)
val_df.to_csv(finetune_val_path, index=False)
test_df.to_csv(finetune_test_path, index=False)
print(f"微调训练集已保存到: {finetune_train_path}")
print(f"微调验证集已保存到: {finetune_val_path}")
print(f"微调测试集已保存到: {finetune_test_path}") 