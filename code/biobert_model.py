import matplotlib.pyplot as plt
import numpy as np

# 提取表格中的数据
models = ["HybridCNN", "BiLSTM", "BioBERT", "DNABERT+CLIP", "ViralMultiNet"]
accuracy = np.array([0.89, 0.87, 0.89, 0.92, 0.92])
f1_score = np.array([0.90, 0.86, 0.90, 0.90, 0.91])

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(accuracy, f1_score, 'o-', linewidth=2, markersize=8)

# 添加数据点标签
for i, model in enumerate(models):
    plt.annotate(model, (accuracy[i], f1_score[i]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

# 设置坐标轴范围和标签
plt.xlabel('Accuracy (ACC)')
plt.ylabel('F1 Score')
plt.title('Model Performance: F1 vs Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)

# 调整坐标轴范围，让数据点更分散
plt.xlim(0.86, 0.93)
plt.ylim(0.85, 0.92)

plt.tight_layout()
plt.show()
