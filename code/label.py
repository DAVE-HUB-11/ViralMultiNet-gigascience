import pandas as pd

# 读取注释结果文件
df = pd.read_excel(r"C:\Users\10785\Desktop\annotations.csv.xlsx")

# 显示前几行确认结构
print(df.head())

def classify_function(text):
    text = str(text).lower()
    if 'enzyme' in text or 'ec=' in text:
        return '酶'
    elif 'transporter' in text:
        return '转运蛋白'
    elif 'structural' in text or 'capsid' in text or 'spike' in text:
        return '结构蛋白'
    else:
        return '其他'

# 应用函数，新增一列 "function_type"
df['function_type'] = df['annotation'].apply(classify_function)

# 打印分类结果
print(df[['subject_id', 'function_type']].head())

# 统计各类别数量
count_by_type = df['function_type'].value_counts()

# 打印结果
print(count_by_type)
import matplotlib.pyplot as plt

# 中文显示（如乱码可取消）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 画柱状图
count_by_type.plot(kind='bar', color=['#4caf50', '#2196f3', '#ff9800', '#9e9e9e'])

plt.title('蛋白功能类型统计')
plt.xlabel('功能类型')
plt.ylabel('数量')
plt.tight_layout()
plt.show()
