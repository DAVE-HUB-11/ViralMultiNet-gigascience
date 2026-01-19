import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 设置重试策略
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
)

# 创建会话并挂载重试策略
session = requests.Session()
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

def get_uniprot_info(uniprot_id):
    url = f'https://www.uniprot.org/uniprot/{uniprot_id}.txt'
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# 读取Excel文件
df = pd.read_excel('C:/Users/10785/Desktop/iasd_cleaned.xlsx')

# 去重后的 ID 列表
unique_ids = df['subject_id'].dropna().unique()

# 尝试加载已保存的注释结果
try:
    annotation_df = pd.read_excel('C:/Users/10785/Desktop/注释结果_uniprot.xlsx')
    completed_ids = annotation_df['subject_id'].unique()
    unique_ids = [uid for uid in unique_ids if uid not in completed_ids]
except FileNotFoundError:
    completed_ids = []

# 收集注释结果
annotations = []

for uid in unique_ids:
    print(f"🔍 正在注释 UniProt ID: {uid}")
    desc = get_uniprot_info(uid)
    annotations.append({'subject_id': uid, 'annotation': desc})

# 合并注释结果
annotation_df = pd.DataFrame(annotations)
result = pd.merge(df, annotation_df, on='subject_id', how='left')

# 保存到桌面
result.to_excel('C:/Users/10785/Desktop/注释结果_uniprot.xlsx', index=False)
print("✅ 所有注释已完成并保存到桌面：注释结果_uniprot.xlsx")

