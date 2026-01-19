import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import ast

df = pd.read_csv(r"C:\Users\10785\Desktop\dataset_balanced_with_domains.csv")
df_nonempty = df[df['domain_labels'].apply(lambda x: len(ast.literal_eval(x)) > 0)]
row = df_nonempty.sample(1).iloc[0]
protein = row['orig_index']
label = row['label']
domains = ast.literal_eval(row['domain_labels'])

G = nx.DiGraph()
G.add_node(protein, type='protein')
G.add_node(label, type='function')
G.add_edge(protein, label, relation='has_function')
for domain in domains:
    # 对长标签做截断
    domain_short = domain if len(domain) <= 20 else domain[:17] + "..."
    G.add_node(domain_short, type='domain')
    G.add_edge(protein, domain_short, relation='has_domain')
    G.add_edge(domain_short, label, relation='domain_to_function')

# 增大画布，调整布局
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42, k=1.5)  # k值越大，节点越分散
node_colors = []
for n in G.nodes:
    t = G.nodes[n]['type']
    if t == 'protein':
        node_colors.append('#1f77b4')
    elif t == 'function':
        node_colors.append('#ff7f0e')
    else:
        node_colors.append('#2ca02c')
nx.draw(
    G, pos, with_labels=True, node_color=node_colors, node_size=1800,
    font_size=9, font_weight='bold', edge_color='gray'
)
plt.title("Example Subgraph: Protein–Domain–Function")
plt.tight_layout()
plt.savefig("supplementary_figure_knowledge_graph.png", dpi=300)
plt.show()