# plot_ncf_results.py
import matplotlib.pyplot as plt
import numpy as np

# Model configurations
configs = ['Emb32-[64,32,16,8]', 'Emb64-[128,64,32,16]', 'Emb128-[256,128,64,32]', 'Emb24-[48,24,12]']

# Results
hit_rate = [0.9504, 0.9495, 0.9507, 0.9478]
ndcg = [0.4882, 0.4908, 0.4933, 0.4806]
recall = [0.4429, 0.4414, 0.4373, 0.4381]

x = np.arange(len(configs))
width = 0.25

fig, ax = plt.subplots(figsize=(10,6))

# Plotting
ax.bar(x - width, hit_rate, width, label='Hit Rate@10')
ax.bar(x, ndcg, width, label='NDCG@10')
ax.bar(x + width, recall, width, label='Recall@10')

# Formatting
ax.set_xlabel('Model Configurations')
ax.set_ylabel('Performance Metrics')
ax.set_title('Performance Comparison of NCF Model Architectures')
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.legend()

# Adding metric values on bars
for i in x:
    ax.text(i - width, hit_rate[i]+0.005, f'{hit_rate[i]:.4f}', ha='center', fontsize=9)
    ax.text(i, ndcg[i]+0.005, f'{ndcg[i]:.4f}', ha='center', fontsize=9)
    ax.text(i + width, recall[i]+0.005, f'{recall[i]:.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('./plots/model_comparison.png', dpi=300)
plt.show()