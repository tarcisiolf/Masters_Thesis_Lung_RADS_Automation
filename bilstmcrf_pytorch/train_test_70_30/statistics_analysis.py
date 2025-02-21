import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import json

"""

list_f1_score_70_30 = [0.8753914028612511, 0.8830617425321264, 0.8846058158718005, 0.8466409241932481, 0.881254195047313, 0.8544953622250852, 0.8823191590972446, 0.836023163118318, 0.875215632490025, 0.8699559857850642, 0.8887171673255899, 
0.863478333879378, 0.8655805849491557, 0.8743447049447743, 0.8624161252154273, 0.8829069854416707, 0.8437717977002285,
0.85603516264226, 0.8743073192849019, 0.8734880274350133, 0.841840879270015, 0.8913111854661439, 0.8659028080364277,
0.8695749222739645, 0.8656607139775655, 0.8681972866917956, 0.8867025335871483]

list_f1_score_80_20 = [0.9183309283309283, 0.8754485897585655, 0.9012868256215388, 0.87378636358341, 0.8412288412801812,
0.867059657902533,  0.899473292580545, 0.8782390323607036, 0.7177169247408486, 0.8633089724838613, 
0.8908422702524336, 0.8981920529724768, 0.8680474264992643, 0.8874952222919908, 0.8904819955917399,
0.8776446938451022, 0.865981155634842, 0.866974545829578, 0.8721507869815147, 0.8930815973077024,
0.9117412055103179, 0.891844010278576, 0.8830804820198354, 0.8815900050136474, 0.8858053276448401, 0.8768349136841214, 0.8744196613362268]

list_f1_score_90_10 = [0.8548052036041945, 0.9044920911717043, 0.8841365798713102, 0.8903159880894623, 0.9235288177664268,
0.8715029911944613, 0.8391861134107005, 0.7571471092577045, 0.8782268238501513, 0.9042877017705394, 0.8987779277252962,
0.8751430717365719, 0.8993712857921402, 0.8901710756656858, 0.9026033597187123, 0.8886501389368363, 0.8642019436530308,
0.9106193444924835, 0.8873346204743381, 0.8662720645176784, 0.9115760861718258, 0.9160885028610197, 0.8739112137209655, 
0.8969004802964978, 0.9034697155128203, 0.8580694747730147, 0.8904477453632055]
"""


list_f1_score_70_30_path = "1_bilstmcrf_pytorch/train_test_70_30/metrics/macro_f1_scores_70_30.json"
list_f1_score_80_20_path = "1_bilstmcrf_pytorch/train_test_80_20/metrics/macro_f1_scores_80_20.json"
list_f1_score_90_10_path = "1_bilstmcrf_pytorch/train_test_90_10/metrics/macro_f1_scores_90_10.json"

# Open and read the JSON file
with open(list_f1_score_70_30_path, 'r') as file:
    list_f1_score_70_30 = json.load(file)

with open(list_f1_score_80_20_path, 'r') as file:
    list_f1_score_80_20 = json.load(file)

with open(list_f1_score_90_10_path, 'r') as file:
    list_f1_score_90_10 = json.load(file)


## Combine data for easier handling
data = {
    "70/30": list_f1_score_70_30,
    "80/20": list_f1_score_80_20,
    "90/10": list_f1_score_90_10
}

# Step 1: Descriptive Statistics
def descriptive_stats(data):
    print("=== Descriptive Statistics ===")
    for split, scores in data.items():
        mean = np.mean(scores)
        median = np.median(scores)
        std = np.std(scores)
        var = np.var(scores)
        min_val = np.min(scores)
        max_val = np.max(scores)
        print(f"{split}: Mean={mean:.4f}, Median={median:.4f}, Std={std:.4f}, Var={var:.4f}, Min={min_val:.4f}, Max={max_val:.4f}")

descriptive_stats(data)

# Step 2: Normality Test (Shapiro-Wilk)
print("\n=== Normality Test (Shapiro-Wilk) ===")
for split, scores in data.items():
    stat, p = stats.shapiro(scores)
    print(f"{split}: Statistic={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print(f"  {split} appears to be normally distributed.")
    else:
        print(f"  {split} does not appear to be normally distributed.")

"""
# Step 8: Visualization
plt.figure(figsize=(10, 6))
plt.boxplot([data["70/30"], data["80/20"], data["90/10"]], labels=["70/30", "80/20", "90/10"])
plt.title("Boxplot of F1 Scores Across Splits")
plt.ylabel("F1 Score")
plt.show()
"""
