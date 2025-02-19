from scipy.stats import shapiro  
from scipy.stats import kstest
import numpy as np
import matplotlib.pyplot as plt

list_f1_scores = [0.8753914028612511, 0.8830617425321264, 0.8846058158718005, 0.8466409241932481, 0.881254195047313, 0.8544953622250852, 0.8823191590972446, 0.836023163118318, 0.875215632490025, 0.8699559857850642, 0.8887171673255899, 
0.863478333879378, 0.8655805849491557, 0.8743447049447743, 0.8624161252154273, 0.8829069854416707, 0.8437717977002285,
0.85603516264226, 0.8743073192849019, 0.8734880274350133, 0.841840879270015, 0.8913111854661439, 0.8659028080364277,
0.8695749222739645, 0.8656607139775655, 0.8681972866917956, 0.8867025335871483]

# Perform Shapiro test
shapiro_test = shapiro(list_f1_scores)  
print(f"p-valor do teste de Shapiro-Wilk: {shapiro_test.pvalue}") 

#If p-valor > 0.05 -> the data is approximately normal
#If p-valor ≤ 0.05 -> the data is not normal.

# We will assume a normal distribution with mean and std from your sample
mean_f1 = np.mean(list_f1_scores)
std_f1 = np.std(list_f1_scores)

# Perform Kolmogorov-Smirnov test
ks_test_result = kstest(list_f1_scores, 'norm', args=(mean_f1, std_f1))
print(f"p-valor do teste de Kolmogorov-Smirnov: {ks_test_result.pvalue}") 

# If the p-value > 0.05, there is not enough evidence to reject the hypothesis that the data comes from a normal distribution.
# If the p-value ≤ 0.05, you reject the null hypothesis and conclude that the data does not follow a normal distribution.

#plt.hist(list_f1_scores)
#plt.show()

