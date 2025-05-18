import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

target_names = iris.target_names
scaler = StandardScaler()
x_scaler = scaler.fit_transform(X)

pca = PCA(n_components=2)
Xpca = pca.fit_transform(x_scaler)

df_pca = pd.DataFrame(Xpca, columns=['PC1', 'PC2'])
df_pca['target'] = y

plt.figure(figsize=(10, 6))
sns.scatterplot(x= df_pca['PC1'], y = df_pca['PC2'], hue=df_pca['target'], palette = 'viridis', legend=True)

plt.xlabel('Principle Component 1')
plt.ylabel('Principle component 2')
plt.title('PCA on iris dataset (4D -> 2D)')

plt.legend(labels = target_names)
plt.show()