import pandas as pd
pd.set_option('display.max_columns',500)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

california_housing=fetch_california_housing()
data=pd.DataFrame(california_housing.data , columns=california_housing.feature_names)
data['MedhHouseVal']=california_housing.target

correlation_matrix=data.corr()
print("Correlation Matrix")
print(correlation_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title('Coorelation Matrix Heatmap')
plt.show()

sns.pairplot(data)
plt.suptitle('Pair Plot of Numerical Features',y=1.02)
plt.show()