import pandas as pd
data=pd.read_csv("C:/Users/Roshan R/Downloads/ecommerce_customers.csv")
x=data.drop(columns=['CustomerID'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print("feature scaled")

print(data.head())
print(pd.DataFrame(x_scaled,columns=x.columns).head())
from sklearn.cluster import AgglomerativeClustering
model=AgglomerativeClustering(n_clusters=3)
data['Clusters']=model.fit_predict(x_scaled)
print("cluster counts is")
print(data['Clusters'].value_counts().sort_index())

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,6))
sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],hue=data['Clusters'],palette='Set1')
plt.title('Hierarchical clustering')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title="clusters")
plt.grid(True)
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
linked=linkage(x_pca,method='ward')
plt.figure(figsize=(10,6))
dendrogram(linked,orientation='top',distance_sort='descending',show_leaf_counts=False)
plt.xlabel('Samples')
plt.ylabel('Distances')
plt.show()