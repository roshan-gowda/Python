import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data=pd.read_csv("C:/Users/Roshan R/Downloads/Mall_Customers.csv")
print(data.head())

x=data[['Annual Income (k$)','Spending Score (1-100)']]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print("feature scaled")

kmeans = KMeans(n_clusters=2,random_state=48)
data['Clusters']=kmeans.fit_predict(x_scaled)
Centroids=scaler.inverse_transform(kmeans.cluster_centers_)
print("\n Centroids")

for i,c in enumerate(Centroids):
    print(f"Cluster{i}:Income ={c[0]},score={c[1]}")
    # print("\n Cluster counts")
print(data['Clusters'].value_counts().sort_index())

import matplotlib.pyplot as plt
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=data['Clusters'])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='black',marker='x')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()