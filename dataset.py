# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 2: Import the Dataset
# Replace this with the actual path to your cleaned dataset
df = pd.read_csv('your_dataset.csv')

# Preview the dataset
print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Step 3: Select Features for Clustering
# Update with the actual features (exclude any label/target columns)
features = df[['feature1', 'feature2', 'feature3']]  # replace with your actual feature columns

# Step 4: Scale the Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 5: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # You can experiment with different k values
clusters = kmeans.fit_predict(scaled_features)

# Add the cluster labels to the original dataframe
df['Cluster'] = clusters

# Step 6: Evaluate Clustering Quality
silhouette_avg = silhouette_score(scaled_features, clusters)
print(f"\nSilhouette Score: {silhouette_avg:.3f}")

# Step 7: Visualize Clusters Using PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)
df['PC1'] = pca_components[:, 0]
df['PC2'] = pca_components[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='viridis')
plt.title('PCA Visualization of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
