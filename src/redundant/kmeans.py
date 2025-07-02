from sklearn.cluster import KMeans
import numpy as np
import os
import matplotlib.pyplot as plt

# Load all embeddings
embedding_files = sorted(os.listdir("embeddings"))
X = np.stack([np.load(f"embeddings/{f}") for f in embedding_files])

# Cluster into 2 groups (maybe stress / no-stress)
kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
labels = kmeans.labels_

# Save cluster labels
with open("cluster_labels.csv", "w") as f:
    f.write("id,cluster\n")
    for i, l in enumerate(labels):
        f.write(f"{i:05d},{l}\n")
