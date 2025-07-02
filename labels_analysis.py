import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset (e.g., labels)
df = pd.read_csv("cluster_labels.csv")  # or your own file
print(df["label"].value_counts())       # count of 0s and 1s

# Plot bar chart
df["label"].value_counts().plot(kind='bar', color=["green", "red"])
plt.title("Class Distribution")
plt.xlabel("Class (0 = Healthy, 1 = Stressed)")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
