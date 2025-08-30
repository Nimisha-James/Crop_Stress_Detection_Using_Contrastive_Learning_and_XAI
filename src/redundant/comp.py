import matplotlib.pyplot as plt

# Data
authors = [
    "Proposed Methodology", 
    "Ahmad", 
    "Judith", 
    "Lehouel",  
]

accuracies = [95.17, 89.10, 97.80, 90.42]

models = [
    "Stacking Classifier", 
    "EigenCL + KNN on NDRE", 
    "FCNN on NDVI (UAV)", 
    "CNN-ViT", 
]

# Define a unique color for each bar
colors = ['#66c2a5', '#fc8d62', '#e78ac3', '#a6d854']

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(authors, accuracies, color=colors)

# Add accuracy value inside each bar
for bar, accuracy in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height - 5, f"{accuracy}%", 
            ha='center', va='top', color='black', fontsize=10, fontweight='bold')

# Add model name on top of each bar
for bar, model in zip(bars, models):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 1, model, 
            ha='center', va='bottom', fontsize=9, color='darkblue')

# Axis labels and title
ax.set_ylabel('Accuracy (%)')
ax.set_title('State-of-the-Art Model Comparison')
plt.xticks(rotation=15)
plt.ylim(0, 105)  # leave space for model names above
plt.tight_layout()
plt.savefig('../../outputs/state-of-the-art-2.png', dpi=300)
plt.show()
