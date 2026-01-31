from utils.data_loader import load_data
from utils.data_loader import load_data
from utils.model_engine import run_tsne, run_kmeans, calculate_metrics, calculate_cluster_accuracy
from utils.viz_engine import create_scatter_plot
import numpy as np

print("Testing Data Loading...")
X, y = load_data(n_samples=100) # Small sample for speed
print(f"Data Loaded: X shape {X.shape}, y shape {y.shape}")

print("Testing t-SNE...")
tsne_2d = run_tsne(X, perplexity=5, max_iter=250)
print(f"t-SNE Output shape: {tsne_2d.shape}")

print("Testing KMeans...")
labels = run_kmeans(tsne_2d, n_clusters=3)
print(f"Labels generated: {np.unique(labels)}")

print("Testing Metrics...")
score = calculate_metrics(tsne_2d, labels)
print(f"Silhouette Score: {score}")

print("Testing Accuracy...")
acc, mask, mapping, _ = calculate_cluster_accuracy(labels, y)
print(f"Accuracy: {acc:.2f}, Misclassified count: {np.sum(mask)}")

print("Testing Viz Engine (with images)...")
try:
    fig = create_scatter_plot(tsne_2d, labels, y, images=X)
    print("Figure created successfully with images.")
except Exception as e:
    print(f"Viz Failed: {e}")
