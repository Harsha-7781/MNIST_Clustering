from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
import streamlit as st
import numpy as np
from scipy.stats import mode

@st.cache_data(show_spinner=True)
def run_tsne(data, n_components=2, perplexity=30.0, max_iter=1000):
    """
    Runs t-SNE reduction on the high-dimensional data.
    
    Args:
        data: High-dimensional data (N, 784).
        n_components: Target dimensions (usually 2).
        perplexity: t-SNE perplexity parameter.
        max_iter: Number of iterations.
        
    Returns:
        np.array: (N, 2) transformed data.
    """
    # Ensure data is float
    data = np.array(data, dtype=np.float64)
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    return tsne.fit_transform(data)

@st.cache_data(show_spinner=True)
def run_kmeans(tsne_data, n_clusters=10):
    """
    Runs KMeans clustering on the t-SNE reduced data.
    
    Args:
        tsne_data: 2D data from t-SNE.
        n_clusters: Number of clusters (K).
        
    Returns:
        np.array: Cluster labels for each point.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(tsne_data)
    return labels

def calculate_metrics(tsne_data, labels):
    """
    Calculates silhouette score for the clustering.
    """
    if len(set(labels)) > 1:
        return silhouette_score(tsne_data, labels)
    return -1

def calculate_cluster_accuracy(predicted_labels, true_labels):
    """
    Calculates unsupervised clustering accuracy by mapping each cluster 
    to its most frequent true label.
    
    Args:
        predicted_labels: (N,)
        true_labels: (N,)
        
    Returns:
        accuracy (float)
        misclassified_mask (bool array): True where prediction != true
        assigned_labels (dict): Mapping from cluster_id -> true_label_guess
    """
    # Create mapping from cluster label to most frequent true label
    labels = np.zeros_like(predicted_labels)
    cluster_mapping = {}
    
    for i in range(len(np.unique(predicted_labels))):
        mask = (predicted_labels == i)
        if np.sum(mask) == 0:
            continue
            
        # Find mode of true labels in this cluster
        # mode returns (mode_array, count_array)
        most_common = mode(true_labels[mask], keepdims=True)[0][0]
        cluster_mapping[i] = most_common
        labels[mask] = most_common
        
    acc = accuracy_score(true_labels, labels)
    misclassified_mask = (labels != true_labels)
    
    return acc, misclassified_mask, cluster_mapping, labels

