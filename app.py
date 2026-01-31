import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_data
from utils.model_engine import run_tsne, run_kmeans, calculate_metrics, calculate_cluster_accuracy
from utils.viz_engine import create_scatter_plot

# Page Config
st.set_page_config(
    page_title="MNIST Clustering",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme consistency
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔢 MNIST Digit Clustering with t-SNE")
st.markdown("""
Unsupervised learning on handwritten digits. 
This tool reduces 784-dimensional images to 2D using **t-SNE** and groups them using **K-Means**.
""")

# Sidebar settings
st.sidebar.header("🔧 Configuration")

n_samples = st.sidebar.slider("Number of Samples", 1000, 10000, 5000, step=500, help="More samples = clearer structure but slower.")
perplexity = st.sidebar.slider("t-SNE Perplexity", 5, 50, 30, help="Balance between local and global structure.")
n_clusters = st.sidebar.slider("K-Means Clusters (K)", 3, 15, 10, help="Expected number of digit groups.")

st.sidebar.markdown("---")
st.sidebar.info("Adjust parameters to re-run the analysis.")

# Application Logic
with st.spinner("Loading basic MNIST Data..."):
    # Load data
    try:
        X, y = load_data(n_samples)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Re-run logic happens automatically on widget change in Streamlit
with st.spinner(f"Running t-SNE (Sample Size: {n_samples})..."):
    tsne_2d = run_tsne(X, n_components=2, perplexity=perplexity)

with st.spinner(f"Clustering with K={n_clusters}..."):
    cluster_labels = run_kmeans(tsne_2d, n_clusters)

# Metrics
silhouette = calculate_metrics(tsne_2d, cluster_labels)
accuracy, misclassified_mask, cluster_mapping, predicted_labels_mapped = calculate_cluster_accuracy(cluster_labels, y)

# Dashboard
col1, col2, col3, col4 = st.columns(4)
col1.metric("Samples", n_samples)
col2.metric("Clusters", n_clusters)
col3.metric("Silhouette", f"{silhouette:.3f}")
col4.metric("Accuracy", f"{accuracy:.1%}")

# Main Chart
st.subheader("2D Projection")
st.info("👆 **Click on any point** to see the digit image in the sidebar!")

fig = create_scatter_plot(tsne_2d, cluster_labels, true_labels=y)

# Use on_select to enable interactivity
# As of Streamlit 1.53, use_container_width is deprecated
event = st.plotly_chart(fig, on_select="rerun", selection_mode="points")

# Handle Selection
if event and len(event.selection['points']) > 0:
    point_index = event.selection['points'][0]['point_index']
    
    # Show in Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Selected Point")
        
        # Get data
        selected_img = X[point_index].reshape(28, 28)
        selected_label = y[point_index]
        selected_cluster = cluster_labels[point_index]
        
        # Display
        st.image(selected_img, width=150, caption=f"Index: {point_index}")
        st.metric("True Digit", selected_label)
        st.metric("Cluster", selected_cluster)
        st.markdown("---")

# Analysis
st.subheader("📊 Cluster Analysis")

df_analysis = pd.DataFrame({'Cluster': cluster_labels, 'Digit': y})

# 1. Cluster Counts
cluster_counts = df_analysis['Cluster'].value_counts().sort_index()

# 2. Dominant Digit per Cluster
# Mode might return multiple values, we take the first one
dominant_digits = df_analysis.groupby('Cluster')['Digit'].agg(lambda x: x.mode()[0])

analysis_df = pd.DataFrame({
    'Count': cluster_counts,
    'Dominant Digit': dominant_digits
})

c1, c2 = st.columns([1, 2])
with c1:
    st.write("Cluster Statistics")
    st.dataframe(analysis_df, use_container_width=True)

with c2:
    st.write("Digit Distribution per Cluster")
    # Cross tab visualization
    ct = pd.crosstab(df_analysis['Cluster'], df_analysis['Digit'])
    st.bar_chart(ct)

# Misclassified Digits Analysis
st.subheader("⚠️ Misclassified Digits")
st.markdown("These digits were assigned to a cluster where the majority were a *different* digit.")

if st.checkbox("Show Misclassified Examples", value=False):
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) > 0:
        # Refresh logic
        if 'misclassified_seed' not in st.session_state:
            st.session_state.misclassified_seed = 0
            
        if st.button("🔄 View Other Misclassified Samples"):
             st.session_state.misclassified_seed += 1
             
        # Set seed for reproducibility of this batch
        np.random.seed(st.session_state.misclassified_seed)
        
        st.info(f"Showing 10 random misclassified digits out of {len(misclassified_indices)} total.")
        
        # Sample 10 random errors
        sample_indices = misclassified_indices[:10]
        if len(misclassified_indices) > 10:
             sample_indices = np.random.choice(misclassified_indices, 10, replace=False)
        
        cols = st.columns(5)
        for i, idx in enumerate(sample_indices):
            with cols[i % 5]:
                # Reshape 784 -> 28x28
                img = X[idx].reshape(28, 28)
                true_label = y[idx]
                pred_label = predicted_labels_mapped[idx]
                
                # Display image
                st.image(img, clamp=True, width=100)
                st.caption(f"True: **{true_label}**\nPred: **{pred_label}**")
    else:
        st.success("Analysis perfect! No misclassified digits found (or K is extremely high).")

