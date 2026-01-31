import plotly.express as px
import pandas as pd
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def numpy_to_base64(img_array):
    """
    Converts a 784-dim or (28,28) numpy array to a base64 png string.
    """
    if img_array.ndim == 1:
        img_array = img_array.reshape(28, 28)
    
    # Normalize to 0-255 uint8 if float
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)
        
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_scatter_plot(tsne_data, cluster_labels, true_labels=None, images=None):
    """
    Creates an interactive 2D scatter plot using Plotly.
    
    Args:
        tsne_data: (N, 2) array of coordinates.
        cluster_labels: (N,) array of cluster assignments.
        true_labels: (N,) array of true digit labels (optional).
        images: (N,) array of flat image vectors (optional).
    """
    # Create DataFrame for Plotly
    df = pd.DataFrame(tsne_data, columns=['x', 'y'])
    df['Cluster'] = cluster_labels.astype(str)
    
    hover_data = {}
    if true_labels is not None:
        df['Digit'] = true_labels.astype(str)
        # We process tooltip manually later for images
    
    # Process images for hover if provided
    # Note: Doing this for 5000+ points can be slow.
    # We will limit the image processing to a reasonable number if needed, 
    # but for local apps 5k is usually okay.
    if images is not None:
        # Convert all to base64
        # Using a list comprehension for speed
        b64_images = ["data:image/png;base64," + numpy_to_base64(img) for img in images]
        df['Image'] = b64_images
        
    # Create plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='Cluster',
        title='t-SNE Visualization with K-Means Clustering',
        hover_data=hover_data, # Use default or empty if custom
        template='plotly_dark', # Dark theme
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=700
    )
    
    # Plotly doesn't support <img> tags in hover templates
    # So we'll just show text information
    if true_labels is not None:
        fig.update_traces(
             hovertemplate="<b>Digit: %{customdata[0]}</b><br>Cluster: %{customdata[1]}<extra></extra>",
             customdata=df[['Digit', 'Cluster']],
             # Prevent blurring of unselected points
             selected=dict(marker=dict(opacity=1.0)),
             unselected=dict(marker=dict(opacity=1.0))
        )
    
    # Improve layout
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_title_text='Cluster'
    )
    
    return fig
