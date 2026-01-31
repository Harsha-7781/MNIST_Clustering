import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_openml

@st.cache_data
def load_data(n_samples=5000):
    """
    Loads MNIST data from openml, samples it, and returns X and y.
    
    Args:
        n_samples (int): Number of samples to return.
        
    Returns:
        tuple: (X_sample, y_sample) as numpy arrays.
        X is normalized to [0, 1].
    """
    # Load 784-dimensional MNIST (28x28)
    # as_frame=False returns numpy arrays directly
    # This might fail if internet is down, but we assume connectivity.
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False, parser='auto')
    
    X = mnist.data
    y = mnist.target.astype(int)
    
    # Random sampling
    np.random.seed(42)
    
    if n_samples > len(X):
        n_samples = len(X)
        
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    X_sample = X[indices] / 255.0  # Normalize
    y_sample = y[indices]
    
    return X_sample, y_sample
