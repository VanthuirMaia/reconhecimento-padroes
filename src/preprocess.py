import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def apply_pcs(X_scaled, n_components=7):
    pca = PCA(n_components=n_components)
    Z_pca = pca.fit_transform(X_scaled)
    return Z_pca, pca

