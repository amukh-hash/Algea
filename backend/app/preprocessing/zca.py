import numpy as np

def zca_whitening(X, epsilon=1e-5):
    """
    ZCA (Zero-phase Component Analysis) Whitening.
    Decorrelates features while minimalizing the mean squared error between 
    original and whitened features (preserving spatial structure).
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        epsilon: Small constant to avoid division by zero
        
    Returns:
        X_zca: Whitened data
        ZCA_mat: Whitening matrix (can be applied to test data)
    """
    # Center the data
    X = X - np.mean(X, axis=0)
    
    # Calculate covariance matrix
    # rowvar=False because rows are samples
    cov = np.cov(X, rowvar=False)
    
    # Singular Value Decomposition
    # cov = U * S * U.T (since cov is symmetric)
    U, S, V = np.linalg.svd(cov)
    
    # Calculate ZCA Matrix: U * diag(1/sqrt(S)) * U.T
    # This rotates to principal components, scales to unit variance, then rotates BACK
    
    inv_sqrt_S = np.diag(1.0 / np.sqrt(S + epsilon))
    
    ZCA_mat = np.dot(U, np.dot(inv_sqrt_S, U.T))
    
    # Apply transformation
    X_zca = np.dot(X, ZCA_mat)
    
    return X_zca, ZCA_mat

def apply_zca_transform(X, ZCA_mat):
    """
    Apply pre-computed ZCA matrix to new data.
    
    Args:
        X: New data matrix (n_samples, n_features)
        ZCA_mat: Pre-computed ZCA matrix
        
    Returns:
        X_whitened: Transformed data
    """
    # Note: X should be centered using the MEAN from training data ideally.
    # For simplicity here we assume caller handles centering or we compute local mean if strictly ZCA
    # Standard ZCA assumes zero mean. 
    # If using in a pipeline, subtract training mean first.
    # For now, we perform dot product directly.
    
    return np.dot(X, ZCA_mat)
