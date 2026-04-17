import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.decomposition import PCA

from src.models import NipalsPCA

def test_nipals_deterministic_2d():
    """
    Test NIPALS on a simple deterministic matrix where we know 
    the principal components by inspection.
    """
    # Data stretched along the y = x line
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4]
    ], dtype=float)
    
    nipals = NipalsPCA(n_components=1)
    nipals.fit(X)
    
    # 1. Check components: The first PC should be [0.707, 0.707] (normalized [1, 1])
    expected_component = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    # We use absolute because of sign ambiguity
    np.testing.assert_allclose(np.abs(nipals.components_[0]), expected_component, atol=1e-5)
    
    # 2. Check Explained Variance Ratio: Should be 1.0 (all data is on one line)
    assert nipals.explained_variance_ratio_[0] == pytest.approx(1.0, abs=1e-5)

def test_nipals_vs_sklearn_pca():
    """
    Compare NIPALS results with standard SVD-based PCA on random data.
    """
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5)
    
    n_components = 3
    nipals = NipalsPCA(n_components=n_components).fit(X)
    sk_pca = PCA(n_components=n_components).fit(X)
    
    # Compare explained variance
    np.testing.assert_allclose(nipals.explained_variance_, sk_pca.explained_variance_, rtol=1e-5)
    
    # Compare components (absolute values due to sign flip possibility)
    np.testing.assert_allclose(np.abs(nipals.components_), np.abs(sk_pca.components_), atol=1e-5)

def test_transform_inverse_consistency():
    """
    Check if inverse_transform(transform(X)) reconstructs the data 
    (for full rank).
    """
    rng = np.random.RandomState(42)
    X = rng.randn(10, 10) 

    nipals = NipalsPCA(n_components=10).fit(X)
    X_transformed = nipals.transform(X)
    X_reconstructed = nipals.inverse_transform(X_transformed)

    # --- DIAGNOSTIC SECTION ---
    # 1. Check if the mean of reconstructed data matches the original mean
    mean_diff = np.mean(X, axis=0) - np.mean(X_reconstructed, axis=0)
    print(f"\nMean Difference (Bias):\n{mean_diff}")

    # 2. Check if the error is indeed constant across rows
    # If the variance of the error is near zero, it's a pure translation (mean) issue
    error = X - X_reconstructed
    print(f"Variance of the error: {np.var(error)}")

    # 3. Check Orthogonality of components
    # NIPALS should produce orthonormal loadings
    dot_product_matrix = np.dot(nipals.components_, nipals.components_.T)
    print(f"Loadings Orthogonality (should be Identity):\n{dot_product_matrix}")

    # 4. Check if transform(X) is centered
    print(f"Mean of transformed data (should be ~0):\n{np.mean(X_transformed, axis=0)}")
    # --------------------------

    np.testing.assert_allclose(X, X_reconstructed, atol=1e-5)
    
# def test_sklearn_compatibility():
    """
    Optional: Runs scikit-learn's internal check_estimator 
    to ensure it plays nice with the API.
    """
    # Note: This might require some minor tweaks to the class to pass every 
    # strict scikit-learn requirement (like handling 0 components), 
    # but it's great for testing robustness.
    # check_estimator(NipalsPCA(n_components=2))