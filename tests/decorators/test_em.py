# Author: Tommaso Zipoli
# License: GNU General Public License v3.0

import pytest
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

from src.decorators import EM
from src.utils.functional import subspace_overlap_score


class TestEMImputer:
    @pytest.fixture
    def noiseless_latent_data(self):
        """Generates a perfectly low-rank matrix."""
        rng = np.random.default_rng(42)
        n_samples, n_features, n_factors = 100, 8, 2
        
        factors = rng.standard_normal((n_samples, n_factors))
        loadings = rng.standard_normal((n_factors, n_features))
        
        # Purely deterministic: X = FW
        X = factors @ loadings
        return X, factors, loadings

    def test_em_noiseless_recovery(self, noiseless_latent_data):
        """
        Test 1: Absolute recovery in a noiseless low-rank setting.
        The imputed values and subspace should be near-perfect.
        """
        X, _, true_loadings = noiseless_latent_data
        rng = np.random.default_rng(42)
        
        # Mask 10% of the data
        mask = rng.random(X.shape) < 0.10
        X_nan = X.copy()
        X_nan[mask] = np.nan

        # Use high precision settings
        em = EM(tf=PCA(n_components=2), max_iter=200, atol=1e-9, rtol=1e-9)
        X_imputed, _ = em(X_nan)

        # 1. Value Recovery: Missing entries should be recovered almost exactly
        assert np.allclose(X_imputed, X, atol=1e-6)

        # 2. Subspace Recovery: The loadings should align perfectly
        pca_est = PCA(n_components=2).fit(X_imputed - X_imputed.mean(axis=0))
        score = subspace_overlap_score(true_loadings.T, pca_est.components_.T)
        
        # In a noiseless case, overlap should be practically 1.0
        assert pytest.approx(score, abs=1e-8) == 1.0

    def test_em_with_noise(self, noiseless_latent_data):
        """
        Test 2: Robustness check with added Gaussian noise.
        """
        X_clean, _, true_loadings = noiseless_latent_data
        rng = np.random.default_rng(42)
        X = X_clean + 0.1 * rng.standard_normal(X_clean.shape)
        
        X_nan = X.copy()
        X_nan[rng.random(X.shape) < 0.1] = np.nan

        em = EM(tf=PCA(n_components=2), max_iter=50)
        X_imputed, _ = em(X_nan)

        # Structural recovery should still be high (>0.9) despite noise
        pca_est = PCA(n_components=2).fit(X_imputed - X_imputed.mean(axis=0))
        score = subspace_overlap_score(true_loadings.T, pca_est.components_.T)
        assert score > 0.9

    def test_no_missing_values(self, noiseless_latent_data):
        """If no NaNs are present, output must be an exact copy."""
        X, _, _ = noiseless_latent_data
        em = EM()
        X_filled, _ = em(X)
        assert np.array_equal(X, X_filled)

    def test_invalid_shapes_raises(self):
        """Ensure ValueError for all-NaN columns or mismatched rows."""
        em = EM()
        # Case: Column of pure NaNs
        X_bad = np.random.rand(10, 2)
        X_bad[:, 0] = np.nan
        with pytest.raises(ValueError, match="column"):
            em(X_bad)

    def test_convergence_warning(self):
        """Ensure ConvergenceWarning triggers if max_iter is too low."""
        X = np.random.rand(20, 5)
        X[0, 0] = np.nan
        em = EM(max_iter=1, atol=1e-12) # Impossible to converge in 1 step
        with pytest.warns(ConvergenceWarning):
            em(X)

    def test_decorator(self):
        """Verify the decorator fills NaNs before calling the target function."""
        em = EM(tf=PCA(n_components=1))
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])

        @em.decorate
        def check_nan(data):
            return np.isnan(data).any()

        assert check_nan(X) == False
