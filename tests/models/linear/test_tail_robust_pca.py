# Author: Tommaso Zipoli
# Licence: GNU General Public License v3.0

import numpy as np
import pytest
from sklearn.decomposition import PCA

from src.models import TailRobustPCA
from src.models import TailRobustPCACV
from src.utils.functional import subspace_overlap_score


class TestTailRobustPCA:

    @pytest.fixture
    def uniform_latent_dgp(self):
        rng = np.random.default_rng(42)
        n_samples, n_features, n_factors = 200, 20, 2
        
        # Clean signal
        factors_true = rng.uniform(-2, 2, (n_samples, n_factors))
        loadings_true = rng.uniform(-1, 1, (n_factors, n_features))
        X = factors_true @ loadings_true + rng.uniform(-0.1, 0.1, (n_samples, n_features))
        
        return X, factors_true, loadings_true

    def test_outlier_rejection_with_uniform_data(self, uniform_latent_dgp):
        X_clean, _, loadings_true = uniform_latent_dgp
        X = X_clean.copy()
        
        # Inject outliers
        X[0, 0] = 1e5
        X[10, 5] = -1e5
        
        cv_model = TailRobustPCACV(n_components=2)
        cv_model.fit(X)
        
        best_thr = cv_model.best_params_['thr']
        
        clean_max = np.max(np.abs(X_clean - np.mean(X_clean, axis=0)))
        
        assert best_thr >= clean_max * 0.8, (
            f"Threshold {best_thr} is too aggressive, cutting clean signal {clean_max}"
        )
        assert best_thr < 1000.0, (
            f"Threshold {best_thr} is too high, likely letting outliers influence the model"
        )
        
        # Robust PCA vs True Loadings
        # Since subspace_overlap_score handles QR, we pass raw components
        score_robust = subspace_overlap_score(
            cv_model.best_estimator_.components_.T, loadings_true.T
        )
        
        # Standard PCA vs True Loadings
        pca_standard = PCA(n_components=2).fit(X)
        score_standard = subspace_overlap_score(
            pca_standard.components_.T, loadings_true.T
        )

        # Robust model should recover the ture loading space sensibly better
        # wrt the baseline
        print(f"Robust Score: {score_robust:.4f}")
        print(f"Standard Score: {score_standard:.4f}")
        assert score_robust > (score_standard * 5)
        assert score_robust > 0.4

    def test_consistency_high_thr(self, uniform_latent_dgp):
        X, _, _ = uniform_latent_dgp
        pca = PCA(n_components=2).fit(X)
        rpca = TailRobustPCA(n_components=2, thr=1e12).fit(X)
        
        corr = subspace_overlap_score(pca.components_.T, rpca.components_.T)
        assert np.isclose(corr, 1.0, atol=1e-10)
        
    def test_consistency_with_standard_pca(self, uniform_latent_dgp):
        """Standard check: High thr should yield standard PCA results."""
        X, _, _ = uniform_latent_dgp
        n_comp = 2
        
        pca = PCA(n_components=n_comp).fit(X)
        rpca = TailRobustPCA(n_components=n_comp, thr=1e12)
        rpca.fit(X)

        corr = subspace_overlap_score(pca.components_.T, rpca.components_.T)
        assert np.isclose(corr, 1, atol=1e-10)

    def test_reconstruction_masking(self, uniform_latent_dgp):
        """Verify the inverse_transform logic on clean bounded data."""
        X, _, _ = uniform_latent_dgp
        # Set threshold high enough to encompass all uniform data
        thr = 10.0 
        
        rpca = TailRobustPCA(n_components=2, thr=thr)
        X_rec = rpca.inverse_transform(rpca.fit_transform(X))
        
        # On clean data with high thr, reconstruction error should be 
        # solely due to PCA dimensionality reduction, not truncation.
        pca = PCA(n_components=2)
        X_rec_pca = pca.inverse_transform(pca.fit_transform(X))
        
        assert np.allclose(X_rec, X_rec_pca, atol=1e-10)
