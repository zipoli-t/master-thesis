# Author: Tommaso Zipoli
# Licence: GNU General Public License v3.0

from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.validation import check_array, check_is_fitted, validate_data


__all__ = ["NipalsPCA"]


class NipalsPCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components):
        self.n_components = n_components
        super().__init__()

    def fit(self, X, y=None, warm_start=None, atol=1e-8, max_iter=100):
        X = validate_data(self, X=X, y=y, reset=True)
        n_samples, n_features = X.shape        
        self.n_components_ = self.n_components or min(n_samples, n_features)
        
        self.mean_ = np.mean(X, axis=0)
        X_h = (X - self.mean_).copy()
        
        total_ss = np.sum(X_h**2)
        loadings_list = []
        ess = np.zeros(self.n_components_)

        for h in range(self.n_components_):
            # 1. Initialization
            if warm_start is not None and h < len(warm_start):
                p_h = warm_start[h].copy()
            else:
                var_idx = np.argmax(np.sum(X_h**2, axis=0))
                p_h = X_h[:, var_idx]
                if n_samples > n_features:
                    p_h = np.dot(X_h.T, p_h)
            
            p_h /= (np.linalg.norm(p_h) + 1e-12)

            for _ in range(max_iter):
                f_h = np.dot(X_h, p_h) 
                f_norm_sq = np.dot(f_h, f_h)
                
                if f_norm_sq < 1e-12: 
                    p_star = p_h
                    break 
                
                p_star = np.dot(X_h.T, f_h) / f_norm_sq
                
                # 2. Sequential MGS for Loadings
                for loading in loadings_list:
                    p_star -= np.dot(p_star, loading) * loading
                
                p_star /= (np.linalg.norm(p_star) + 1e-12)

                # Convergence check
                if min(np.linalg.norm(p_h - p_star), np.linalg.norm(p_h + p_star)) < atol:
                    p_h = p_star
                    break
                p_h = p_star
            
            # 3. Final Score
            f_star = np.dot(X_h, p_h)
            ess[h] = np.dot(f_star, f_star)
            
            # 4. Double-Pass Deflation (The fix for scores orthogonality)
            # First pass removes the variance
            X_h -= np.outer(f_star, p_h)
            # Second pass removes numerical noise/leakage
            X_h -= np.outer(np.dot(X_h, p_h), p_h)
            
            loadings_list.append(p_h)
        
        self.components_ = np.vstack(loadings_list)
        self.explained_variance_ = ess / (n_samples - 1)
        self.explained_variance_ratio_ = ess / total_ss
        
        return self
    
    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, y="no_validation", reset=False)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X):
        """Map latent factors back to the original space."""
        check_is_fitted(self)
        return (X @ self.components_) + self.mean_
            