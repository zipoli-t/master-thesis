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

    def fit(self, X, y=None, warm_start=None, rtol=1e-5, atol=1e-8, max_iter=100):
        X = validate_data(self, X=X, y=y, reset=True)
        n_samples, n_features = X.shape        
        self.n_components_ = self.n_components or min(n_samples, n_features)
        
        self.mean_ = np.mean(X, axis=0)
        X_h = np.copy(X) - self.mean_
        
        total_ss = np.sum(X_h**2)
        loadings = np.zeros((self.n_components_, n_features))
        ess = np.zeros((self.n_components_,))

        for h in range(self.n_components_):
            # Stability: Start with the column of X_h with max variance
            if warm_start is not None:
                p_h = warm_start[h]
            else:
                # Initialize p_h with a column of X_h (must be n_features long)
                # A common trick is taking a row or a unit vector
                p_h = np.ones(n_features)
                p_h /= np.linalg.norm(p_h)

            for _ in range(max_iter):
                f_h = np.dot(X_h, p_h) 
                
                denom = np.dot(f_h.T, f_h)
                if denom < 1e-12: 
                    p_star = p_h # Avoid break without assignment
                    break 
                
                p_star = np.dot(f_h.T, X_h) / denom
                p_star /= np.linalg.norm(p_star)
                f_h = np.dot(X_h, p_star)

                # Sign-agnostic convergence check
                diff_pos = np.linalg.norm(p_h - p_star)
                diff_neg = np.linalg.norm(p_h + p_star)
                
                if min(diff_pos, diff_neg) < atol:
                    # Align sign to p_h for consistency
                    if diff_neg < diff_pos:
                        p_star = -p_star
                    break
                p_h = p_star
            
            f_star = np.dot(X_h, p_star)
            ess[h] = np.dot(f_star.T, f_star)
            
            # Deflation
            X_h = X_h - np.outer(f_star, p_star)
            loadings[h] = p_star
        
        self.components_ = loadings
        # n_samples - 1 for unbiased variance estimation
        self.explained_variance_ = ess / (n_samples - 1)
        self.explained_variance_ratio_ = ess / total_ss
        return self
    
    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, y="no_validation", reset=False)
        Xc = X - self.mean_
        return np.dot(Xc, self.components_.T)

    def inverse_transform(self, X):
        """Map latent factors back to the original space."""
        check_is_fitted(self)
        return np.dot(X, self.components_) + self.mean_