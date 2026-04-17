# Author: Tommaso Zipoli
# Licence: GNU General Public License v3.0

from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.validation import check_is_fitted, validate_data

from ...utils.functional import subspace_overlap_score


class TailRobustPCA(TransformerMixin, BaseEstimator):
    r"""Tail-robust Principal Component Analysis (PCA).
    
    Refined implementation strictly following Barigozzi, Cho, and Maeng (2025).
    
    This version implements the "double-truncation" logic:
    1. Truncate raw data to estimate robust loadings (components).
    2. Truncate the projected factor scores to ensure robust latent estimates.
    """
    def __init__(self, n_components=None, thr=1.0, thr_factors=None):
        self.n_components = n_components
        self.thr = thr
        # Paper suggests a second truncation level for factors
        self.thr_factors = thr_factors 

    def fit(self, X, y=None):
        X = validate_data(self, X=X, y=y, reset=True)
        n_samples, n_features = X.shape
        
        self.n_components_ = self.n_components or min(n_samples, n_features)

        # 1. Robust Location
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_

        # 2. First Truncation (Data level)
        # Used to stabilize the sample covariance matrix/SVD
        X_truncated = np.clip(Xc, -self.thr, self.thr)

        # 3. SVD for Loadings
        _, s, Vh = np.linalg.svd(X_truncated, full_matrices=False)
        
        # Eigenvalues of the truncated sample covariance
        eigenvalues = (s**2) / n_samples
        
        self.components_ = Vh[:self.n_components_]
        self.singular_values_ = s[:self.n_components_]
        self.explained_variance_ = eigenvalues[:self.n_components_]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / np.sum(eigenvalues)
        )
        
        return self
        
    def transform(self, X):
        """Project and apply the second truncation to factors."""
        check_is_fitted(self)
        X = validate_data(self, X=X, y="no_validation", reset=False)
        
        Xc = X - self.mean_
        
        # Step A: Projection using truncated data (as per Section 3.1)
        X_truncated = np.clip(Xc, -self.thr, self.thr)
        factors = np.dot(X_truncated, self.components_.T)
        
        # Step B: Second Truncation (Factor level)
        # If thr_factors is not set, we default to the same thr or no clipping
        if self.thr_factors is not None:
            factors = np.clip(factors, -self.thr_factors, self.thr_factors)
            
        return factors

    def inverse_transform(self, X):
        """Map latent factors back to the original space."""
        check_is_fitted(self)
        # X here are the 'double-truncated' factors.
        # Reconstruction: chi = F * Lambda'
        return np.dot(X, self.components_) + self.mean_

    def score_samples(self, X):
        raise NotImplementedError
    
    def score(self, X, y = None):
        check_is_fitted(self)
        
        # Train set
        train_load_subspace = self.components_.T

        # Validation set
        X = validate_data(self, X=X, y="no_validation", reset=False)
        Xc = X - self.mean_
        X_truncated = np.clip(Xc, -self.thr, self.thr)
        _, _, Vh = np.linalg.svd(X_truncated, full_matrices=False, compute_uv=True)
        val_load_subspace = Vh[:self.n_components_].T

        # Cross-fold stability of loading space
        return subspace_overlap_score(train_load_subspace, val_load_subspace)
        

class TailRobustPCACV(GridSearchCV):
    def __init__(
            self,
            n_components=None, 
            cv=TimeSeriesSplit(n_splits=3), 
            param_grid=None,
            **kwargs
        ):
        # 1. Store the parameter on 'self' so sklearn can find it
        self.n_components = n_components
        self.cv = cv
        
        # 2. Initialize the base estimator with that parameter
        estimator = TailRobustPCA(n_components=self.n_components)
        
        pg = param_grid if param_grid is not None else {}
        super().__init__(estimator=estimator, param_grid=pg, cv=self.cv, **kwargs)
    
    def fit(self, X, y=None, **fit_params):
        """Dynamic grid generation and fitting."""
        # 1. Generate the threshold grid based on the actual data scale
        Xabs = np.abs(X - np.mean(X, axis=0))
        # Start at median of absolute deviations, end at max (no truncation)
        start, stop = np.median(Xabs), np.max(Xabs)
        
        # Guard against zero-variance features
        if start == 0: start = 1e-6 
        
        # Populate the grid for 'thr'
        self.param_grid['thr'] = np.geomspace(start, stop, num=50)

        # 2. Run the standard GridSearchCV fit
        super().fit(X, y, **fit_params)

        # 3. Post-fit adjustment: Set thr_factors = thr as per paper recommendation
        # This ensures the best model found by CV uses the dual-truncation
        best_thr = self.best_params_['thr']
        self.best_estimator_.set_params(thr_factors=best_thr)
        
        # Refit the best estimator on the full data with the finalized thr_factors
        self.best_estimator_.fit(X, y)

        return self