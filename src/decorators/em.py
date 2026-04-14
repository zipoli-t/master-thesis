# Author: Tommaso Zipoli
# License: GNU General Public License v3.0

from __future__ import annotations
import numpy as np
import warnings
from functools import wraps
from sklearn.base import TransformerMixin, clone
from sklearn.utils import check_array
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning


__all__ = ["EM"]


class EM:
    """Expectation-Maximization imputation using a transformer.

    This class provides iterative imputation for missing values by 
    alternating between estimating latent factors and updating 
    the missing entries of the input matrix.

    Args:
        tf (TransformerMixin): Transformer (e.g., PCA) used to 
            extract factors. Defaults to PCA().
        max_iter (int): Maximum number of iterations. Defaults to 100.
        atol (float): Absolute tolerance for convergence. 
            Defaults to 1e-3.
        rtol (float): Relative tolerance for convergence. 
            Defaults to 1e-4.

    Returns:
        Ximputed (np.ndarray): The input matrix with missing values 
            filled.
        factors (np.ndarray): The latent factors extracted during 
            the last iteration.

    Raises:
        TypeError: If tf is not a scikit-learn transformer.
        ValueError: If max_iter, atol, or rtol are invalid, or if 
            the input contains rows/columns of only NaNs.

    Warns:
        ConvergenceWarning: If the algorithm fails to converge 
            within max_iter.

    Examples:
        Direct call usage:
        >>> import numpy as np
        >>> from sklearn.decomposition import PCA
        >>> X = [[1, 2], [np.nan, 4], [5, 6]]
        >>> imputer = EM(tf=PCA(n_components=1), max_iter=50)
        >>> X_filled, factors = imputer(X)

        Decorator usage:
        >>> @imputer.decorate
        ... def calculate_mean(data):
        ...     return data.mean()
        >>> result = calculate_mean(X)
    """

    def __init__(
            self, 
            tf: TransformerMixin = PCA(), 
            max_iter: int = 100, 
            atol: float = 1e-3, 
            rtol: float = 1e-4
        ) -> None:
        self.tf = tf
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol

    def __call__(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Execute the EM imputation algorithm."""
        if not isinstance(self.tf, TransformerMixin):
            raise TypeError("tf must be a scikit-learn transformer.")
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer.")
        
        for name, val in [("atol", self.atol), ("rtol", self.rtol)]:
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(f"{name} must be a non-negative number.")

        X = check_array(X, ensure_all_finite='allow-nan')
        nan_mask = np.isnan(X)
        
        if np.any(np.all(nan_mask, axis=0)):
            raise ValueError("One or more columns contain only NaN values.")
        if np.any(np.all(nan_mask, axis=1)):
            raise ValueError("One or more rows contain only NaN values.")
        
        if not np.any(nan_mask):
            tf_fitted = clone(self.tf).fit(X)
            return X, tf_fitted.transform(X)

        tf = clone(self.tf)
        Ximputed = X.copy()
        
        # Initial imputation using column means
        mu = np.nanmean(X, axis=0, keepdims=True)
        Ximputed[nan_mask] = np.take(mu, np.where(nan_mask)[1])
        
        Xhat_prev = None
        converged = False

        for _ in range(self.max_iter):
            Xc = Ximputed - mu
            factors = tf.fit_transform(Xc)
            Xhat = mu + tf.inverse_transform(factors)
            
            # Robust convergence test on Xhat
            if Xhat_prev is not None:
                diff = np.linalg.norm(Xhat - Xhat_prev)
                threshold = self.atol + self.rtol * np.linalg.norm(Xhat_prev)
                if diff < threshold:
                    converged = True
                    break
            
            Xhat_prev = Xhat.copy()
            Ximputed[nan_mask] = Xhat[nan_mask]
            mu = Ximputed.mean(axis=0, keepdims=True)

        if self.max_iter > 0 and not converged:
            warnings.warn(
                f"EM did not converge after {self.max_iter} iterations.",
                ConvergenceWarning
            )

        return Ximputed, factors
    
    def decorate(self, func):
        """Decorator to impute X before calling the decorated function."""
        @wraps(func)
        def wrapper(X, *args, **kwargs):
            Ximputed, _ = self(X)
            return func(Ximputed, *args, **kwargs)
        return wrapper
