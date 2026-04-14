# Author: Tommaso Zipoli
# License: GNU General Public License v3.0

import numpy as np
from sklearn.utils import check_array


__all__ = ["subspace_overlap_score"]


def subspace_overlap_score(A, B):
    r"""Compute the CCA-based overlap score between two subspaces.

    The score is derived from the principal angles between the column
    spaces of $A \in \mathbb{R}^{n \times p}$ and $B \in
    \mathbb{R}^{n \times q}$ ($p \geq q$). It is computed as the mean
    squared canonical correlation:

    .. math::
        \mathcal{S} = \frac{1}{q} \sum_{i=1}^q \rho_i^2

    Principal angles are calculated via QR factorization followed by 
    SVD (Golub & Van Loan, Alg. 6.4.3).

    Args:
        A (array_like): First subspace matrix of shape (n, p).
        B (array_like): Second subspace matrix of shape (n, q).

    Returns:
        float: Overlap score $\mathcal{S} \in [0, 1]$. A value of 1.0 
            indicates identity; 0.0 indicates orthogonality.

    Raises:
        ValueError: If A and B have a different number of rows.

    References:
        Golub, G. H., & Van Loan, C. F. (2013). "Matrix Computations". 
        4th Ed.
    """
    A = check_array(A, ensure_2d=True)
    B = check_array(B, ensure_2d=True)

    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"Subspaces must have the same number of rows. "
            f"Found {A.shape[0]} and {B.shape[0]}."
        )

    # Orthonormalize bases using reduced QR
    qa, _ = np.linalg.qr(A, mode='reduced')
    qb, _ = np.linalg.qr(B, mode='reduced')

    # Singular values of the product of orthonormal bases are the 
    # cosines of the principal angles (canonical correlations).
    s = np.linalg.svd(np.dot(qa.T, qb), compute_uv=False)
    rho = np.clip(s, 0.0, 1.0)

    return np.mean(rho**2)
