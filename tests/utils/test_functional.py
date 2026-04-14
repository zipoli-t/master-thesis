# Author: Tommaso Zipoli
# License: GNU General Public License v3.0

import pytest
import numpy as np
from numpy.testing import assert_allclose

from src.utils.functional import subspace_overlap_score


class TestSubspaceOverlapScore:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_identity(self, rng):
        """Test that identical subspaces return a score of 1.0."""
        A = rng.standard_normal((100, 10))
        # B is a shuffled/scaled version of A, but spans the same space
        B = A @ rng.standard_normal((10, 10)) 
        
        score = subspace_overlap_score(A, B)
        assert_allclose(score, 1.0, atol=1e-12)

    def test_orthogonality(self):
        """Test that perfectly orthogonal subspaces return a score of 0.0."""
        # Use identity blocks to guarantee orthogonality
        A = np.zeros((10, 2))
        A[:2, :] = np.eye(2)
        
        B = np.zeros((10, 2))
        B[2:4, :] = np.eye(2)
        
        score = subspace_overlap_score(A, B)
        assert_allclose(score, 0.0, atol=1e-12)

    def test_p_greater_than_q(self, rng):
        """Test that the score is normalized by q (the smaller subspace)."""
        # A is 5D subspace, B is 2D subspace
        # Let B be a perfect subset of A
        A = rng.standard_normal((100, 5))
        B = A[:, :2]  
        
        score = subspace_overlap_score(A, B)
        # Should be 1.0 because mean is over q=2
        assert_allclose(score, 1.0, atol=1e-12)

    def test_different_row_counts(self):
        """Test that ValueError is raised for mismatched sample counts."""
        A = np.random.rand(10, 3)
        B = np.random.rand(11, 3)
        with pytest.raises(ValueError, match="same number of rows"):
            subspace_overlap_score(A, B)

    def test_numerical_stability_clip(self, rng):
        """Verify that scores stay within [0, 1] even with noisy inputs."""
        # Create nearly identical subspaces
        A = rng.standard_normal((50, 5))
        B = A + 1e-15 
        
        score = subspace_overlap_score(A, B)
        assert 0.0 <= score <= 1.0

    def test_rotation_invariance(self, rng):
        """Score should be invariant to orthonormal rotation of the basis."""
        A = rng.standard_normal((100, 5))
        B = rng.standard_normal((100, 5))
        
        score1 = subspace_overlap_score(A, B)
        
        # Apply random rotation to B
        q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        B_rotated = B @ q
        
        score2 = subspace_overlap_score(A, B_rotated)
        assert_allclose(score1, score2, atol=1e-12)