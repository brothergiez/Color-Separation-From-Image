import unittest
import numpy as np
from sklearn.mixture import GaussianMixture
from utils.gmm_cpu import gmm_cpu

class TestGMMCPU(unittest.TestCase):
    def setUp(self):
        self.sample_data = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
            [[128, 128, 128], [255, 255, 255], [0, 0, 0]]
        ], dtype=np.uint8)
        
        self.large_data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

    def test_basic_gmm(self):
        """Test basic GMM functionality"""
        labels, centers, probs = gmm_cpu(self.sample_data, k=3, covariance_type='full')
        
        self.assertEqual(labels.shape, (9,))
        self.assertEqual(centers.shape, (3, 3))
        self.assertEqual(probs.shape, (9, 3))
        
        self.assertTrue(np.all(centers >= 0))
        self.assertTrue(np.all(centers <= 255))
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1))

    def test_different_covariance_types(self):
        """Test different covariance types"""
        for cov_type in ['full', 'tied', 'diag', 'spherical']:
            labels, centers, probs = gmm_cpu(self.sample_data, k=2, covariance_type=cov_type)
            self.assertEqual(labels.shape, (9,))
            self.assertEqual(centers.shape, (2, 3))

    def test_output_ranges(self):
        """Test output values are valid"""
        labels, centers, probs = gmm_cpu(self.large_data, k=3, covariance_type='full')
        
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels <= 2))
        
        self.assertTrue(np.all(centers >= 0))
        self.assertTrue(np.all(centers <= 255))
        self.assertEqual(centers.dtype, np.uint8)
        
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1))

    def test_deterministic_results(self):
        """Test results are deterministic with fixed random state"""
        labels1, centers1, probs1 = gmm_cpu(self.sample_data, k=2, covariance_type='full')
        labels2, centers2, probs2 = gmm_cpu(self.sample_data, k=2, covariance_type='full')
        
        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_equal(centers1, centers2)
        np.testing.assert_array_almost_equal(probs1, probs2)

    def test_edge_cases(self):
        """Test edge cases like single color image"""
        solid_red = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)
        labels, centers, probs = gmm_cpu(solid_red, k=2, covariance_type='full')
        
        self.assertEqual(len(centers), 2)
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1))

    def test_cluster_count(self):
        """Test correct number of clusters returned"""
        for k in [1, 2, 3]:
            _, centers, _ = gmm_cpu(self.sample_data, k=k, covariance_type='full')
            self.assertEqual(len(centers), k)

if __name__ == '__main__':
    unittest.main()