import unittest
import numpy as np
from skimage.color import rgb2lab
from utils.kmeans_cpu import kmeans_cpu
import warnings
from sklearn.exceptions import ConvergenceWarning

class TestKMeansCPU(unittest.TestCase):
    def setUp(self):
        self.sample_data = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 0], [0, 255, 0], [0, 255, 255]],
            [[128, 128, 128], [0, 255, 0], [0, 0, 0]]
        ], dtype=np.uint8)
        
        self.large_data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

    def test_ensure_green(self):
        """Test that ensure_green forces a green center"""
        for _ in range(5): 
            _, centers = kmeans_cpu(self.sample_data, k=3, ensure_green=True)
            
            lab_centers = rgb2lab(centers.reshape(1, -1, 3)).reshape(-1, 3)
            green_lab = rgb2lab(np.array([[[0, 255, 0]]])).reshape(3)
            
            distances = np.linalg.norm(lab_centers - green_lab, axis=1)
            
            if np.any(distances < 10):
                return
        
        self.fail("No green center found after 5 attempts with ensure_green=True")

    def test_basic_kmeans(self):
        """Test basic k-means clustering"""
        labels, centers = kmeans_cpu(self.sample_data, k=2)
        self.assertEqual(labels.shape, self.sample_data.shape[:2])
        self.assertEqual(centers.shape, (2, 3))

    def test_precision_mode(self):
        """Test precision mode uses all pixels for final prediction"""
        labels, _ = kmeans_cpu(self.large_data, k=2, precision_mode=True)
        self.assertEqual(labels.shape, self.large_data.shape[:2])

    def test_lab_space(self):
        """Test LAB colorspace conversion"""
        _, centers_rgb = kmeans_cpu(self.sample_data, k=2, use_lab_space=False)
        _, centers_lab = kmeans_cpu(self.sample_data, k=2, use_lab_space=True)
        self.assertFalse(np.allclose(centers_rgb, centers_lab, atol=10))

    def test_cluster_count(self):
        """Test correct number of clusters returned"""
        for k in [1, 2, 3]:
            _, centers = kmeans_cpu(self.sample_data, k=k)
            self.assertEqual(len(centers), k)

    def test_output_ranges(self):
        """Test output values are valid"""
        labels, centers = kmeans_cpu(self.large_data, k=3)
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels <= 2))
        self.assertTrue(np.all(centers >= 0))
        self.assertTrue(np.all(centers <= 255))
        self.assertEqual(centers.dtype, np.uint8)

    def test_deterministic_results(self):
        """Test results are deterministic with fixed random state"""
        labels1, centers1 = kmeans_cpu(self.sample_data, k=2)
        labels2, centers2 = kmeans_cpu(self.sample_data, k=2)
        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_equal(centers1, centers2)

    def test_edge_cases(self):
        """Test edge cases like single color image"""
        solid_red = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            labels, centers = kmeans_cpu(solid_red, k=2)
        self.assertEqual(len(centers), 2)
        self.assertTrue(np.all(labels >= 0))

if __name__ == '__main__':
    unittest.main()