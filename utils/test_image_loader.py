import unittest
import os
import tempfile
import numpy as np
from PIL import Image
from utils.image_loader import load_image, save_image

class TestImageFunctions(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        self.test_image_path = os.path.join(self.test_dir, 'test.png')
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with Image.fromarray(arr) as img:
            img.save(self.test_image_path, dpi=(300, 300))
        
        self.test_jpg_path = os.path.join(self.test_dir, 'test.jpg')
        with Image.fromarray(arr) as img:
            img.save(self.test_jpg_path, dpi=(150, 150), quality=90, format='JPEG')

    def tearDown(self):
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                filepath = os.path.join(root, name)
                self._safe_remove(filepath)
        
        for _ in range(3):
            try:
                os.rmdir(self.test_dir)
                break
            except OSError:
                import time
                time.sleep(0.1)

    def _safe_remove(self, path):
        """Helper method to safely remove files"""
        for _ in range(3):
            try:
                os.remove(path)
                return
            except PermissionError:
                import time
                time.sleep(0.1)
        raise PermissionError(f"Could not remove {path} after multiple attempts")

    def test_load_image_metadata(self):
        """Test that metadata is correctly loaded"""
        result = load_image(self.test_image_path)
        self.assertAlmostEqual(result['dpi'][0], 300, places=2)
        self.assertAlmostEqual(result['dpi'][1], 300, places=2)

    def test_save_image_preserves_metadata(self):
        """Test that saving preserves metadata"""
        loaded = load_image(self.test_image_path)
        output_path = os.path.join(self.test_dir, 'output.png')
        
        save_image(loaded['array'], output_path, loaded)
        
        with Image.open(output_path) as saved_img:
            self.assertAlmostEqual(saved_img.info['dpi'][0], 300, places=2)
            self.assertAlmostEqual(saved_img.info['dpi'][1], 300, places=2)

    def test_save_with_default_metadata(self):
        """Test saving when some metadata is missing"""
        loaded = load_image(self.test_image_path)
        output_path = os.path.join(self.test_dir, 'output_default.png')
        
        save_image(loaded['array'], output_path, {'format': 'PNG'})
        
        with Image.open(output_path) as saved_img:
            dpi = saved_img.info['dpi']
            self.assertEqual(round(dpi[0]), 72)
            self.assertEqual(round(dpi[1]), 72)
        
    def test_invalid_image_path(self):
        """Test loading with invalid path raises appropriate exception"""
        with self.assertRaises(FileNotFoundError):
            load_image('nonexistent_path.png')

if __name__ == '__main__':
    unittest.main()