import unittest
import os
import tempfile
import numpy as np
from PIL import Image, ImageFilter
from utils.layer_exporter import rgb_to_hex, export_layers, export_smooth_layers

class TestLayerExport(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, 'output')
        
        self.labels = np.array([
            [0, 0, 1],
            [1, 1, 2],
            [2, 2, 0]
        ])
        
        self.centers = np.array([
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255]     # Blue
        ])
        
        self.probs = np.random.rand(3, 3, 3)
        self.metadata = {'dpi': (300, 300), 'format': 'PNG'}
        
    def tearDown(self):
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

    def test_rgb_to_hex(self):
        self.assertEqual(rgb_to_hex((255, 0, 0)), 'ff0000')
        self.assertEqual(rgb_to_hex((0, 255, 0)), '00ff00')
        self.assertEqual(rgb_to_hex((0, 0, 255)), '0000ff')

    def test_export_layers_basic(self):
        export_layers(self.labels, self.centers, self.output_dir, self.metadata)
        
        files = os.listdir(self.output_dir)
        self.assertIn('layer_1_ff0000.png', files)
        
        with Image.open(os.path.join(self.output_dir, 'layer_1_ff0000.png')) as img:
            self.assertAlmostEqual(img.info['dpi'][0], 300, places=2)
            self.assertAlmostEqual(img.info['dpi'][1], 300, places=2)
            pixels = list(img.getdata())
            self.assertIn((255, 0, 0, 255), pixels)

    def test_export_layers_empty_mask(self):
        empty_labels = np.zeros_like(self.labels)
        export_layers(empty_labels, self.centers, self.output_dir, self.metadata)
        files = os.listdir(self.output_dir)
        self.assertEqual(len(files), 1)

    def test_export_layers_dot_size(self):
        export_layers(self.labels, self.centers, self.output_dir, self.metadata, dot_size=2)
        with Image.open(os.path.join(self.output_dir, 'layer_1_ff0000.png')) as img:
            pixels = np.array(img)
            self.assertTrue(np.any(pixels[0,0] == [255, 0, 0, 255]))

    def test_export_smooth_layers(self):
        export_smooth_layers(
            self.labels, 
            self.centers, 
            self.probs, 
            self.output_dir, 
            blur_radius=2,
            original_metadata=self.metadata
        )
        files = os.listdir(self.output_dir)
        self.assertIn('layer_0_#ff0000_hard.png', files)

    def test_export_with_default_metadata(self):
        export_layers(self.labels, self.centers, self.output_dir, {})
        with Image.open(os.path.join(self.output_dir, 'layer_1_ff0000.png')) as img:
            self.assertEqual(tuple(round(x) for x in img.info['dpi']), (72, 72))

if __name__ == '__main__':
    unittest.main()