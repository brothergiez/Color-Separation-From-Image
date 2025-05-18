from PIL import Image
import numpy as np

def load_image(path):
    """Load image with metadata"""
    img = Image.open(path)
    return {
        'array': np.array(img.convert('RGB')),
        'dpi': img.info.get('dpi', (72, 72)),
        'mode': img.mode,
        'format': img.format
    }

def save_image(array, path, metadata):
    """Save image with original metadata"""
    img = Image.fromarray(array.astype(np.uint8))
    img.save(path, dpi=metadata['dpi'], format=metadata.get('format', 'PNG'))