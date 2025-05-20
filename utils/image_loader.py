from PIL import Image
import numpy as np

def load_image(path):
    """Load image with metadata"""
    with Image.open(path) as img:
        return {
            'array': np.array(img.convert('RGB')),
            'dpi': img.info.get('dpi', (72, 72)),
            'mode': img.mode,
            'format': img.format
        }

def save_image(array, path, metadata=None):
    """Save image with original metadata"""
    if metadata is None:
        metadata = {}
    
    img = Image.fromarray(array.astype(np.uint8))
    save_args = {
        'format': metadata.get('format', 'PNG'),
        'dpi': metadata.get('dpi', (72, 72))
    }
    
    img.save(path, **save_args)
    img.close()