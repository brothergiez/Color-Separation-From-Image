import numpy as np
from PIL import Image
import os

def rgb_to_hex(rgb):
    return '{:02x}{:02x}{:02x}'.format(*rgb)

def export_layers(labels, centers, out_dir, original_metadata, dot_size=1):
    os.makedirs(out_dir, exist_ok=True)
    h, w = labels.shape
    
    for i, color in enumerate(centers):
        mask = (labels == i).astype(np.uint8) * 255
        hex_color = rgb_to_hex(color)

        if mask.sum() == 0:
            print(f"Skipping layer {i+1} - {hex_color} (empty mask)")
            continue
        
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        pixels = img.load()
        
        if 'dpi' in original_metadata:
            img.info['dpi'] = original_metadata['dpi']
        
        y_indices, x_indices = np.where(mask > 0)
        for y, x in zip(y_indices, x_indices):
            for dy in range(dot_size):
                for dx in range(dot_size):
                    if x+dx < w and y+dy < h:
                        pixels[x+dx, y+dy] = (*color, 255)
        
        save_kwargs = {
            'format': 'PNG',
            'compress_level': 6,
            'dpi': original_metadata.get('dpi', (72, 72))
        }
        
        img.save(f"{out_dir}/layer_{i+1}_{hex_color}.png", **save_kwargs)