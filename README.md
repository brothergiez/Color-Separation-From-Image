# Color Separation From Image
A Python tool for automated color separation of images using clustering algorithms. This project analyzes an input image and separates it into distinct color layers, which can be useful for screen printing, digital art preparation, or image analysis tasks.

## Features

- Clustering Algorithms: Supports both K-Means and Gaussian Mixture Models (GMM) for color separation
- Color Layer Export: Generates individual PNG layers for each dominant color
- Green Channel Enhancement: Optional prioritization of green tones in color detection
- Image Metadata Preservation: Maintains original DPI and format information
- Configurable Parameters: Adjustable cluster count and dot size for output

## Technical Details

- Implemented with Python and scikit-learn
- Processes standard image formats (JPG, PNG, etc.)
- Lightweight with minimal dependencies (NumPy, Pillow, scikit-learn)

## Use Cases

- Preparing artwork for screen printing
- Creating color separations for digital art
- Image analysis and dominant color extraction
- Educational purposes for studying color spaces

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Configure settings in config.py
3. Run: `python main.py`

