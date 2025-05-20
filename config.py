INPUT_IMAGE = "image-target.png"
OUTPUT_DIR = "output_layers"
OUTPUT_SMOOTH_DIR = "output_smooth_layers"

NUM_COLORS = 15
DOT_SIZE = 1
CLUSTER_ALGORITHM = "kmeans"  # "kmeans" or "gmm"
GMM_BLUR_RADIUS = 1.5  # Atur level blur
GMM_COVARIANCE_TYPE = 'full'  # 'full'/'tied'/'diag'
GMM_EXPORT_SMOOTH = False

KMEANS_PRECISION_MODE = True  # False untuk lebih cepat
FORCE_GREEN_COLOR = True      # Prioritaskan hijau
USE_LAB_COLORSPACE = True 