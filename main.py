from utils.image_loader import load_image
from utils.kmeans_cpu import kmeans_cpu
from utils.gmm_cpu import gmm_cpu
from utils.layer_exporter import export_layers, export_smooth_layers

import config

def main():
    path = config.INPUT_IMAGE
    image_data = load_image(path)

    if config.CLUSTER_ALGORITHM == "kmeans":
        labels, centers = kmeans_cpu(
            image_data['array'],
            config.NUM_COLORS,
            ensure_green=config.FORCE_GREEN_COLOR,
            precision_mode=config.KMEANS_PRECISION_MODE,
            use_lab_space=config.USE_LAB_COLORSPACE
        )
        export_layers(
            labels,
            centers,
            config.OUTPUT_DIR,
            original_metadata=image_data,
            dot_size=config.DOT_SIZE
        )
    elif config.CLUSTER_ALGORITHM == "gmm":
        labels, centers, probs = gmm_cpu(
            image_data['array'],
            config.NUM_COLORS,
            config.GMM_COVARIANCE_TYPE
        )
        if config.GMM_EXPORT_SMOOTH:
            export_smooth_layers(
                labels, centers, probs,
                config.OUTPUT_SMOOTH_DIR,
                config.GMM_BLUR_RADIUS,
                original_metadata=image_data
            )
        else:
            export_layers(
            labels,
            centers,
            config.OUTPUT_DIR,
            original_metadata=image_data,
            dot_size=config.DOT_SIZE
        )
    else:
        raise ValueError("Unsupported clustering algorithm: " + config.CLUSTER_ALGORITHM)
    
    

if __name__ == "__main__":
    main()