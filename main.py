from utils.image_loader import load_image
from utils.kmeans_cpu import kmeans_cpu
from utils.gmm_cpu import gmm_cpu
from utils.layer_exporter import export_layers
import config

def main():
    path = "sample-input.jpg"
    image_data = load_image(path)

    if config.CLUSTER_ALGORITHM == "kmeans":
        labels, centers = kmeans_cpu(
            image_data['array'],
            config.NUM_COLORS,
            ensure_green=True
        )
    elif config.CLUSTER_ALGORITHM == "gmm":
        labels, centers = gmm_cpu(
            image_data['array'],
            config.NUM_COLORS
        )
    else:
        raise ValueError("Unsupported clustering algorithm: " + config.CLUSTER_ALGORITHM)
    
    export_layers(
        labels,
        centers,
        "output_layers",
        original_metadata=image_data,
        dot_size=config.DOT_SIZE
    )

if __name__ == "__main__":
    main()