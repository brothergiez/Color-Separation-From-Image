import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb


def kmeans_cpu(data, k, max_iter=20, ensure_green=False, precision_mode=False, use_lab_space=False):
    original_shape = data.shape[:2]
    flat_data = data.reshape((-1, 3))
    
    if precision_mode:
        sample_size = len(flat_data) // 5
        sampled_indices = np.random.choice(len(flat_data), sample_size, replace=False)
        sampled_data = flat_data[sampled_indices]
    else:
        sampled_data = flat_data
    
    if use_lab_space:
        sampled_data = rgb2lab(sampled_data)
    
    if ensure_green:
        init_centers = np.zeros((k, sampled_data.shape[1]))
        init_centers[0] = [0, 128, 0] if use_lab_space else [0, 255, 0]
        if k > 1:
            init_centers[1:] = sampled_data[np.random.choice(len(sampled_data), k-1)]
        n_init = 1
    else:
        init_centers = 'k-means++'
        n_init = 5
    
    kmeans = KMeans(
        n_clusters=k,
        init=init_centers,
        max_iter=max_iter,
        tol=1e-6,
        n_init=n_init,
        random_state=42
    ).fit(sampled_data)
    
    if precision_mode:
        if use_lab_space:
            full_data = rgb2lab(flat_data)
        else:
            full_data = flat_data
        labels = kmeans.predict(full_data)
    else:
        labels = kmeans.labels_
    
    centers = kmeans.cluster_centers_
    if use_lab_space:
        centers = lab2rgb(centers) * 255
    
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    print(f"🕜 Please wait... image separation proccessing to {k} layers...")

    return labels.reshape(original_shape), centers