import numpy as np
from sklearn.cluster import KMeans

def is_green_dominant(color, threshold=30):
    r, g, b = color
    return (g > r + threshold) and (g > b + threshold)

def kmeans_cpu(data, k, max_iter=20, ensure_green=False):
    flat_data = data.reshape((-1, data.shape[-1]))
    
    kmeans = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        n_init=3,
        init='k-means++',
        tol=1e-4
    )
    
    kmeans.fit(flat_data)
    
    centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)
    labels = kmeans.labels_.reshape(data.shape[:2])
    
    if ensure_green:
        green_found = any(is_green_dominant(center) for center in centers)
        print("✅ Cluster warna hijau terdeteksi." if green_found 
              else "⚠️ Tidak ditemukan cluster warna hijau yang dominan.")

    return labels, centers