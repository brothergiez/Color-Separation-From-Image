import numpy as np
from sklearn.mixture import GaussianMixture

def gmm_cpu(data, k, max_iter=100):
    flat_data = data.reshape((-1, data.shape[-1]))

    gmm = GaussianMixture(
        n_components=k,
        max_iter=max_iter,
        covariance_type='tied',
        random_state=42
    )
    gmm.fit(flat_data)

    labels = gmm.predict(flat_data)
    labels = labels.reshape(data.shape[:2])

    centers = gmm.means_
    centers = np.clip(centers, 0, 255).astype(np.uint8)

    return labels, centers