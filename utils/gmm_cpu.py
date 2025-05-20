import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def gmm_cpu(data, k, covariance_type, max_iter=200):
    """GMM dengan covariance_type='full' dan normalisasi data"""
    flat_data = data.reshape((-1, 3))
    
    scaler = StandardScaler()
    flat_data_scaled = scaler.fit_transform(flat_data)
    
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type, 
        max_iter=max_iter,
        random_state=42,
        tol=1e-5
    )
    gmm.fit(flat_data_scaled)
    
    labels = gmm.predict(flat_data_scaled)
    probs = gmm.predict_proba(flat_data_scaled)
    
    centers = scaler.inverse_transform(gmm.means_)
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    
    return labels, centers, probs