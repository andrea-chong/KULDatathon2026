import numpy as np

from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors


def hopkins_statistic(X, sampling_size=0.1):
    """
    Calculates the Hopkins Statistic for a high-dimensional dataset.
    X: your 5634-column normalized dataframe or array.
    """
    d = X.shape[1]
    n = len(X)
    m = int(n * sampling_size) # Sample 10% of the data
    
    # 1. Fit Nearest Neighbors on the real data
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    
    # 2. Generate 'm' random points within the feature space
    # (Using the min/max of each feature to define the space)
    rand_points = np.random.uniform(X.min(axis=0), X.max(axis=0), (m, d))
    
    # 3. Distance from random points to nearest real data points
    dist_rand, _ = nbrs.kneighbors(rand_points, n_neighbors=1)
    
    # 4. Distance from a sample of real points to their nearest neighbors
    real_samples = resample(X, n_samples=m, replace=False)
    dist_real, _ = nbrs.kneighbors(real_samples, n_neighbors=2) # 2 because index 0 is the point itself
    
    # Sum of distances
    sum_dist_rand = np.sum(dist_rand)
    sum_dist_real = np.sum(dist_real[:, 1]) # Take the second neighbor
    
    hopkins_val = sum_dist_rand / (sum_dist_rand + sum_dist_real)
    return hopkins_val

# Usage
# score = hopkins_statistic(your_large_dataframe.values)
# print(f"Hopkins Statistic: {score:.4f}")