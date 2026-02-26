import os
import pickle
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from scipy.cluster.hierarchy import linkage, fcluster

def run_ahc_pipeline(df, user_col='user_id', n_components=30, n_clusters=10, linkage_method='ward', distance_threshold=None):
    """
    Runs Agglomerative Hierarchical Clustering on a dataframe.
    
    Parameters:
        df: DataFrame with user_col + feature columns
        n_components: PCA target dims (only applied if features > n_components)
        n_clusters: Number of clusters to form (ignored if distance_threshold is set)
        linkage_method: 'ward', 'complete', 'average', or 'single'
        distance_threshold: If set, n_clusters is ignored and the tree is cut at this distance
    """
    # 1. Separate ID from features
    features = df.drop(columns=[user_col])

    # 2. Dimensionality Reduction
    if features.shape[1] > n_components:
        print(f"Reducing dimensions from {features.shape[1]} to {n_components}...")
        reducer = PCA(n_components=n_components)
        data_to_cluster = reducer.fit_transform(features)
    else:
        data_to_cluster = features.values

    # 3. Compute full linkage matrix (needed for dendrogram + flexible cutting)
    print(f"Computing linkage matrix (method={linkage_method})...")
    Z = linkage(data_to_cluster, method=linkage_method)

    # 4. Fit AgglomerativeClustering
    if distance_threshold is not None:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage=linkage_method
        )
    else:
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )

    labels = clusterer.fit_predict(data_to_cluster)

    # 5. Attach results
    result_df = df.copy()
    result_df['cluster_label'] = labels

    n_found = len(set(labels))
    print(f"Agglomerative Clustering found {n_found} clusters.")
    for c in sorted(set(labels)):
        count = (labels == c).sum()
        print(f"  Cluster {c}: {count} users ({count/len(labels):.1%})")

    return result_df, clusterer, Z


def save_clustering_results(df, clusterer, Z, name, save_dir="data/tmp/ahc"):
    """
    Saves the dataframe to CSV and the clusterer + linkage matrix for later visualization.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save the labeled data
    df.to_csv(f'{save_dir}/{name}_results.csv', index=False)

    # Save the clusterer object and linkage matrix
    with open(f'{save_dir}/{name}_model.pkl', 'wb') as f:
        pickle.dump({'clusterer': clusterer, 'linkage_matrix': Z}, f)

    print(f"Saved results for {name} to '{save_dir}/' folder.")


def get_all_cluster_importances(df, features_list):
    """
    Identifies top features for each cluster using a Random Forest (same as HDBSCAN version).
    """
    results = {}
    unique_clusters = sorted(df['cluster_label'].unique())

    for cluster_id in unique_clusters:
        y = (df['cluster_label'] == cluster_id).astype(int)
        X = df[features_list]

        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=features_list)
        results[cluster_id] = importances.sort_values(ascending=False).head(5)

    return results