import os
import hdbscan
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def run_hdbscan_pipeline(df, user_col='user_id', n_components=30, epsilon=0, min_cluster_size=15, min_samples=5):
    """
    Runs clustering on a dataframe and returns the DF with labels and the clusterer object.
    """
    # 1. Separate ID from features
    features = df.drop(columns=[user_col])
    
    # 2. Dimensionality Reduction
    # We use PCA here as a fast baseline; swap for UMAP if you have it installed.
    if features.shape[1] > n_components:
        print(f"Reducing dimensions from {features.shape[1]} to {n_components}...")
        reducer = PCA(n_components=n_components)
        data_to_cluster = reducer.fit_transform(features)
    else:
        data_to_cluster = features.values

    # 3. Initialize and Fit HDBSCAN
    # 'allow_single_cluster' helps if your data is very dense/homogeneous
    clusterer = hdbscan.HDBSCAN(
        cluster_selection_epsilon=epsilon,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True, 
        gen_min_span_tree=True,
        cluster_selection_method='leaf'
    )

    fit = clusterer.fit(data_to_cluster) # Use .fit() to ensure the model is ready for soft clustering
    
    # 4. Standard Results
    labels = fit.labels_
    probs = fit.probabilities_

    # 5. Generate Soft Clustering Results
    # This generates a matrix of [n_samples, n_clusters]
    membership_vectors = hdbscan.all_points_membership_vectors(clusterer)
    # Assign each point to its most likely cluster (even if it was noise)
    soft_labels = np.argmax(membership_vectors, axis=1)
    # Get the probability score for that specific assignment
    soft_scores = np.max(membership_vectors, axis=1)
    
    # 6. Attach results
    result_df = df.copy()
    result_df['cluster_label'] = labels
    result_df['cluster_probability'] = probs
    result_df['soft_cluster_label'] = soft_labels
    result_df['soft_cluster_score'] = soft_scores
    
    print(f"Original Noise: {list(labels).count(-1)} ({list(labels).count(-1)/len(labels):.2%})")
    print(f"Soft Clustering has re-assigned all points to {len(set(soft_labels))} clusters.")
    
    return result_df, fit

def save_clustering_results(df, clusterer, name, save_dir="data/tmp"):
    """
    Saves the dataframe to CSV and the clusterer object for tree visualization.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the labeled data
    df.to_csv(f'{save_dir}/{name}_results.csv', index=False)
    
    # Save the clusterer object (to plot trees later without re-running)
    with open(f'{save_dir}/{name}_model.pkl', 'wb') as f:
        import pickle
        pickle.dump(clusterer, f)
        
    print(f"Saved results for {name} to '{save_dir}/' folder.")

def get_all_cluster_importances(df, features_list):
    """
    Identifies top features for each cluster.
    """
    results = {}
    # Filter for only clustered points (exclude noise -1)
    clustered_df = df[df['cluster_label'] >= 0].copy()
    
    unique_clusters = sorted(clustered_df['cluster_label'].unique())
    
    for cluster_id in unique_clusters:
        # Target: 1 if in the cluster, 0 if in any other cluster
        y = (clustered_df['cluster_label'] == cluster_id).astype(int)
        X = clustered_df[features_list]
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X, y)
        
        # Pull importance and store
        importances = pd.Series(rf.feature_importances_, index=features_list)
        results[cluster_id] = importances.sort_values(ascending=False).head(5)
        
    return results