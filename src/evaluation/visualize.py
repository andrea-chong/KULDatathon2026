import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import umap
import os
from sklearn.preprocessing import MinMaxScaler

def visualize_hdbscan_with_labels(df, clusterer, name, save_dir):
    reducer = umap.UMAP(random_state=42)
    # Ensure we only use numeric features for UMAP
    features = df.drop(columns=['user_id', 'cluster_label', 'cluster_probability', 'soft_cluster_label','soft_cluster_score'], errors='ignore')
    embedding = reducer.fit_transform(features)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- 1. UMAP Plot with Labels ---
    labels = df['cluster_label']
    clustered = (labels >= 0)
    
    # Plot Noise
    ax1.scatter(embedding[~clustered, 0], embedding[~clustered, 1], 
                color='lightgray', s=10, alpha=0.3, label='Noise')
    
    # Plot Clusters
    scatter = ax1.scatter(embedding[clustered, 0], embedding[clustered, 1], 
                         c=labels[clustered], s=15, cmap='Spectral', alpha=0.8)
    
    # Add Centroid Labels
    unique_labels = set(labels[clustered])
    for label in unique_labels:
        # Calculate the median position for the label
        mask = (labels == label)
        median_x, median_y = np.median(embedding[mask], axis=0)
        ax1.text(median_x, median_y, str(label), fontsize=12, 
                 fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax1.set_title(f'UMAP Projection with Cluster IDs: {name}')

    # Get the color palette from the UMAP
    n_clusters = len(unique_labels)
    colors = sns.color_palette('Spectral', n_clusters)
    
    # --- 2. Condensed Tree ---
    plt.sca(ax2) # Set current axis for HDBSCAN
    dark_grey_cmap = LinearSegmentedColormap.from_list('dark_grey', ['0.9', '0.0'])
    clusterer.condensed_tree_.plot(select_clusters=True,
                                   selection_palette=colors, # Colors the selection lines like the UMAP
                                   label_clusters=False,
                                   cmap=dark_grey_cmap                 
                                   )
    ax2.set_title(f'Condensed Tree (Stability): {name}')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{name}_dashboard.png')
    plt.show()

def plot_cluster_top_features_boxplot(df, cluster_id, importance_dict):
    top_features = importance_dict[cluster_id].index[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, feature in enumerate(top_features):
        # Create a 'Focus' column for visualization
        df['is_this_cluster'] = (df['cluster_label'] == cluster_id).map({True: f'Cluster {cluster_id}', False: 'Others'})
        
        sns.boxplot(data=df, x='is_this_cluster', y=feature, ax=axes[i], palette='Set2', hue='is_this_cluster', legend=False, showfliers=False)
        axes[i].set_title(f'Importance of {feature}')
        
    plt.tight_layout()
    plt.show()

def plot_cluster_top_features_radar(df, cluster_id, importance_dict):
    # 1. Get top 5 features for this cluster
    top_features = list(importance_dict[cluster_id].index[:5])
    
    # 2. Normalize features for the plot
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[top_features] = scaler.fit_transform(df[top_features])
    
    # 3. Calculate means for this cluster vs everyone else
    cluster_stats = df_norm[df_norm['cluster_label'] == cluster_id][top_features].mean()
    global_stats = df_norm[top_features].mean()
    
    # 4. Prepare Radar Chart Geometry
    angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
    # "Close" the loop for the radar chart
    angles += angles[:1]
    stats = cluster_stats.tolist() + cluster_stats.tolist()[:1]
    global_mean = global_stats.tolist() + global_stats.tolist()[:1]
    
    # 5. Plotting
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Cluster Shape
    ax.fill(angles, stats, color='teal', alpha=0.25)
    ax.plot(angles, stats, color='teal', linewidth=2, label=f'Cluster {cluster_id}')
    
    # Global Baseline (for comparison)
    ax.plot(angles, global_mean, color='gray', linewidth=1, linestyle='--', label='Dataset Average')
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.title(f'Feature Profile: Cluster {cluster_id}', size=15, pad=20)
    plt.show()