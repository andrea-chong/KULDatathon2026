import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram


def visualize_ahc_with_labels(df, Z, name, save_dir, n_clusters=None):
    """
    Creates a 2-panel dashboard:
      1. UMAP projection colored by cluster labels
      2. Dendrogram from the linkage matrix
    """
    reducer = umap.UMAP(random_state=42)
    features = df.drop(columns=['user_id', 'cluster_label'], errors='ignore')
    embedding = reducer.fit_transform(features)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- 1. UMAP Plot with Labels ---
    labels = df['cluster_label']
    unique_labels = sorted(set(labels))
    n_clusters_found = len(unique_labels)

    scatter = ax1.scatter(embedding[:, 0], embedding[:, 1],
                          c=labels, s=15, cmap='Spectral', alpha=0.8)

    # Add Centroid Labels
    for label in unique_labels:
        mask = (labels == label)
        median_x, median_y = np.median(embedding[mask], axis=0)
        ax1.text(median_x, median_y, str(label), fontsize=12,
                 fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax1.set_title(f'UMAP Projection with Cluster IDs: {name}')

    # --- 2. Dendrogram ---
    # Truncate to last p merges for readability
    p = min(30, n_clusters_found * 3) if n_clusters_found else 30
    dendrogram(Z, ax=ax2, truncate_mode='lastp', p=p,
               leaf_rotation=90, leaf_font_size=8,
               color_threshold=0)  # color_threshold=0 gives uniform color; set to 'default' for auto-coloring
    ax2.set_title(f'Dendrogram: {name}')
    ax2.set_xlabel('Cluster Size (or Sample Index)')
    ax2.set_ylabel('Distance')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{name}_dashboard.png', dpi=150)
    plt.show()


def plot_cluster_top_features_boxplot(df, cluster_id, importance_dict):
    """Same as HDBSCAN version — works with any cluster_label column."""
    top_features = importance_dict[cluster_id].index[:3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, feature in enumerate(top_features):
        df['is_this_cluster'] = (df['cluster_label'] == cluster_id).map(
            {True: f'Cluster {cluster_id}', False: 'Others'})

        sns.boxplot(data=df, x='is_this_cluster', y=feature, ax=axes[i],
                    palette='Set2', hue='is_this_cluster', legend=False, showfliers=False)
        axes[i].set_title(f'Importance of {feature}')

    plt.tight_layout()
    plt.show()


def plot_cluster_top_features_radar(df, cluster_id, importance_dict):
    """Same as HDBSCAN version — works with any cluster_label column."""
    top_features = list(importance_dict[cluster_id].index[:5])

    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[top_features] = scaler.fit_transform(df[top_features])

    cluster_stats = df_norm[df_norm['cluster_label'] == cluster_id][top_features].mean()
    global_stats = df_norm[top_features].mean()

    angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
    angles += angles[:1]
    stats = cluster_stats.tolist() + cluster_stats.tolist()[:1]
    global_mean = global_stats.tolist() + global_stats.tolist()[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, stats, color='teal', alpha=0.25)
    ax.plot(angles, stats, color='teal', linewidth=2, label=f'Cluster {cluster_id}')
    ax.plot(angles, global_mean, color='gray', linewidth=1, linestyle='--', label='Dataset Average')

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features)

    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.title(f'Feature Profile: Cluster {cluster_id}', size=15, pad=20)
    plt.show()