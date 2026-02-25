import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram

def visualize_ahc_with_labels(df, Z, name, save_dir, n_clusters=None):
    """
    Creates a 3-row vertical dashboard:
      1. PCA projection colored by cluster labels
      2. UMAP projection colored by cluster labels
      3. Dendrogram from the linkage matrix
    """
    from sklearn.decomposition import PCA
    import umap

    features = df.drop(columns=['user_id', 'cluster_label'], errors='ignore')
    labels = df['cluster_label']
    unique_labels = sorted(set(labels))
    n_clusters_found = len(unique_labels)

    pca_embedding = PCA(n_components=2).fit_transform(features)
    umap_embedding = umap.UMAP(random_state=42).fit_transform(features)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 22))

    # --- 1. PCA Plot ---
    ax1.scatter(pca_embedding[:, 0], pca_embedding[:, 1],
                c=labels, s=15, cmap='Spectral', alpha=0.8)
    for label in unique_labels:
        mask = (labels == label)
        mx, my = np.median(pca_embedding[mask], axis=0)
        ax1.text(mx, my, str(label), fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax1.set_title(f'PCA Projection: {name}')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')

    # --- 2. UMAP Plot ---
    ax2.scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                c=labels, s=15, cmap='Spectral', alpha=0.8)
    for label in unique_labels:
        mask = (labels == label)
        mx, my = np.median(umap_embedding[mask], axis=0)
        ax2.text(mx, my, str(label), fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax2.set_title(f'UMAP Projection: {name}')

    # --- 3. Dendrogram ---
    p = min(30, n_clusters_found * 3) if n_clusters_found else 30
    dendrogram(Z, ax=ax3, truncate_mode='lastp', p=p,
               leaf_rotation=90, leaf_font_size=8, color_threshold=0)
    ax3.set_title(f'Dendrogram: {name}')
    ax3.set_xlabel('Cluster Size (or Sample Index)')
    ax3.set_ylabel('Distance')

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