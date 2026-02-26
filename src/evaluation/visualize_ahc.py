import os
import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram

def visualize_ahc_with_labels(df, Z, name, save_dir, n_clusters=None, outlier_percentile=2):
    """
    Creates a 5-row vertical dashboard:
      1. PCA 2D
      2. PCA 3D
      3. UMAP 2D
      4. UMAP 3D
      5. Dendrogram
    
    outlier_percentile: removes the top/bottom X% of points per axis to declutter the plots.
                        Set to 0 to disable.
    """
    from sklearn.decomposition import PCA
    import umap

    features = df.drop(columns=['user_id', 'cluster_label'], errors='ignore')
    labels = df['cluster_label'].values
    unique_labels = sorted(set(labels))
    n_clusters_found = len(unique_labels)

    # Compute projections
    pca_2d = PCA(n_components=2).fit_transform(features)
    pca_3d = PCA(n_components=3).fit_transform(features)
    umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(features)
    umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(features)

    # --- Outlier mask: keep points within percentile range across all projections ---
    def inlier_mask(embedding, percentile):
        if percentile <= 0:
            return np.ones(len(embedding), dtype=bool)
        low = np.percentile(embedding, percentile, axis=0)
        high = np.percentile(embedding, 100 - percentile, axis=0)
        return np.all((embedding >= low) & (embedding <= high), axis=1)

    mask_pca_2d = inlier_mask(pca_2d, outlier_percentile)
    mask_pca_3d = inlier_mask(pca_3d, outlier_percentile)
    mask_umap_2d = inlier_mask(umap_2d, outlier_percentile)
    mask_umap_3d = inlier_mask(umap_3d, outlier_percentile)

    removed = (~mask_pca_2d).sum() + (~mask_umap_2d).sum()
    print(f"Outlier removal ({outlier_percentile}th percentile): hiding ~{(~mask_pca_2d).sum()} PCA / ~{(~mask_umap_2d).sum()} UMAP outlier points from plots")

    fig = plt.figure(figsize=(14, 36))

    # --- 1. PCA 2D ---
    ax1 = fig.add_subplot(5, 1, 1)
    m = mask_pca_2d
    ax1.scatter(pca_2d[m, 0], pca_2d[m, 1],
                c=labels[m], s=15, cmap='Spectral', alpha=0.8)
    for label in unique_labels:
        mask = (labels == label) & m
        if mask.any():
            mx, my = np.median(pca_2d[mask], axis=0)
            ax1.text(mx, my, str(label), fontsize=12, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax1.set_title(f'PCA 2D: {name}')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')

    # --- 2. PCA 3D ---
    ax2 = fig.add_subplot(5, 1, 2, projection='3d')
    m = mask_pca_3d
    ax2.scatter(pca_3d[m, 0], pca_3d[m, 1], pca_3d[m, 2],
                c=labels[m], s=15, cmap='Spectral', alpha=0.8)
    ax2.set_title(f'PCA 3D: {name}')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')

    # --- 3. UMAP 2D ---
    ax3 = fig.add_subplot(5, 1, 3)
    m = mask_umap_2d
    ax3.scatter(umap_2d[m, 0], umap_2d[m, 1],
                c=labels[m], s=15, cmap='Spectral', alpha=0.8)
    for label in unique_labels:
        mask = (labels == label) & m
        if mask.any():
            mx, my = np.median(umap_2d[mask], axis=0)
            ax3.text(mx, my, str(label), fontsize=12, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax3.set_title(f'UMAP 2D: {name}')

    # --- 4. UMAP 3D ---
    ax4 = fig.add_subplot(5, 1, 4, projection='3d')
    m = mask_umap_3d
    ax4.scatter(umap_3d[m, 0], umap_3d[m, 1], umap_3d[m, 2],
                c=labels[m], s=15, cmap='Spectral', alpha=0.8)
    ax4.set_title(f'UMAP 3D: {name}')
    ax4.set_xlabel('UMAP1')
    ax4.set_ylabel('UMAP2')
    ax4.set_zlabel('UMAP3')

    # --- 5. Dendrogram ---
    ax5 = fig.add_subplot(5, 1, 5)
    p = min(30, n_clusters_found * 3) if n_clusters_found else 30
    dendrogram(Z, ax=ax5, truncate_mode='lastp', p=p,
               leaf_rotation=90, leaf_font_size=8, color_threshold=0)
    ax5.set_title(f'Dendrogram: {name}')
    ax5.set_xlabel('Cluster Size (or Sample Index)')
    ax5.set_ylabel('Distance')

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