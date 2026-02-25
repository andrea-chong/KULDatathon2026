import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import umap
import os
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 必須匯入 3D 繪圖工具
import umap
import numpy as np
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def visualize_hdbscan_3d(df, clusterer, name, save_dir):
    BG_WHITE = '#FFFFFF'
    DUO_GREEN = '#58CC02'      # 標誌綠
    DUO_DARK_TEXT = '#4B4B4B'  # 深灰文字
    GRID_LIGHT = '#F5F5F5'     # 極輕背景格線
    
    reducer = umap.UMAP(n_components=3, random_state=42)
    exclude_cols = ['user_id', 'cluster_label', 'cluster_probability', 'soft_cluster_label','soft_cluster_score']
    features = df.drop(columns=exclude_cols, errors='ignore')
    embedding = reducer.fit_transform(features)
    
    labels = df['cluster_label']
    clustered = (labels >= 0)
    unique_labels = sorted(list(set(labels[clustered])))
    
    fig = plt.figure(figsize=(14, 10), facecolor=BG_WHITE)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_facecolor(BG_WHITE)
    
    ax.scatter(embedding[~clustered, 0], embedding[~clustered, 1], embedding[~clustered, 2], 
                color='#EFEFEF', s=10, alpha=0.2, label='Noise')
    
    scatter = ax.scatter(embedding[clustered, 0], embedding[clustered, 1], embedding[clustered, 2], 
                         c=labels[clustered], s=40, cmap='Spectral', alpha=0.85, edgecolors='none')
    
    for label in unique_labels:
        mask = (labels == label)
        if np.any(mask):
            pos = np.median(embedding[mask], axis=0)
            z_offset = (embedding[:, 2].max() - embedding[:, 2].min()) * 0.05
            
            ax.text(pos[0], pos[1], pos[2] + z_offset, 
                    str(label), 
                    fontsize=10, 
                    fontweight='black', 
                    color=DUO_DARK_TEXT,
                    zorder=100,           
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='white', 
                              alpha=0.9, 
                              edgecolor=DUO_GREEN, 
                              boxstyle='round,pad=0.4', 
                              linewidth=1.5))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax.xaxis._axinfo["grid"]['color'] = GRID_LIGHT
    ax.yaxis._axinfo["grid"]['color'] = GRID_LIGHT
    ax.zaxis._axinfo["grid"]['color'] = GRID_LIGHT
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_title('UMAP 3D PERSPECTIVE', size=24, fontweight='black', color=DUO_DARK_TEXT, pad=30)
    
    ax.view_init(elev=25, azim=40)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{name}_3D_UMAP_Duo.png', dpi=300, facecolor=BG_WHITE)
    plt.show()
    
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

def plot_all_clusters_radar_grid(df, importance_dict, n_cols=5):
    BG_WHITE = '#FFFFFF'       
    DUO_GREEN = '#58CC02'      
    DUO_DARK_TEXT = '#4B4B4B'  
    ACCENT_YELLOW = '#FFC800'  
    GRID_GRAY = '#E5E5E5'      

    unique_clusters = sorted([c for c in df['cluster_label'].unique() if c >= 0])
    n_rows = (len(unique_clusters) + n_cols - 1) // n_cols

    plt.rcParams['text.color'] = DUO_DARK_TEXT
    plt.rcParams['axes.labelcolor'] = DUO_DARK_TEXT
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), 
                             subplot_kw=dict(polar=True), facecolor=BG_WHITE)
    axes = axes.flatten()
    
    all_feats = list(set([f for sub in importance_dict.values() for f in sub.index[:5]]))
    df_norm = df.copy()
    df_norm[all_feats] = MinMaxScaler().fit_transform(df[all_feats])
    
    for i, cluster_id in enumerate(unique_clusters):
        ax = axes[i]
        ax.set_facecolor(BG_WHITE)
        
        top_features = list(importance_dict[cluster_id].index[:5])
        clean_features = [f.replace('_', ' ').title() for f in top_features]
        
        stats = df_norm[df_norm['cluster_label'] == cluster_id][top_features].mean().tolist()
        g_mean = df_norm[top_features].mean().tolist()
        
        angles = np.linspace(0, 2*np.pi, len(top_features), endpoint=False).tolist()
        angles += angles[:1]; stats += stats[:1]; g_mean += g_mean[:1]
        
        ax.fill(angles, stats, color=DUO_GREEN, alpha=0.2)
        
        ax.plot(angles, stats, color=DUO_GREEN, linewidth=3.5, zorder=5)
        
        ax.scatter(angles, stats, color=ACCENT_YELLOW, s=40, zorder=10, 
                   edgecolors=DUO_GREEN, linewidth=1.5)
        
        ax.plot(angles, g_mean, color=DUO_DARK_TEXT, linewidth=1, linestyle='--', alpha=0.3)
        
        ax.spines['polar'].set_visible(False) 
        ax.xaxis.grid(True, color=GRID_GRAY, linestyle='-', linewidth=1)
        ax.yaxis.grid(True, color=GRID_GRAY, linestyle='-', linewidth=1)
        
        ax.set_ylim(0, 1.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(clean_features, fontsize=10, fontweight='bold', color=DUO_DARK_TEXT)
        ax.set_yticklabels([])
        
        ax.set_title(f'Group {cluster_id}', size=18, pad=30, fontweight='black', color=DUO_GREEN)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle('SKILL PROFICIENCY DASHBOARD', 
                 size=28, weight='black', color=DUO_DARK_TEXT, y=0.98)
    
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

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