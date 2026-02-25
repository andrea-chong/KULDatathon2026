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
    # 1. 將 UMAP 設定為 3 維輸出
    reducer = umap.UMAP(n_components=3, random_state=42)
    
    # 排除非數值列
    exclude_cols = ['user_id', 'cluster_label', 'cluster_probability', 'soft_cluster_label','soft_cluster_score']
    features = df.drop(columns=exclude_cols, errors='ignore')
    embedding = reducer.fit_transform(features)
    
    # 建立畫布：左邊 3D 散點圖，右邊 2D 壓縮樹
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d') # 指定 3D 投影
    ax2 = fig.add_subplot(1, 2, 2)
    
    labels = df['cluster_label']
    clustered = (labels >= 0)
    
    # --- 1. 3D UMAP Plot ---
    # 繪製雜訊 (Noise)
    ax1.scatter(embedding[~clustered, 0], 
                embedding[~clustered, 1], 
                embedding[~clustered, 2], 
                color='lightgray', s=5, alpha=0.2, label='Noise')
    
    # 繪製聚類 (Clusters)
    unique_labels = sorted(list(set(labels[clustered])))
    n_clusters = len(unique_labels)
    # 使用與樹狀圖一致的顏色盤
    colors = sns.color_palette('Spectral', n_clusters)
    
    scatter = ax1.scatter(embedding[clustered, 0], 
                         embedding[clustered, 1], 
                         embedding[clustered, 2], 
                         c=labels[clustered], s=20, cmap='Spectral', alpha=0.7)
    
    # 在 3D 空間添加中心點標籤
    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        # 計算 3D 中位數位置
        median_pos = np.median(embedding[mask], axis=0)
        ax1.text(median_pos[0], median_pos[1], median_pos[2], 
                 str(label), fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax1.set_title(f'3D UMAP Projection: {name}')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_zlabel('UMAP 3')
    # 調整視角，讓初始呈現更有立體感
    ax1.view_init(elev=20, azim=45)

    # --- 2. Condensed Tree (維持 2D，因為樹狀結構不適合 3D) ---
    plt.sca(ax2) 
    dark_grey_cmap = LinearSegmentedColormap.from_list('dark_grey', ['0.9', '0.0'])
    clusterer.condensed_tree_.plot(select_clusters=True,
                                   selection_palette=colors,
                                   label_clusters=False,
                                   cmap=dark_grey_cmap                 
                                   )
    ax2.set_title(f'Condensed Tree (Stability): {name}')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{name}_3D_dashboard.png')
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
    # 1. 取得所有群集 ID (排除雜訊 -1)
    unique_clusters = sorted([c for c in df['cluster_label'].unique() if c >= 0])
    n_clusters = len(unique_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols  # 自動計算需要的行數
    
    # 2. 建立大圖
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), 
                             subplot_kw=dict(polar=True))
    axes = axes.flatten()  # 將多維矩陣拉平，方便迴圈讀取
    
    # 3. 事先標準化所有可能用到的特徵 (全域標準化，比較才有意義)
    all_features = set()
    for features in importance_dict.values():
        all_features.update(features.index[:5])
    all_features = list(all_features)
    
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[all_features] = scaler.fit_transform(df[all_features])
    
    # 4. 開始繪製每個群集
    for i, cluster_id in enumerate(unique_clusters):
        ax = axes[i]
        
        # 取得該群集的前 5 大特徵
        top_features = list(importance_dict[cluster_id].index[:5])
        
        # 計算數值
        cluster_stats = df_norm[df_norm['cluster_label'] == cluster_id][top_features].mean()
        global_stats = df_norm[top_features].mean()
        
        # 雷達圖幾何設定
        angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
        angles += angles[:1]
        stats = cluster_stats.tolist() + cluster_stats.tolist()[:1]
        global_mean = global_stats.tolist() + global_stats.tolist()[:1]
        
        # 繪圖
        ax.fill(angles, stats, color='teal', alpha=0.25)
        ax.plot(angles, stats, color='teal', linewidth=2)
        ax.plot(angles, global_mean, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        
        # 設定刻度與標籤
        ax.set_ylim(0, 1.1) # 統一比例尺
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_features, fontsize=8)
        ax.set_title(f'Cluster {cluster_id}', size=12, pad=10)

    # 5. 隱藏沒用到的空白子圖
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.suptitle('All Clusters Feature Profiles', size=20, y=1.02)
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