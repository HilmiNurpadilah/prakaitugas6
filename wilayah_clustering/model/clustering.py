import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import io
import base64

# Fitur numerik yang tersedia setelah preprocessing
NUMERIC_FEATURES = [
    'kematian_anak', 'ekspor', 'kesehatan', 'impor',
    'pendapatan', 'inflasi', 'harapan_hidup'
]

# Load dan scaling data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    X = df[NUMERIC_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled

# Grafik elbow (inertia/WCSS vs K)
def plot_elbow(X_scaled, k_min=2, k_max=10, filename='elbow.png'):
    inertias = []
    K = range(k_min, k_max+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(K, inertias, marker='o')
    plt.xlabel('Jumlah Cluster (K)')
    plt.ylabel('Inertia / WCSS')
    plt.title('Elbow Method (Inertia/Within-Cluster Sum of Squares vs K)')
    plt.tight_layout()
    path = os.path.join('static', filename)
    plt.savefig(path)
    plt.close()
    return filename

# Grafik silhouette score vs K
def plot_silhouette(X_scaled, k_min=2, k_max=10, filename='silhouette.png'):
    silhouettes = []
    K = range(k_min, k_max+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouettes.append(score)
    plt.figure(figsize=(6,4))
    plt.plot(K, silhouettes, marker='o', color='orange')
    plt.xlabel('Jumlah Cluster (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs K')
    plt.tight_layout()
    path = os.path.join('static', filename)
    plt.savefig(path)
    plt.close()
    return filename

# Clustering dengan jumlah cluster optimal (misal K=3 atau input)
def run_kmeans(X_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans

# Scatter plot hasil clustering (pakai income vs life_expec)
def plot_scatter(df, labels, filename='scatter.png'):
    plt.figure(figsize=(6,4))
    scatter = plt.scatter(df['pendapatan'], df['harapan_hidup'], c=labels, cmap='viridis', s=40)
    plt.xlabel('Pendapatan')
    plt.ylabel('Harapan Hidup')
    plt.title('Hasil Clustering (Pendapatan vs Harapan Hidup)')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    path = os.path.join('static', filename)
    plt.savefig(path)
    plt.close()
    return filename

# Tabel hasil cluster
def get_clustered_table(df, labels):
    df_out = df.copy()
    df_out['Cluster'] = labels
    return df_out

if __name__ == "__main__":
    import os
    # Pastikan folder static ada
    os.makedirs('static', exist_ok=True)
    # --- SETUP ---
    # Path data hasil preprocessing
    data_path = os.path.join('dataset', 'data_preprocessed.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} tidak ditemukan. Jalankan preprocessing.py terlebih dahulu.")
    df, X_scaled = load_and_preprocess_data(data_path)

    # --- GRAFIK 1: Elbow (Inertia vs K) ---
    plot_elbow(X_scaled, k_min=2, k_max=10, filename='elbow_inertia.png')
    # --- GRAFIK 2: Elbow (WCSS vs K) (judul berbeda saja) ---
    plot_elbow(X_scaled, k_min=2, k_max=10, filename='elbow_wcss.png')
    # --- GRAFIK 3: Silhouette ---
    plot_silhouette(X_scaled, k_min=2, k_max=10, filename='silhouette.png')
    # --- CLUSTERING ---
    K = 3  # Default K, bisa diubah sesuai kebutuhan
    labels, kmeans = run_kmeans(X_scaled, n_clusters=K)
    # --- GRAFIK 4: Scatter plot hasil clustering ---
    labels, kmeans = run_kmeans(X_scaled, n_clusters=3)
    # Scatter plot hasil clustering
    plot_scatter(df, labels, filename='scatter.png')
    # Simpan tabel hasil cluster
    df_clustered = get_clustered_table(df, labels)
    df_clustered.to_csv(os.path.join('static', 'hasil_cluster.csv'), index=False)
    print('Grafik dan hasil cluster berhasil disimpan di folder static.')

def calculate_silhouette(X_scaled, k_range):
    silhouette_scores = []
    for k in k_range:
        if k == 1:
            silhouette_scores.append(np.nan)
            continue
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    return silhouette_scores

def perform_kmeans(X_scaled, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return cluster_labels, kmeans
