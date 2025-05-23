import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from model import clustering
from sklearn.cluster import KMeans

UPLOAD_FOLDER = os.path.join('dataset')
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

# Helper untuk cek ekstensi

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    elbow_plot = None
    wcss_plot = None
    silhouette_plot = None
    table_preview = None
    scatter_plot = None
    cluster_table = None
    filename = None
    message = None
    k_range = list(range(2, 11))  # Mulai dari 2 karena silhouette tidak valid untuk k=1
    selected_k = 3  # Default
    if request.method == 'POST':
        try:
            selected_k = int(request.form.get('selected_k', 3))
        except (TypeError, ValueError):
            selected_k = 3

    # Untuk gambar statis base64
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    images_base64 = []
    for fname in os.listdir(static_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            with open(os.path.join(static_dir, fname), 'rb') as imgf:
                encoded = base64.b64encode(imgf.read()).decode()
                images_base64.append({'filename': fname, 'data': encoded})
    # Untuk hasil_cluster.csv
    hasil_cluster_table = None
    hasil_cluster_path = os.path.join(static_dir, 'hasil_cluster.csv')
    if os.path.exists(hasil_cluster_path):
        df_hasil = pd.read_csv(hasil_cluster_path)
        hasil_cluster_table = df_hasil.to_html(classes='table table-bordered table-striped', index=False)

    # Path default dataset
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data_preprocessed.csv')
    # Tidak ada fitur upload, tidak perlu cek request.method
    if os.path.exists(data_path):
        df, X_scaled = clustering.load_and_preprocess_data(data_path)
        table_preview = df.head(10).to_html(classes='table table-bordered', index=False)
        # Elbow (Inertia vs K)
        inertia = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
        # Grafik 1: Elbow (Inertia vs K)
        plt.figure(figsize=(5,4))
        plt.plot(k_range, inertia, marker='o')
        plt.xlabel('K')
        plt.ylabel('Inertia')
        plt.title('Elbow Method (Inertia vs K)')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        elbow_plot = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        # Grafik 2: Elbow (WCSS vs K) - sama dengan inertia, hanya judul beda
        plt.figure(figsize=(5,4))
        plt.plot(k_range, inertia, marker='o', color='red')
        plt.xlabel('K')
        plt.ylabel('WCSS')
        plt.title('Elbow Method (WCSS vs K)')
        plt.tight_layout()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        wcss_plot = base64.b64encode(buf2.getvalue()).decode()
        plt.close()
        # Grafik 3: Silhouette Method
        silhouette_scores = clustering.calculate_silhouette(X_scaled, k_range)
        plt.figure(figsize=(5,4))
        plt.plot(k_range, silhouette_scores, marker='o', color='orange')
        plt.xlabel('K')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method')
        plt.tight_layout()
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png')
        buf3.seek(0)
        silhouette_plot = base64.b64encode(buf3.getvalue()).decode()
        plt.close()
        # Jika user sudah memilih K, lakukan clustering dan scatter plot
        if selected_k:
            labels, kmeans = clustering.run_kmeans(X_scaled, n_clusters=selected_k)
            # Grafik 4: Scatter plot hasil clustering
            plt.figure(figsize=(6,4))
            scatter = plt.scatter(df['pendapatan'], df['harapan_hidup'], c=labels, cmap='viridis', s=40)
            plt.xlabel('Pendapatan')
            plt.ylabel('Harapan Hidup')
            plt.title(f'Hasil Clustering (Pendapatan vs Harapan Hidup), K={selected_k}')
            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()
            buf4 = io.BytesIO()
            plt.savefig(buf4, format='png')
            buf4.seek(0)
            scatter_plot = base64.b64encode(buf4.getvalue()).decode()
            plt.close()
            # Tabel hasil cluster
            df_clustered = clustering.get_clustered_table(df, labels)
            cluster_table = df_clustered.to_html(classes='table table-bordered table-sm', index=False)
    return render_template('index.html',
        elbow_plot=elbow_plot,
        wcss_plot=wcss_plot,
        silhouette_plot=silhouette_plot,
        table_preview=table_preview,
        scatter_plot=scatter_plot,
        cluster_table=cluster_table,
        filename=filename,
        message=message,
        k_range=k_range,
        selected_k=selected_k,
        images_base64=images_base64,
        hasil_cluster_table=hasil_cluster_table
    )

    if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
