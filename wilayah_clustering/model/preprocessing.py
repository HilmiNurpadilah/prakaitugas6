import pandas as pd
import os

def pra_proses(jalur_masuk, jalur_keluar):
    # Kolom yang akan diambil dari data
    kolom = [
        'country', 'child_mort', 'exports', 'health', 'imports',
        'income', 'inflation', 'life_expec'
    ]
    # Nama kolom baru dalam Bahasa Indonesia
    ganti_nama = {
        'country': 'negara',
        'child_mort': 'kematian_anak',
        'exports': 'ekspor',
        'health': 'kesehatan',
        'imports': 'impor',
        'income': 'pendapatan',
        'inflation': 'inflasi',
        'life_expec': 'harapan_hidup',
    }
    print("Mencari file data di:", os.path.abspath(jalur_masuk))
    df = pd.read_csv(jalur_masuk)
    df = df[kolom].copy()
    df = df.rename(columns=ganti_nama)
    # Membuat folder output jika belum ada
    os.makedirs(os.path.dirname(jalur_keluar), exist_ok=True)
    df.to_csv(jalur_keluar, index=False)
    print(f"Pra-proses selesai! Data telah disimpan di: {jalur_keluar}")

if __name__ == "__main__":
    jalur_masuk = 'Data_Pengelompokkan_Negara.csv'
    jalur_keluar = os.path.join('dataset', 'data_preprocessed.csv')
    pra_proses(jalur_masuk, jalur_keluar)
