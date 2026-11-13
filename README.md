# Dashboard Prediksi Konsumsi Ikan Jawa Barat

Implementasi sistem dari skripsi *"Prediksi Tingkat Konsumsi Ikan Masyarakat di Jawa Barat Menggunakan
Metode XGBoost dan LightGBM"* (Akbar Putra Syarif, 535230076, Teknik Informatika, Universitas Tarumanagara).

## Struktur

```
Dashboard/
├── app.py                  # Dashboard Streamlit (Panduan & Unggah, Halaman Utama, Tentang)
├── train_model.py          # (opsional, tidak dipakai dashboard) latih model di luar dashboard untuk laporan
├── src/
│   ├── config.py           # Link Google Drive, hyperparameter, path file
│   ├── preprocessing.py    # Validasi + cleansing, merge, top-10 ikan, lag features, one-hot jenis ikan
│   ├── modeling.py         # Walk-forward validation, MAE/RMSE/R², model final
│   ├── prediksi.py         # Proses Data: praproses -> latih & bandingkan model -> prediksi
│   └── sesi.py             # Simpan/muat hasil per sesi (tahan reload)
├── data/                   # (opsional) jabar_kabkota.geojson untuk peta choropleth
├── models/                 # (opsional) hasil train_model.py, tidak dipakai dashboard
├── sesi/                   # Dibuat otomatis: hasil prediksi per sesi, terhapus setelah 24 jam
├── preprocessing_jabar.py  # Skrip asli (tetap dipertahankan)
└── modeling_jabar.py       # Skrip asli (tetap dipertahankan)
```

## Cara menjalankan

1. Install library:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan dashboard:
   ```bash
   streamlit run app.py
   ```
3. Di halaman **Panduan & Unggah Data**, unggah ketiga file lalu tekan **Proses Data**. Sistem langsung
   melatih XGBoost dan LightGBM dari data tersebut (walk-forward validation), memilih model dengan RMSE
   rata-rata terendah, melatih ulang model terpilih dengan seluruh data, lalu menampilkan prediksi.
   Model tidak disimpan ke file; yang disimpan hanya hasil prediksi per sesi.

## Catatan

- **Link Google Drive** data input diatur di `src/config.py` (`LINK_GOOGLE_DRIVE`).
- **Peta choropleth** memakai `data/jabar_kabkota.geojson`: batas 27 kabupaten/kota Jawa Barat dari
  [geoBoundaries](https://www.geoboundaries.org) (IDN ADM2, sumber batas BPS 2020), sudah difilter khusus Jawa Barat.
  Peta dasar (laut, nama kota) dimuat dari CARTO, jadi butuh internet. Tanpa internet, bentuk wilayah tetap tampil.
- Fitur model: Lag_1 & Lag_2 konsumsi, Lag_1 jumlah penduduk, Lag_1 PDRB per kapita, dan one-hot
  **jenis ikan** (Kabupaten/Kota tidak di-encode dan tidak menjadi fitur).
- Prediksi tahun berikutnya (misal 2026) memakai Lag_1/Lag_2 konsumsi 2025/2024 serta penduduk & PDRB 2025.
- Nilai per wilayah di peta dan panel rekomendasi = **total** konsumsi 10 komoditas ikan (Kg/Kapita).
- **Sesi & navigasi:** tiap halaman punya URL sendiri (`/`, `/halaman-utama`, `/tentang`), jadi tombol Back
  browser berpindah antarhalaman. Setelah data diproses, URL membawa `?sesi=<id>` dan hasilnya disimpan di
  folder `sesi/`, sehingga reload tidak menghilangkan data. Sesi kedaluwarsa setelah 24 jam
  (`SESI_KEDALUWARSA_JAM` di `src/config.py`). Siapa pun yang memegang link berisi `?sesi=` bisa melihat hasilnya.
