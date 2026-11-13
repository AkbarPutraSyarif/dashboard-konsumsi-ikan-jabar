"""
KONFIGURASI GLOBAL DASHBOARD PREDIKSI KONSUMSI IKAN JAWA BARAT
Ubah nilai di file ini kalau ada yang perlu disesuaikan (link Google Drive,
lokasi file model, dsb.) tanpa perlu menyentuh kode lain.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

FILE_MODEL = MODEL_DIR / "model_terbaik.joblib"
FILE_GEOJSON = DATA_DIR / "jabar_kabkota.geojson"

# Hasil prediksi tiap pengguna disimpan sementara di sini supaya tidak hilang
# saat halaman di-reload (lihat src/sesi.py)
SESI_DIR = BASE_DIR / "sesi"
SESI_KEDALUWARSA_JAM = 24

# TODO: ganti dengan link folder Google Drive berisi 3 file data input
LINK_GOOGLE_DRIVE = "https://drive.google.com/drive/folders/1CAYE-dMZUQ_S4cuRj-ShqBaxgvRpq7Zl?usp=sharing"

# Rentang tahun yang dipakai (sesuai cakupan data KKP 2015-2025)
TAHUN_MIN = 2015
TAHUN_MAX = 2025

# Hyperparameter terbaik hasil eksperimen (RMSE rata-rata walk-forward terendah):
#   XGBoost  : n_estimators=100, learning_rate=0.03, max_depth=3
#   LightGBM : n_estimators=100, learning_rate=0.03, num_leaves=7
# Parameter lain disamakan dengan notebook eksperimen Bab 4.
PARAM_XGB = dict(
    n_estimators=100, learning_rate=0.03, max_depth=3,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
)
PARAM_LGBM = dict(
    n_estimators=100, learning_rate=0.03, num_leaves=7, max_depth=-1,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=-1,
)

TARGET_COL = "Konsumsi Ikan (Kg/Kapita)"
COL_LAG1_POP = "Lag_1_Penduduk (Jiwa)"
COL_LAG1_PDRB = "Lag_1_PDRB_perKapita (Ribu Rp)"

# Kategori agregat/generik yang bukan spesies tunggal -- dibuang sebelum
# seleksi Top 10 (sama dengan preprocessing_jabar.py)
IKAN_TIDAK_SPESIFIK = [
    'ikan matang', 'ikan segar lainnya', 'ikan diawetkan lainnya',
    'tongkol, tuna, cakalang diawetkan', 'cumi-cumi, sotong, gurita',
    'cumi-cumi, sotong, gurita diawetkan', 'ketam, kepiting, rajungan',
    'udang dan hewan air diawetkan lainnya', 'udang dan hewan air segar lainnya',
]

# Kategori jenis ikan untuk halaman Tentang Sistem. Kunci = nama dasar ikan (huruf
# kecil, tanpa kata "diawetkan"). Daftar 10 ikan yang ditampilkan diambil otomatis
# dari model; ikan yang belum ada di sini akan masuk kelompok "Lainnya".
KATEGORI_IKAN = {
    **dict.fromkeys([
        'bandeng', 'kembung', 'teri', 'tongkol', 'cakalang', 'tuna', 'tenggiri', 'layang', 'selar',
        'lemuru', 'kakap', 'kerapu', 'layur', 'ekor kuning', 'belanak', 'peperek', 'japuh', 'kuro',
        'manyung', 'sarden', 'baronang', 'swanggi', 'bawal putih', 'bawal hitam', 'tembang', 'julung-julung',
    ], 'Ikan Laut'),
    **dict.fromkeys([
        'lele', 'mas', 'mujair', 'nila', 'gurame', 'gurami', 'patin', 'bawal', 'tawes', 'nilem', 'belut',
        'gabus', 'sepat', 'betok', 'tambakan', 'baung', 'jambal', 'sidat',
    ], 'Ikan Air Tawar'),
}

# Titik tengah (perkiraan) tiap Kabupaten/Kota -- dipakai sebagai peta
# cadangan (bubble map) kalau file GeoJSON batas wilayah belum tersedia.
KOORDINAT_WILAYAH = {
    "BANDUNG": (-7.10, 107.60), "BANDUNG BARAT": (-6.86, 107.45),
    "BEKASI": (-6.25, 107.15), "BOGOR": (-6.55, 106.80),
    "CIAMIS": (-7.33, 108.35), "CIANJUR": (-7.10, 107.15),
    "CIREBON": (-6.76, 108.48), "GARUT": (-7.38, 107.76),
    "INDRAMAYU": (-6.45, 108.15), "KARAWANG": (-6.25, 107.40),
    "KUNINGAN": (-7.00, 108.55), "MAJALENGKA": (-6.85, 108.23),
    "PANGANDARAN": (-7.65, 108.55), "PURWAKARTA": (-6.58, 107.45),
    "SUBANG": (-6.50, 107.75), "SUKABUMI": (-7.05, 106.70),
    "SUMEDANG": (-6.85, 107.92), "TASIKMALAYA": (-7.50, 108.13),
    "KOTA BANDUNG": (-6.91, 107.61), "KOTA BANJAR": (-7.37, 108.54),
    "KOTA BEKASI": (-6.24, 106.99), "KOTA BOGOR": (-6.60, 106.80),
    "KOTA CIMAHI": (-6.88, 107.54), "KOTA CIREBON": (-6.73, 108.55),
    "KOTA DEPOK": (-6.40, 106.82), "KOTA SUKABUMI": (-6.92, 106.93),
    "KOTA TASIKMALAYA": (-7.33, 108.22),
}
