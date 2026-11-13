"""
MODUL PEMROSESAN DATA (dipanggil dashboard saat tombol "Proses Data" ditekan).

Logikanya sama dengan preprocessing_jabar.py, tapi dipecah jadi fungsi-fungsi
supaya bisa menerima file hasil unggahan pengguna (bukan cuma path lokal),
dan setiap file divalidasi dulu formatnya sebelum diproses.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    TAHUN_MIN, TAHUN_MAX, TARGET_COL, COL_LAG1_POP, COL_LAG1_PDRB,
    IKAN_TIDAK_SPESIFIK,
)


class DataTidakValid(Exception):
    """Dilempar kalau file unggahan tidak sesuai format yang dibutuhkan."""


def _ke_buffer(sumber):
    """Path / UploadedFile / bytes -> BytesIO, supaya file bisa dibaca berkali-kali."""
    if isinstance(sumber, (str, Path)):
        return io.BytesIO(Path(sumber).read_bytes())
    if isinstance(sumber, (bytes, bytearray)):
        return io.BytesIO(sumber)
    sumber.seek(0)
    return io.BytesIO(sumber.read())


def _normalisasi_wilayah(series):
    # Huruf kapital + buang prefix "KABUPATEN " / "KAB. " (prefix "KOTA "
    # dipertahankan karena data ikan juga memakai "KOTA X")
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace(r'^(KABUPATEN|KAB\.?)\s+', '', regex=True)
    )


def _ke_numerik(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')


# ==============================================================================
# 1. DATA PENDUDUK (OPEN DATA JABAR / BPS)
# ==============================================================================
KOLOM_PENDUDUK = ['tahun', 'nama_kabupaten_kota', 'jumlah_penduduk', 'satuan']


def load_penduduk(sumber):
    buf = _ke_buffer(sumber)
    try:
        xls = pd.ExcelFile(buf)
    except Exception:
        raise DataTidakValid("File Data Jumlah Penduduk tidak dapat dibaca. Pastikan formatnya .xlsx.")

    sheet = "data" if "data" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = [str(c).strip().lower() for c in df.columns]

    kurang = [c for c in KOLOM_PENDUDUK if c not in df.columns]
    if kurang:
        raise DataTidakValid(
            f"File Data Jumlah Penduduk tidak memiliki kolom: {', '.join(kurang)}. "
            f"Kolom wajib: {', '.join(KOLOM_PENDUDUK)}."
        )

    df = df[KOLOM_PENDUDUK].dropna(subset=['tahun', 'nama_kabupaten_kota', 'jumlah_penduduk']).copy()
    df.rename(columns={
        'tahun': 'Tahun',
        'nama_kabupaten_kota': 'Kabupaten/Kota',
        'jumlah_penduduk': 'Penduduk',
    }, inplace=True)

    df['Tahun'] = _ke_numerik(df['Tahun']).astype('Int64')
    df['Penduduk'] = _ke_numerik(df['Penduduk'])
    df = df.dropna(subset=['Tahun', 'Penduduk'])

    # Konversi "Ribu Orang" -> "Jiwa" (dikali 1.000). Dicek per baris lewat
    # kolom satuan, bukan diasumsikan buta.
    faktor = np.where(df['satuan'].astype(str).str.upper().str.contains('RIBU'), 1000, 1)
    df['Penduduk'] = df['Penduduk'] * faktor

    df['Kabupaten/Kota'] = _normalisasi_wilayah(df['Kabupaten/Kota'])
    df['Tahun'] = df['Tahun'].astype(int)
    df = df[df['Tahun'].between(TAHUN_MIN, TAHUN_MAX)].reset_index(drop=True)

    if df.empty:
        raise DataTidakValid(f"File Data Jumlah Penduduk tidak berisi data tahun {TAHUN_MIN}-{TAHUN_MAX}.")
    return df[['Tahun', 'Kabupaten/Kota', 'Penduduk']]


# ==============================================================================
# 2. DATA PDRB PER KAPITA (BPS) -- MULTI-SHEET, 1 SHEET PER TAHUN
# ==============================================================================
def load_pdrb(sumber):
    buf = _ke_buffer(sumber)
    try:
        xls = pd.ExcelFile(buf)
    except Exception:
        raise DataTidakValid("File Data PDRB per Kapita tidak dapat dibaca. Pastikan formatnya .xlsx.")

    dfs = []
    for sheet_name in xls.sheet_names:
        try:
            tahun = int(str(sheet_name).strip())
        except ValueError:
            continue  # skip sheet yang bukan angka tahun

        df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
        if df_sheet.shape[1] < 2:
            continue

        # Ambil berdasarkan posisi kolom (nama kolom beda-beda antar sheet)
        df_sheet = df_sheet.iloc[:, :2].copy()
        df_sheet.columns = ['Kabupaten/Kota', 'PDRB_raw']
        df_sheet = df_sheet.dropna(subset=['Kabupaten/Kota', 'PDRB_raw'], how='any')
        df_sheet['Kabupaten/Kota'] = df_sheet['Kabupaten/Kota'].astype(str)

        # Buang baris catatan kaki / baris agregat
        df_sheet = df_sheet[
            ~df_sheet['Kabupaten/Kota'].str.lower().str.contains(
                'keterangan|perbedaan|diskrepansi|provinsi|jumlah|indonesia', na=False
            )
        ]

        # Hapus tanda (*)/(**) lalu sisakan hanya angka dan titik desimal
        df_sheet['PDRB_perKapita (Ribu Rp)'] = pd.to_numeric(
            df_sheet['PDRB_raw'].astype(str)
            .str.replace(r'\(\*+\)', '', regex=True)
            .str.replace(r'[^\d.]', '', regex=True),
            errors='coerce',
        )
        df_sheet['Kabupaten/Kota'] = _normalisasi_wilayah(df_sheet['Kabupaten/Kota'])
        df_sheet['Tahun'] = tahun
        dfs.append(df_sheet[['Tahun', 'Kabupaten/Kota', 'PDRB_perKapita (Ribu Rp)']])

    if not dfs:
        raise DataTidakValid(
            "File Data PDRB per Kapita tidak memiliki sheet bernama tahun (contoh: '2016', '2017', ...). "
            "Setiap sheet harus berisi kolom Kabupaten/Kota dan PDRB per Kapita."
        )

    df = pd.concat(dfs, ignore_index=True).dropna(subset=['PDRB_perKapita (Ribu Rp)'])
    if df.empty:
        raise DataTidakValid("File Data PDRB per Kapita tidak berisi nilai PDRB yang valid.")
    return df.reset_index(drop=True)


# ==============================================================================
# 3. DATA KONSUMSI IKAN (KKP RI)
# ==============================================================================
KOLOM_IKAN = ['Tahun', 'Kabupaten/Kota', 'Jenis Ikan']


def _cari_baris_header(df_mentah):
    # File unduhan KKP punya beberapa baris judul di atas tabel (biasanya 7).
    # Cari otomatis baris yang memuat nama kolom, supaya tetap aman kalau
    # jumlah baris judulnya berubah.
    for i in range(min(30, len(df_mentah))):
        isi = [str(v).strip() for v in df_mentah.iloc[i].tolist()]
        if 'Jenis Ikan' in isi and 'Tahun' in isi:
            return i
    return None


def load_ikan(sumber):
    buf = _ke_buffer(sumber)
    try:
        df_mentah = pd.read_excel(buf, sheet_name=0, header=None)
    except Exception:
        raise DataTidakValid("File Data Konsumsi Ikan tidak dapat dibaca. Pastikan formatnya .xlsx.")

    baris = _cari_baris_header(df_mentah)
    if baris is None:
        raise DataTidakValid(
            "File Data Konsumsi Ikan tidak sesuai format KKP: kolom 'Tahun', 'Kabupaten/Kota', "
            "'Jenis Ikan', dan 'Konsumsi Ikan Masyarakat (Kg/Kapita)' tidak ditemukan."
        )

    df = df_mentah.iloc[baris + 1:].copy()
    df.columns = [str(c).strip() for c in df_mentah.iloc[baris].tolist()]

    kolom_konsumsi = next((c for c in df.columns if 'konsumsi' in c.lower()), None)
    kurang = [c for c in KOLOM_IKAN if c not in df.columns]
    if kurang or kolom_konsumsi is None:
        raise DataTidakValid(
            "File Data Konsumsi Ikan tidak memiliki kolom: "
            f"{', '.join(kurang + ([] if kolom_konsumsi else ['Konsumsi Ikan Masyarakat (Kg/Kapita)']))}."
        )

    df = df[KOLOM_IKAN + [kolom_konsumsi]].rename(columns={kolom_konsumsi: 'Konsumsi'})
    df = df.dropna(subset=['Kabupaten/Kota', 'Konsumsi'])
    df['Tahun'] = _ke_numerik(df['Tahun'])
    df['Konsumsi'] = _ke_numerik(df['Konsumsi'])
    df = df.dropna(subset=['Tahun', 'Konsumsi'])
    df['Tahun'] = df['Tahun'].astype(int)

    df['Kabupaten/Kota'] = df['Kabupaten/Kota'].astype(str).str.strip().str.upper()
    # Cegah duplikasi akibat beda kapitalisasi ("Ekor Kuning" vs "Ekor kuning")
    df['Jenis Ikan'] = df['Jenis Ikan'].astype(str).str.strip().str.title()

    if df.empty:
        raise DataTidakValid("File Data Konsumsi Ikan tidak berisi baris data yang valid.")
    return df.reset_index(drop=True)


# ==============================================================================
# 4. SELEKSI TOP 10 JENIS IKAN
# ==============================================================================
def pilih_top10(df_ikan, n=10):
    df_bersih = df_ikan[~df_ikan['Jenis Ikan'].str.lower().isin(IKAN_TIDAK_SPESIFIK)]
    rata2 = df_bersih.groupby('Jenis Ikan')['Konsumsi'].mean().sort_values(ascending=False)
    return rata2.head(n).index.tolist(), rata2


# ==============================================================================
# 5. FEATURE ENGINEERING + INTEGRASI VARIABEL EKSTERNAL
# ==============================================================================
def _gabung_eksternal(df, df_pop, df_pdrb):
    # Populasi & PDRB tahun T dipakai sebagai input (Lag 1) untuk target T+1
    pop_lag = df_pop.assign(Tahun=df_pop['Tahun'] + 1).rename(columns={'Penduduk': COL_LAG1_POP})
    pdrb_lag = df_pdrb.assign(Tahun=df_pdrb['Tahun'] + 1).rename(
        columns={'PDRB_perKapita (Ribu Rp)': COL_LAG1_PDRB})
    df = df.merge(pop_lag, on=['Tahun', 'Kabupaten/Kota'], how='left')
    df = df.merge(pdrb_lag, on=['Tahun', 'Kabupaten/Kota'], how='left')
    return df


def bangun_dataset(df_ikan, df_pop, df_pdrb, top_10_ikan):
    """
    Menghasilkan:
      - df_multivariate : data siap model (baris historis lengkap dengan lag)
      - df_forecast     : baris fitur untuk memprediksi tahun berikutnya
      - laporan         : list pesan diagnostik (untuk ditampilkan ke pengguna)
    """
    laporan = []

    nama_ikan = set(df_ikan['Kabupaten/Kota'])
    for nama, df_lain in [('Jumlah Penduduk', df_pop), ('PDRB per Kapita', df_pdrb)]:
        tidak_match = sorted(nama_ikan - set(df_lain['Kabupaten/Kota']))
        if tidak_match:
            laporan.append(f"Wilayah di data ikan yang tidak ditemukan di data {nama}: {', '.join(tidak_match)}")

    df_top10 = df_ikan[df_ikan['Jenis Ikan'].isin(top_10_ikan)].copy()
    if df_top10.empty:
        raise DataTidakValid(
            "Data Konsumsi Ikan tidak memuat satu pun dari 10 jenis ikan yang dipakai model: "
            f"{', '.join(top_10_ikan)}."
        )

    # Kalau ada duplikat (Tahun, Wilayah, Ikan) -- misal akibat beda kapitalisasi
    # nama ikan -- ambil rata-ratanya supaya deret waktunya tetap unik
    df_top10 = (
        df_top10.groupby(['Tahun', 'Kabupaten/Kota', 'Jenis Ikan'], as_index=False)['Konsumsi'].mean()
        .sort_values(['Kabupaten/Kota', 'Jenis Ikan', 'Tahun'])
        .reset_index(drop=True)
    )

    grup = df_top10.groupby(['Kabupaten/Kota', 'Jenis Ikan'])['Konsumsi']
    df_top10['Lag_1_Konsumsi'] = grup.shift(1)
    df_top10['Lag_2_Konsumsi'] = grup.shift(2)

    # --- Baris historis (untuk training / perbandingan aktual vs prediksi) ---
    df_hist = df_top10.dropna(subset=['Lag_1_Konsumsi', 'Lag_2_Konsumsi'])
    df_hist = df_hist.rename(columns={'Konsumsi': TARGET_COL})
    df_hist = _gabung_eksternal(df_hist, df_pop, df_pdrb)

    n_gagal = df_hist[[COL_LAG1_POP, COL_LAG1_PDRB]].isna().any(axis=1).sum()
    if n_gagal:
        laporan.append(f"{n_gagal} baris dibuang karena tidak memiliki pasangan data penduduk/PDRB tahun sebelumnya.")
    df_multivariate = df_hist.dropna().reset_index(drop=True)

    # --- Baris forecast tahun berikutnya ---
    # Lag_1 = konsumsi tahun terakhir, Lag_2 = konsumsi 1 tahun sebelumnya,
    # penduduk & PDRB = nilai tahun terakhir.
    tahun_akhir = int(df_top10['Tahun'].max())
    tahun_forecast = tahun_akhir + 1

    pivot = df_top10.pivot_table(index=['Kabupaten/Kota', 'Jenis Ikan'], columns='Tahun', values='Konsumsi')
    df_fc = pd.DataFrame(index=pivot.index).reset_index()
    df_fc['Tahun'] = tahun_forecast
    df_fc['Lag_1_Konsumsi'] = pivot.get(tahun_akhir).values if tahun_akhir in pivot else np.nan
    df_fc['Lag_2_Konsumsi'] = pivot.get(tahun_akhir - 1).values if (tahun_akhir - 1) in pivot else np.nan

    # Kalau data penduduk/PDRB tahun terakhir belum rilis, pakai nilai
    # terbaru yang tersedia per wilayah (dan beri tahu pengguna).
    for df_src, kolom_src, kolom_lag, label in [
        (df_pop, 'Penduduk', COL_LAG1_POP, 'Jumlah Penduduk'),
        (df_pdrb, 'PDRB_perKapita (Ribu Rp)', COL_LAG1_PDRB, 'PDRB per Kapita'),
    ]:
        terbaru = (
            df_src[df_src['Tahun'] <= tahun_akhir]
            .sort_values('Tahun')
            .groupby('Kabupaten/Kota')
            .tail(1)
        )
        tahun_dipakai = sorted(terbaru['Tahun'].unique())
        if tahun_dipakai and tahun_dipakai != [tahun_akhir]:
            laporan.append(
                f"Data {label} tahun {tahun_akhir} tidak lengkap; prediksi {tahun_forecast} "
                f"memakai nilai terbaru yang tersedia (tahun {', '.join(map(str, tahun_dipakai))})."
            )
        df_fc = df_fc.merge(
            terbaru[['Kabupaten/Kota', kolom_src]].rename(columns={kolom_src: kolom_lag}),
            on='Kabupaten/Kota', how='left',
        )

    n_sebelum = len(df_fc)
    df_fc = df_fc.dropna().reset_index(drop=True)
    if len(df_fc) < n_sebelum:
        laporan.append(
            f"{n_sebelum - len(df_fc)} kombinasi wilayah-ikan tidak bisa diprediksi untuk {tahun_forecast} "
            f"karena data konsumsi {tahun_akhir - 1}-{tahun_akhir} atau data eksternalnya tidak lengkap."
        )

    return df_multivariate, df_fc, laporan


# ==============================================================================
# 6. ONE-HOT ENCODING
# ==============================================================================
def encode(df, feature_cols=None):
    """
    One-hot encoding Jenis Ikan saja. Kabupaten/Kota TIDAK di-encode dan tidak
    menjadi fitur (hanya dipakai sebagai identitas baris), sehingga perbedaan
    antarwilayah ditangkap lewat lag konsumsi, jumlah penduduk, dan PDRB.
    Kalau feature_cols diberikan (saat prediksi), kolom disamakan dengan urutan
    fitur saat model dilatih -- kategori yang tidak dikenal menjadi 0 semua.
    """
    df_enc = pd.get_dummies(df, columns=['Jenis Ikan'], prefix=['Ikan'], dtype=int)
    if feature_cols is None:
        feature_cols = [c for c in df_enc.columns if c not in [TARGET_COL, 'Tahun', 'Kabupaten/Kota']]
    X = df_enc.reindex(columns=feature_cols, fill_value=0)
    return X, feature_cols
