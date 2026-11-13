"""
MODUL PROSES DATA: dipanggil dashboard saat pengguna menekan tombol "Proses Data".

Alur: validasi file -> praproses -> seleksi 10 jenis ikan -> walk-forward
validation XGBoost vs LightGBM -> pilih model dengan RMSE rata-rata terendah ->
latih ulang model terpilih dengan seluruh data -> prediksi tahun berikutnya.

Model dilatih langsung dari data yang diunggah pengguna setiap kali tombol
ditekan dan TIDAK disimpan ke file; yang disimpan hanya hasil prediksi dan
hasil evaluasinya (lewat src/sesi.py).
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.config import TARGET_COL
from src.modeling import walk_forward, latih_final
from src.preprocessing import (
    DataTidakValid, load_ikan, load_penduduk, load_pdrb, pilih_top10, bangun_dataset, encode,
)


def proses_data(file_ikan, file_penduduk, file_pdrb):
    # --- 1. Validasi + muat ketiga file. Semua error dikumpulkan sekaligus supaya
    # pengguna bisa memperbaiki semua file dalam satu kali unggah ulang.
    data, errors = {}, []
    for kunci, fungsi, f in [
        ('ikan', load_ikan, file_ikan),
        ('pop', load_penduduk, file_penduduk),
        ('pdrb', load_pdrb, file_pdrb),
    ]:
        try:
            data[kunci] = fungsi(f)
        except DataTidakValid as e:
            errors.append(str(e))
    if errors:
        raise DataTidakValid("\n".join(errors))

    # --- 2. Praproses: seleksi 10 jenis ikan + feature engineering (lag & variabel eksternal)
    top_10_ikan, _ = pilih_top10(data['ikan'])
    df_multi, df_fc, laporan = bangun_dataset(data['ikan'], data['pop'], data['pdrb'], top_10_ikan)

    # Walk-forward butuh minimal 2 tahun data latih yang sudah punya lag
    # (= minimal 4 tahun data konsumsi berurutan)
    if df_multi['Tahun'].nunique() < 2 or df_fc.empty:
        raise DataTidakValid(
            "Data tidak cukup untuk melatih model. Pastikan data konsumsi ikan mencakup minimal 4 tahun "
            "berurutan dan tahunnya beririsan dengan data jumlah penduduk dan PDRB per kapita."
        )

    # --- 3. Pelatihan: walk-forward validation XGBoost vs LightGBM, pilih RMSE terendah
    df_fold, ringkasan, nama_model, _, feature_cols = walk_forward(df_multi, log=lambda *_: None)

    # --- 4. Latih ulang model terpilih dengan seluruh data
    model = latih_final(df_multi, nama_model, feature_cols)

    # --- 5. Prediksi
    # a) data historis (untuk grafik aktual vs prediksi)
    df_hist = df_multi[['Tahun', 'Kabupaten/Kota', 'Jenis Ikan', TARGET_COL]].rename(columns={TARGET_COL: 'Aktual'})
    X_hist, _ = encode(df_multi, feature_cols)
    df_hist['Prediksi'] = np.clip(model.predict(X_hist), 0, None)

    # b) tahun berikutnya
    df_forecast = df_fc[['Tahun', 'Kabupaten/Kota', 'Jenis Ikan']].copy()
    df_forecast['Konsumsi_Tahun_Terakhir'] = df_fc['Lag_1_Konsumsi'].values
    X_fc, _ = encode(df_fc, feature_cols)
    df_forecast['Prediksi'] = np.clip(model.predict(X_fc), 0, None)

    # Konsumsi aktual per tahun (semua tahun, termasuk tahun awal yang tidak
    # punya lag) -- dipakai supaya garis aktual di grafik tren tetap utuh
    df_aktual = (
        data['ikan'][data['ikan']['Jenis Ikan'].isin(top_10_ikan)]
        .groupby(['Tahun', 'Kabupaten/Kota', 'Jenis Ikan'], as_index=False)['Konsumsi'].mean()
        .rename(columns={'Konsumsi': 'Aktual'})
    )

    evaluasi = {
        'nama_model': nama_model,
        'ringkasan': ringkasan,
        'hasil_per_fold': df_fold,
        'feature_importance': pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False),
        'top_10_ikan': top_10_ikan,
        'n_fitur': len(feature_cols),
        'n_baris_latih': len(df_multi),
        'rentang_tahun': (int(df_multi['Tahun'].min()), int(df_multi['Tahun'].max())),
        'dilatih_pada': datetime.now().strftime("%d-%m-%Y %H:%M"),
    }

    return {
        'hist': df_hist.reset_index(drop=True),
        'forecast': df_forecast.reset_index(drop=True),
        'aktual': df_aktual,
        'tahun_forecast': int(df_fc['Tahun'].iloc[0]),
        'laporan': laporan,
        'evaluasi': evaluasi,
    }


def ringkas_wilayah(df_forecast):
    """
    Agregasi per Kabupaten/Kota: total prediksi konsumsi 10 komoditas
    (Kg/Kapita) tahun depan vs total konsumsi aktual tahun terakhir.
    """
    per_wil = df_forecast.groupby('Kabupaten/Kota').agg(
        Prediksi=('Prediksi', 'sum'),
        Aktual_Terakhir=('Konsumsi_Tahun_Terakhir', 'sum'),
        Jumlah_Ikan=('Jenis Ikan', 'nunique'),
    ).reset_index()
    per_wil['Perubahan_%'] = np.where(
        per_wil['Aktual_Terakhir'] > 0,
        (per_wil['Prediksi'] - per_wil['Aktual_Terakhir']) / per_wil['Aktual_Terakhir'] * 100,
        np.nan,
    )

    # Jenis ikan dengan prediksi konsumsi tertinggi & penurunan terbesar per wilayah
    df = df_forecast.assign(Selisih=df_forecast['Prediksi'] - df_forecast['Konsumsi_Tahun_Terakhir'])
    teratas = df.loc[df.groupby('Kabupaten/Kota')['Prediksi'].idxmax(), ['Kabupaten/Kota', 'Jenis Ikan']]
    turun = df.loc[df.groupby('Kabupaten/Kota')['Selisih'].idxmin(), ['Kabupaten/Kota', 'Jenis Ikan', 'Selisih']]
    per_wil = per_wil.merge(teratas.rename(columns={'Jenis Ikan': 'Ikan_Teratas'}), on='Kabupaten/Kota')
    per_wil = per_wil.merge(
        turun.rename(columns={'Jenis Ikan': 'Ikan_Turun_Terbesar', 'Selisih': 'Selisih_Ikan_Turun'}),
        on='Kabupaten/Kota',
    )
    return per_wil
