"""
DASHBOARD PREDIKSI KONSUMSI IKAN JAWA BARAT (Streamlit)

Halaman:
  1. Panduan & Unggah Data  (halaman awal)
  2. Halaman Utama          (peta choropleth, evaluasi model, tren, rekomendasi)
  3. Tentang Sistem

Jalankan: streamlit run app.py
Model dilatih langsung dari data yang diunggah saat tombol "Proses Data" ditekan
(tidak ada file model yang disimpan).
"""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    FILE_GEOJSON, LINK_GOOGLE_DRIVE, KOORDINAT_WILAYAH, SESI_KEDALUWARSA_JAM, KATEGORI_IKAN,
)
from src.prediksi import proses_data, ringkas_wilayah
from src.preprocessing import DataTidakValid
from src.sesi import simpan_sesi, muat_sesi, hapus_sesi, bersihkan_sesi_lama

JUDUL = "Dashboard Prediksi Konsumsi Ikan Jawa Barat"
SEMUA_WILAYAH, SEMUA_IKAN = "Semua Wilayah", "Semua Jenis Ikan"
# Skala biru tanpa ujung putih supaya wilayah dengan konsumsi terendah tetap terlihat
SKALA_WARNA = px.colors.sequential.Blues[2:]

st.set_page_config(page_title=JUDUL, page_icon=":material/set_meal:", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.kartu-prioritas {border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
                  border-left: 6px solid; color: #1f2937;}
.kartu-turun {background: #fdecec; border-color: #d64545;}
.kartu-naik  {background: #e8f6ec; border-color: #2f9e55;}
.kartu-prioritas .nama {font-weight: 600; font-size: 0.95rem;}
.kartu-prioritas .nilai {font-size: 0.82rem; color: #4b5563;}
.kartu-prioritas .persen {float: right; font-weight: 700; font-size: 1rem;}
/* Tombol "Lihat rincian per wilayah" di kartu metrik dibuat berbentuk pil seperti st.badge */
.st-key-btn_daftar_turun button {
    min-height: 0; padding: 0 0.5rem; border: none; border-radius: 0.5rem;
    background: rgba(255, 164, 33, 0.15); color: #b45309; font-size: 0.875rem; line-height: 1.6;
    position: relative; top: 0.26rem;  /* sejajarkan dengan pil st.badge di kartu lain */
}
.st-key-btn_daftar_turun button:hover {background: rgba(255, 164, 33, 0.3); color: #92400e;}
.st-key-btn_daftar_turun button p {font-size: 0.875rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# UTILITAS
# ==============================================================================
@st.cache_data
def muat_geojson():
    """
    Muat GeoJSON batas Kabupaten/Kota dan tambahkan properti 'WILAYAH_NORM'
    yang formatnya sama dengan nama wilayah di data (mis. 'BANDUNG', 'KOTA BANDUNG').
    Mendukung berbagai sumber GeoJSON (GADM, BIG/Ina-Geoportal, dll.).
    """
    if not FILE_GEOJSON.exists():
        return None
    gj = json.loads(FILE_GEOJSON.read_text(encoding="utf-8"))

    kunci_nama = ['WILAYAH', 'KABKOT', 'NAME_2', 'WADMKK', 'NAMOBJ', 'nama', 'name', 'Name',
                  'kabkot', 'KAB_KOTA', 'nm_kabkota', 'kab_kota']
    kunci_tipe = ['TYPE_2', 'ENGTYPE_2', 'tipe', 'TIPE', 'type']
    kunci_prov = ['NAME_1', 'PROVINSI', 'WADMPR', 'provinsi', 'Propinsi', 'PROVINCE']

    fitur = []
    for f in gj.get('features', []):
        p = f.get('properties', {})
        prov = next((str(p[k]) for k in kunci_prov if k in p and p[k]), '')
        if prov and not any(s in prov.upper() for s in ['JAWA BARAT', 'WEST JAVA']):
            continue
        nama = next((str(p[k]) for k in kunci_nama if k in p and p[k]), None)
        if nama is None:
            continue
        tipe = next((str(p[k]).upper() for k in kunci_tipe if k in p and p[k]), '')
        nama = nama.strip().upper()
        for awalan in ['KABUPATEN ', 'KAB. ', 'KAB ']:
            if nama.startswith(awalan):
                nama = nama[len(awalan):]
        if ('KOTA' in tipe or 'CITY' in tipe) and not nama.startswith('KOTA '):
            nama = 'KOTA ' + nama
        p['WILAYAH_NORM'] = nama
        fitur.append(f)
    gj['features'] = fitur
    return gj


def judul_wilayah(nama):
    return nama.title() if nama.startswith('KOTA ') else f"Kab. {nama.title()}"


def fmt(x, d=2):
    """Format angka gaya Indonesia (koma desimal)."""
    if pd.isna(x):
        return "-"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def ikan_turun(b):
    """Jenis ikan dengan penurunan terbesar di suatu wilayah, atau None kalau tidak ada yang turun."""
    return b['Ikan_Turun_Terbesar'] if b['Selisih_Ikan_Turun'] < 0 else None


def status_perubahan(perubahan):
    """Series persentase perubahan -> list 'Naik' / 'Turun' / 'Tetap' ('-' kalau tidak bisa dihitung)."""
    return ['-' if pd.isna(v) else 'Turun' if v < 0 else 'Naik' if v > 0 else 'Tetap' for v in perubahan]


def warna_perubahan(nilai):
    """Gaya sel tabel: merah kalau turun, hijau kalau naik (untuk kolom Perubahan & Status)."""
    if isinstance(nilai, str):
        if nilai == 'Turun':
            return 'color: #dc2626; font-weight: 600'
        return 'color: #16a34a; font-weight: 600' if nilai == 'Naik' else ''
    if pd.isna(nilai) or nilai == 0:
        return ''
    return 'color: #dc2626' if nilai < 0 else 'color: #16a34a'


def fmt_persen(v):
    return "-" if pd.isna(v) else f"{'+' if v >= 0 else '−'}{fmt(abs(v), 1)}%"


def tabel_rincian_ikan(forecast, wilayah, tahun_fc):
    """Tabel naik-turun tiap jenis ikan di satu wilayah, urut dari penurunan terbesar."""
    df = forecast[forecast['Kabupaten/Kota'] == wilayah]
    akt, pred = df['Konsumsi_Tahun_Terakhir'], df['Prediksi']
    perubahan = np.where(akt > 0, (pred - akt) / akt * 100, np.nan)
    kol_akt, kol_pred = f'Aktual {tahun_fc - 1} (Kg/Kapita)', f'Prediksi {tahun_fc} (Kg/Kapita)'
    tabel = pd.DataFrame({
        'Jenis Ikan': df['Jenis Ikan'].values,
        kol_akt: akt.values,
        kol_pred: pred.values,
        'Perubahan (%)': perubahan,
        'Status': status_perubahan(perubahan),
    }).sort_values('Perubahan (%)', na_position='last').reset_index(drop=True)

    n_turun = (tabel['Status'] == 'Turun').sum()
    st.caption(f"**{judul_wilayah(wilayah)}**: {n_turun} dari {len(tabel)} jenis ikan diprediksi turun "
               f"pada {tahun_fc} dibanding {tahun_fc - 1}.")
    gaya = (
        tabel.style
        .map(warna_perubahan, subset=['Perubahan (%)', 'Status'])
        .format({kol_akt: lambda v: fmt(v, 2), kol_pred: lambda v: fmt(v, 2), 'Perubahan (%)': fmt_persen})
    )
    st.dataframe(gaya, hide_index=True, width="stretch", height=35 * (len(tabel) + 1) + 3)


def tooltip_wilayah(b, tahun_fc, n_wilayah, ikan_peta):
    """Isi tooltip peta (HTML sederhana yang didukung Plotly) untuk satu wilayah."""
    perubahan = b['Perubahan_%']
    if pd.isna(perubahan):
        badge = ""
    else:
        warna = "#16a34a" if perubahan >= 0 else "#dc2626"
        panah = "▲" if perubahan >= 0 else "▼"
        badge = f"   <b><span style='color:{warna}'>{panah} {fmt(abs(perubahan), 1)}%</span></b>"
    subjudul = "total 10 komoditas" if ikan_peta == SEMUA_IKAN else ikan_peta

    baris = [
        f"<b><span style='font-size:15px;color:#0f172a'>{judul_wilayah(b['Kabupaten/Kota'])}</span></b>",
        f"<span style='font-size:11px;color:#64748b'>Prediksi konsumsi {tahun_fc} · {subjudul}</span>",
        f"<b><span style='font-size:15px;color:#1e3a8a'>{fmt(b['Prediksi'], 2)}</span></b>"
        f"<span style='color:#64748b'> kg/kapita</span>{badge}",
        "<span style='color:#cbd5e1'>──────────────────────</span>",
        f"Aktual {tahun_fc - 1}: <b>{fmt(b['Aktual_Terakhir'], 2)}</b> kg/kapita",
        f"Peringkat konsumsi: <b>{b['Peringkat']}</b> dari {n_wilayah}",
    ]
    if ikan_peta == SEMUA_IKAN:
        turun = ikan_turun(b)
        baris.append(f"Paling dikonsumsi: <b>{b['Ikan_Teratas']}</b>")
        baris.append(f"Penurunan terbesar: <b><span style='color:#dc2626'>{turun}</span></b>" if turun
                     else "Penurunan terbesar: <span style='color:#16a34a'>tidak ada, semua naik</span>")
    return "<br>".join(baris)


@st.cache_resource(ttl=3600)
def _jadwal_bersih_sesi():
    # Dijalankan paling banyak sekali per jam: hapus file sesi yang sudah kedaluwarsa
    bersihkan_sesi_lama()


def query_sesi():
    """Query param ?sesi=<id> untuk link navigasi (None kalau belum ada data)."""
    sid = st.session_state.get("sesi_id")
    return {"sesi": sid} if sid else None


# ==============================================================================
# HALAMAN 1: PANDUAN & UNGGAH DATA
# ==============================================================================
def halaman_unggah():
    with st.container(border=True):
        st.subheader("Panduan Penggunaan")
        st.markdown("""
1. Unduh tiga file data input melalui tombol **Buka Google Drive** di bawah: data konsumsi ikan (KKP RI),
   data jumlah penduduk (Open Data Jabar), dan data PDRB per kapita (BPS Jawa Barat).
2. Pastikan format file **.xlsx** dan strukturnya tidak diubah (nama kolom, nama sheet tahun pada file PDRB).
3. Unggah masing-masing file pada kolom unggah yang sesuai.
4. Tekan tombol **Proses Data**. Sistem akan memvalidasi file, menjalankan praproses, melatih dan
   membandingkan model XGBoost dan LightGBM, lalu menghasilkan prediksi menggunakan model terbaik.
5. Setelah berhasil, buka **Halaman Utama** untuk melihat peta, evaluasi model, grafik tren, dan rekomendasi
   wilayah prioritas.
""")

    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.markdown("**Data terbaru dapat diunduh melalui Google Drive:**")
        c2.link_button("Buka Google Drive", LINK_GOOGLE_DRIVE, icon=":material/folder_open:",
                       width="stretch")

    st.subheader("Unggah Data")
    c1, c2, c3 = st.columns(3)
    f_ikan = c1.file_uploader("Data Konsumsi Ikan (.xlsx)", type=["xlsx"], key="up_ikan")
    f_pop = c2.file_uploader("Data Jumlah Penduduk (.xlsx)", type=["xlsx"], key="up_pop")
    f_pdrb = c3.file_uploader("Data PDRB per Kapita (.xlsx)", type=["xlsx"], key="up_pdrb")

    _, tengah, _ = st.columns([2, 1, 2])
    tekan = tengah.button("Proses Data", type="primary", icon=":material/play_arrow:", width="stretch")

    if tekan:
        kosong = [n for n, f in [("Data Konsumsi Ikan", f_ikan), ("Data Jumlah Penduduk", f_pop),
                                 ("Data PDRB per Kapita", f_pdrb)] if f is None]
        if kosong:
            st.error(f"File berikut belum diunggah: {', '.join(kosong)}.")
            return
        try:
            with st.spinner("Memvalidasi file, menjalankan praproses, melatih model XGBoost dan LightGBM, "
                            "lalu menghasilkan prediksi. Mohon tunggu..."):
                hasil = proses_data(f_ikan, f_pop, f_pdrb)
        except DataTidakValid as e:
            st.error("**Data tidak valid.** Perbaiki file berikut lalu unggah ulang:\n\n" +
                     "\n".join(f"- {baris}" for baris in str(e).splitlines()))
            return
        except Exception as e:  # error tak terduga -- tetap tampilkan pesan yang ramah
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
            return
        # Simpan ke session state + ke file sesi (supaya tidak hilang saat reload),
        # lalu pindah ke Halaman Utama dengan ID sesi di URL
        hapus_sesi(st.session_state.get("sesi_id"))
        sesi_id = simpan_sesi(hasil)
        st.session_state.hasil = hasil
        st.session_state.sesi_id = sesi_id
        st.session_state.baru_diproses = True
        st.switch_page(PAGE_UTAMA, query_params={"sesi": sesi_id})

    if "hasil" in st.session_state:
        st.success("Data sudah diproses. Buka **Halaman Utama** untuk melihat hasil prediksi. "
                   "Mengunggah data baru akan menggantikan hasil sebelumnya.", icon=":material/check_circle:")
        st.page_link(PAGE_UTAMA, label="Buka Halaman Utama", icon=":material/arrow_forward:",
                     query_params=query_sesi())


# ==============================================================================
# HALAMAN 2: HALAMAN UTAMA
# ==============================================================================
def komponen_peta(hasil):
    forecast = hasil['forecast']
    tahun_fc = hasil['tahun_forecast']

    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    c1.markdown(f"**Peta Prediksi Konsumsi Ikan {tahun_fc}**")
    ikan_peta = c2.selectbox("Jenis ikan (peta)", [SEMUA_IKAN] + sorted(forecast['Jenis Ikan'].unique()),
                             label_visibility="collapsed", key="ikan_peta")

    df = forecast if ikan_peta == SEMUA_IKAN else forecast[forecast['Jenis Ikan'] == ikan_peta]
    per_wil = ringkas_wilayah(df)
    per_wil['Peringkat'] = per_wil['Prediksi'].rank(ascending=False, method='min').astype(int)
    per_wil['Tooltip'] = [tooltip_wilayah(b, tahun_fc, len(per_wil), ikan_peta) for _, b in per_wil.iterrows()]

    hover = "%{customdata[0]}<extra></extra>"
    custom = ['Tooltip']
    label_warna = {'Prediksi': 'Kg/Kapita'}

    gj = muat_geojson()
    if gj and gj['features']:
        # choropleth_map (MapLibre) dipakai, bukan px.choropleth (geo), karena tidak sensitif
        # terhadap arah urutan titik poligon (winding order) di file GeoJSON
        fig = px.choropleth_map(
            per_wil, geojson=gj, locations='Kabupaten/Kota', featureidkey='properties.WILAYAH_NORM',
            color='Prediksi', color_continuous_scale=SKALA_WARNA, custom_data=custom, labels=label_warna,
            zoom=6.9, center={'lat': -6.92, 'lon': 107.62}, map_style="carto-positron", opacity=0.85,
        )
        fig.update_traces(hovertemplate=hover, marker_line_color="white", marker_line_width=0.8)
        tidak_ada = sorted(set(per_wil['Kabupaten/Kota']) - {f['properties']['WILAYAH_NORM'] for f in gj['features']})
    else:
        per_wil['lat'] = per_wil['Kabupaten/Kota'].map(lambda w: KOORDINAT_WILAYAH.get(w, (None, None))[0])
        per_wil['lon'] = per_wil['Kabupaten/Kota'].map(lambda w: KOORDINAT_WILAYAH.get(w, (None, None))[1])
        fig = px.scatter_map(
            per_wil.dropna(subset=['lat']), lat='lat', lon='lon', color='Prediksi', size='Prediksi',
            color_continuous_scale=SKALA_WARNA, custom_data=custom, labels=label_warna,
            size_max=28, zoom=7, center={'lat': -6.95, 'lon': 107.6}, map_style="carto-positron",
        )
        fig.update_traces(hovertemplate=hover)
        tidak_ada = []

    fig.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_colorbar=dict(title="Tingkat<br>Konsumsi", thickness=12, len=0.6),
                      hoverlabel=dict(bgcolor="white", bordercolor="#cbd5e1", align="left",
                                      font=dict(family="Source Sans Pro, sans-serif", size=12, color="#334155")))
    st.plotly_chart(fig, width="stretch")

    if gj is None:
        st.caption(f":material/info: File batas wilayah `{FILE_GEOJSON.relative_to(FILE_GEOJSON.parents[1])}` "
                   "belum ada, peta ditampilkan sebagai titik per Kabupaten/Kota.")
    elif tidak_ada:
        st.caption(f":material/warning: Tidak ditemukan di GeoJSON: {', '.join(map(judul_wilayah, tidak_ada))}")


def komponen_evaluasi(hasil):
    ev = hasil['evaluasi']
    st.markdown("**Evaluasi Model Prediksi**")
    r = ev['ringkasan'].rename(columns={'MAE_rata2': 'MAE', 'RMSE_rata2': 'RMSE', 'R2_rata2': 'R²'})
    terbaik = ev['nama_model']

    def sorot(baris):
        warna = 'background-color: rgba(59, 130, 246, 0.18); font-weight: 600' if baris['Model'] == terbaik else ''
        return [warna] * len(baris)

    st.dataframe(r.style.apply(sorot, axis=1).format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R²': '{:.4f}'}),
                 hide_index=True, width="stretch")
    a, b = ev['rentang_tahun']
    st.caption(f"Rata-rata seluruh fold Walk-Forward Validation (data {a}–{b}). "
               f"Model terpilih (RMSE terendah): **{terbaik}**.")

    with st.expander("Detail per fold"):
        fold = ev['hasil_per_fold'][['Tahun_Uji', 'N_Train', 'N_Test',
                                         'MAE_XGBoost', 'MAE_LightGBM', 'RMSE_XGBoost', 'RMSE_LightGBM',
                                         'R2_XGBoost', 'R2_LightGBM']]
        st.dataframe(fold.style.format(precision=4), hide_index=True, width="stretch")


def _agregasi_tren(df, kolom, wilayah, ikan):
    if wilayah != SEMUA_WILAYAH:
        df = df[df['Kabupaten/Kota'] == wilayah]
    if ikan != SEMUA_IKAN:
        df = df[df['Jenis Ikan'] == ikan]
    # Jumlahkan antar jenis ikan per wilayah, lalu rata-rata antar wilayah
    per_wil = df.groupby(['Tahun', 'Kabupaten/Kota'])[kolom].sum().reset_index()
    return per_wil.groupby('Tahun')[kolom].mean()


def komponen_tren(hasil):
    hist, aktual, forecast = hasil['hist'], hasil['aktual'], hasil['forecast']
    tahun_fc = hasil['tahun_forecast']

    c0, c1, c2 = st.columns([2, 2, 2], vertical_alignment="bottom")
    c0.markdown("**Tren Konsumsi Ikan**")
    daftar_wil = sorted(aktual['Kabupaten/Kota'].unique())
    wilayah = c1.selectbox("Wilayah", [SEMUA_WILAYAH] + daftar_wil, format_func=lambda w: w if w == SEMUA_WILAYAH
                           else judul_wilayah(w), label_visibility="collapsed")
    ikan = c2.selectbox("Jenis ikan", [SEMUA_IKAN] + sorted(aktual['Jenis Ikan'].unique()),
                        label_visibility="collapsed")

    s_akt = _agregasi_tren(aktual, 'Aktual', wilayah, ikan)
    s_pred = _agregasi_tren(hist, 'Prediksi', wilayah, ikan)
    s_fc = _agregasi_tren(forecast, 'Prediksi', wilayah, ikan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_akt.index, y=s_akt.values, name="Aktual", mode="lines+markers",
                             line=dict(color="#1f4e8c", width=2.5)))
    fig.add_trace(go.Scatter(x=s_pred.index, y=s_pred.values, name="Prediksi Model", mode="lines+markers",
                             line=dict(color="#f08c2e", width=2, dash="dash")))
    if not s_fc.empty and not s_akt.empty:
        x0 = s_akt.index.max()
        fig.add_trace(go.Scatter(
            x=[x0, tahun_fc], y=[s_akt.loc[x0], s_fc.iloc[0]], name=f"Proyeksi {tahun_fc}",
            mode="lines+markers", line=dict(color="#d64545", width=2, dash="dot"),
            marker=dict(size=[0, 11], symbol="star"),
        ))
    fig.update_layout(
        height=360, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
        yaxis_title="Konsumsi (Kg/Kapita)", xaxis=dict(dtick=1, title=None),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig, width="stretch")
    keterangan = "total 10 komoditas" if ikan == SEMUA_IKAN else ikan
    lingkup = "rata-rata 27 kabupaten/kota" if wilayah == SEMUA_WILAYAH else judul_wilayah(wilayah)
    st.caption(f"Konsumsi ikan per kapita ({keterangan}, {lingkup}).")


def komponen_rekomendasi(hasil):
    st.markdown("**Panel Rekomendasi Wilayah Prioritas**")
    tahun_fc = hasil['tahun_forecast']
    per_wil = ringkas_wilayah(hasil['forecast'])

    st.caption(f"5 wilayah dengan penurunan konsumsi ikan paling tajam pada {tahun_fc} dibanding {tahun_fc - 1}.")
    urut = per_wil.sort_values('Perubahan_%').reset_index(drop=True)

    for _, b in urut.head(5).iterrows():
        perubahan = b['Perubahan_%']
        kelas = "kartu-turun" if (pd.notna(perubahan) and perubahan < 0) else "kartu-naik"
        persen = "-" if pd.isna(perubahan) else f"{perubahan:+.1f}%".replace(".", ",")
        st.markdown(f"""
<div class="kartu-prioritas {kelas}">
  <span class="persen">{persen}</span>
  <div class="nama">{judul_wilayah(b['Kabupaten/Kota'])}</div>
  <div class="nilai">Prediksi {tahun_fc}: {fmt(b['Prediksi'], 2)} kg/kapita · {f"turun terbesar: {ikan_turun(b)}" if ikan_turun(b) else "semua jenis ikan naik"}</div>
</div>""", unsafe_allow_html=True)

    st.caption(f"Perubahan = prediksi {tahun_fc} dibanding konsumsi aktual {tahun_fc - 1} "
               "(total 10 komoditas ikan).")

    kol_pred, kol_akt = f'Prediksi_{tahun_fc} (Kg/Kapita)', f'Aktual_{tahun_fc - 1} (Kg/Kapita)'

    # File 1: ringkasan per wilayah, urut dari penurunan paling tajam
    rekom = pd.DataFrame({
        'Peringkat_Prioritas': range(1, len(urut) + 1),
        'Kabupaten/Kota': urut['Kabupaten/Kota'].values,
        kol_akt: urut['Aktual_Terakhir'].round(4).values,
        kol_pred: urut['Prediksi'].round(4).values,
        'Perubahan (%)': urut['Perubahan_%'].round(2).values,
        'Status': status_perubahan(urut['Perubahan_%']),
        'Ikan_Paling_Dikonsumsi': urut['Ikan_Teratas'].values,
        'Ikan_Penurunan_Terbesar': [ikan_turun(b) or '' for _, b in urut.iterrows()],
    })

    # File 2: detail per wilayah x jenis ikan
    fc = hasil['forecast']
    perubahan_ikan = np.where(
        fc['Konsumsi_Tahun_Terakhir'] > 0,
        (fc['Prediksi'] - fc['Konsumsi_Tahun_Terakhir']) / fc['Konsumsi_Tahun_Terakhir'] * 100,
        np.nan,
    )
    detail = pd.DataFrame({
        'Tahun': fc['Tahun'],
        'Kabupaten/Kota': fc['Kabupaten/Kota'],
        'Jenis Ikan': fc['Jenis Ikan'],
        kol_akt: fc['Konsumsi_Tahun_Terakhir'].round(4),
        kol_pred: fc['Prediksi'].round(4),
        'Perubahan (%)': np.round(perubahan_ikan, 2),
        'Status': status_perubahan(pd.Series(perubahan_ikan)),
    })

    # Format CSV Excel Indonesia: pemisah kolom ";" dan desimal koma. Kalau memakai
    # titik desimal, Excel berbahasa Indonesia membaca 13.6938 sebagai 136.938 (ribuan).
    # utf-8-sig (dengan BOM) supaya Excel mengenali encoding UTF-8.
    def csv_excel(df):
        return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")

    st.download_button("Unduh Rekomendasi (CSV)", csv_excel(rekom),
                       file_name=f"rekomendasi_prioritas_wilayah_{tahun_fc}.csv", mime="text/csv",
                       icon=":material/download:", width="stretch")
    st.download_button("Unduh Prediksi Lengkap (CSV)", csv_excel(detail),
                       file_name=f"prediksi_konsumsi_ikan_{tahun_fc}.csv", mime="text/csv",
                       icon=":material/download:", width="stretch")


@st.dialog("Perubahan Konsumsi per Wilayah", width="large", icon=":material/compare_arrows:")
def dialog_wilayah(per_wil, forecast, tahun_fc):
    semua = per_wil.sort_values('Perubahan_%').reset_index(drop=True)
    turun = semua['Perubahan_%'] < 0
    n_turun, n_naik = int(turun.sum()), int((~turun).sum())

    st.caption(f"Prediksi total konsumsi 10 komoditas ikan {tahun_fc} dibanding konsumsi aktual {tahun_fc - 1} "
               f"untuk {len(semua)} kabupaten/kota: **{n_turun} turun** dan **{n_naik} naik**. "
               "Diurutkan dari penurunan terbesar. Klik salah satu baris untuk melihat rincian per jenis ikan.")
    pilihan = st.segmented_control(
        "Tampilkan", ["Semua", f"Turun ({n_turun})", f"Naik ({n_naik})"],
        default="Semua", label_visibility="collapsed", key="filter_dialog_wilayah",
    )
    if pilihan and pilihan.startswith("Turun"):
        semua = semua[turun]
    elif pilihan and pilihan.startswith("Naik"):
        semua = semua[~turun]

    kol_akt, kol_pred = f'Aktual {tahun_fc - 1} (Kg/Kapita)', f'Prediksi {tahun_fc} (Kg/Kapita)'
    tabel = pd.DataFrame({
        'Kabupaten/Kota': semua['Kabupaten/Kota'].map(judul_wilayah),
        kol_akt: semua['Aktual_Terakhir'],
        kol_pred: semua['Prediksi'],
        'Perubahan (%)': semua['Perubahan_%'],
        'Status': ['Turun' if v < 0 else 'Naik' for v in semua['Perubahan_%']],
    })

    gaya = (
        tabel.style
        .map(warna_perubahan, subset=['Perubahan (%)', 'Status'])
        .format({kol_akt: lambda v: fmt(v, 2), kol_pred: lambda v: fmt(v, 2), 'Perubahan (%)': fmt_persen})
    )
    # Tinggi dibatasi ~11 baris (sisanya di-scroll) supaya rincian di bawahnya tetap terlihat
    pilih = st.dataframe(gaya, hide_index=True, width="stretch", height=min(35 * (len(tabel) + 1) + 3, 423),
                         on_select="rerun", selection_mode="single-row", key=f"tabel_dialog_wilayah_{pilihan}")

    st.markdown("**Rincian per Jenis Ikan**")
    if pilih.selection.rows:
        tabel_rincian_ikan(forecast, semua.iloc[pilih.selection.rows[0]]['Kabupaten/Kota'], tahun_fc)
    else:
        st.caption("Belum ada wilayah yang dipilih. Klik salah satu baris pada tabel di atas.")


def halaman_utama():
    if "hasil" not in st.session_state:
        if st.session_state.pop("sesi_hilang", False):
            st.warning(f"Sesi pada link ini sudah kedaluwarsa (lebih dari {SESI_KEDALUWARSA_JAM} jam) atau "
                       "tidak ditemukan. Silakan unggah data kembali.", icon=":material/history:")
        else:
            st.warning("Belum ada data yang diproses. Silakan unggah data terlebih dahulu pada halaman "
                       "**Panduan & Unggah Data**.", icon=":material/warning:")
        st.page_link(PAGE_UNGGAH, label="Ke Halaman Unggah Data", icon=":material/upload_file:")
        return

    hasil = st.session_state.hasil
    if st.session_state.pop("baru_diproses", False):
        st.toast("Data berhasil diproses!", icon=":material/check_circle:")
    if hasil['laporan']:
        with st.expander(f"Catatan praproses ({len(hasil['laporan'])})", icon=":material/info:"):
            for pesan in hasil['laporan']:
                st.markdown(f"- {pesan}")
    if hasil['forecast'].empty:
        st.error("Tidak ada kombinasi wilayah dan jenis ikan yang dapat diprediksi untuk tahun berikutnya.")
        return

    per_wil = ringkas_wilayah(hasil['forecast'])
    tahun_fc = hasil['tahun_forecast']
    rata_pred, rata_akt = per_wil['Prediksi'].mean(), per_wil['Aktual_Terakhir'].mean()
    m1, m2, m3, m4 = st.columns(4)
    # Setiap kartu = container ber-border, sama tinggi (height="stretch"), berisi angka
    # di atas dan satu "pil" keterangan di bawah (vertical_alignment="distribute"),
    # supaya keempat kartu simetris. Pil kartu 3 adalah tombol pembuka daftar wilayah.
    def kartu(kolom):
        return kolom.container(border=True, height="stretch", vertical_alignment="distribute")

    tahun_data = hasil['aktual']['Tahun']
    perubahan_rata = (rata_pred - rata_akt) / rata_akt * 100 if rata_akt else None
    ev = hasil['evaluasi']
    r_terbaik = ev['ringkasan'].set_index('Model').loc[ev['nama_model']]

    with kartu(m1):
        st.metric("Tahun Prediksi", tahun_fc)
        st.badge(f"Data historis {tahun_data.min()}–{tahun_data.max()}", icon=":material/history:", color="gray")
    with kartu(m2):
        st.metric("Rata-rata Prediksi (Kg/Kapita)", fmt(rata_pred))
        if perubahan_rata is not None:
            naik = perubahan_rata >= 0
            st.badge(f"{fmt(perubahan_rata, 1)}% vs {tahun_fc - 1}".replace("-", "−") if not naik
                     else f"+{fmt(perubahan_rata, 1)}% vs {tahun_fc - 1}",
                     icon=":material/arrow_upward:" if naik else ":material/arrow_downward:",
                     color="green" if naik else "red")
    with kartu(m3):
        st.metric("Wilayah Diprediksi Turun", f"{(per_wil['Perubahan_%'] < 0).sum()} dari {len(per_wil)}")
        if st.button("Lihat rincian per wilayah", icon=":material/list:", key="btn_daftar_turun"):
            dialog_wilayah(per_wil, hasil['forecast'], tahun_fc)
    with kartu(m4):
        st.metric("Model Digunakan", ev['nama_model'])
        st.badge(f"RMSE {fmt(r_terbaik['RMSE_rata2'], 4)} · R² {fmt(r_terbaik['R2_rata2'], 4)}",
                 icon=":material/verified:", color="blue")

    kiri, kanan = st.columns([2, 1])
    with kiri:
        with st.container(border=True):
            komponen_peta(hasil)
        with st.container(border=True):
            komponen_tren(hasil)
    with kanan:
        with st.container(border=True):
            komponen_evaluasi(hasil)
        with st.container(border=True):
            komponen_rekomendasi(hasil)


# ==============================================================================
# HALAMAN 3: TENTANG SISTEM
# ==============================================================================
def kategori_ikan(nama):
    dasar = nama.lower().replace('diawetkan', '').strip()
    return KATEGORI_IKAN.get(dasar, 'Lainnya')


def komponen_jenis_ikan():
    """
    Daftar jenis ikan diambil dari hasil seleksi 10 ikan pada data yang terakhir
    diproses, bukan ditulis statis -- sehingga otomatis mengikuti data unggahan.
    """
    hasil = st.session_state.get("hasil")
    if hasil is None:
        st.info("Daftar jenis ikan akan tampil setelah data diunggah dan diproses pada halaman "
                "**Panduan & Unggah Data**.", icon=":material/info:")
        return
    ev = hasil['evaluasi']

    daftar = sorted(ev['top_10_ikan'])
    grup = {}
    for ikan in daftar:
        grup.setdefault(kategori_ikan(ikan), []).append(ikan)
    urutan = [k for k in ['Ikan Laut', 'Ikan Air Tawar', 'Lainnya'] if k in grup]

    nama_kategori = [k.lower() for k in urutan if k != 'Lainnya']
    terdiri = f", terdiri dari {' dan '.join(nama_kategori)}" if nama_kategori else ""
    teks = (f"Sistem memprediksi **{len(daftar)} jenis ikan** dengan rata-rata tingkat konsumsi tertinggi di "
            f"Jawa Barat{terdiri}. Daftar ini dipilih otomatis dari data yang diunggah "
            f"(diproses {ev['dilatih_pada']}).")
    if any('diawetkan' in i.lower() for i in daftar):
        teks += (" Jenis ikan *diawetkan* adalah ikan olahan seperti pindang, ikan asin, dan sejenisnya.")
    st.markdown(teks)

    for kolom, kategori in zip(st.columns(len(urutan)), urutan):
        kolom.markdown(f"**{kategori}**\n\n" + "\n".join(f"- {i}" for i in grup[kategori]))


def halaman_tentang():
    kiri, kanan = st.columns([2, 1])
    with kiri:
        with st.container(border=True):
            st.subheader("Latar Belakang")
            st.markdown("""
Ikan merupakan salah satu sumber protein hewani utama di Indonesia dan komoditas strategis dalam ketahanan
pangan nasional. Jawa Barat adalah pusat produksi perikanan yang penting, namun tingkat konsumsi ikan rumah
tangganya masih tergolong rendah dan tidak merata antar kabupaten/kota.

Proyeksi tren konsumsi ikan diperlukan agar kebijakan ketahanan pangan dapat disusun lebih tepat sasaran,
misalnya untuk mengantisipasi wilayah yang berpotensi mengalami penurunan konsumsi. Karena data konsumsi
ikan bersifat deret waktu dan dipengaruhi banyak faktor, sistem ini menerapkan dan membandingkan dua
algoritma *gradient boosting*, yaitu **XGBoost** dan **LightGBM**, untuk memprediksi tingkat konsumsi ikan
masyarakat di **27 kabupaten/kota Provinsi Jawa Barat** berdasarkan data historis KKP RI periode 2015–2025.
""")
            st.markdown("#### Jenis Ikan yang Diprediksi")
            komponen_jenis_ikan()
            st.markdown("#### Wilayah yang Dicakup")
            st.markdown("Prediksi dilakukan untuk seluruh **27 kabupaten/kota** di Provinsi Jawa Barat, "
                        "yaitu 18 kabupaten dan 9 kota:")
            kab, kota = st.columns(2)
            kab.markdown("""
**Kabupaten (18)**

Bandung, Bandung Barat, Bekasi, Bogor, Ciamis, Cianjur, Cirebon, Garut, Indramayu, Karawang, Kuningan,
Majalengka, Pangandaran, Purwakarta, Subang, Sukabumi, Sumedang, Tasikmalaya
""")
            kota.markdown("""
**Kota (9)**

Kota Bandung, Kota Banjar, Kota Bekasi, Kota Bogor, Kota Cimahi, Kota Cirebon, Kota Depok,
Kota Sukabumi, Kota Tasikmalaya
""")
            st.subheader("Metode")
            st.markdown("""
- **Praproses data:** pembersihan data, standarisasi nama wilayah, penggabungan tiga dataset, seleksi
  10 jenis ikan dengan rata-rata konsumsi tertinggi, *lag features* (Lag_1 & Lag_2 konsumsi, Lag_1 jumlah
  penduduk, Lag_1 PDRB per kapita), dan *one-hot encoding* jenis ikan.
- **Pemodelan:** satu model global untuk seluruh wilayah dan jenis ikan, dilatih langsung dari data yang
  diunggah setiap kali tombol **Proses Data** ditekan, dan divalidasi dengan *Walk-Forward Validation*
  (expanding window per tahun) untuk mencegah kebocoran data.
- **Evaluasi:** MAE, RMSE, dan R². Model dengan RMSE rata-rata terendah dipilih otomatis, lalu dilatih ulang
  dengan seluruh data untuk menghasilkan prediksi.
""")
            st.subheader("Tujuan")
            st.markdown("""
Membantu instansi pemerintah daerah terkait ketahanan pangan dan perikanan dalam menentukan prioritas
wilayah untuk program intervensi, seperti Gerakan Memasyarakatkan Makan Ikan (Gemarikan), edukasi
konsumsi ikan, perbaikan distribusi, maupun perencanaan bantuan induk ikan. Sistem ini mendukung
**SDG 2 (Zero Hunger)** dan **SDG 3 (Good Health and Well-being)**.
""")
    with kanan:
        with st.container(border=True):
            st.subheader("Perancang Sistem")
            st.markdown("  \n".join([
                "**Akbar Putra Syarif**",
                "NIM 535230076",
                "Program Studi Teknik Informatika",
                "Fakultas Teknologi Informasi",
                "Universitas Tarumanagara",
            ]))
            st.markdown("**Dosen Pembimbing**  \nDr. Bagus Mulyawan, S.Kom., M.M.")
            st.markdown("**Pembimbing Pendamping**  \nDr. Meirista Wulandari, S.T., M.Eng.")
        with st.container(border=True):
            st.subheader("Sumber Data")
            st.markdown("""
1. **Konsumsi Ikan Masyarakat** (Kg/Kapita), 2015–2025 —
   [Portal Data KKP RI](https://portaldata.kkp.go.id/portals/data-statistik/konsumsi-ikan-rt/tbl-dinamis)
2. **Jumlah Penduduk** per Kabupaten/Kota —
   [Galura Data Jabar](https://galura.jabarprov.go.id/dataset/proyeksi-penduduk-berdasarkan-kabupatenkota-di-jawa-barat)
3. **PDRB per Kapita ADHB** (Ribu Rupiah), 2016–2025 —
   [BPS Provinsi Jawa Barat](https://jabar.bps.go.id/id/statistics-table/3/YWtoQlRVZzNiMU5qU1VOSlRFeFZiRTR4VDJOTVVUMDkjMyMzMjAw/produk-domestik-regional-bruto-per-kapita-atas-dasar-harga-berlaku-menurut-kabupaten-kota-di-provinsi-jawa-barat--ribu-rupiah-.html?year=2025)
""")


# ==============================================================================
# NAVIGASI & SESI
# ==============================================================================
# Tiap halaman punya URL sendiri (/panduan, /halaman-utama, /tentang) sehingga
# tombol Back/Forward browser berpindah antarhalaman dashboard.
PAGE_UNGGAH = st.Page(halaman_unggah, title="Panduan & Unggah Data", icon=":material/upload_file:",
                      url_path="panduan", default=True)
PAGE_UTAMA = st.Page(halaman_utama, title="Halaman Utama", icon=":material/dashboard:",
                     url_path="halaman-utama")
PAGE_TENTANG = st.Page(halaman_tentang, title="Tentang Sistem", icon=":material/info:", url_path="tentang")

halaman_aktif = st.navigation([PAGE_UNGGAH, PAGE_UTAMA, PAGE_TENTANG], position="hidden")

# Pulihkan hasil dari file sesi kalau session state kosong (mis. setelah reload)
# tapi URL masih membawa ?sesi=<id>
_jadwal_bersih_sesi()
sid_url = st.query_params.get("sesi")
if sid_url and sid_url != st.session_state.get("sesi_id"):
    hasil_sesi = muat_sesi(sid_url)
    # File sesi format lama (sebelum model dilatih saat Proses Data) tidak punya
    # 'evaluasi' -- dianggap kedaluwarsa supaya pengguna mengunggah ulang
    if hasil_sesi is not None and 'evaluasi' in hasil_sesi:
        st.session_state.hasil = hasil_sesi
        st.session_state.sesi_id = sid_url
    else:
        st.session_state.sesi_hilang = True
        del st.query_params["sesi"]
elif not sid_url and st.session_state.get("sesi_id"):
    # Jaga agar ID sesi selalu ada di URL, supaya reload di halaman mana pun aman
    st.query_params["sesi"] = st.session_state.sesi_id

# Menu navigasi (sidebar, persisten). Link halaman ditata seperti tombol; link
# halaman aktif (dikenali dari href-nya) diberi warna primer.
# href berbentuk "<url_path>?sesi=..." (halaman default: "" atau "?sesi=...")
if halaman_aktif.title == PAGE_UNGGAH.title:
    sel_aktif = ['[href=""]', '[href^="?"]']
else:
    sel_aktif = [f'[href^="{halaman_aktif.url_path}"]']
link_nav = '[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]'
css_aktif = ", ".join(link_nav + sel for sel in sel_aktif)
css_aktif_teks = ", ".join(link_nav + sel + " span" for sel in sel_aktif)
st.markdown(f"""
<style>
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{
    border: 1px solid #d0d7e2; border-radius: 8px; background: #ffffff;
    justify-content: center; padding: 0.45rem 0.75rem; margin-bottom: 0.15rem;
}}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {{ border-color: #1f6feb; }}
{css_aktif} {{ background: #1f6feb; border-color: #1f6feb; }}
{css_aktif_teks} {{ color: #ffffff !important; }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### :material/set_meal: Konsumsi Ikan Jawa Barat")
    st.divider()
    for page in [PAGE_UNGGAH, PAGE_UTAMA, PAGE_TENTANG]:
        st.page_link(page, query_params=query_sesi(), width="stretch")
    st.divider()
    if "hasil" in st.session_state:
        ev_sidebar = st.session_state.hasil['evaluasi']
        st.caption(f"Model terpilih: **{ev_sidebar['nama_model']}**  \n"
                   f"Dilatih: {ev_sidebar['dilatih_pada']}")
    else:
        st.caption("Model dilatih saat data diproses")

st.title(JUDUL)
halaman_aktif.run()
