"""
PENYIMPANAN SESI HASIL PREDIKSI

st.session_state milik Streamlit hilang setiap kali halaman di-reload (session
terikat ke satu koneksi tab browser). Supaya hasil unggahan tidak hilang, hasil
proses juga disimpan sementara di server (folder sesi/) dengan ID acak, dan ID
itu ditaruh di URL (?sesi=<id>). Saat reload, dashboard memuat ulang hasil dari
file berdasarkan ID di URL. File sesi dihapus otomatis setelah kedaluwarsa.
"""

import re
import time
import uuid

import joblib

from src.config import SESI_DIR, SESI_KEDALUWARSA_JAM

_POLA_ID = re.compile(r"^[0-9a-f]{32}$")


def _path(sesi_id):
    # ID divalidasi ketat (32 karakter heksadesimal) supaya nilai dari URL
    # tidak bisa dipakai untuk membaca file lain di server
    if not sesi_id or not _POLA_ID.match(sesi_id):
        return None
    return SESI_DIR / f"{sesi_id}.joblib"


def simpan_sesi(hasil):
    SESI_DIR.mkdir(exist_ok=True)
    sesi_id = uuid.uuid4().hex
    joblib.dump(hasil, _path(sesi_id))
    return sesi_id


def muat_sesi(sesi_id):
    """Kembalikan hasil prediksi untuk sesi_id, atau None kalau tidak ada / kedaluwarsa."""
    path = _path(sesi_id)
    if path is None or not path.exists():
        return None
    if time.time() - path.stat().st_mtime > SESI_KEDALUWARSA_JAM * 3600:
        path.unlink(missing_ok=True)
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def hapus_sesi(sesi_id):
    path = _path(sesi_id)
    if path is not None:
        path.unlink(missing_ok=True)


def bersihkan_sesi_lama():
    if not SESI_DIR.exists():
        return
    batas = time.time() - SESI_KEDALUWARSA_JAM * 3600
    for f in SESI_DIR.glob("*.joblib"):
        if f.stat().st_mtime < batas:
            f.unlink(missing_ok=True)
