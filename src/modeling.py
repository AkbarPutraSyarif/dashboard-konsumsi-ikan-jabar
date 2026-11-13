"""
MODUL PEMODELAN & EVALUASI (XGBoost vs LightGBM).
Logikanya sama dengan modeling_jabar.py: walk-forward validation (expanding
window per tahun), evaluasi MAE/RMSE/R2, pilih model dengan RMSE rata-rata
terendah, lalu latih ulang model final dengan seluruh data.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import PARAM_XGB, PARAM_LGBM, TARGET_COL
from src.preprocessing import encode


def buat_model(nama):
    return XGBRegressor(**PARAM_XGB) if nama == 'XGBoost' else LGBMRegressor(**PARAM_LGBM)


def _metrik(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred),
    )


def walk_forward(df_multivariate, log=print):
    """
    Setiap fold: data latih = seluruh tahun SEBELUM tahun uji,
    data uji = satu tahun tunggal setelahnya (tanpa pengacakan -> tanpa leakage).
    Variabel Tahun tidak masuk fitur, hanya dipakai untuk membagi data.
    """
    X_all, feature_cols = encode(df_multivariate)
    y_all = df_multivariate[TARGET_COL]
    tahun = df_multivariate['Tahun']

    tahun_tersedia = sorted(tahun.unique())
    hasil_fold = []
    pred_oos = []

    for tahun_uji in tahun_tersedia[1:]:
        train_mask = tahun < tahun_uji
        test_mask = tahun == tahun_uji
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        baris = {'Tahun_Uji': tahun_uji, 'N_Train': len(X_train), 'N_Test': len(X_test)}
        df_pred = df_multivariate.loc[test_mask, ['Tahun', 'Kabupaten/Kota', 'Jenis Ikan']].copy()
        df_pred['Aktual'] = y_test.values

        for nama in ['XGBoost', 'LightGBM']:
            model = buat_model(nama)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mae, rmse, r2 = _metrik(y_test, pred)
            baris.update({f'MAE_{nama}': mae, f'RMSE_{nama}': rmse, f'R2_{nama}': r2})
            df_pred[f'Prediksi_{nama}'] = pred

        hasil_fold.append(baris)
        pred_oos.append(df_pred)
        log(f"   Fold tahun uji {tahun_uji}: "
            f"XGB(MAE={baris['MAE_XGBoost']:.4f}, RMSE={baris['RMSE_XGBoost']:.4f}, R2={baris['R2_XGBoost']:.4f}) | "
            f"LGBM(MAE={baris['MAE_LightGBM']:.4f}, RMSE={baris['RMSE_LightGBM']:.4f}, R2={baris['R2_LightGBM']:.4f})")

    df_hasil_fold = pd.DataFrame(hasil_fold)
    ringkasan = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM'],
        'MAE_rata2': [df_hasil_fold['MAE_XGBoost'].mean(), df_hasil_fold['MAE_LightGBM'].mean()],
        'RMSE_rata2': [df_hasil_fold['RMSE_XGBoost'].mean(), df_hasil_fold['RMSE_LightGBM'].mean()],
        'R2_rata2': [df_hasil_fold['R2_XGBoost'].mean(), df_hasil_fold['R2_LightGBM'].mean()],
    })
    model_terbaik = ringkasan.loc[ringkasan['RMSE_rata2'].idxmin(), 'Model']
    return df_hasil_fold, ringkasan, model_terbaik, pd.concat(pred_oos, ignore_index=True), feature_cols


def latih_final(df_multivariate, nama_model, feature_cols):
    X_full, _ = encode(df_multivariate, feature_cols)
    model = buat_model(nama_model)
    model.fit(X_full, df_multivariate[TARGET_COL])
    return model
