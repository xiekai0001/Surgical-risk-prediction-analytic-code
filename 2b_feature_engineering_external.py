"""
External validation feature engineering — eICU + PIC
Aligns features to MIMIC training set using MIMIC-derived scaler parameters.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import time

DATA_DIR = Path(r"Z:\本地数据\asa_data")

LAB_ITEMS = [
    'creatinine', 'bun', 'albumin', 'hemoglobin', 'wbc', 'platelet',
    'inr', 'pt', 'ptt', 'glucose', 'sodium', 'potassium', 'bicarbonate',
    'alt', 'ast', 'bilirubin_total', 'lactate',
]

SURG_COLS = [
    'surg_cardiac', 'surg_vascular', 'surg_thoracic', 'surg_digestive',
    'surg_hepatobiliary', 'surg_urologic', 'surg_orthopedic',
    'surg_neurosurgery', 'surg_obgyn', 'surg_ent',
]

CHARLSON_COLS = [
    'charlson_mi', 'charlson_chf', 'charlson_pvd', 'charlson_cvd',
    'charlson_dementia', 'charlson_copd', 'charlson_rheumatic',
    'charlson_pud', 'charlson_mld', 'charlson_dm', 'charlson_dmcx',
    'charlson_hp', 'charlson_rend', 'charlson_canc', 'charlson_msld',
    'charlson_mets', 'charlson_aids',
]


def compute_derived_labs(df, gender_col='gender_male'):
    """Compute derived lab features (same logic as MIMIC)"""
    cr = df['creatinine_value'].clip(lower=0.1)
    age = df['age'].clip(lower=18)

    is_female = 1 - df[gender_col]
    k = np.where(df[gender_col] == 0, 0.7, 0.9)
    alpha = np.where(df[gender_col] == 0, -0.241, -0.302)

    cr_k = cr / k
    egfr = 142 * np.power(cr_k.clip(lower=0.01), alpha) * np.power(0.9938, age)
    egfr = egfr * (1.012 ** is_female)
    df['egfr'] = egfr.clip(upper=200)

    df['egfr_stage'] = 0
    df.loc[egfr < 90, 'egfr_stage'] = 1
    df.loc[egfr < 60, 'egfr_stage'] = 2
    df.loc[egfr < 45, 'egfr_stage'] = 3
    df.loc[egfr < 30, 'egfr_stage'] = 4
    df.loc[egfr < 15, 'egfr_stage'] = 5

    anemia_f = (df[gender_col] == 0) & (df['hemoglobin_value'] < 12)
    anemia_m = (df[gender_col] == 1) & (df['hemoglobin_value'] < 13)
    df['anemia'] = (anemia_f | anemia_m).astype(int)

    df['bun_cr_ratio'] = df['bun_value'] / df['creatinine_value'].clip(lower=0.01)
    df['bun_cr_ratio'] = df['bun_cr_ratio'].clip(upper=100)

    df['hypoalbuminemia'] = (df['albumin_value'] < 3.5).astype(int)
    df['leukocytosis'] = (df['wbc_value'] > 11).astype(int)
    df['leukopenia'] = (df['wbc_value'] < 4).astype(int)
    df['thrombocytopenia'] = (df['platelet_value'] < 150).astype(int)
    df['coagulopathy'] = (df['inr_value'] > 1.5).astype(int)
    df['hyperglycemia'] = (df['glucose_value'] > 180).astype(int)
    df['hyponatremia'] = (df['sodium_value'] < 135).astype(int)
    df['hypernatremia'] = (df['sodium_value'] > 145).astype(int)
    df['hypokalemia'] = (df['potassium_value'] < 3.5).astype(int)
    df['hyperkalemia'] = (df['potassium_value'] > 5.2).astype(int)
    df['acidosis'] = (df['bicarbonate_value'] < 22).astype(int)
    df['lactate_elevated'] = (df['lactate_value'] > 2.0).astype(int)
    df['liver_injury'] = ((df['alt_value'] > 80) | (df['ast_value'] > 80)).astype(int)
    df['hyperbilirubinemia'] = (df['bilirubin_total_value'] > 1.2).astype(int)

    return df


def build_scaler_params():
    """Compute scaler parameters from MIMIC raw lab values"""
    print("\n[0] Computing MIMIC scaler parameters...")
    labs = pd.read_csv(DATA_DIR / 'cohort_preop_labs_main.csv')
    baseline = pd.read_csv(DATA_DIR / 'cohort_baseline.csv')

    scaler_params = {}

    # Age scaler
    age_mask = baseline['age'] < 89
    scaler_params['age_mean'] = baseline.loc[age_mask, 'age'].mean()
    scaler_params['age_std'] = baseline.loc[age_mask, 'age'].std()

    # Lab scalers
    for item in LAB_ITEMS:
        col = f'{item}_value'
        if col in labs.columns:
            vals = labs[col].replace(-1, np.nan)
            lo = vals.quantile(0.01)
            hi = vals.quantile(0.99)
            clipped = vals.clip(lo, hi)
            scaler_params[f'{item}_mean'] = clipped.mean()
            scaler_params[f'{item}_std'] = clipped.std()
        else:
            scaler_params[f'{item}_mean'] = 0
            scaler_params[f'{item}_std'] = 1

    print(f"  Age: mean={scaler_params['age_mean']:.1f}, std={scaler_params['age_std']:.1f}")
    for item in LAB_ITEMS:
        m = scaler_params[f'{item}_mean']
        s = scaler_params[f'{item}_std']
        print(f"  {item:20s}: mean={m:8.2f}, std={s:8.2f}")

    return scaler_params


def process_external(df, scaler, id_col, db_name):
    """Apply MIMIC feature pipeline to external dataset"""
    print(f"\n[{db_name}] Processing {len(df):,} rows...")

    # --- Clean lab values ---
    for item in LAB_ITEMS:
        val_col = f'{item}_value'
        if val_col in df.columns:
            # -1 and extreme values → NaN
            df[val_col] = df[val_col].replace(-1, np.nan)
            # Clip at MIMIC percentile range
            lo = df[val_col].quantile(0.01)
            hi = df[val_col].quantile(0.99)
            df[val_col] = df[val_col].clip(lo, hi)

    # --- Age standardization ---
    df['age_std'] = (df['age'] - scaler['age_mean']) / scaler['age_std']

    # --- Lab z-score standardization ---
    lab_std_cols = []
    for item in LAB_ITEMS:
        val_col = f'{item}_value'
        std_col = f'{item}_std'
        m = scaler[f'{item}_mean']
        s = scaler[f'{item}_std']
        if s and s > 0 and val_col in df.columns:
            df[std_col] = (df[val_col] - m) / s
            lab_std_cols.append(std_col)

    # Fill missing standardized labs with 0
    for col in lab_std_cols:
        df[col] = df[col].fillna(0)

    # --- Missing indicators ---
    miss_cols = []
    for item in LAB_ITEMS:
        val_col = f'{item}_value'
        miss_col = f'{item}_missing'
        df[miss_col] = df[val_col].isna().astype(int)
        miss_cols.append(miss_col)

    # --- Derived variables ---
    gender_for_derived = 'gender_male'
    if gender_for_derived not in df.columns:
        df['gender_male'] = 0  # fallback

    print(f"  Computing derived labs...")
    df = compute_derived_labs(df, gender_for_derived)

    # Fill NaN in derived: binary → 0, continuous → median
    derived_cols = ['egfr', 'egfr_stage', 'anemia', 'bun_cr_ratio',
                    'hypoalbuminemia', 'leukocytosis', 'leukopenia',
                    'thrombocytopenia', 'coagulopathy', 'hyperglycemia',
                    'hyponatremia', 'hypernatremia', 'hypokalemia', 'hyperkalemia',
                    'acidosis', 'lactate_elevated', 'liver_injury', 'hyperbilirubinemia']
    for col in derived_cols:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                if df[col].nunique() <= 2 or col.endswith('stage'):
                    df[col] = df[col].fillna(0).astype(int)
                else:
                    df[col] = df[col].fillna(df[col].median())
            else:
                if df[col].dtype == 'float64' and df[col].nunique() <= 3:
                    df[col] = df[col].fillna(0).astype(int)

    # --- Align feature columns to MIMIC ---
    feature_cols = [
        'age', 'age_std', 'age_gt89', 'gender_male', 'emergency',
    ] + CHARLSON_COLS + ['charlson_score'] + SURG_COLS + \
        [f'{item}_std' for item in LAB_ITEMS] + \
        [f'{item}_missing' for item in LAB_ITEMS] + derived_cols

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Outcomes
    if 'died_hosp' in df.columns:
        df['died_30d'] = df['died_hosp']  # eICU uses in-hospital as proxy
    if 'died_30d' not in df.columns:
        df['died_30d'] = 0

    result = df[[id_col] + feature_cols + ['died_30d']].copy()

    # Final NaN check
    nan_cols = result.columns[result.isna().any()].tolist()
    if nan_cols:
        print(f"  Filling remaining NaN in: {nan_cols}")
        for c in nan_cols:
            if c == 'died_30d':
                continue
            if result[c].nunique() <= 3:
                result[c] = result[c].fillna(0)
            else:
                result[c] = result[c].fillna(result[c].median())

    print(f"  {db_name}: {len(result):,} rows x {len(feature_cols)} features")
    print(f"  Outcome: {result['died_30d'].sum():,} ({result['died_30d'].mean()*100:.1f}%)")

    return result


def main():
    t_start = time.time()
    print("=" * 60)
    print("External Validation Feature Engineering")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 0: Compute MIMIC scaler params
    scaler = build_scaler_params()

    # Step 1: Process eICU
    print("\n[1] Loading eICU raw data...")
    eicu = pd.read_csv(DATA_DIR / 'eicu_raw.csv')
    id_col_eicu = 'patientunitstayid'
    eicu_feat = process_external(eicu, scaler, id_col_eicu, 'eICU')
    # Rename ID to hadm_id for consistency
    eicu_feat = eicu_feat.rename(columns={'patientunitstayid': 'hadm_id'})
    eicu_feat.to_csv(DATA_DIR / 'features_eicu.csv', index=False)
    sz = os.path.getsize(DATA_DIR / 'features_eicu.csv') / 1024**2
    print(f"  Saved: features_eicu.csv ({sz:.1f} MB)")

    # Step 2: Process PIC
    print("\n[2] Loading PIC raw data...")
    pic = pd.read_csv(DATA_DIR / 'pic_raw.csv')
    id_col_pic = 'hadm_id'
    pic_feat = process_external(pic, scaler, id_col_pic, 'PIC')
    pic_feat.to_csv(DATA_DIR / 'features_pic.csv', index=False)
    sz = os.path.getsize(DATA_DIR / 'features_pic.csv') / 1024**2
    print(f"  Saved: features_pic.csv ({sz:.1f} MB)")

    # Step 3: Load MIMIC features for reference
    print("\n[3] Loading MIMIC features (reference)...")
    mimic = pd.read_csv(DATA_DIR / 'features_full.csv')
    print(f"  MIMIC: {len(mimic):,} rows x {len(mimic.columns)} cols")

    # Summary
    print("\n" + "=" * 70)
    print("External Validation Feature Engineering Summary")
    print("=" * 70)
    for name, df in [('MIMIC', mimic), ('eICU', eicu_feat), ('PIC', pic_feat)]:
        d = df['died_30d'].sum()
        print(f"  {name:8s}: {len(df):>8,} rows, {d:>8,} deaths ({d/len(df)*100:5.2f}%)")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
    return eicu_feat, pic_feat


if __name__ == '__main__':
    eicu_feat, pic_feat = main()
