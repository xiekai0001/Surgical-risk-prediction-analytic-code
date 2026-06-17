"""
MIMIC-IV 手术风险评分 — 特征工程 (完整版)
人口学 + Charlson + 手术类型 + 17项化验
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import time

DATA_DIR = Path(r"Z:\本地数据\asa_data")

# ICD 手术大类 (同 baseline)
ICD9_SURGERY = {
    'surg_cardiac':       ['35', '36', '37'],
    'surg_vascular':      ['38', '39'],
    'surg_thoracic':      ['32', '33', '34'],
    'surg_digestive':     ['42', '43', '44', '45', '46', '47', '48', '49', '53', '54'],
    'surg_hepatobiliary': ['50', '51', '52'],
    'surg_urologic':      ['55', '56', '57', '58', '59'],
    'surg_orthopedic':    ['76', '77', '78', '79', '80', '81', '82', '83', '84'],
    'surg_neurosurgery':  ['01', '02', '03', '04', '05'],
    'surg_obgyn':         ['65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75'],
    'surg_ent':           ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
}

ICD10_SURGERY = {
    'surg_cardiac':       ['2'],
    'surg_vascular':      ['3', '4', '5', '6'],
    'surg_thoracic':      ['B'],
    'surg_digestive':     ['D'],
    'surg_hepatobiliary': ['F'],
    'surg_urologic':      ['T'],
    'surg_orthopedic':    ['K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S'],
    'surg_neurosurgery':  ['0', '1'],
    'surg_obgyn':         ['U', 'V'],
    'surg_ent':           ['9', 'C'],
}

# 化验列 (与 1d_extract_labs_main.py 一致)
LAB_ITEMS = [
    'creatinine', 'bun', 'albumin', 'hemoglobin', 'wbc', 'platelet',
    'inr', 'pt', 'ptt', 'glucose', 'sodium', 'potassium', 'bicarbonate',
    'alt', 'ast', 'bilirubin_total', 'lactate',
]


def classify_surgery_icd9(code: str) -> dict:
    prefix = str(code)[:2].zfill(2)
    return {cat: int(any(prefix.startswith(p) for p in prefixes))
            for cat, prefixes in ICD9_SURGERY.items()}


def classify_surgery_icd10(code: str) -> dict:
    code = str(code).upper()
    if len(code) < 2:
        return {k: 0 for k in ICD10_SURGERY}
    body_system = code[1]
    return {cat: int(body_system in prefixes)
            for cat, prefixes in ICD10_SURGERY.items()}


def build_surgery_features(surgery_df: pd.DataFrame) -> pd.DataFrame:
    primary = surgery_df[surgery_df['seq_num'] == 1].copy()
    primary['icd_code'] = primary['icd_code'].astype(str)
    primary['icd_version'] = primary['icd_version'].astype(int)

    rows = []
    all_cats = set(ICD9_SURGERY.keys()) | set(ICD10_SURGERY.keys())
    for hadm_id, grp in primary.groupby('hadm_id'):
        row = {'hadm_id': hadm_id}
        for c in all_cats:
            row[c] = 0
        for _, rec in grp.iterrows():
            code = rec['icd_code']
            ver = rec['icd_version']
            cats = classify_surgery_icd9(code) if ver == 9 else classify_surgery_icd10(code)
            for k, v in cats.items():
                if v == 1:
                    row[k] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def compute_derived_labs(df):
    """计算衍生化验指标"""
    # eGFR (CKD-EPI 2021, 简化版)
    # eGFR = 142 * (Cr/k)^alpha * 0.9938^age * (1.012 if female)
    # k=0.9(male)/0.7(female), alpha=-0.302(f)/-0.241(m) for Cr<=k
    cr = df['creatinine_value'].clip(lower=0.1)
    age = df['age'].clip(lower=18)

    is_female = (df['gender'] == 'F').astype(float)
    k = np.where(df['gender'] == 'F', 0.7, 0.9)
    alpha = np.where(df['gender'] == 'F', -0.241, -0.302)  # simplified

    cr_k = cr / k
    # Use single formula for simplicity (most Cr values > k actually)
    egfr = 142 * np.power(cr_k.clip(lower=0.01), alpha) * np.power(0.9938, age)
    egfr = egfr * (1.012 ** is_female)
    df['egfr'] = egfr.clip(upper=200)

    # eGFR categories
    df['egfr_stage'] = 0
    df.loc[egfr < 90, 'egfr_stage'] = 1
    df.loc[egfr < 60, 'egfr_stage'] = 2
    df.loc[egfr < 45, 'egfr_stage'] = 3
    df.loc[egfr < 30, 'egfr_stage'] = 4
    df.loc[egfr < 15, 'egfr_stage'] = 5

    # Anemia: Hb < 12 (F) / < 13 (M)
    df['anemia'] = 0
    df.loc[(df['gender'] == 'F') & (df['hemoglobin_value'] < 12), 'anemia'] = 1
    df.loc[(df['gender'] == 'M') & (df['hemoglobin_value'] < 13), 'anemia'] = 1

    # BUN/Cr ratio (dehydration / prerenal signal)
    df['bun_cr_ratio'] = df['bun_value'] / df['creatinine_value'].clip(lower=0.01)
    df['bun_cr_ratio'] = df['bun_cr_ratio'].clip(upper=100)

    # Hypoalbuminemia
    df['hypoalbuminemia'] = (df['albumin_value'] < 3.5).astype(int)

    # Leukocytosis / Leukopenia
    df['leukocytosis'] = (df['wbc_value'] > 11).astype(int)
    df['leukopenia'] = (df['wbc_value'] < 4).astype(int)

    # Thrombocytopenia
    df['thrombocytopenia'] = (df['platelet_value'] < 150).astype(int)

    # Coagulopathy: INR > 1.5
    df['coagulopathy'] = (df['inr_value'] > 1.5).astype(int)

    # Hyperglycemia
    df['hyperglycemia'] = (df['glucose_value'] > 180).astype(int)

    # Hyponatremia / Hypernatremia
    df['hyponatremia'] = (df['sodium_value'] < 135).astype(int)
    df['hypernatremia'] = (df['sodium_value'] > 145).astype(int)

    # Hyperkalemia / Hypokalemia
    df['hypokalemia'] = (df['potassium_value'] < 3.5).astype(int)
    df['hyperkalemia'] = (df['potassium_value'] > 5.2).astype(int)

    # Acidosis (bicarb < 22)
    df['acidosis'] = (df['bicarbonate_value'] < 22).astype(int)

    # Elevated lactate (>2)
    df['lactate_elevated'] = (df['lactate_value'] > 2.0).astype(int)

    # Liver injury (ALT or AST > 2x upper normal)
    df['liver_injury'] = ((df['alt_value'] > 80) | (df['ast_value'] > 80)).astype(int)

    # Hyperbilirubinemia
    df['hyperbilirubinemia'] = (df['bilirubin_total_value'] > 1.2).astype(int)

    return df


def main():
    t_start = time.time()
    print("=" * 60)
    print("MIMIC-IV 特征工程 v2 (完整版: + 17项化验)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # === 1. 加载 ===
    print("\n[1] 加载输入文件...")
    baseline = pd.read_csv(DATA_DIR / 'cohort_baseline.csv')
    comorb = pd.read_csv(DATA_DIR / 'cohort_comorbidities.csv')
    outcomes = pd.read_csv(DATA_DIR / 'cohort_outcomes.csv')
    surgery = pd.read_csv(DATA_DIR / 'surgery_procedures.csv', dtype={'icd_code': str})
    labs = pd.read_csv(DATA_DIR / 'cohort_preop_labs_main.csv')

    print(f"  baseline:   {len(baseline):,} rows")
    print(f"  comorbidities: {len(comorb):,} rows")
    print(f"  outcomes:   {len(outcomes):,} rows")
    print(f"  surgery:    {len(surgery):,} rows")
    print(f"  labs:       {len(labs):,} rows x {len(labs.columns)} cols")

    # === 2. 手术类型 ===
    print("\n[2] 构建手术类型特征...")
    surg_feat = build_surgery_features(surgery)
    surg_cols = [c for c in surg_feat.columns if c != 'hadm_id']
    print(f"  手术类型: {len(surg_cols)} 类")
    for col in surg_cols:
        n = surg_feat[col].sum()
        print(f"    {col:25s}: {n:>8,} ({n/len(surg_feat)*100:5.1f}%)")
    unknown_n = (surg_feat[surg_cols].sum(axis=1) == 0).sum()
    print(f"    Unknown: {unknown_n:>8,} ({unknown_n/len(surg_feat)*100:5.1f}%)")

    # === 3. 化验清洗 ===
    print("\n[3] 化验数据清洗 & 衍生变量...")
    # 提取化验值列
    lab_value_cols = [f'{item}_value' for item in LAB_ITEMS]
    lab_df = labs[['hadm_id'] + lab_value_cols].copy()

    # -1 -> NaN
    for col in lab_value_cols:
        lab_df[col] = lab_df[col].replace(-1, np.nan)

    # 缺失率
    for item in LAB_ITEMS:
        col = f'{item}_value'
        present = lab_df[col].notna().sum()
        if present > 0:
            med = lab_df[col].median()
            print(f"  {item:20s}: {present:>8,}/{len(lab_df):,} "
                  f"({present/len(lab_df)*100:5.1f}%) median={med:.2f}")
        else:
            print(f"  {item:20s}: ALL MISSING")

    # === 4. 合并 ===
    print("\n[4] 合并特征矩阵...")
    n_start = len(baseline)

    # Build base
    merged = baseline[['hadm_id', 'age', 'gender', 'emergency', 'admission_type']].copy()
    merged['age_gt89'] = (merged['age'] == 91).astype(int)

    # Merge comorbidities
    comorb_m = comorb.drop(columns=['subject_id'], errors='ignore')
    merged = merged.merge(comorb_m, on='hadm_id', how='left')
    assert len(merged) == n_start

    # Merge surgery
    merged = merged.merge(surg_feat, on='hadm_id', how='left')
    assert len(merged) == n_start

    # Merge labs
    merged = merged.merge(lab_df, on='hadm_id', how='left')
    assert len(merged) == n_start

    # Merge outcomes (skip duplicate cols)
    out_existing = [c for c in outcomes.columns if c not in merged.columns or c == 'hadm_id']
    merged = merged.merge(outcomes[out_existing], on='hadm_id', how='left')
    assert len(merged) == n_start

    print(f"  合并后: {len(merged):,} rows x {len(merged.columns)} cols")

    # === 5. 衍生变量 ===
    print("\n[5] 计算衍生化验指标...")
    merged = compute_derived_labs(merged)
    derived_cols = ['egfr', 'egfr_stage', 'anemia', 'bun_cr_ratio',
                    'hypoalbuminemia', 'leukocytosis', 'leukopenia',
                    'thrombocytopenia', 'coagulopathy', 'hyperglycemia',
                    'hyponatremia', 'hypernatremia', 'hypokalemia', 'hyperkalemia',
                    'acidosis', 'lactate_elevated', 'liver_injury', 'hyperbilirubinemia']
    print(f"  衍生变量: {len(derived_cols)}")

    # Fill NaN in derived columns: binary flags → 0, continuous → median
    for col in derived_cols:
        if col in merged.columns and merged[col].isna().any():
            if merged[col].nunique() <= 2:
                merged[col] = merged[col].fillna(0).astype(int)
            else:
                merged[col] = merged[col].fillna(merged[col].median())

    # === 6. 标准化 ===
    print("\n[6] 特征标准化...")

    # Age
    age_mask = merged['age'] < 89
    age_mean = merged.loc[age_mask, 'age'].mean()
    age_std = merged.loc[age_mask, 'age'].std()
    merged['age_std'] = (merged['age'] - age_mean) / age_std

    # Gender
    merged['gender_male'] = (merged['gender'] == 'M').astype(int)

    # Lab value clipping + z-score (on non-missing values, then fill missing as 0)
    lab_std_cols = []
    for item in LAB_ITEMS:
        col = f'{item}_value'
        # Clip extreme outliers (1st and 99th percentile, then fill remaining extremes)
        lo = merged[col].quantile(0.01)
        hi = merged[col].quantile(0.99)
        clipped = merged[col].clip(lo, hi)
        m = clipped.mean()
        s = clipped.std()
        if s > 0:
            std_col = f'{item}_std'
            merged[std_col] = (clipped - m) / s
            lab_std_cols.append(std_col)

    # Fill missing standardized labs with 0 (mean imputation in z-space)
    for col in lab_std_cols:
        merged[col] = merged[col].fillna(0)

    # Missing indicators
    miss_cols = []
    for item in LAB_ITEMS:
        col = f'{item}_value'
        miss_col = f'{item}_missing'
        merged[miss_col] = merged[col].isna().astype(int)
        miss_cols.append(miss_col)
        pct = merged[miss_col].mean() * 100
        if pct > 0.1:
            print(f"  {miss_col}: {pct:.1f}% missing")

    # Fill surgery NA
    for c in surg_cols:
        merged[c] = merged[c].fillna(0).astype(int)

    print(f"  标准化化验: {len(lab_std_cols)} 列 + {len(miss_cols)} 缺失指示列")

    # === 7. 整理输出 ===
    print("\n[7] 整理输出列...")

    feature_cols = [
        'hadm_id',
        # Demographics
        'age', 'age_std', 'age_gt89', 'gender_male', 'emergency',
        # Charlson
        'charlson_mi', 'charlson_chf', 'charlson_pvd', 'charlson_cvd',
        'charlson_dementia', 'charlson_copd', 'charlson_rheumatic',
        'charlson_pud', 'charlson_mld', 'charlson_dm', 'charlson_dmcx',
        'charlson_hp', 'charlson_rend', 'charlson_canc', 'charlson_msld',
        'charlson_mets', 'charlson_aids', 'charlson_score',
        # Surgery
    ] + surg_cols + [
        # Lab standardized values
    ] + lab_std_cols + [
        # Lab missing indicators
    ] + miss_cols + [
        # Derived lab features
    ] + derived_cols + [
        # Outcomes
        'died_30d', 'unplanned_icu', 'aki',
        'comp_cardiac', 'comp_respiratory', 'comp_sepsis',
        'comp_bleeding', 'comp_vte', 'comp_any',
    ]

    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = 0

    features = merged[feature_cols].copy()
    # Final NaN check: fill any remaining NaN with 0
    nan_cols = features.columns[features.isna().any()].tolist()
    if nan_cols:
        print(f"  Filling NaN in: {nan_cols}")
        features[nan_cols] = features[nan_cols].fillna(0)
    print(f"  输出: {len(features):,} rows x {len(features.columns)} cols")

    # === 8. 保存 ===
    print("\n[8] 保存...")
    fpath = DATA_DIR / 'features_full.csv'
    features.to_csv(fpath, index=False)
    sz = os.path.getsize(fpath) / 1024**2
    print(f"  features_full.csv: {len(features):,} rows, {sz:.1f} MB")

    # === 9. 报告 ===
    print("\n[9] 特征摘要...")
    lines = []
    L = lines.append
    L("=" * 70)
    L("MIMIC-IV 特征工程摘要 (完整版)")
    L(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L("=" * 70)

    L(f"\n【样本量】")
    L(f"  训练样本: {len(features):,}")
    L(f"  特征总数: {len(features.columns) - 5}")  # minus hadm_id + outcomes

    L(f"\n【人口学】")
    L(f"  年龄: median={features['age'].median():.0f}, mean={features['age'].mean():.1f}")
    L(f"  男性: {features['gender_male'].sum():,} ({features['gender_male'].mean()*100:.1f}%)")
    L(f"  急诊: {features['emergency'].sum():,} ({features['emergency'].mean()*100:.1f}%)")

    L(f"\n【Charlson总分】")
    L(f"  mean={features['charlson_score'].mean():.2f}, median={features['charlson_score'].median():.0f}")

    L(f"\n【化验覆盖率】")
    for item in LAB_ITEMS:
        col = f'{item}_value'
        if col in merged.columns:
            present = merged[col].notna().sum()
            L(f"  {item:20s}: {present:>8,}/{len(merged):,} ({present/len(merged)*100:5.1f}%)")

    L(f"\n【衍生指标发生率】")
    for col in derived_cols:
        if col in features.columns:
            if 'egfr' not in col or col == 'egfr_stage':
                n = features[col].sum() if features[col].dtype in ['int64', 'int32', 'float64'] else features[col].notna().sum()
                pct = features[col].mean() * 100 if features[col].dtype in ['int64', 'float64'] else 0
                L(f"  {col:25s}: {n:>8,} ({pct:5.1f}%)")

    L(f"\n【结局】")
    L(f"  30天死亡: {features['died_30d'].sum():,} ({features['died_30d'].mean()*100:.2f}%)")
    L(f"  非计划ICU: {features['unplanned_icu'].sum():,} ({features['unplanned_icu'].mean()*100:.2f}%)")
    L(f"  AKI: {features['aki'].sum():,} ({features['aki'].mean()*100:.2f}%)")

    L(f"\n【特征缺失率】")
    n_missing_cols = 0
    for col in features.columns:
        missing = features[col].isna().sum()
        if missing > 0:
            L(f"  {col}: {missing} missing ({missing/len(features)*100:.1f}%)")
            n_missing_cols += 1
    if n_missing_cols == 0:
        L(f"  (无缺失)")

    L("\n" + "=" * 70)

    report_path = DATA_DIR / 'feature_summary_full.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))

    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed:.1f} 秒")
    return features


if __name__ == '__main__':
    features = main()
