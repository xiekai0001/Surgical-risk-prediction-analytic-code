"""
MIMIC-IV 手术风险评分 — 特征工程 (基线版)
无化验数据版本，基于人口学 + Charlson合并症 + 手术类型
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import time

DATA_DIR = Path(r"Z:\本地数据\asa_data")
OUT_DIR = DATA_DIR  # 输出到同一目录

# ═══════════════════════════════════════════════════════════════
# ICD 手术大类映射
# ═══════════════════════════════════════════════════════════════

# ICD-9-CM 手术编码: 前2位匹配
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

# ICD-10-PCS: 第2位字符 = 身体系统
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


def classify_surgery_icd9(code: str) -> dict:
    """ICD-9: 取前2位匹配大类"""
    prefix = str(code)[:2].zfill(2)
    cats = {}
    for cat_name, prefixes in ICD9_SURGERY.items():
        cats[cat_name] = int(any(prefix.startswith(p) for p in prefixes))
    return cats


def classify_surgery_icd10(code: str) -> dict:
    """ICD-10-PCS: 第2位字符匹配 -- 第2位即 index 1"""
    code = str(code).upper()
    if len(code) < 2:
        return {k: 0 for k in ICD10_SURGERY}
    body_system = code[1]
    cats = {}
    for cat_name, prefixes in ICD10_SURGERY.items():
        cats[cat_name] = int(body_system in prefixes)
    return cats


def build_surgery_features(surgery_df: pd.DataFrame) -> pd.DataFrame:
    """从手术编码构建手术类型特征矩阵

    策略: 对每个 hadm_id, 取 seq_num=1 (主要手术) 做分类.
    若存在多个主要手术 (罕见), 任一类匹配即标记为1.
    """
    primary = surgery_df[surgery_df['seq_num'] == 1].copy()
    primary['icd_code'] = primary['icd_code'].astype(str)
    primary['icd_version'] = primary['icd_version'].astype(int)

    rows = []
    for hadm_id, grp in primary.groupby('hadm_id'):
        row = {'hadm_id': hadm_id}
        # 初始化所有类别为0
        all_cats = set(ICD9_SURGERY.keys()) | set(ICD10_SURGERY.keys())
        for c in all_cats:
            row[c] = 0

        for _, rec in grp.iterrows():
            code = rec['icd_code']
            ver = rec['icd_version']
            if ver == 9:
                cats = classify_surgery_icd9(code)
            else:
                cats = classify_surgery_icd10(code)
            for k, v in cats.items():
                if v == 1:
                    row[k] = 1
        rows.append(row)

    surg_feat = pd.DataFrame(rows)
    return surg_feat


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 60)
    print("MIMIC-IV 特征工程 v1 (基线版)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # === 1. 加载 ===
    print("\n[1] 加载输入文件...")
    baseline = pd.read_csv(DATA_DIR / 'cohort_baseline.csv')
    comorb = pd.read_csv(DATA_DIR / 'cohort_comorbidities.csv')
    outcomes = pd.read_csv(DATA_DIR / 'cohort_outcomes.csv')
    surgery = pd.read_csv(DATA_DIR / 'surgery_procedures.csv', dtype={'icd_code': str})

    print(f"  baseline:    {len(baseline):,} rows × {len(baseline.columns)} cols")
    print(f"  comorbidities: {len(comorb):,} rows × {len(comorb.columns)} cols")
    print(f"  outcomes:    {len(outcomes):,} rows × {len(outcomes.columns)} cols")
    print(f"  surgery:     {len(surgery):,} rows × {len(surgery.columns)} cols")

    # === 2. 手术类型特征 ===
    print("\n[2] 构建手术类型特征...")
    surg_feat = build_surgery_features(surgery)
    print(f"  手术类型维度: {len(surg_feat.columns)-1}")  # -1 for hadm_id
    for col in surg_feat.columns:
        if col == 'hadm_id':
            continue
        n = surg_feat[col].sum()
        pct = n / len(surg_feat) * 100
        print(f"    {col:25s}: {n:>8,} ({pct:5.1f}%)")

    # 检查多标签
    surg_cols = [c for c in surg_feat.columns if c != 'hadm_id']
    multi_label = (surg_feat[surg_cols].sum(axis=1) > 1).sum()
    no_label = (surg_feat[surg_cols].sum(axis=1) == 0).sum()
    print(f"  多标签 (>1类): {multi_label:,}")
    print(f"  无标签 (Unknown): {no_label:,}")

    # === 3. 合并 ===
    print("\n[3] 合并特征矩阵...")
    n_start = len(baseline)

    # 基线人口学 (精简列)
    base_cols = ['hadm_id', 'age', 'gender', 'emergency']
    merged = baseline[base_cols].copy()
    # 年龄 >89 在MIMIC中被截断为91, 检测并标记
    merged['age_gt89'] = (merged['age'] == 91).astype(int)
    print(f"  age>89 截断标记: {merged['age_gt89'].sum():,} ({merged['age_gt89'].mean()*100:.1f}%)")

    # 合并症: 去掉 subject_id 避免重复
    comorb_for_merge = comorb.drop(columns=['subject_id'], errors='ignore')
    merged = merged.merge(comorb_for_merge, on='hadm_id', how='left')
    assert len(merged) == n_start, f"合并症merge行数变化: {n_start} -> {len(merged)}"

    # 手术类型
    merged = merged.merge(surg_feat, on='hadm_id', how='left')
    assert len(merged) == n_start, f"手术类型merge行数变化: {n_start} -> {len(merged)}"

    # 结局
    out_cols_to_merge = [c for c in outcomes.columns if c not in merged.columns or c == 'hadm_id']
    merged = merged.merge(outcomes[out_cols_to_merge], on='hadm_id', how='left')
    assert len(merged) == n_start, f"结局merge行数变化: {n_start} -> {len(merged)}"

    print(f"  合并后: {len(merged):,} rows × {len(merged.columns)} cols")

    # === 4. 特征标准化 ===
    print("\n[4] 特征处理...")

    # 年龄 z-score
    age_mask = merged['age'] < 89  # >89截断的不参与标准化
    age_mean = merged.loc[age_mask, 'age'].mean()
    age_std = merged.loc[age_mask, 'age'].std()
    merged['age_std'] = (merged['age'] - age_mean) / age_std
    # 对 age>89 的，使用91作为原始值标准化 (保守估计)
    # 实际上 age<89 才是常规值
    print(f"  age: mean={age_mean:.1f}, std={age_std:.2f}, z-score range=[{merged['age_std'].min():.2f}, {merged['age_std'].max():.2f}]")

    # 性别: F->0, M->1
    merged['gender_male'] = (merged['gender'] == 'M').astype(int)

    # 填充手术类型缺失为0
    for c in surg_cols:
        merged[c] = merged[c].fillna(0).astype(int)

    # === 5. 特征列整理 ===
    print("\n[5] 整理输出列...")

    feature_cols = [
        'hadm_id',
        # 人口学
        'age', 'age_std', 'age_gt89', 'gender_male', 'emergency',
        # Charlson 17维 + 总分
        'charlson_mi', 'charlson_chf', 'charlson_pvd', 'charlson_cvd',
        'charlson_dementia', 'charlson_copd', 'charlson_rheumatic',
        'charlson_pud', 'charlson_mld', 'charlson_dm', 'charlson_dmcx',
        'charlson_hp', 'charlson_rend', 'charlson_canc', 'charlson_msld',
        'charlson_mets', 'charlson_aids', 'charlson_score',
        # 手术类型
    ] + surg_cols + [
        # 结局
        'died_30d', 'unplanned_icu', 'aki',
        'comp_cardiac', 'comp_respiratory', 'comp_sepsis',
        'comp_bleeding', 'comp_vte', 'comp_any',
    ]

    # 确保所有列存在
    for col in feature_cols:
        if col not in merged.columns:
            print(f"  WARNING: {col} not found, filling with 0")
            merged[col] = 0

    features = merged[feature_cols].copy()
    print(f"  输出: {len(features):,} rows × {len(features.columns)} cols")

    # === 6. 保存 ===
    print("\n[6] 保存...")
    fpath = OUT_DIR / 'features_baseline.csv'
    features.to_csv(fpath, index=False)
    sz = os.path.getsize(fpath) / 1024**2
    print(f"  features_baseline.csv: {len(features):,} rows, {sz:.1f} MB")

    # === 7. 报告 ===
    print("\n[7] 生成特征摘要...")
    lines = []
    L = lines.append
    L("=" * 70)
    L("MIMIC-IV 特征工程摘要 (基线版)")
    L(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L("=" * 70)

    L(f"\n【样本量】")
    L(f"  训练样本: {len(features):,}")
    L(f"  特征维度: {len(features.columns) - 5}")  # 减去 hadm_id + 4个结局

    L(f"\n【人口学】")
    L(f"  年龄: median={features['age'].median():.0f}, mean={features['age'].mean():.1f}, "
      f"std={features['age'].std():.1f}")
    L(f"  男性: {features['gender_male'].sum():,} ({features['gender_male'].mean()*100:.1f}%)")
    L(f"  急诊/紧急: {features['emergency'].sum():,} ({features['emergency'].mean()*100:.1f}%)")
    L(f"  age>89截断: {features['age_gt89'].sum():,} ({features['age_gt89'].mean()*100:.1f}%)")

    L(f"\n【Charlson合并症总分】")
    L(f"  mean={features['charlson_score'].mean():.2f}, median={features['charlson_score'].median():.0f}")
    for score in range(0, 8):
        if score <= 6:
            n_s = (features['charlson_score'] == score).sum()
            L(f"  {score}分: {n_s:>8,} ({n_s/len(features)*100:5.1f}%)")
        else:
            n_s = (features['charlson_score'] >= 7).sum()
            L(f"  >=7分: {n_s:>8,} ({n_s/len(features)*100:5.1f}%)")
            break

    # 各合并症发生率
    charlson_items = [c for c in features.columns if c.startswith('charlson_') and c != 'charlson_score']
    L(f"\n【合并症明细 — 发生率 (Charlson 17维)】")
    # 用可读名称
    readable = {
        'charlson_mi': '心肌梗死', 'charlson_chf': '心衰', 'charlson_pvd': '外周血管病',
        'charlson_cvd': '脑血管病', 'charlson_dementia': '痴呆', 'charlson_copd': 'COPD',
        'charlson_rheumatic': '结缔组织病', 'charlson_pud': '溃疡病', 'charlson_mld': '轻中度肝病',
        'charlson_dm': '糖尿病', 'charlson_dmcx': '糖尿病并发症', 'charlson_hp': '偏瘫',
        'charlson_rend': '肾病', 'charlson_canc': '实体瘤', 'charlson_msld': '重度肝病',
        'charlson_mets': '转移癌', 'charlson_aids': 'AIDS',
    }
    for col in charlson_items:
        n = features[col].sum()
        label = readable.get(col, col)
        L(f"  {label:12s}: {n:>8,} ({n/len(features)*100:5.1f}%)")

    L(f"\n【手术类型分布】")
    for col in surg_cols:
        n = features[col].sum()
        label = col.replace('surg_', '')
        L(f"  {label:15s}: {n:>8,} ({n/len(features)*100:5.1f}%)")

    L(f"\n【结局】")
    L(f"  30天死亡:    {features['died_30d'].sum():>8,} ({features['died_30d'].mean()*100:.2f}%)")
    L(f"  非计划ICU:   {features['unplanned_icu'].sum():>8,} ({features['unplanned_icu'].mean()*100:.2f}%)")
    L(f"  AKI:         {features['aki'].sum():>8,} ({features['aki'].mean()*100:.2f}%)")
    L(f"  心脏并发症:  {features['comp_cardiac'].sum():>8,} ({features['comp_cardiac'].mean()*100:.2f}%)")
    L(f"  呼吸并发症:  {features['comp_respiratory'].sum():>8,} ({features['comp_respiratory'].mean()*100:.2f}%)")
    L(f"  脓毒症:      {features['comp_sepsis'].sum():>8,} ({features['comp_sepsis'].mean()*100:.2f}%)")
    L(f"  出血:        {features['comp_bleeding'].sum():>8,} ({features['comp_bleeding'].mean()*100:.2f}%)")
    L(f"  VTE:         {features['comp_vte'].sum():>8,} ({features['comp_vte'].mean()*100:.2f}%)")
    L(f"  任何并发症:  {features['comp_any'].sum():>8,} ({features['comp_any'].mean()*100:.2f}%)")

    L(f"\n【按Charlson分层 — 30天死亡率】")
    for score in range(0, 7):
        subset = features[features['charlson_score'] == score]
        if len(subset) > 0:
            dr = subset['died_30d'].mean() * 100
            L(f"  Charlson {score}分: {len(subset):,}人, 30d死亡 {dr:.2f}%")
    subset = features[features['charlson_score'] >= 7]
    if len(subset) > 0:
        dr = subset['died_30d'].mean() * 100
        L(f"  Charlson >=7分: {len(subset):,}人, 30d死亡 {dr:.2f}%")

    L(f"\n【按手术类型分层 — 30天死亡率】")
    for col in surg_cols:
        subset = features[features[col] == 1]
        if len(subset) > 0:
            dr = subset['died_30d'].mean() * 100
            L(f"  {col.replace('surg_',''):15s}: {len(subset):>7,}人, 30d死亡 {dr:.2f}%")

    L(f"\n【特征缺失率】")
    for col in features.columns:
        missing = features[col].isna().sum()
        if missing > 0:
            L(f"  {col}: {missing} missing ({missing/len(features)*100:.1f}%)")
    L(f"  (未列出列无缺失)")

    L(f"\n【数据限制】")
    L(f"  1. 无化验数据 — MIMIC主模块labevents待下载")
    L(f"  2. 手术分类基于ICD编码前缀, 部分手术可能分类错误或归入Unknown")
    L(f"  3. age>89 在MIMIC中被截断为91, 标记为 age_gt89=1")

    L("\n" + "=" * 70)

    report_path = OUT_DIR / 'feature_summary.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))

    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed:.1f} 秒")
    return features


if __name__ == '__main__':
    features = main()
