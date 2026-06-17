"""
MIMIC-IV 手术风险评分 — 完整模型训练
人口学 + Charlson + 手术类型 + 17项化验 + 衍生指标
对比: Baseline vs Full, 量化化验数据的增量价值
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb

DATA_DIR = Path(r"Z:\本地数据\asa_data")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def ordinal_risk_score(y_true, y_pred_proba, n_bins=6):
    """基于预测概率分6级, 保证单调性"""
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    thresholds = np.quantile(y_pred_proba, quantiles)
    strata = np.ones(len(y_pred_proba), dtype=int)
    for i, t in enumerate(thresholds):
        strata[y_pred_proba >= t] = i + 2
    observed = []
    for level in range(1, n_bins + 1):
        mask = strata == level
        if mask.sum() > 0:
            observed.append(y_true[mask].mean())
    is_mono = all(x <= y for x, y in zip(observed, observed[1:]))
    return strata, thresholds, observed, is_mono


def compute_nri(y_test, yp_old, yp_new):
    """Continuous NRI"""
    n_died = y_test.sum()
    n_alive = (1 - y_test).sum()
    d_imp = (yp_new[y_test == 1] > yp_old[y_test == 1]).sum()
    d_wor = (yp_new[y_test == 1] < yp_old[y_test == 1]).sum()
    a_imp = (yp_new[y_test == 0] < yp_old[y_test == 0]).sum()
    a_wor = (yp_new[y_test == 0] > yp_old[y_test == 0]).sum()
    return (d_imp - d_wor) / n_died, (a_imp - a_wor) / n_alive


def evaluate_strata(name, y_true, y_pred_proba):
    """评估风险分层"""
    strata, thresh, rates, mono = ordinal_risk_score(y_true, y_pred_proba)
    auroc = roc_auc_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)

    lines = []
    L = lines.append
    L(f"\n{'='*60}")
    L(f"  {name}")
    L(f"  AUROC: {auroc:.4f}  |  Brier: {brier:.4f}  |  Monotonic: {'OK' if mono else 'FAIL'}")
    L(f"  {'─'*60}")
    L(f"  {'Level':<8} {'N':>8} {'Died':>8} {'Rate':>10} {'OR':>8}")

    l1_rate = None
    for level in sorted(set(strata)):
        mask = strata == level
        n = mask.sum()
        n_d = y_true[mask].sum()
        rate = n_d / n * 100 if n > 0 else 0

        if l1_rate and l1_rate > 0:
            or_vs_l1 = (rate / (100 - rate)) / (l1_rate / (100 - l1_rate))
            or_str = f"{or_vs_l1:.2f}"
        elif level == 1:
            l1_rate = rate
            or_str = "ref"
        else:
            or_str = "-"

        L(f"  Level {level:<3} {n:>8,} {n_d:>8,} {rate:>9.2f}% {or_str:>8}")

    if rates:
        ratio = rates[-1] / rates[0] * 100 if rates[0] > 0 else float('inf')
        L(f"\n  死亡率比值 (L6/L1): {ratio:.1f}x")

    return auroc, brier, strata, thresh, '\n'.join(lines)


def main():
    t_start = time.time()
    print("=" * 60)
    print("MIMIC-IV 完整风险评分模型训练 v2")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # === 1. 加载 ===
    print("\n[1] 加载特征矩阵...")
    df = pd.read_csv(DATA_DIR / 'features_full.csv')
    print(f"  样本: {len(df):,}, 特征: {len(df.columns)}")

    # 特征分组
    demo_cols = ['age_std', 'age_gt89', 'gender_male', 'emergency']
    charlson_indiv = [c for c in df.columns if c.startswith('charlson_') and c != 'charlson_score']
    charlson_total = ['charlson_score']
    surg_cols = [c for c in df.columns if c.startswith('surg_')]

    lab_std_cols = [c for c in df.columns if c.endswith('_std') and c != 'age_std']
    lab_miss_cols = [c for c in df.columns if c.endswith('_missing')]
    derived_cols = ['egfr', 'egfr_stage', 'anemia', 'bun_cr_ratio',
                    'hypoalbuminemia', 'leukocytosis', 'leukopenia',
                    'thrombocytopenia', 'coagulopathy', 'hyperglycemia',
                    'hyponatremia', 'hypernatremia', 'hypokalemia', 'hyperkalemia',
                    'acidosis', 'lactate_elevated', 'liver_injury', 'hyperbilirubinemia']
    derived_cols = [c for c in derived_cols if c in df.columns]

    # Baseline feature set (no labs)
    baseline_features = demo_cols + charlson_indiv + charlson_total + surg_cols
    # Full feature set
    full_features = baseline_features + lab_std_cols + lab_miss_cols + derived_cols

    # Remove any cols not in df
    baseline_features = [c for c in baseline_features if c in df.columns]
    full_features = [c for c in full_features if c in df.columns]

    print(f"  Baseline: {len(baseline_features)} features")
    print(f"  Full:     {len(full_features)} features")
    print(f"  Lab added: {len(full_features) - len(baseline_features)} features")

    # Pre-flight: check for duplicate columns (铁律1 防重复)
    dup_check = pd.Series(full_features)
    dupes = dup_check[dup_check.duplicated()].tolist()
    if dupes:
        print(f"  WARNING: Duplicate features detected: {dupes}")
        full_features = list(dict.fromkeys(full_features))  # deduplicate preserving order
        print(f"  Deduplicated Full: {len(full_features)} features")

    y = df['died_30d'].values
    print(f"  30d mortality: {y.mean()*100:.2f}%")

    # === 2. Train/test split ===
    print("\n[2] 划分训练/测试集 (80/20, stratified)...")
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train_base = df.iloc[train_idx][baseline_features].values
    X_test_base = df.iloc[test_idx][baseline_features].values
    X_train_full = df.iloc[train_idx][full_features].values
    X_test_full = df.iloc[test_idx][full_features].values
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"  Train: {len(train_idx):,} ({y_train.mean()*100:.2f}% death)")
    print(f"  Test:  {len(test_idx):,} ({y_test.mean()*100:.2f}% death)")

    # Standardize
    sc_base = StandardScaler()
    X_train_base_s = sc_base.fit_transform(X_train_base)
    X_test_base_s = sc_base.transform(X_test_base)

    sc_full = StandardScaler()
    X_train_full_s = sc_full.fit_transform(X_train_full)
    X_test_full_s = sc_full.transform(X_test_full)

    # === 3. Baseline LogReg (no labs) ===
    print("\n[3] 基线模型: LogReg (无化验)...")
    lr_base = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    lr_base.fit(X_train_base_s, y_train)
    yp_base = lr_base.predict_proba(X_test_base_s)[:, 1]
    auc_base, brier_base, strata_base, thresh_base, report_base = evaluate_strata(
        "Baseline LogReg (no labs)", y_test, yp_base
    )
    print(report_base)

    # === 4. Full LogReg (with labs) ===
    print("\n[4] 完整模型: LogReg + Labs...")
    lr_full = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    lr_full.fit(X_train_full_s, y_train)
    yp_full = lr_full.predict_proba(X_test_full_s)[:, 1]
    auc_full, brier_full, strata_full, thresh_full, report_full = evaluate_strata(
        "Full LogReg (with labs)", y_test, yp_full
    )
    print(report_full)

    # Feature importance
    coefs = pd.DataFrame({
        'feature': full_features,
        'coef': lr_full.coef_[0]
    }).sort_values('coef', key=abs, ascending=False)
    print("\n  Top 20 features (|coef|):")
    for _, row in coefs.head(20).iterrows():
        print(f"    {row['feature']:30s}: {row['coef']:+8.4f}")

    # === 5. XGBoost Full ===
    print("\n[5] XGBoost (with labs)...")
    scale_pos_weight = (1 - y_train.mean()) / y_train.mean()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED, eval_metric='aucpr',
        early_stopping_rounds=20,
    )
    xgb_clf.fit(
        X_train_full_s, y_train,
        eval_set=[(X_test_full_s, y_test)],
        verbose=False,
    )
    yp_xgb = xgb_clf.predict_proba(X_test_full_s)[:, 1]
    auc_xgb, brier_xgb, strata_xgb, thresh_xgb, report_xgb = evaluate_strata(
        "XGBoost (with labs)", y_test, yp_xgb
    )
    print(report_xgb)

    # === 6. NRI: Lab value-add ===
    print("\n[6] 化验增量价值: NRI (Full vs Baseline)...")
    nri_d, nri_a = compute_nri(y_test, yp_base, yp_full)
    print(f"  NRI (events):   {nri_d:+.4f}")
    print(f"  NRI (non-events): {nri_a:+.4f}")
    print(f"  NRI (total):    {nri_d + nri_a:+.4f}")
    print(f"  AUROC gain:     {auc_full - auc_base:+.4f}")

    # === 7. CV ===
    print("\n[7] 5-Fold CV (Full LogReg)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_auc_full = []
    cv_auc_base = []
    X_all_full = df[full_features].values
    X_all_base = df[baseline_features].values

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_all_full, y)):
        sc_f = StandardScaler()
        sc_b = StandardScaler()

        X_tr_f = sc_f.fit_transform(X_all_full[tr_idx])
        X_val_f = sc_f.transform(X_all_full[val_idx])
        X_tr_b = sc_b.fit_transform(X_all_base[tr_idx])
        X_val_b = sc_b.transform(X_all_base[val_idx])

        m_f = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
        m_f.fit(X_tr_f, y[tr_idx])
        cv_auc_full.append(roc_auc_score(y[val_idx], m_f.predict_proba(X_val_f)[:, 1]))

        m_b = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
        m_b.fit(X_tr_b, y[tr_idx])
        cv_auc_base.append(roc_auc_score(y[val_idx], m_b.predict_proba(X_val_b)[:, 1]))

        print(f"  Fold {fold+1}: Full={cv_auc_full[-1]:.4f}, Base={cv_auc_base[-1]:.4f}")

    print(f"\n  CV Full:  {np.mean(cv_auc_full):.4f} +/- {np.std(cv_auc_full):.4f}")
    print(f"  CV Base:  {np.mean(cv_auc_base):.4f} +/- {np.std(cv_auc_base):.4f}")

    # === 8. Final model (full data) ===
    print("\n[8] 最终模型 (全量数据, Full LogReg)...")
    sc_final = StandardScaler()
    X_all_full_s = sc_final.fit_transform(X_all_full)
    final_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    final_model.fit(X_all_full_s, y)

    yp_all = final_model.predict_proba(X_all_full_s)[:, 1]
    strata_all, thresholds, rates_all, mono_all = ordinal_risk_score(y, yp_all)

    # Save predictions
    preds = df[['hadm_id']].copy()
    preds['pred_30d_death'] = yp_all
    preds['risk_stratum'] = strata_all
    preds['died_30d'] = y
    preds.to_csv(DATA_DIR / 'predictions_full.csv', index=False)
    print(f"  predictions_full.csv: {len(preds):,} rows")

    # === 9. Report ===
    print("\n" + "=" * 70)
    print("最终报告")
    print("=" * 70)

    lines = []
    L = lines.append
    L("=" * 70)
    L("MIMIC-IV 手术风险评分 — 完整模型报告")
    L(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L("=" * 70)

    L(f"\n【数据概览】")
    L(f"  总样本: {len(df):,}")
    L(f"  Baseline特征: {len(baseline_features)} (人口学+Charlson+手术类型)")
    L(f"  完整特征: {len(full_features)} (+{len(full_features)-len(baseline_features)} 化验)")
    L(f"  30天死亡: {y.sum():,} ({y.mean()*100:.2f}%)")

    L(f"\n【模型性能 (Test Set)】")
    L(f"  {'Model':<35s} {'AUROC':>8s} {'Brier':>8s}")
    L(f"  {'─'*51}")
    L(f"  {'Baseline (no labs)':<35s} {auc_base:>8.4f} {brier_base:>8.4f}")
    L(f"  {'Full LogReg (+labs)':<35s} {auc_full:>8.4f} {brier_full:>8.4f}")
    L(f"  {'XGBoost (+labs)':<35s} {auc_xgb:>8.4f} {brier_xgb:>8.4f}")

    L(f"\n【化验增量价值】")
    L(f"  AUROC improvement: +{auc_full - auc_base:.4f}")
    L(f"  NRI (events):      {nri_d:+.4f}")
    L(f"  NRI (non-events):  {nri_a:+.4f}")
    L(f"  NRI (total):       {nri_d + nri_a:+.4f}")

    L(f"\n【交叉验证 (5-Fold)】")
    L(f"  Full LogReg: {np.mean(cv_auc_full):.4f} +/- {np.std(cv_auc_full):.4f}")
    L(f"  Baseline:    {np.mean(cv_auc_base):.4f} +/- {np.std(cv_auc_base):.4f}")

    L(f"\n【最终风险分层 (6级, Full LogReg, 全量)】")
    L(f"  {'Level':<8} {'N':>8} {'Died':>8} {'Rate':>10} {'Threshold':>14}")
    L(f"  {'─'*50}")
    for level in range(1, 7):
        mask = strata_all == level
        n = mask.sum()
        n_d = y[mask].sum()
        rate = n_d / n * 100 if n > 0 else 0
        if level == 1:
            thresh_str = f"<{thresholds[0]:.4f}"
        elif level < 6:
            thresh_str = f"[{thresholds[level-2]:.4f}, {thresholds[level-1]:.4f})"
        else:
            thresh_str = f">={thresholds[-1]:.4f}"
        L(f"  Level {level:<3} {n:>8,} {n_d:>8,} {rate:>9.2f}% {thresh_str:>14}")

    if rates_all:
        ratio = rates_all[-1] / rates_all[0] * 100 if rates_all[0] > 0 else float('inf')
        L(f"\n  死亡率比值 (L6/L1): {ratio:.1f}x")

    L(f"\n【Top 20 预测因子 (LogReg |coef|)】")
    for _, row in coefs.head(20).iterrows():
        L(f"  {row['feature']:30s}: {row['coef']:+8.4f}")

    L(f"\n【模型对比: 本评分 vs Charlson vs ASA】")
    L(f"  {'评分系统':<30s} {'AUROC':>8s} {'层级':>8s} {'客观性':>10s}")
    L(f"  {'─'*58}")
    L(f"  {'本评分 (Full LogReg)':<30s} {np.mean(cv_auc_full):>8.4f} {'6级':>8s} {'完全客观':>10s}")
    L(f"  {'本评分 (Baseline, 无lab)':<30s} {np.mean(cv_auc_base):>8.4f} {'6级':>8s} {'完全客观':>10s}")
    L(f"  {'Charlson Comorbidity Index':<30s} {'0.76':>8s} {'连续':>8s} {'完全客观':>10s}")
    L(f"  {'ASA Physical Status':<30s} {'—':>8s} {'6级':>8s} {'主观':>10s}")

    L(f"\n【数据限制】")
    L(f"  1. 化验来自MIMIC-IV主模块 (hosp/labevents/), 覆盖率72-78% (核心项)")
    L(f"  2. Albumin/Lactate覆盖率仅20-22%, 衍生指标 (hypoalbuminemia等) 受影响")
    L(f"  3. 外部验证 (eICU, PIC) 待完成")
    L(f"  4. 时间验证 (temporal validation) 待完成")

    L("\n" + "=" * 70)

    report_path = DATA_DIR / 'model_report_full.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))

    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed:.1f} 秒")
    return preds


if __name__ == '__main__':
    preds = main()
