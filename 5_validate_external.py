"""
Dual-center external validation — MIMIC (train) → eICU (validate) + PIC (validate)
Evaluates model transportability across databases and patient populations.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import xgboost as xgb

DATA_DIR = Path(r"Z:\本地数据\asa_data")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def ordinal_risk_score(y_true, y_pred_proba, n_bins=6):
    """6-level risk stratification with monotonicity check"""
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


def compute_metrics(name, y_true, y_pred_proba):
    """Comprehensive metrics"""
    auroc = roc_auc_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    strata, thresh, rates, mono = ordinal_risk_score(y_true, y_pred_proba)

    # Calibration: predicted vs observed by decile
    decile_edges = np.quantile(y_pred_proba, np.linspace(0, 1, 11))
    calib_rows = []
    for i in range(10):
        mask = (y_pred_proba >= decile_edges[i]) & (y_pred_proba < decile_edges[i+1])
        if mask.sum() >= 10:
            calib_rows.append({
                'decile': i + 1,
                'n': mask.sum(),
                'pred_mean': y_pred_proba[mask].mean(),
                'obs_rate': y_true[mask].mean(),
            })

    lines = []
    L = lines.append
    L(f"\n{'='*65}")
    L(f"  {name}")
    L(f"  AUROC: {auroc:.4f}  |  Brier: {brier:.4f}  |  Monotonic: {'OK' if mono else 'FAIL'}")
    L(f"  {'─'*65}")
    L(f"  {'Level':<8} {'N':>8} {'Died':>8} {'Rate':>10} {'OR_vs_L1':>10}")
    l1_rate = rates[0] if rates else 0
    for level in sorted(set(strata)):
        mask = strata == level
        n = mask.sum()
        n_d = y_true[mask].sum()
        rate = n_d / n * 100 if n > 0 else 0
        if l1_rate > 0 and level > 1:
            or_vs_l1 = (rate / (100 - rate)) / (l1_rate / (100 - l1_rate))
            or_str = f"{or_vs_l1:.2f}"
        elif level == 1:
            or_str = "ref"
        else:
            or_str = "-"
        L(f"  Level {level:<3} {n:>8,} {n_d:>8,} {rate:>9.2f}% {or_str:>10}")

    if len(rates) >= 2 and rates[0] > 0:
        ratio = rates[-1] / rates[0]
        L(f"\n  Risk ratio (L6/L1): {ratio:.1f}x")

    # Calibration summary
    if calib_rows:
        calib_df = pd.DataFrame(calib_rows)
        calib_error = np.mean(np.abs(calib_df['pred_mean'] - calib_df['obs_rate']))
        L(f"  Calibration error (MAE): {calib_error:.4f}")

    return auroc, brier, '\n'.join(lines)


def compute_nri_continuous(y_true, yp_old, yp_new):
    """Continuous NRI"""
    n_event = y_true.sum()
    n_nonevent = (1 - y_true).sum()
    if n_event == 0 or n_nonevent == 0:
        return 0, 0
    d_up = (yp_new[y_true == 1] > yp_old[y_true == 1]).sum()
    d_down = (yp_new[y_true == 1] < yp_old[y_true == 1]).sum()
    a_up = (yp_new[y_true == 0] < yp_old[y_true == 0]).sum()
    a_down = (yp_new[y_true == 0] > yp_old[y_true == 0]).sum()
    return (d_up - d_down) / n_event, (a_up - a_down) / n_nonevent


def main():
    t_start = time.time()
    print("=" * 65)
    print("Dual-Center External Validation")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # === 1. Load data ===
    print("\n[1] Loading feature matrices...")
    mimic = pd.read_csv(DATA_DIR / 'features_full.csv')
    eicu = pd.read_csv(DATA_DIR / 'features_eicu.csv')
    pic = pd.read_csv(DATA_DIR / 'features_pic.csv')

    print(f"  MIMIC: {len(mimic):,} x {len(mimic.columns)}")
    print(f"  eICU:  {len(eicu):,} x {len(eicu.columns)}")
    print(f"  PIC:   {len(pic):,} x {len(pic.columns)}")

    # === 2. Define feature sets ===
    print("\n[2] Defining feature sets...")
    demo_cols = ['age_std', 'age_gt89', 'gender_male', 'emergency']
    charlson_indiv = [c for c in mimic.columns if c.startswith('charlson_') and c != 'charlson_score']
    charlson_total = ['charlson_score']
    surg_cols = [c for c in mimic.columns if c.startswith('surg_')]
    lab_std_cols = [c for c in mimic.columns if c.endswith('_std') and c != 'age_std']
    lab_miss_cols = [c for c in mimic.columns if c.endswith('_missing')]
    derived_cols = ['egfr', 'egfr_stage', 'anemia', 'bun_cr_ratio',
                    'hypoalbuminemia', 'leukocytosis', 'leukopenia',
                    'thrombocytopenia', 'coagulopathy', 'hyperglycemia',
                    'hyponatremia', 'hypernatremia', 'hypokalemia', 'hyperkalemia',
                    'acidosis', 'lactate_elevated', 'liver_injury', 'hyperbilirubinemia']
    derived_cols = [c for c in derived_cols if c in mimic.columns]

    baseline_features = demo_cols + charlson_indiv + charlson_total + surg_cols
    full_features = baseline_features + lab_std_cols + lab_miss_cols + derived_cols

    # Ensure all features exist
    baseline_features = [c for c in baseline_features if c in mimic.columns]
    full_features = [c for c in full_features if c in mimic.columns]

    # Dedup
    full_features = list(dict.fromkeys(full_features))
    baseline_features = list(dict.fromkeys(baseline_features))

    print(f"  Baseline: {len(baseline_features)} features (no labs)")
    print(f"  Full:     {len(full_features)} features")

    # === 3. Train on MIMIC (80/20 split), Validate on eICU + PIC ===
    print("\n[3] Training on MIMIC, external validation on eICU + PIC...")

    # Standardize MIMIC features
    X_mimic_full = mimic[full_features].values
    X_mimic_base = mimic[baseline_features].values
    y_mimic = mimic['died_30d'].values

    scaler_full = StandardScaler()
    scaler_base = StandardScaler()
    X_mimic_full_s = scaler_full.fit_transform(X_mimic_full)
    X_mimic_base_s = scaler_base.fit_transform(X_mimic_base)

    # Train full model
    model_full = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    model_full.fit(X_mimic_full_s, y_mimic)

    model_base = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    model_base.fit(X_mimic_base_s, y_mimic)

    # XGBoost
    scale_pos_weight = (1 - y_mimic.mean()) / y_mimic.mean()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED, eval_metric='aucpr',
    )
    xgb_clf.fit(X_mimic_full_s, y_mimic)

    # Stratified split for MIMIC internal test
    idx = np.arange(len(mimic))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y_mimic
    )

    # Refit on training split only (for unbiased internal eval)
    sc_f_int = StandardScaler()
    sc_b_int = StandardScaler()
    X_tr_full_s = sc_f_int.fit_transform(X_mimic_full[tr_idx])
    X_te_full_s = sc_f_int.transform(X_mimic_full[te_idx])
    X_tr_base_s = sc_b_int.fit_transform(X_mimic_base[tr_idx])
    X_te_base_s = sc_b_int.transform(X_mimic_base[te_idx])

    m_full_int = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    m_full_int.fit(X_tr_full_s, y_mimic[tr_idx])
    m_base_int = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    m_base_int.fit(X_tr_base_s, y_mimic[tr_idx])

    yp_mimic_test_full = m_full_int.predict_proba(X_te_full_s)[:, 1]
    yp_mimic_test_base = m_base_int.predict_proba(X_te_base_s)[:, 1]

    # === 4. External validation ===
    print("\n[4] External validation predictions...")

    all_reports = []

    datasets = [
        ('MIMIC (Internal Test)', y_mimic[te_idx], yp_mimic_test_full, yp_mimic_test_base),
        ('eICU (External)', eicu['died_30d'].values,
         model_full.predict_proba(scaler_full.transform(eicu[full_features].values))[:, 1],
         model_base.predict_proba(scaler_base.transform(eicu[baseline_features].values))[:, 1]),
        ('PIC (External)', pic['died_30d'].values,
         model_full.predict_proba(scaler_full.transform(pic[full_features].values))[:, 1],
         model_base.predict_proba(scaler_base.transform(pic[baseline_features].values))[:, 1]),
    ]

    results = []
    report_lines = []
    RL = report_lines.append

    for name, y_true, yp_full, yp_base in datasets:
        print(f"\n  --- {name} ---")
        auc_f, brier_f, report_f = compute_metrics(f"Full LogReg — {name}", y_true, yp_full)
        print(report_f)

        auc_b, brier_b, report_b = compute_metrics(f"Baseline — {name}", y_true, yp_base)
        print(report_b)

        nri_e, nri_ne = compute_nri_continuous(y_true, yp_base, yp_full)
        results.append({
            'dataset': name,
            'n': len(y_true),
            'n_deaths': y_true.sum(),
            'mortality_rate': y_true.mean(),
            'auc_base': auc_b,
            'auc_full': auc_f,
            'auc_gain': auc_f - auc_b,
            'brier_base': brier_b,
            'brier_full': brier_f,
            'nri_events': nri_e,
            'nri_nonevents': nri_ne,
            'nri_total': nri_e + nri_ne,
        })

    # === 5. XGBoost external validation ===
    print("\n[5] XGBoost external validation...")
    for name, df in [('eICU', eicu), ('PIC', pic)]:
        X_ext = scaler_full.transform(df[full_features].values)
        y_ext = df['died_30d'].values
        yp_xgb = xgb_clf.predict_proba(X_ext)[:, 1]
        auc_xgb = roc_auc_score(y_ext, yp_xgb)
        brier_xgb = brier_score_loss(y_ext, yp_xgb)
        print(f"  XGBoost — {name}: AUROC={auc_xgb:.4f}, Brier={brier_xgb:.4f}")
        # Add to results
        for r in results:
            if r['dataset'] == f'{name} (External)':
                r['auc_xgb'] = auc_xgb
                r['brier_xgb'] = brier_xgb

    # === 6. Report ===
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    RL("=" * 70)
    RL("Dual-Center External Validation Report")
    RL(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    RL("=" * 70)
    RL("")
    RL("Model: Logistic Regression with L2 regularization")
    RL("Training: MIMIC-IV 3.0 (n=287,504 surgeries)")
    RL("External validation: eICU (n=200,764 ICU stays), PIC (n=6,356 pediatric surgeries)")
    RL("")

    RL(f"{'Dataset':<30s} {'N':>8s} {'Deaths':>8s} {'Rate':>8s} "
       f"{'Base AUC':>10s} {'Full AUC':>10s} {'ΔAUC':>8s} "
       f"{'Base Brier':>11s} {'Full Brier':>11s} {'NRI total':>10s}")
    RL("-" * 115)

    for r in results:
        RL(f"{r['dataset']:<30s} {r['n']:>8,} {r['n_deaths']:>8,} "
           f"{r['mortality_rate']:>7.2%} "
           f"{r['auc_base']:>10.4f} {r['auc_full']:>10.4f} {r['auc_gain']:>+8.4f} "
           f"{r['brier_base']:>11.4f} {r['brier_full']:>11.4f} {r['nri_total']:>+10.4f}")

    RL("")
    RL("-" * 115)
    RL("")

    # Key findings
    eicu_res = [r for r in results if 'eICU' in r['dataset']][0]
    pic_res = [r for r in results if 'PIC' in r['dataset']][0]
    mimic_res = [r for r in results if 'MIMIC' in r['dataset']][0]

    RL("Key Findings:")
    RL(f"  1. External transportability (eICU):")
    RL(f"     - Full model AUROC: {eicu_res['auc_full']:.4f}")
    RL(f"     - Δ vs MIMIC internal: {eicu_res['auc_full'] - mimic_res['auc_full']:+.4f}")
    RL(f"     - Lab value NRI: {eicu_res['nri_total']:+.4f}")
    RL(f"  2. Pediatric generalizability (PIC):")
    RL(f"     - Full model AUROC: {pic_res['auc_full']:.4f}")
    RL(f"     - Applicable to children: {'Yes' if pic_res['auc_full'] > 0.70 else 'Limited'}")
    RL(f"  3. Lab data contribution:")
    RL(f"     - MIMIC ΔAUROC: {mimic_res['auc_gain']:+.4f}")
    RL(f"     - eICU ΔAUROC: {eicu_res['auc_gain']:+.4f}")
    RL(f"     - PIC ΔAUROC: {pic_res['auc_gain']:+.4f}")

    RL("")
    RL("=" * 70)

    report = '\n'.join(report_lines)
    print(report)

    report_path = DATA_DIR / 'external_validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved: external_validation_report.txt")

    # === 7. Save predictions ===
    print("\n[7] Saving predictions...")
    for name, df in [('eICU', eicu), ('PIC', pic)]:
        X_ext = scaler_full.transform(df[full_features].values)
        yp = model_full.predict_proba(X_ext)[:, 1]
        yp_base = model_base.predict_proba(scaler_base.transform(df[baseline_features].values))[:, 1]
        strata, thresh, rates, mono = ordinal_risk_score(df['died_30d'].values, yp)

        preds = df[['hadm_id']].copy()
        preds['pred_baseline'] = yp_base
        preds['pred_full'] = yp
        preds['risk_stratum'] = strata
        preds['died_30d'] = df['died_30d'].values
        preds.to_csv(DATA_DIR / f'predictions_{name}.csv', index=False)
        print(f"  predictions_{name}.csv: {len(preds):,} rows")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
    return results


if __name__ == '__main__':
    results = main()
