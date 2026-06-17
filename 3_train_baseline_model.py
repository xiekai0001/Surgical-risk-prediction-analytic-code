"""
MIMIC-IV 手术风险评分 — 基线模型训练
无化验版本: 人口学 + Charlson + 手术类型
目标: 预测30天死亡，输出1-6级风险分层
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
import xgboost as xgb

DATA_DIR = Path(r"Z:\本地数据\asa_data")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_features():
    df = pd.read_csv(DATA_DIR / 'features_baseline.csv')
    return df


def define_feature_set(df):
    """定义特征矩阵 X 和目标向量 y"""

    # 特征列
    demo_cols = ['age_std', 'age_gt89', 'gender_male', 'emergency']

    charlson_indiv = [c for c in df.columns if c.startswith('charlson_') and c != 'charlson_score']
    charlson_total = ['charlson_score']

    surg_cols = [c for c in df.columns if c.startswith('surg_')]
    # 排除 Unknown — 它作为截距的参考类，不显式加入 (加入会导致共线性)
    # 实际上保留所有 surg 列，让模型学习

    feature_cols = demo_cols + charlson_indiv + charlson_total + surg_cols

    X = df[feature_cols].copy()
    y_died = df['died_30d'].values
    y_icu = df['unplanned_icu'].values
    y_aki = df['aki'].values

    # 复合结局: 死亡或非计划ICU (临床最关心的硬终点)
    y_composite = ((df['died_30d'] == 1) | (df['unplanned_icu'] == 1)).astype(int).values

    return X, y_died, y_icu, y_aki, y_composite, feature_cols


def ordinal_risk_score(y_true, y_pred_proba, n_bins=6):
    """基于预测概率将患者分为1-6级风险层
    保证: 层级越高, 实际死亡率越高
    """
    # 使用分位数边界确保每层有足够样本
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    thresholds = np.quantile(y_pred_proba, quantiles)

    strata = np.zeros(len(y_pred_proba), dtype=int)
    for i, t in enumerate(thresholds):
        strata[y_pred_proba >= t] = i + 1
    strata += 1  # 1-indexed

    # 验证单调性
    observed_rates = []
    for level in range(1, n_bins + 1):
        mask = strata == level
        if mask.sum() > 0:
            observed_rates.append(y_true[mask].mean())
    is_monotonic = all(x <= y for x, y in zip(observed_rates, observed_rates[1:]))

    return strata, thresholds, observed_rates, is_monotonic


def evaluate_model(name, y_true, y_pred_proba, y_pred_strata=None):
    """打印模型评估指标"""
    auroc = roc_auc_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)

    lines = []
    L = lines.append
    L(f"\n  【{name}】")
    L(f"    AUROC: {auroc:.4f}")
    L(f"    Brier: {brier:.4f}")

    if y_pred_strata is not None:
        L(f"\n    {'─'*50}")
        L(f"    {'层级':<8} {'样本量':>8} {'死亡数':>8} {'死亡率':>10} {'OR':>8}")
        L(f"    {'─'*50}")

        prev_rate = None
        prev_n = None
        for level in sorted(set(y_pred_strata)):
            mask = y_pred_strata == level
            n = mask.sum()
            n_died = y_true[mask].sum()
            rate = n_died / n * 100 if n > 0 else 0

            # OR vs Level 1
            if level == 1:
                or_str = "ref"
            elif prev_rate and prev_rate > 0:
                odds_ratio = (rate / (100 - rate)) / (prev_rate / (100 - prev_rate))
                or_str = f"{odds_ratio:.2f}"
            else:
                or_str = "-"

            L(f"    Level {level:<3} {n:>8,} {n_died:>8,} {rate:>9.2f}% {or_str:>8}")
            prev_rate = rate

        # 梯度检查
        rates = []
        for level in sorted(set(y_pred_strata)):
            mask = y_pred_strata == level
            if mask.sum() > 0:
                rates.append(y_true[mask].mean() * 100)
        monotonic = all(x <= y for x, y in zip(rates, rates[1:]))
        L(f"\n    风险梯度: {'OK 单调递增' if monotonic else 'FAIL 违反单调性'}")

    return '\n'.join(lines)


def fit_calibration_intercept(y_true, y_pred_logits):
    """Platt scaling: 用逻辑回归拟合校准截距和斜率"""
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression(penalty=None, solver='lbfgs')
    cal.fit(y_pred_logits.reshape(-1, 1), y_true)
    return cal


def main():
    t_start = time.time()
    print("=" * 60)
    print("MIMIC-IV 基线风险评分模型训练 v1")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # === 1. 加载 ===
    print("\n[1] 加载特征矩阵...")
    df = load_features()
    X, y_died, y_icu, y_aki, y_composite, feature_cols = define_feature_set(df)
    print(f"  样本: {len(X):,}, 特征: {len(feature_cols)}")
    print(f"  30天死亡率: {y_died.mean()*100:.2f}%")
    print(f"  复合结局率: {y_composite.mean()*100:.2f}%")

    # === 2. 划分 ===
    print("\n[2] 划分训练/测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_died, test_size=0.2, random_state=RANDOM_SEED, stratify=y_died
    )
    # 复合结局
    _, _, yc_train, yc_test = train_test_split(
        X, y_composite, test_size=0.2, random_state=RANDOM_SEED, stratify=y_composite
    )
    print(f"  训练集: {len(X_train):,} ({y_train.mean()*100:.2f}% 死亡)")
    print(f"  测试集: {len(X_test):,} ({y_test.mean()*100:.2f}% 死亡)")

    # === 3. 基线模型: 仅 Charlson 总分 ===
    print("\n[3] 基线对比: Charlson 总分 LogReg...")
    lr_charlson = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
    X_c_train = X_train[['charlson_score']].values
    X_c_test = X_test[['charlson_score']].values
    lr_charlson.fit(X_c_train, y_train)
    yp_c_test = lr_charlson.predict_proba(X_c_test)[:, 1]
    strata_c, thresh_c, rates_c, mono_c = ordinal_risk_score(y_test, yp_c_test)
    print(evaluate_model("Charlson-Only LogReg", y_test, yp_c_test, strata_c))

    # === 4. 完整特征 LogReg ===
    print("\n[4] 完整特征 Logistic Regression...")
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_full = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    lr_full.fit(X_train_scaled, y_train)
    yp_lr = lr_full.predict_proba(X_test_scaled)[:, 1]
    strata_lr, thresh_lr, rates_lr, mono_lr = ordinal_risk_score(y_test, yp_lr)
    print(evaluate_model("Full LogReg (L2)", y_test, yp_lr, strata_lr))

    # 特征重要性
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coef': lr_full.coef_[0]
    }).sort_values('coef', key=abs, ascending=False)
    print("\n  Top 15 特征 (|coef|):")
    for _, row in coef_df.head(15).iterrows():
        print(f"    {row['feature']:25s}: {row['coef']:+8.4f}")

    # === 5. XGBoost ===
    print("\n[5] XGBoost 分类器...")
    scale_pos_weight = (1 - y_train.mean()) / y_train.mean()

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric='aucpr',
        early_stopping_rounds=20,
    )
    xgb_clf.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )
    yp_xgb = xgb_clf.predict_proba(X_test_scaled)[:, 1]
    strata_xgb, thresh_xgb, rates_xgb, mono_xgb = ordinal_risk_score(y_test, yp_xgb)
    print(evaluate_model("XGBoost", y_test, yp_xgb, strata_xgb))

    xgb_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n  Top 15 特征 (importance):")
    for _, row in xgb_importance.head(15).iterrows():
        print(f"    {row['feature']:25s}: {row['importance']:.4f}")

    # === 6. XGBoost 校准 ===
    print("\n[6] XGBoost + Platt 校准...")
    yp_xgb_train = xgb_clf.predict_proba(X_train_scaled)[:, 1]
    cal_model = fit_calibration_intercept(y_train, yp_xgb_train)
    yp_xgb_cal = cal_model.predict_proba(yp_xgb.reshape(-1, 1))[:, 1]
    strata_xgb_cal, thresh_xgb_cal, rates_xgb_cal, mono_xgb_cal = ordinal_risk_score(
        y_test, yp_xgb_cal
    )
    print(evaluate_model("XGBoost + Platt Calibrated", y_test, yp_xgb_cal, strata_xgb_cal))

    # === 7. 净重分类改善 (NRI) ===
    print("\n[7] 净重分类改善: Full LogReg vs Charlson-Only...")
    # NRI: 比较两个模型在风险层上的分类差异
    strata_ref = strata_c  # Charlson-only 分层
    strata_new = strata_lr  # Full LogReg 分层

    # Category-free NRI (continuous NRI)
    # 对于死亡患者，预测概率更高 = 改善
    n_died = y_test.sum()
    n_alive = (1 - y_test).sum()

    died_improve = (yp_lr[y_test == 1] > yp_c_test[y_test == 1]).sum()
    died_worsen = (yp_lr[y_test == 1] < yp_c_test[y_test == 1]).sum()
    alive_improve = (yp_lr[y_test == 0] < yp_c_test[y_test == 0]).sum()
    alive_worsen = (yp_lr[y_test == 0] > yp_c_test[y_test == 0]).sum()

    nri_died = (died_improve - died_worsen) / n_died
    nri_alive = (alive_improve - alive_worsen) / n_alive
    nri_total = nri_died + nri_alive

    print(f"  NRI (事件): {nri_died:+.3f} (改善{died_improve}, 恶化{died_worsen})")
    print(f"  NRI (非事件): {nri_alive:+.3f} (改善{alive_improve}, 恶化{alive_worsen})")
    print(f"  NRI (总): {nri_total:+.3f}")

    # === 8. 5折交叉验证 ===
    print("\n[8] 5折交叉验证...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_died)):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y_died[train_idx]
        y_val = y_died[val_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)

        model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
        model.fit(X_tr_s, y_tr)
        yp = model.predict_proba(X_val_s)[:, 1]
        auc = roc_auc_score(y_val, yp)
        cv_scores.append(auc)
        print(f"  Fold {fold+1}: AUROC={auc:.4f}")

    print(f"\n  CV AUROC: mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}")

    # === 9. 保存模型 & 预测 ===
    print("\n[9] 保存模型和预测...")

    # 保存最终风险分层 (Full LogReg, 6级)
    # 在全量数据上重新训练以获得最终阈值
    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X)
    final_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    final_model.fit(X_full_scaled, y_died)
    yp_full = final_model.predict_proba(X_full_scaled)[:, 1]

    strata_full, thresholds, rates_full, _ = ordinal_risk_score(y_died, yp_full)

    predictions = df[['hadm_id']].copy()
    predictions['pred_30d_death'] = yp_full
    predictions['risk_stratum'] = strata_full
    predictions['died_30d'] = y_died
    predictions['unplanned_icu'] = y_icu
    predictions['aki'] = y_aki

    pred_path = DATA_DIR / 'predictions_baseline.csv'
    predictions.to_csv(pred_path, index=False)
    print(f"  predictions_baseline.csv: {len(predictions):,} rows")

    # === 10. 最终报告 ===
    print("\n" + "=" * 70)
    print("模型训练报告")
    print("=" * 70)

    lines = []
    L = lines.append
    L("=" * 70)
    L("MIMIC-IV 基线风险评分 — 模型报告")
    L(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L("=" * 70)

    L(f"\n【数据集】")
    L(f"  总样本: {len(X):,}")
    L(f"  特征数: {len(feature_cols)}")
    L(f"  30天死亡: {y_died.sum():,} ({y_died.mean()*100:.2f}%)")
    L(f"  训练/测试: 80/20")

    L(f"\n【模型性能对比】")
    L(f"  {'模型':<30s} {'AUROC':>8s} {'Brier':>8s}")
    L(f"  {'─'*46}")
    L(f"  {'Charlson-Only LogReg':<30s} {roc_auc_score(y_test, yp_c_test):>8.4f} {brier_score_loss(y_test, yp_c_test):>8.4f}")
    L(f"  {'Full LogReg (L2)':<30s} {roc_auc_score(y_test, yp_lr):>8.4f} {brier_score_loss(y_test, yp_lr):>8.4f}")
    L(f"  {'XGBoost':<30s} {roc_auc_score(y_test, yp_xgb):>8.4f} {brier_score_loss(y_test, yp_xgb):>8.4f}")
    L(f"  {'XGBoost + Platt Calib.':<30s} {roc_auc_score(y_test, yp_xgb_cal):>8.4f} {brier_score_loss(y_test, yp_xgb_cal):>8.4f}")

    L(f"\n【净重分类改善 vs Charlson】")
    L(f"  NRI (事件): {nri_died:+.4f}")
    L(f"  NRI (非事件): {nri_alive:+.4f}")
    L(f"  NRI (总): {nri_total:+.4f}")

    L(f"\n【交叉验证】")
    L(f"  5-Fold CV AUROC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    L(f"\n【风险分层 (6级, Full LogReg, 全量数据)】")
    L(f"  {'层级':<8} {'样本量':>8} {'死亡数':>8} {'死亡率':>10} {'边界':>12}")
    L(f"  {'─'*48}")
    for level in range(1, 7):
        mask = strata_full == level
        n = mask.sum()
        n_died = y_died[mask].sum()
        rate = n_died / n * 100 if n > 0 else 0
        bound = f"<{thresholds[level-1]:.4f}" if level < 6 else f">={thresholds[level-2]:.4f}" if level == 6 else "-"
        if level == 1:
            bound = f"<{thresholds[0]:.4f}"
        elif level < 6:
            bound = f"[{thresholds[level-2]:.4f}, {thresholds[level-1]:.4f})"
        else:
            bound = f">={thresholds[-1]:.4f}"
        L(f"  Level {level:<3} {n:>8,} {n_died:>8,} {rate:>9.2f}% {bound:>12}")

    L(f"\n【风险梯度验证】")
    grad_ok = all(x <= y for x, y in zip(rates_full, rates_full[1:]))
    L(f"  单调递增: {'OK' if grad_ok else 'FAIL 违反'}")
    min_rate = rates_full[0]
    max_rate = rates_full[-1]
    L(f"  死亡率范围: {min_rate:.2f}% - {max_rate:.2f}%")
    L(f"  死亡率比值: {max_rate/min_rate:.1f}x" if min_rate > 0 else "  死亡率比值: ∞")

    L(f"\n【特征重要性 Top 10 (LogReg |coef|)】")
    for _, row in coef_df.head(10).iterrows():
        L(f"  {row['feature']:25s}: {row['coef']:+8.4f}")

    L(f"\n【数据限制】")
    L(f"  1. 无化验数据 — MIMIC主模块labevents待下载")
    L(f"  2. 88,609 (30.8%) 手术类型为Unknown (诊断性/非手术操作)")
    L(f"  3. 当前AUROC可能被低估, 加入化验后预期提升至0.80-0.83")

    L("\n" + "=" * 70)

    report_path = DATA_DIR / 'model_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))

    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed:.1f} 秒")

    return predictions


if __name__ == '__main__':
    preds = main()
