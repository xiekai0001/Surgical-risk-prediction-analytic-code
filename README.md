# Analytic Code for Surgical Risk Prediction Model


## Overview

This repository contains the core analytic code for developing and validating a machine learning-based surgical risk prediction model using the MIMIC-IV database with external validation in eICU and PIC databases.

**Note**: Data extraction scripts and figure generation scripts are not included in this repository, as they are specific to local database configurations and visualization requirements.

---

## File Structure

### Core Analysis Pipeline

```
analytic_code_for_github/
├── README.md                                  (This file)
├── 2_feature_engineering_baseline.py          (Baseline feature engineering)
├── 2_feature_engineering_full.py              (Full model feature engineering)
├── 2b_feature_engineering_external.py         (External validation feature engineering)
├── 3_train_baseline_model.py                  (Baseline model training)
├── 3_train_full_model.py                      (Full model training)
└── 5_validate_external.py                     (External validation)
```

---

## Analysis Workflow

### Step 1: Feature Engineering - Baseline Model

**Script**: `2_feature_engineering_baseline.py`

**Purpose**: Generate features for the baseline model (demographics + comorbidities + surgery type)

**Input**:
- `cohort_baseline.csv` - Patient demographics
- `cohort_comorbidities.csv` - Charlson comorbidities
- `cohort_outcomes.csv` - 30-day mortality and complications
- `surgery_procedures.csv` - ICD-9/ICD-10 procedure codes

**Output**:
- `features_baseline.csv` - Baseline feature matrix (32 features)

**Features**:
- Demographics: age (z-score), age>89 flag, gender, emergency admission
- Charlson comorbidities: 17 individual components + total score
- Surgery categories: 10 major surgery types (ICD-9/ICD-10 mapped)

**Key Functions**:
- `classify_surgery_icd9()` - ICD-9-CM procedure code classification
- `classify_surgery_icd10()` - ICD-10-PCS procedure code classification
- `build_surgery_features()` - Extract surgery type from primary procedure

---

### Step 2: Feature Engineering - Full Model

**Script**: `2_feature_engineering_full.py`

**Purpose**: Generate features for the full model (baseline + 52 laboratory features)

**Input**:
- Same as baseline, plus:
- `cohort_labs.csv` - Preoperative laboratory values (17 tests)

**Output**:
- `features_full.csv` - Full feature matrix (84 features)

**Additional Features** (52):
- 17 standardized lab values (z-scores)
- 17 missing indicators (binary)
- 18 derived clinical variables:
  - eGFR and CKD stage
  - Anemia, BUN/Cr ratio
  - Electrolyte disturbances (hypo/hypernatremia, hypo/hyperkalemia)
  - Coagulopathy, acidosis
  - Liver injury, hyperbilirubinemia
  - Others

**Key Functions**:
- `standardize_labs()` - Z-score normalization
- `create_derived_features()` - Clinical interpretation of lab values

---

### Step 3: Feature Engineering - External Validation

**Script**: `2b_feature_engineering_external.py`

**Purpose**: Generate features for external validation cohorts (eICU and PIC)

**Input**:
- External cohort data with same structure as MIMIC-IV

**Output**:
- `features_eicu.csv` - eICU feature matrix
- `features_pic.csv` - PIC feature matrix

**Note**: Uses the same feature definitions as the MIMIC-IV full model to ensure consistency.

---

### Step 4: Train Baseline Model

**Script**: `3_train_baseline_model.py`

**Purpose**: Train and evaluate the baseline model (32 features)

**Input**:
- `features_baseline.csv`

**Output**:
- `predictions_baseline.csv` - Risk predictions and 6-tier stratification
- `model_report.txt` - Performance metrics

**Models**:
1. Charlson-only Logistic Regression (reference)
2. Full baseline Logistic Regression (L2 regularization)
3. XGBoost classifier
4. XGBoost + Platt calibration

**Evaluation Metrics**:
- AUROC (Area Under ROC Curve)
- Brier Score
- 6-tier risk stratification (ordinal scale)
- Net Reclassification Improvement (NRI) vs Charlson-only
- 5-fold cross-validation

**Key Functions**:
- `ordinal_risk_score()` - Generate 6-tier risk stratification
- `evaluate_model()` - Comprehensive model evaluation
- `fit_calibration_intercept()` - Platt scaling for calibration

---

### Step 5: Train Full Model

**Script**: `3_train_full_model.py`

**Purpose**: Train and evaluate the full model (84 features) and compare with baseline

**Input**:
- `features_full.csv`
- `features_baseline.csv` (for comparison)

**Output**:
- `predictions_full.csv` - Risk predictions and stratification
- `model_comparison.txt` - Baseline vs Full performance

**Analysis**:
- Train both baseline (32) and full (84) models
- Quantify incremental value of laboratory data (Δ AUROC)
- Net Reclassification Improvement (NRI)
- Feature importance ranking
- 5-fold cross-validation with stability metrics (mean ± SD)

**Key Metrics**:
- AUROC: Baseline vs Full
- Δ AUROC: Incremental value of laboratory data
- NRI: Reclassification improvement
- Calibration slope and intercept

---

### Step 6: External Validation

**Script**: `5_validate_external.py`

**Purpose**: Validate the trained model on external cohorts (eICU and PIC)

**Input**:
- Trained model from Step 5 (parameters and thresholds)
- `features_eicu.csv`
- `features_pic.csv`

**Output**:
- `predictions_eICU.csv` - eICU predictions
- `predictions_PIC.csv` - PIC predictions
- `external_validation_report.txt` - Performance metrics

**Validation Metrics**:
- AUROC and 95% CI (bootstrap)
- Calibration curves (observed vs predicted)
- Decision curve analysis (net benefit)
- Risk stratification performance
- Subgroup analyses

**Key Findings**:
- eICU (ICU population): AUROC 0.784 [0.780-0.788]
- PIC (pediatric): AUROC 0.643 [0.590-0.696] - expected degradation due to age mismatch

---

## Statistical Methods

### Primary Outcome
30-day all-cause mortality

### Model Development
- **Algorithm**: Logistic Regression with L2 regularization (C=1.0)
- **Train/Test Split**: 80/20 stratified by outcome
- **Cross-Validation**: 5-fold stratified
- **Feature Standardization**: Z-score normalization (mean=0, SD=1)

### Risk Stratification
- **Levels**: 6-tier ordinal scale (L1-L6)
- **Method**: Quantile-based thresholds ensuring monotonicity
- **Validation**: Mortality rate increases monotonically across levels

### Model Comparison
- **AUROC**: C-statistic for discrimination
- **Brier Score**: Overall performance
- **NRI**: Continuous net reclassification improvement
- **Calibration**: Calibration plots (Loess smoothed)
- **Clinical Utility**: Decision curve analysis

### External Validation
- **Calibration**: No recalibration applied (assess generalizability)
- **Discrimination**: AUROC with 95% bootstrap CI (200 iterations)
- **Clinical Utility**: Net benefit across decision thresholds

---

## Software Requirements

### Python Version
Python 3.8 or higher

### Required Packages
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
scipy>=1.7.0
```

### Installation
```bash
pip install pandas numpy scikit-learn xgboost scipy
```

---

## Usage

### Running the Analysis Pipeline

```bash
# Step 1: Feature engineering (baseline)
python 2_feature_engineering_baseline.py

# Step 2: Feature engineering (full model)
python 2_feature_engineering_full.py

# Step 3: Feature engineering (external validation)
python 2b_feature_engineering_external.py

# Step 4: Train baseline model
python 3_train_baseline_model.py

# Step 5: Train full model
python 3_train_full_model.py

# Step 6: External validation
python 5_validate_external.py
```

**Note**: Each script expects input files in a directory specified by `DATA_DIR` variable (default: `Z:\本地数据\asa_data`). Modify this path according to your local setup.

---

## Key Results

### MIMIC-IV Internal Validation
- **Baseline Model (32 features)**: AUROC 0.838 [0.830-0.846]
- **Full Model (84 features)**: AUROC 0.893 [0.888-0.898]
- **Incremental Value**: Δ AUROC +0.055

### External Validation - eICU
- **Baseline**: AUROC 0.672 [0.668-0.676]
- **Full Model**: AUROC 0.784 [0.780-0.788]
- **Incremental Value**: Δ AUROC +0.112 (largest gain in ICU population)

### External Validation - PIC
- **Baseline**: AUROC 0.562 [0.510-0.614]
- **Full Model**: AUROC 0.643 [0.590-0.696]
- **Note**: Expected degradation due to age distribution mismatch (adult model applied to 83% pediatric cohort)

### Cross-Validation Stability
- **5-Fold CV**: Mean AUROC 0.895, SD 0.002
- Demonstrates excellent model stability

---

## Important Notes

### Data Privacy
- This repository contains **code only**, no patient data
- All data must be obtained through proper channels:
  - MIMIC-IV: https://physionet.org/content/mimiciv/
  - eICU: https://physionet.org/content/eicu-crd/
  - PIC: Contact authors for access

### Data Paths
- All scripts use local file paths (`DATA_DIR`)
- **You must modify these paths** to match your local data directory
- Default path: `Z:\本地数据\asa_data` (Windows-specific)

### Excluded Scripts
The following scripts are **not included** in this repository:
- `1_extract_surgical_cohort.py` - Database-specific data extraction
- `1d_extract_labs_main.py` - Laboratory data extraction
- `4a_extract_eicu_icu.py` - eICU data extraction
- `4b_extract_pic_surgery.py` - PIC data extraction
- `6_generate_figures.py` - Manuscript figure generation
- `export_manuscript.py` - Manuscript export utilities

**Reason**: These scripts are highly specific to local database configurations, authentication methods, and output formatting requirements.

---

## Reproducibility

### Random Seed
All scripts use `RANDOM_SEED = 42` for reproducibility

### Model Hyperparameters
- **Logistic Regression**: L2 penalty, C=1.0, max_iter=5000
- **XGBoost**: n_estimators=200, max_depth=5, learning_rate=0.03

### Feature Definitions
- All feature engineering logic is explicitly coded (no external dependencies)
- ICD-9/ICD-10 surgery classification mappings are hardcoded
- Laboratory reference ranges follow standard clinical guidelines

---

## Citation

If you use this code, please cite:

```
[Full citation will be added upon publication]
```

---

## Contact

For questions about the code or methodology:
- [Your contact information]

For data access:
- MIMIC-IV: https://physionet.org/content/mimiciv/
- eICU: https://physionet.org/content/eicu-crd/
- PIC: [Contact information]

---

## License

[Your license choice - suggest MIT or Apache 2.0 for academic code]

---

## Acknowledgments

This research was conducted using the MIMIC-IV database provided by the MIT Laboratory for Computational Physiology.
