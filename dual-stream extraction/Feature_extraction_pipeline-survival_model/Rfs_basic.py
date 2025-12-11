# Improved survival modeling pipeline for radiomics + Random Survival Forest (RSF)
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lifelines is still used for KaplanMeier and the Univariate Filter (it's great for that)
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

# Scikit-Survival for RSF
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.inspection import permutation_importance

from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm.auto import tqdm

import shap
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)


# ----------------------------
# Utility functions
# ----------------------------
def compute_vif(df, thresh=5.0):
    """
    Iteratively remove features with VIF > thresh.
    """
    X = df.copy()
    dropped = []
    pbar = tqdm(desc="Reducing Multicollinearity (VIF)", unit="feat")

    while True:
        if X.shape[1] <= 1:
            break
        try:
            vif = pd.Series(
                [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns
            )
        except Exception as e:
            print(f"VIF Error: {e}")
            break

        max_vif = vif.max()
        pbar.set_postfix({"Max VIF": f"{max_vif:.1f}", "Features Left": X.shape[1]})

        if max_vif > thresh:
            drop_feature = vif.idxmax()
            dropped.append(drop_feature)
            X = X.drop(columns=[drop_feature])
            pbar.update(1)
        else:
            break
    pbar.close()
    return X, dropped


def univariate_cox_filter(df, duration_col, event_col, features, p_threshold=0.2):
    """
    Uses Lifelines CoxPH to filter features.
    Even for RSF, checking univariate association with survival is a valid selection step.
    """
    keep = []
    for f in features:
        tmp = df[[duration_col, event_col, f]].dropna()
        if tmp.shape[0] < 10:
            continue
        cph = CoxPHFitter()
        try:
            cph.fit(tmp, duration_col=duration_col, event_col=event_col, show_progress=False)
            p = float(cph.summary.loc[f, "p"])
            if p < p_threshold:
                keep.append(f)
        except Exception:
            continue
    return keep


def bootstrap_concordance_rsf(model, X_test, y_test_struct, n_boot=200):
    """
    Compute bootstrap confidence interval for concordance on test set for RSF.
    """
    scores = []
    n = len(X_test)
    # y_test_struct is a structured array
    for _ in range(n_boot):
        idx = np.random.choice(range(n), size=n, replace=True)
        sample_X = X_test.iloc[idx]
        sample_y = y_test_struct[idx]

        # RSF predict returns risk scores (higher = worse)
        preds = model.predict(sample_X)

        # concordance_index_censored returns (c-index, concordant_pairs, discordant_pairs, tied_pairs, tied_risk)
        c = concordance_index_censored(sample_y["Status"], sample_y["Survival_in_days"], preds)[0]
        scores.append(c)
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


# ----------------------------
# 1. Load & Prepare Data
# ----------------------------
radiomics_df = pd.read_csv("/Users/mdsaibhossain/code/python/survival_model_analysis/radiomics_features_422_patients_seg_rtstruct.csv").set_index("patient_id")
clinical_df = pd.read_csv("/Users/mdsaibhossain/code/python/survival_model_analysis/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
clinical_df = clinical_df.rename(columns={"PatientID": "patient_id"}).set_index("patient_id")
survival_df = clinical_df[["Survival.time", "deadstatus.event"]]

df = radiomics_df.join(survival_df, how="inner").dropna(subset=["Survival.time", "deadstatus.event"])
print(f" Merged dataset: {len(df)} patients")

# Feature columns
duration_col = "Survival.time"
event_col = "deadstatus.event"
exclude_cols = {duration_col, event_col, "ct_path", "seg_path", "ct_path.1", "seg_path.1"}
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Impute & drop zero variance
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
zero_var = df[feature_cols].columns[df[feature_cols].var() == 0].tolist()
if zero_var:
    print(" Dropping zero-variance features:", zero_var)
    df = df.drop(columns=zero_var)
    feature_cols = [c for c in feature_cols if c not in zero_var]

df = df.dropna()
print(f"➡️ Final: {len(df)} patients, {len(feature_cols)} features")

# ----------------------------
# 2. Scaling & Train/Test Split
# ----------------------------
X = df[feature_cols]
y_time = df[duration_col]
y_event = df[event_col]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

X_train, X_test, y_t_train, y_t_test, y_e_train, y_e_test = train_test_split(
    X_scaled, y_time, y_event, test_size=0.2, stratify=y_event, random_state=42
)

# Convert targets to scikit-survival structured array format
# Format: [(boolean_event, time), ...]
y_train_struct = Surv.from_arrays(event=y_e_train.astype(bool).values, time=y_t_train.values)
y_test_struct = Surv.from_arrays(event=y_e_test.astype(bool).values, time=y_t_test.values)

# Make names friendly for sksurv
dtype_names = [('Status', '?'), ('Survival_in_days', '<f8')]
y_train_struct = y_train_struct.astype(dtype_names)
y_test_struct = y_test_struct.astype(dtype_names)

# ----------------------------
# 3. Feature Selection: VIF (Train) + Univariate Cox (Train)
# ----------------------------
print("\n🔎 Removing multicollinearity (VIF) on training data...")
X_train_for_vif = X_train.copy().reset_index(drop=True)
X_vif_reduced, dropped_vif = compute_vif(X_train_for_vif, thresh=5.0)
print(" Dropped due to VIF >", 5.0, ":", dropped_vif)
vif_features = X_vif_reduced.columns.tolist()

# Univariate Cox filtering (using Lifelines for filtering only)
print("🔎 Univariate Cox filtering (p < 0.20)...")
# Construct temporary df for lifelines filtering
temp_train_df = pd.concat([
    X_train_for_vif[vif_features].reset_index(drop=True),
    pd.Series(y_t_train.reset_index(drop=True).values, name=duration_col),
    pd.Series(y_e_train.reset_index(drop=True).values, name=event_col)
], axis=1)

uni_keep = univariate_cox_filter(
    df=temp_train_df,
    duration_col=duration_col, event_col=event_col, features=vif_features, p_threshold=0.20
)
print(" Kept after univariate filter:", len(uni_keep))

if len(uni_keep) == 0:
    print(" WARNING: fallback to top-10 variance features.")
    uni_keep = X_train_for_vif.var().sort_values(ascending=False).index[:10].tolist()

final_features = uni_keep.copy()
print(" Final features used for modeling:", final_features)

# Subset X to final features
X_train_sel = X_train[final_features]
X_test_sel = X_test[final_features]

# ----------------------------
# 4. Hyperparameter tuning via CV (Min Samples Leaf)
# ----------------------------
print("\n Tuning RSF (min_samples_leaf) via 5-fold CV...")
# RSF doesn't use "penalizer". It uses tree params. min_samples_leaf is crucial for survival trees.
param_grid = [1, 3, 5, 10, 20]
best_leaf = 1
best_score = -np.inf

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for leaf in param_grid:
    # Use C-index as score
    scores = []
    for train_idx, val_idx in kf.split(X_train_sel):
        X_tr_fold, X_val_fold = X_train_sel.iloc[train_idx], X_train_sel.iloc[val_idx]
        y_tr_fold, y_val_fold = y_train_struct[train_idx], y_train_struct[val_idx]

        rsf_temp = RandomSurvivalForest(n_estimators=100, min_samples_leaf=leaf, random_state=42, n_jobs=-1)
        rsf_temp.fit(X_tr_fold, y_tr_fold)
        scores.append(rsf_temp.score(X_val_fold, y_val_fold))

    mean_score = np.mean(scores)
    print(f"  min_samples_leaf={leaf:>2} → mean CV C-index = {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_leaf = leaf

print(f" Best min_samples_leaf: {best_leaf} → mean CV C-index = {best_score:.4f}")

# ----------------------------
# 5. Fit final RSF
# ----------------------------
print("\n Fitting final RSF model on training data...")
rsf = RandomSurvivalForest(n_estimators=1000, min_samples_leaf=best_leaf, random_state=42, n_jobs=-1)
rsf.fit(X_train_sel, y_train_struct)
print(" Model fitted.")

# ----------------------------
# 6. Model Evaluation (C-Index & Feature Importance)
# ----------------------------
print("\n" + "=" * 60)
print("RADIOMICS-ONLY RSF MODEL: FULL EVALUATION")
print("=" * 60)

# C-index
# rsf.score returns Harrell's concordance index
train_c = rsf.score(X_train_sel, y_train_struct)
test_c = rsf.score(X_test_sel, y_test_struct)

print(f"\n Concordance Index:")
print(f"   - Training: {train_c:.3f}")
print(f"   - Test:     {test_c:.3f}")

# Bootstrap C-index CI
mean_c, ci_low, ci_high = bootstrap_concordance_rsf(rsf, X_test_sel, y_test_struct, n_boot=200)
print(f"   - Test (bootstrap mean ± 95% CI): {mean_c:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

# Feature Importance (Permutation) - RSF does not have coefficients
print("\n📈 Computing Permutation Feature Importance...")
result = permutation_importance(rsf, X_test_sel, y_test_struct, n_repeats=10, random_state=42, n_jobs=-1)
perm_sorted_idx = result.importances_mean.argsort()[::-1]

print("\n Top 10 Features by Permutation Importance:")
for i in perm_sorted_idx[:10]:
    print(f"   - {X_test_sel.columns[i]}: {result.importances_mean[i]:.4f} +/- {result.importances_std[i]:.4f}")

# ----------------------------
# 8. Kaplan-Meier curves (risk groups) + log-rank
# ----------------------------
# Predict risk scores (Total number of expected events)
test_preds = rsf.predict(X_test_sel)

median_risk = np.median(test_preds)
# High score = High Risk (RSF outputs cumulative hazard/risk)
high_mask = test_preds >= median_risk
low_mask = test_preds < median_risk

high_time = y_t_test[high_mask]
high_event = y_e_test[high_mask]
low_time = y_t_test[low_mask]
low_event = y_e_test[low_mask]

kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 5))
kmf.fit(high_time, high_event, label="High Risk")
ax = kmf.plot_survival_function()
kmf.fit(low_time, low_event, label="Low Risk")
kmf.plot_survival_function(ax=ax)

plt.title(f"Radiomics RSF Risk Groups (Test C-index = {test_c:.3f})")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('c-index-RSF.png', dpi=300, bbox_inches='tight')
plt.show()

# Log-rank test
lr = logrank_test(high_time, low_time, high_event, low_event)
print(f"\n Log-rank Test: p = {lr.p_value:.4f}")

# ----------------------------
# 9. SHAP interpretability (Optimized for RSF)
# ----------------------------
print("\n Computing SHAP values (TreeExplainer for RSF)...")

try:
    explainer = shap.TreeExplainer(rsf)
    shap_values = explainer.shap_values(X_test_sel)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

        # 4. Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sel, show=False)
    plt.title("SHAP (Tree) summary — RSF")
    plt.tight_layout()
    plt.savefig('shap(tree)-RSF.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(" SHAP Plot Generated.")

except Exception as e:
    print(f"TreeExplainer failed ({e}). Falling back to fast KernelExplainer...")
    background_summary = shap.kmeans(X_train_sel, 10)


    def predict_risk(x):
        return rsf.predict(x)


    explainer = shap.KernelExplainer(predict_risk, background_summary)
    shap_values = explainer.shap_values(X_test_sel.iloc[:50], nsamples=100)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sel.iloc[:50], show=False)
    plt.title("SHAP (Kernel-Fast) summary — RSF")
    plt.tight_layout()
    plt.savefig('shap(kernel)-RSF.png', dpi=300, bbox_inches='tight')
    plt.show()

# ----------------------------
# 11. CALIBRATION & BRIER SCORE
# ----------------------------
print("\n" + "=" * 60)
print("CLINICAL VALIDITY METRICS (Q1 JOURNAL REQUIREMENT)")
print("=" * 60)

eval_time = y_t_test.median()

# Predict Survival Probability at median time for Test Set
# sksurv returns an array of StepFunction objects
surv_funcs_test = rsf.predict_survival_function(X_test_sel)


# Helper to evaluate step functions at a specific time
def get_survival_prob(surv_funcs, time_point):
    probs = []
    for fn in surv_funcs:
        probs.append(fn(time_point))
    return np.array(probs)


surv_prob_at_t = get_survival_prob(surv_funcs_test, eval_time)

# Create Calibration Plot using Risk Deciles/Groups
n_bins = 3
test_df_calib = pd.DataFrame()
test_df_calib['pred_survival'] = surv_prob_at_t
test_df_calib[duration_col] = y_t_test.values
test_df_calib[event_col] = y_e_test.values

test_df_calib['risk_group'] = pd.qcut(test_df_calib['pred_survival'], n_bins,
                                      labels=["High Risk", "Medium Risk", "Low Risk"])

observed_survival = []
predicted_survival = []
risk_labels = []

kmf_calib = KaplanMeierFitter()

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'r--', label="Perfect Calibration")

for group in ["High Risk", "Medium Risk", "Low Risk"]:
    subset = test_df_calib[test_df_calib['risk_group'] == group]

    # Mean Predicted Survival
    mean_pred = subset['pred_survival'].mean()

    # Observed Survival (KM) at eval_time
    kmf_calib.fit(subset[duration_col], subset[event_col])
    # Find closest time in KM index
    if len(kmf_calib.survival_function_.index) > 0:
        idx = np.argmin(np.abs(kmf_calib.survival_function_.index - eval_time))
        mean_obs = kmf_calib.survival_function_.iloc[idx].values[0]
    else:
        mean_obs = np.nan

    predicted_survival.append(mean_pred)
    observed_survival.append(mean_obs)
    risk_labels.append(group)

    if not pd.isna(mean_obs):
        # Error bars
        ci = kmf_calib.confidence_interval_survival_function_
        ci_idx = np.argmin(np.abs(ci.index - eval_time))
        lower = ci.iloc[ci_idx, 0]
        upper = ci.iloc[ci_idx, 1]
        yerr = [[mean_obs - lower], [upper - mean_obs]]
        plt.errorbar(mean_pred, mean_obs, yerr=yerr, fmt='o', capsize=5, label=f"{group} (Obs vs Pred)", markersize=8)

plt.plot(predicted_survival, observed_survival, 'b-', alpha=0.5)
plt.title(f"RSF Calibration Plot at Median Time ({eval_time:.0f} days)")
plt.xlabel("Predicted Survival Probability")
plt.ylabel("Observed Survival Probability (KM)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('survival_probability_calibration-RSF.png', dpi=300, bbox_inches='tight')
plt.show()


# ----------------------------
# 12. DECISION CURVE ANALYSIS (DCA)
# ----------------------------
def calculate_net_benefit(y_true, y_pred_risk, thresh):
    n = len(y_true)
    net_benefits = []
    for pt in thresh:
        high_risk_idx = y_pred_risk >= pt
        tp = np.sum((y_true == 1) & high_risk_idx)
        fp = np.sum((y_true == 0) & high_risk_idx)
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    return net_benefits


# Risk = 1 - Survival Probability at Median Time
predicted_risk = 1 - surv_prob_at_t
true_event = y_e_test.values

thresholds = np.linspace(0.01, 0.9, 100)
nb_model = calculate_net_benefit(true_event, predicted_risk, thresholds)
nb_all = calculate_net_benefit(true_event, np.ones_like(true_event), thresholds)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, nb_model, label="RSF Model", color='blue', linewidth=2)
plt.plot(thresholds, nb_all, label="Treat All", color='gray', linestyle='--')
plt.axhline(y=0, color='black', linestyle='-', label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (RSF)")
plt.legend()
plt.ylim(-0.05, 0.25)
plt.grid(True, alpha=0.3)
plt.savefig('decision_curv_analysis-RSF.png', dpi=300, bbox_inches='tight')
plt.show()

print(" RSF DCA Plot generated.")