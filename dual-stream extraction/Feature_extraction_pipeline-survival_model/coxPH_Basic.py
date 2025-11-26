# Improved survival modeling pipeline for radiomics + CoxPH
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index, k_fold_cross_validation
from lifelines.statistics import logrank_test

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm.auto import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import shap
import random

random.seed(42)
np.random.seed(42)


# ----------------------------
# Utility functions
# ----------------------------
def compute_vif(df, thresh=5.0):
    """
    Iteratively remove features with VIF > thresh.
    Includes a Progress Bar to track the reduction process.
    """
    X = df.copy()
    dropped = []

    # Initialize progress bar
    # We don't know exactly when it ends, so we leave total=None or manage manually
    # 'leave=True' keeps the bar on screen after it finishes
    pbar = tqdm(desc="Reducing Multicollinearity (VIF)", unit="feat")

    while True:
        if X.shape[1] <= 1:
            break

        # Calculate VIF for all features (The slow part)
        try:
            vif = pd.Series(
                [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns
            )
        except Exception as e:
            print(f"VIF Error: {e}")
            break

        max_vif = vif.max()

        # Update progress bar with live stats
        pbar.set_postfix({
            "Max VIF": f"{max_vif:.1f}",
            "Features Left": X.shape[1]
        })

        if max_vif > thresh:
            drop_feature = vif.idxmax()
            dropped.append(drop_feature)
            X = X.drop(columns=[drop_feature])

            # Update the bar visually to show work is happening
            pbar.update(1)
        else:
            # Stop if Max VIF is acceptable
            break

    pbar.close()
    return X, dropped


def univariate_cox_filter(df, duration_col, event_col, features, p_threshold=0.2):
    """
    Fit single-variable CoxPH for each feature and keep those with p < p_threshold.
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


def bootstrap_concordance(model, X_test_df, duration_col, event_col, n_boot=200):
    """
    Compute bootstrap confidence interval for concordance on test set.
    """
    scores = []
    n = len(X_test_df)
    for _ in range(n_boot):
        idx = np.random.choice(range(n), size=n, replace=True)
        sample = X_test_df.iloc[idx]
        preds = model.predict_partial_hazard(sample).values.ravel()
        c = concordance_index(sample[duration_col], -preds, sample[event_col])
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

# Feature columns (exclude common path-like columns if present)
duration_col = "Survival.time"
event_col = "deadstatus.event"
exclude_cols = {duration_col, event_col, "ct_path", "seg_path", "ct_path.1", "seg_path.1"}
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Impute & drop zero variance features
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

# split (stratify by event to preserve censoring proportion)
X_train, X_test, y_t_train, y_t_test, y_e_train, y_e_test = train_test_split(
    X_scaled, y_time, y_event, test_size=0.2, stratify=y_event, random_state=42
)

# Build DataFrames that lifelines expects (we'll modify columns after filtering)
train_df_full = pd.concat([
    X_train.reset_index(drop=True),
    pd.Series(y_t_train.values, name=duration_col),
    pd.Series(y_e_train.values, name=event_col)
], axis=1)

test_df_full = pd.concat([
    X_test.reset_index(drop=True),
    pd.Series(y_t_test.values, name=duration_col),
    pd.Series(y_e_test.values, name=event_col)
], axis=1)

# ----------------------------
# 3. Feature Selection: VIF (on training set), then univariate Cox filter
# ----------------------------
print("\n🔎 Removing multicollinearity (VIF) on training data...")
X_train_for_vif = X_train.copy().reset_index(drop=True)
X_vif_reduced, dropped_vif = compute_vif(X_train_for_vif, thresh=5.0)
print(" Dropped due to VIF >", 5.0, ":", dropped_vif)
vif_features = X_vif_reduced.columns.tolist()

# Univariate Cox filtering (keep p < 0.20)
print("🔎 Univariate Cox filtering (p < 0.20)...")
uni_keep = univariate_cox_filter(
    df=pd.concat([X_train_for_vif[vif_features].reset_index(drop=True),
                  pd.Series(y_t_train.reset_index(drop=True).values, name=duration_col),
                  pd.Series(y_e_train.reset_index(drop=True).values, name=event_col)], axis=1),
    duration_col=duration_col, event_col=event_col, features=vif_features, p_threshold=0.20
)
print(" Kept after univariate filter:", len(uni_keep))
if len(uni_keep) == 0:
    # fallback: keep top 10 by variance if nothing passes
    print(" WARNING: univariate filter removed all features, falling back to top-10 variance features.")
    uni_keep = X_train_for_vif.var().sort_values(ascending=False).index[:10].tolist()

# Final feature list to use
final_features = uni_keep.copy()
print(" Final features used for modeling:", final_features)

# Prepare final train/test DataFrames with selected features
train_df = pd.concat([
    X_train_for_vif[final_features].reset_index(drop=True),
    pd.Series(y_t_train.reset_index(drop=True).values, name=duration_col),
    pd.Series(y_e_train.reset_index(drop=True).values, name=event_col)
], axis=1)

# For test, ensure same columns (reindex)
X_test_sel = X_test.reset_index(drop=True)[final_features]
test_df = pd.concat([
    X_test_sel,
    pd.Series(y_t_test.reset_index(drop=True).values, name=duration_col),
    pd.Series(y_e_test.reset_index(drop=True).values, name=event_col)
], axis=1)

# ----------------------------
# 4. Penalizer tuning via k-fold CV (on training data)
# ----------------------------
print("\n Tuning penalizer via 5-fold CV (concordance index)...")
penalties = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
best_pen = None
best_score = -np.inf
cv_results = {}
for p in penalties:
    cph_temp = CoxPHFitter(penalizer=p)
    try:
        scores = k_fold_cross_validation(cph_temp, train_df, duration_col=duration_col,
                                         event_col=event_col, k=5, scoring_method="concordance_index")
        mean_score = np.mean(scores)
    except Exception as e:
        mean_score = -np.inf
    cv_results[p] = mean_score
    print(f"  λ={p:>6} → mean CV C-index = {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_pen = p

print(f" Best penalizer (by CV): {best_pen} → mean CV C-index = {best_score:.4f}")

# ----------------------------
# 5. Fit final CoxPH with best penalizer
# ----------------------------
print("\n Fitting final CoxPH model on training data...")
cph = CoxPHFitter(penalizer=best_pen)
cph.fit(train_df, duration_col=duration_col, event_col=event_col, show_progress=False)
print(" Model fitted.")

# ----------------------------
# 6. Model Summary & Concordance
# ----------------------------
print("\n" + "=" * 60)
print("RADIOMICS-ONLY COXPH MODEL: FULL EVALUATION")
print("=" * 60)

# Model summary (top 10 by |coef|)
summary = cph.summary
summary_sorted = summary.reindex(summary['coef'].abs().sort_values(ascending=False).index)
print("\n📈 MODEL SUMMARY (Top 10 Features by |coef|):")
print(summary_sorted[['coef', 'exp(coef)', 'p']].head(10))

# Concordance indices
train_preds = cph.predict_partial_hazard(train_df).values.ravel()
train_c = concordance_index(train_df[duration_col], -train_preds, train_df[event_col])

test_preds = cph.predict_partial_hazard(test_df).values.ravel()
test_c = concordance_index(test_df[duration_col], -test_preds, test_df[event_col])

print(f"\n Concordance Index:")
print(f"   - Training: {train_c:.3f}")
print(f"   - Test:     {test_c:.3f}")

# Bootstrap C-index CI
mean_c, ci_low, ci_high = bootstrap_concordance(cph, test_df, duration_col, event_col, n_boot=200)
print(f"   - Test (bootstrap mean ± 95% CI): {mean_c:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

# Significant features
sig_features = summary[summary['p'] < 0.05].index.tolist()
print(f"\n🔬 Significant Features (p < 0.05):")
if sig_features:
    for feat in sig_features:
        hr = summary.loc[feat, 'exp(coef)']
        direction = "↑ risk" if hr > 1 else "↓ risk"
        print(f"   - {feat}: HR = {hr:.2f} ({direction}), p = {summary.loc[feat, 'p']:.4g}")
else:
    print("   - None")

# ----------------------------
# 7. Proportional Hazards assumption check
# ----------------------------
print("\n Checking proportional hazards assumption (training set).")
try:
    # This prints warnings/violations if any. set p_value_threshold as needed.
    cph.check_assumptions(train_df, show_plots=False, p_value_threshold=0.05)
except Exception as e:
    print("  check_assumptions raised an exception (some diagnostics may not be available):", e)

# ----------------------------
# 8. Kaplan-Meier curves (risk groups) + log-rank
# ----------------------------
# split test into high/low risk using median partial hazard
median_risk = np.median(test_preds)
test_df_plot = test_df.copy()
test_df_plot['partial_hazard'] = test_preds
high = test_df_plot[test_df_plot['partial_hazard'] >= median_risk]
low = test_df_plot[test_df_plot['partial_hazard'] < median_risk]

kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 5))
kmf.fit(high[duration_col], high[event_col], label="High Risk")
ax = kmf.plot_survival_function()
kmf.fit(low[duration_col], low[event_col], label="Low Risk")
kmf.plot_survival_function(ax=ax)

plt.title(f"Radiomics Risk Groups (Test C-index = {test_c:.3f})")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('c-index-coxPH.png', dpi=300, bbox_inches='tight')
plt.show()

# Log-rank test
lr = logrank_test(high[duration_col], low[duration_col], high[event_col], low[event_col])
print(f"\n Log-rank Test: p = {lr.p_value:.4f}")

# ----------------------------
# 9. SHAP interpretability
# ----------------------------
print("\n Computing SHAP values (exact linear SHAP + optional Kernel SHAP verification)...")

# Exact linear SHAP (for linear Cox: contribution to linear predictor = coef * (x - mean))
coefs = cph.params_.loc[final_features].values  # aligned with final_features order
X_train_mean = train_df[final_features].mean().values
X_test_arr = test_df[final_features].values
linear_shap_values = (X_test_arr - X_train_mean) * coefs.reshape(1, -1)  # shape (n_samples, n_features)

# Global importance by mean(|SHAP|)
mean_abs_shap = np.mean(np.abs(linear_shap_values), axis=0)
shap_ranking = pd.Series(mean_abs_shap, index=final_features).sort_values(ascending=False)
print("\n Top features by mean |linear SHAP|:")
print(shap_ranking.head(10))

# SHAP summary plot (uses the computed matrix)
try:
    plt.figure(figsize=(10, 6))
    # shap.summary_plot expects either SHAP object or array; here we provide array and DataFrame
    shap.summary_plot(linear_shap_values, pd.DataFrame(test_df[final_features], columns=final_features), show=False)
    plt.title("SHAP (linear exact) summary — contributions to linear predictor")
    plt.tight_layout()
    plt.savefig('shap(linear)-coxPH.png', dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print("Could not draw SHAP summary plot (matplotlib / shap issue):", e)

# Optional: Kernel SHAP verification on a small subset (more costly). We'll run it on up-to-50 samples.
run_kernel_shap = True
if run_kernel_shap:
    try:
        subset_bg = train_df[final_features].sample(n=min(50, len(train_df)), random_state=42)


        def model_partial_hazard(X_df):
            # lifelines returns a series-like; convert to 1d numpy
            return cph.predict_partial_hazard(pd.DataFrame(X_df, columns=final_features)).values.ravel()


        print("\nRunning Kernel SHAP on a small subset (slow) — this helps verify linear SHAP patterns...")
        explainer = shap.KernelExplainer(model_partial_hazard, subset_bg, link="identity")
        subset_test = test_df[final_features].sample(n=min(50, len(test_df)), random_state=1)
        kernel_shap_vals = explainer.shap_values(subset_test, nsamples=200)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(kernel_shap_vals, subset_test, show=False)
        plt.title("SHAP (Kernel) on subset — verify linear SHAP")
        plt.tight_layout()
        plt.savefig('shap(kernel)-coxPH.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(" Kernel SHAP failed or was interrupted (this is optional). Error:", e)

print("\n Done. Pipeline completed.")

# ----------------------------
# 11. CALIBRATION & BRIER SCORE
# ----------------------------
from sksurv.metrics import integrated_brier_score, brier_score
from sksurv.util import Surv

print("\n" + "=" * 60)
print("CLINICAL VALIDITY METRICS (Q1 JOURNAL REQUIREMENT)")
print("=" * 60)

# 1. Get Evaluation Time (Median)
eval_time = y_t_test.median()

# 2. Get Actual Survival Status at Evaluation Time
# A patient is "Dead at t" if they died (Event=1) AND Time <= t
# A patient is "Alive at t" if Time > t
# Censored before t are ambiguous, but standard calibration plots often drop them or use KM estimates per bin.
# Here we use the standard approach: Compare Mean Predicted Survival vs Observed KM Survival in groups.

# Predict Survival Probability at median time for Test Set
surv_funcs_test = cph.predict_survival_function(test_df) # Corrected coxnet to cph
# Find the closest time point in surv_funcs_test.index to eval_time
if len(surv_funcs_test.index) > 0:
    closest_time_label = surv_funcs_test.index[np.argmin(np.abs(surv_funcs_test.index - eval_time))]
    surv_prob_at_t = surv_funcs_test.loc[closest_time_label].values
else:
    raise ValueError("cph.predict_survival_function returned an empty DataFrame for calibration.")

# 3. Create Calibration Plot using Risk Deciles/Groups
# We split patients into 3 groups: Low Risk, Medium Risk, High Risk
n_bins = 3
test_df_calib = test_df.copy()
test_df_calib['pred_survival'] = surv_prob_at_t
test_df_calib['risk_group'] = pd.qcut(test_df_calib['pred_survival'], n_bins, labels=["High Risk", "Medium Risk", "Low Risk"])
# Note: High predicted survival = Low Risk

observed_survival = []
predicted_survival = []
risk_labels = []

kmf_calib = KaplanMeierFitter()

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'r--', label="Perfect Calibration")

for group in ["High Risk", "Medium Risk", "Low Risk"]:
    subset = test_df_calib[test_df_calib['risk_group'] == group]

    # Mean Predicted Survival for this group
    mean_pred = subset['pred_survival'].mean()

    # Observed Survival (Kaplan-Meier) at eval_time
    kmf_calib.fit(subset[duration_col], subset[event_col])
    # Get survival prob at exactly eval_time (handling if time not in index)
    # Use closest_time_label found above to ensure consistency
    if len(kmf_calib.survival_function_.index) > 0:
        idx = np.argmin(np.abs(kmf_calib.survival_function_.index - closest_time_label))
        mean_obs = kmf_calib.survival_function_.iloc[idx].values[0]
    else:
        mean_obs = np.nan # Handle case where KM survival function is empty

    predicted_survival.append(mean_pred)
    observed_survival.append(mean_obs)
    risk_labels.append(group)

    # Plot point with error bars (approximate)
    if not pd.isna(mean_obs):
        ci = kmf_calib.confidence_interval_survival_function_
        ci_idx = np.argmin(np.abs(ci.index - closest_time_label))
        lower = ci.iloc[ci_idx, 0]
        upper = ci.iloc[ci_idx, 1]
        yerr = [[mean_obs - lower], [upper - mean_obs]]

        plt.errorbar(mean_pred, mean_obs, yerr=yerr, fmt='o', capsize=5, label=f"{group} (Obs vs Pred)", markersize=8)

# Plot line connecting groups
plt.plot(predicted_survival, observed_survival, 'b-', alpha=0.5)

plt.title(f"Calibration Plot at Median Time ({eval_time:.0f} days)\n(Binned by Risk Group)")
plt.xlabel("Predicted Survival Probability")
plt.ylabel("Observed Survival Probability (KM)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('survival_probability_calibration-coxPH.png', dpi=300, bbox_inches='tight')
plt.show()

print(f" Calibration Plot generated. Points close to the red diagonal indicate a well-calibrated model.")


# ----------------------------
# 12. DECISION CURVE ANALYSIS (DCA)
# ----------------------------
# Simple DCA implementation for Survival Models
def calculate_net_benefit(y_true, y_pred_risk, thresh):
    # Net Benefit = (True Positives / N) - (False Positives / N) * (pt / (1 - pt))
    # For survival, we weight by Time. This is a simplified approximation.
    n = len(y_true)
    net_benefits = []
    for pt in thresh:
        # Who does model say is high risk?
        high_risk_idx = y_pred_risk >= pt

        # True Positives (High Risk AND Event happened) - Weighted by inverse probability of censoring (simplified here)
        # In pure python without IPCW, we approximate:
        tp = np.sum((y_true == 1) & high_risk_idx)
        fp = np.sum((y_true == 0) & high_risk_idx)

        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    return net_benefits


# Calculate Risk Scores (1 - Survival at Median Time)
median_surv_funcs = cph.predict_survival_function(test_df) # Corrected coxnet to cph
# Find the closest time point in median_surv_funcs.index to eval_time
if len(median_surv_funcs.index) > 0:
    closest_time_label_dca = median_surv_funcs.index[np.argmin(np.abs(median_surv_funcs.index - eval_time))]
    median_surv = median_surv_funcs.loc[closest_time_label_dca]
else:
    raise ValueError("cph.predict_survival_function returned an empty DataFrame for DCA.")

predicted_risk = 1 - median_surv.values
true_event = y_e_test.values  # Binary event status (simplification for DCA)

thresholds = np.linspace(0.01, 0.9, 100)
nb_model = calculate_net_benefit(true_event, predicted_risk, thresholds)
nb_all = calculate_net_benefit(true_event, np.ones_like(true_event), thresholds)  # Treat all

plt.figure(figsize=(8, 6))
plt.plot(thresholds, nb_model, label="Radiomics Model", color='blue', linewidth=2)
plt.plot(thresholds, nb_all, label="Treat All", color='gray', linestyle='--')
plt.axhline(y=0, color='black', linestyle='-', label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (DCA)")
plt.legend()
plt.ylim(-0.05, 0.25)  # Zoom in on relevant area
plt.grid(True, alpha=0.3)
plt.savefig('decision_curv_analysis-coxPH.png', dpi=300, bbox_inches='tight')
plt.show()

print(" DCA Plot generated. Model line should be higher than 'Treat All' and 'Treat None'.")