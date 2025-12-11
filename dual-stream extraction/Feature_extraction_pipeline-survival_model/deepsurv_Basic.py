# Improved survival modeling pipeline for radiomics + DeepSurv (PyCox/PyTorch)
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lifelines for Filtering and KM plots
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

# PyTorch and PyCox for DeepSurv
import torch
import torchtuples as tt # Helper for pycox
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm.auto import tqdm

import shap
import random

# ----------------------------
# Reproducibility Seeds
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# Utility functions
# ----------------------------
def compute_vif(df, thresh=5.0):
    """Iteratively remove features with VIF > thresh."""
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
    """Uses Lifelines CoxPH to filter features (valid for Deep Learning input selection too)."""
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

def bootstrap_concordance_deepsurv(model, X_test, duration_test, event_test, n_boot=200):
    """Bootstrap CI for DeepSurv."""
    scores = []
    n = len(X_test)
    X_test_np = X_test.astype('float32') # Ensure numpy float32

    for _ in range(n_boot):
        idx = np.random.choice(range(n), size=n, replace=True)
        sample_X = X_test_np[idx]
        sample_dur = duration_test[idx]
        sample_evt = event_test[idx]

        # Predict partial hazard (negative because pycox predicts log-hazard, higher = risk)
        # However, CoxPH model in pycox: predict returns log-hazard.
        # Concordance expects risk scores.
        log_hazards = model.predict(sample_X).flatten()

        c = concordance_index(sample_dur, -log_hazards, sample_evt)
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
    df = df.drop(columns=zero_var)
    feature_cols = [c for c in feature_cols if c not in zero_var]

df = df.dropna()
print(f"Final: {len(df)} patients, {len(feature_cols)} features")

# ----------------------------
# 2. Scaling & Train/Test Split
# ----------------------------
X = df[feature_cols]
y_time = df[duration_col]
y_event = df[event_col]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

# Split
X_train, X_test, y_t_train, y_t_test, y_e_train, y_e_test = train_test_split(
    X_scaled, y_time, y_event, test_size=0.2, stratify=y_event, random_state=42
)

# Further split Train into Train/Val for Early Stopping (Crucial for Deep Learning)
X_train, X_val, y_t_train, y_t_val, y_e_train, y_e_val = train_test_split(
    X_train, y_t_train, y_e_train, test_size=0.2, stratify=y_e_train, random_state=42
)

# ----------------------------
# 3. Feature Selection (VIF + Univariate Cox)
# ----------------------------
print("\n Removing multicollinearity (VIF) on training data...")
X_train_vif = X_train.copy().reset_index(drop=True)
X_vif_reduced, dropped_vif = compute_vif(X_train_vif, thresh=5.0)
vif_features = X_vif_reduced.columns.tolist()

print(" Univariate Cox filtering (p < 0.20)...")
temp_train_df = pd.concat([
    X_train_vif[vif_features].reset_index(drop=True),
    pd.Series(y_t_train.reset_index(drop=True).values, name=duration_col),
    pd.Series(y_e_train.reset_index(drop=True).values, name=event_col)
], axis=1)

uni_keep = univariate_cox_filter(
    df=temp_train_df,
    duration_col=duration_col, event_col=event_col, features=vif_features, p_threshold=0.20
)
print(" Kept after univariate filter:", len(uni_keep))
final_features = uni_keep.copy()
print(" Final features used:", final_features)

# ----------------------------
# 4. Preparing Data for PyCox
# ----------------------------
# Convert to float32 numpy arrays (Required for PyTorch)
x_train = X_train[final_features].values.astype('float32')
x_val = X_val[final_features].values.astype('float32')
x_test = X_test[final_features].values.astype('float32')

# Target tuples (Duration, Event)
get_target = lambda df: (df[duration_col].values.astype('float32'), df[event_col].values.astype('float32'))
y_train = get_target(pd.concat([y_t_train, y_e_train], axis=1))
y_val = get_target(pd.concat([y_t_val, y_e_val], axis=1))
val = (x_val, y_val)

# ----------------------------
# 5. Define DeepSurv Architecture
# ----------------------------
in_features = x_train.shape[1]
num_nodes = [32, 32] # Two hidden layers with 32 neurons
out_features = 1 # Single output (log hazard)
batch_norm = True
dropout = 0.1 # Helps prevent overfitting on small datasets

net = tt.practical.MLPVanilla(
    in_features, num_nodes, out_features, batch_norm, dropout,
    output_bias=False # Standard for CoxPH
)

# ----------------------------
# 6. Train DeepSurv
# ----------------------------
print("\n Training DeepSurv Network...")
model = CoxPH(net, tt.optim.Adam)

# Learning rate finder (optional, but good practice. We use standard LR here for stability)
lr = 0.01
model.optimizer.set_lr(lr)

epochs = 512
batch_size = 64
callbacks = [tt.callbacks.EarlyStopping()] # Stop if val_loss doesn't improve

log = model.fit(
    x_train, y_train, batch_size, epochs, callbacks, val_data=val, verbose=False
)

# Plot training loss
plt.figure(figsize=(6,4))
log.plot()
plt.title("DeepSurv Training Curve (Loss)")
plt.xlabel("Epochs")
plt.show()

# ----------------------------
# 7. Model Evaluation
# ----------------------------
print("\n" + "="*60)
print("RADIOMICS-ONLY DEEPSURV MODEL: FULL EVALUATION")
print("="*60)

# Predictions (Log Hazards)
log_hazard_train = model.predict(x_train).flatten()
log_hazard_test = model.predict(x_test).flatten()

# C-Index (Note: In standard Cox, hazard increases risk. Higher prediction = Higher Risk)
train_c = concordance_index(y_train[0], -log_hazard_train, y_train[1])
test_c = concordance_index(y_t_test, -log_hazard_test, y_e_test)

print(f"\n Concordance Index:")
print(f"   - Training: {train_c:.3f}")
print(f"   - Test:     {test_c:.3f}")

# Bootstrap CI
mean_c, ci_low, ci_high = bootstrap_concordance_deepsurv(
    model, x_test, y_t_test.values, y_e_test.values
)
print(f"   - Test (bootstrap mean ± 95% CI): {mean_c:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

# ----------------------------
# 8. Kaplan-Meier Risk Groups
# ----------------------------
# Compute Baseline Hazard to get Survival Functions
_ = model.compute_baseline_hazards()

# Predict Survival Dataframe (Index=Time, Columns=Patients)
surv_df = model.predict_surv_df(x_test)

# Risk Groups based on Median Log Hazard (Predicted Risk)
median_risk = np.median(log_hazard_test)
high_mask = log_hazard_test >= median_risk
low_mask = log_hazard_test < median_risk

plt.figure(figsize=(8, 5))
kmf = KaplanMeierFitter()

kmf.fit(y_t_test[high_mask], y_e_test[high_mask], label="High Risk")
ax = kmf.plot_survival_function()
kmf.fit(y_t_test[low_mask], y_e_test[low_mask], label="Low Risk")
kmf.plot_survival_function(ax=ax)

plt.title(f"DeepSurv Risk Groups (Test C-index = {test_c:.3f})")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('c-index-DeepSurv.png', dpi=300, bbox_inches='tight')
plt.show()

# Log-rank test
lr = logrank_test(y_t_test[high_mask], y_t_test[low_mask], y_e_test[high_mask], y_e_test[low_mask])
print(f"\n Log-rank Test: p = {lr.p_value:.4f}")

# ----------------------------
# 9. SHAP Interpretability
# ----------------------------
print("\n Computing SHAP values (DeepExplainer)...")
# We explain the underlying PyTorch network
# DeepExplainer requires a background dataset (we use a subset of train)
background = torch.tensor(x_train[:100])
e = shap.DeepExplainer(model.net, background)

# Calculate SHAP on test set
shap_values = e.shap_values(torch.tensor(x_test))

# Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, pd.DataFrame(x_test, columns=final_features), show=False)
plt.title("SHAP (Deep) summary — DeepSurv")
plt.tight_layout()
plt.savefig('shap(deep)-DeepSurv.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Done. Pipeline completed.")

# ----------------------------
# 11. CALIBRATION & BRIER SCORE
# ----------------------------
print("\n" + "=" * 60)
print("CLINICAL VALIDITY METRICS (Q1 JOURNAL REQUIREMENT)")
print("=" * 60)

eval_time = y_t_test.median()

# Get Survival Probs at median time
# surv_df index is time, columns are patients.
# We find the row index closest to eval_time
closest_time_idx = np.argmin(np.abs(surv_df.index - eval_time))
surv_prob_at_t = surv_df.iloc[closest_time_idx].values

# Calibration Plot
n_bins = 3
test_df_calib = pd.DataFrame()
test_df_calib['pred_survival'] = surv_prob_at_t
test_df_calib[duration_col] = y_t_test.values
test_df_calib[event_col] = y_e_test.values

# Binning
test_df_calib['risk_group'] = pd.qcut(test_df_calib['pred_survival'], n_bins, labels=["High Risk", "Medium Risk", "Low Risk"])

observed_survival = []
predicted_survival = []
risk_labels = []

kmf_calib = KaplanMeierFitter()

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'r--', label="Perfect Calibration")

for group in ["High Risk", "Medium Risk", "Low Risk"]:
    subset = test_df_calib[test_df_calib['risk_group'] == group]
    mean_pred = subset['pred_survival'].mean()

    kmf_calib.fit(subset[duration_col], subset[event_col])
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
plt.title(f"DeepSurv Calibration at Median Time ({eval_time:.0f} days)")
plt.xlabel("Predicted Survival Probability")
plt.ylabel("Observed Survival Probability (KM)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('survival_probability_calibration-DeepSurv.png', dpi=300, bbox_inches='tight')
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

# Risk = 1 - Survival Prob
predicted_risk = 1 - surv_prob_at_t
true_event = y_e_test.values

thresholds = np.linspace(0.01, 0.9, 100)
nb_model = calculate_net_benefit(true_event, predicted_risk, thresholds)
nb_all = calculate_net_benefit(true_event, np.ones_like(true_event), thresholds)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, nb_model, label="DeepSurv Model", color='blue', linewidth=2)
plt.plot(thresholds, nb_all, label="Treat All", color='gray', linestyle='--')
plt.axhline(y=0, color='black', linestyle='-', label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (DeepSurv)")
plt.legend()
plt.ylim(-0.05, 0.25)
plt.grid(True, alpha=0.3)
plt.savefig('decision_curv_analysis-DeepSurv.png', dpi=300, bbox_inches='tight')
plt.show()

print(" DeepSurv DCA Plot generated.")