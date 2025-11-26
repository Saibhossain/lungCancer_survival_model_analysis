# =============================================================================
# COMPREHENSIVE SURVIVAL ANALYSIS PIPELINE
# Models: CoxPH vs CoxNet (ElasticNet) vs RSF vs DeepSurv
# Metrics: C-Index, Bootstrap CI, KM Curves, Calibration, DCA, SHAP
# =============================================================================

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# --- 1. Libraries ---
# Lifelines (Standard Cox)
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

# Scikit-Survival (CoxNet, RSF)
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# PyCox (DeepSurv)
import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

# General ML / Stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm.auto import tqdm
import shap

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# PART 1: UTILITY FUNCTIONS
# =============================================================================

def compute_vif(df, thresh=5.0):
    """Iteratively remove features with VIF > threshold."""
    X = df.copy()
    dropped = []
    pbar = tqdm(desc="Reducing Multicollinearity (VIF)", unit="feat")
    while True:
        if X.shape[1] <= 1: break
        try:
            vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
        except:
            break
        max_vif = vif.max()
        pbar.set_postfix({"Max VIF": f"{max_vif:.1f}", "Features": X.shape[1]})
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
    """Keep features with p-value < threshold in univariate CoxPH."""
    keep = []
    for f in features:
        tmp = df[[duration_col, event_col, f]].dropna()
        if tmp.shape[0] < 10: continue
        cph = CoxPHFitter()
        try:
            cph.fit(tmp, duration_col=duration_col, event_col=event_col, show_progress=False)
            if cph.summary.loc[f, "p"] < p_threshold:
                keep.append(f)
        except:
            continue
    return keep


def get_risk_scores(model, X, model_type):
    """Unified risk score prediction (Higher Score = Higher Risk)."""
    if model_type == 'lifelines':
        return model.predict_partial_hazard(X).values.ravel()
    elif model_type == 'sksurv':
        return model.predict(X)
    elif model_type == 'pycox':
        return model.predict(X).flatten()
    return None


def get_survival_at_t(model, X, t, model_type):
    """Unified survival probability prediction at time t."""
    if model_type == 'lifelines':
        surv_df = model.predict_survival_function(X)
        idx = (np.abs(surv_df.index - t)).argmin()
        return surv_df.iloc[idx].values
    elif model_type == 'sksurv':
        surv_funcs = model.predict_survival_function(X)
        return np.array([fn(t) for fn in surv_funcs])
    elif model_type == 'pycox':
        surv_df = model.predict_surv_df(X)
        idx = (np.abs(surv_df.index - t)).argmin()
        return surv_df.iloc[idx].values
    return None


def calculate_net_benefit(y_true, y_pred_risk, thresh):
    """Calculate Net Benefit for DCA."""
    n = len(y_true)
    net_benefits = []
    for pt in thresh:
        high_risk_idx = y_pred_risk >= pt
        tp = np.sum((y_true == 1) & high_risk_idx)
        fp = np.sum((y_true == 0) & high_risk_idx)
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    return net_benefits


# =============================================================================
# PART 2: DATA LOADING & PREPROCESSING
# =============================================================================
print("--- 1. Loading & Preprocessing Data ---")
# Adjust paths as needed
radiomics_df = pd.read_csv("/Users/mdsaibhossain/code/python/survival_model_analysis/radiomics_features_422_patients_seg_rtstruct.csv").set_index("patient_id")
clinical_df = pd.read_csv("/Users/mdsaibhossain/code/python/survival_model_analysis/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv").rename(
    columns={"PatientID": "patient_id"}).set_index("patient_id")
survival_df = clinical_df[["Survival.time", "deadstatus.event"]]

df = radiomics_df.join(survival_df, how="inner").dropna(subset=["Survival.time", "deadstatus.event"])
duration_col, event_col = "Survival.time", "deadstatus.event"

# Remove non-feature columns
exclude = {duration_col, event_col, "ct_path", "seg_path", "ct_path.1", "seg_path.1"}
feature_cols = [c for c in df.columns if c not in exclude]

# Impute & Drop Zero Variance
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
df = df.drop(columns=df[feature_cols].columns[df[feature_cols].var() == 0])
feature_cols = [c for c in feature_cols if c in df.columns]
df = df.dropna()

print(f"Dataset: {len(df)} patients, {len(feature_cols)} original features")

# Scaling
X = df[feature_cols]
y_t = df[duration_col]
y_e = df[event_col]
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

# Split (Train/Test)
X_train_full, X_test, y_t_train_full, y_t_test, y_e_train_full, y_e_test = train_test_split(
    X_scaled, y_t, y_e, test_size=0.2, stratify=y_e, random_state=SEED
)

# =============================================================================
# PART 3: GLOBAL FEATURE SELECTION
# =============================================================================
print("\n--- 2. Global Feature Selection (VIF + Univariate) ---")
# 1. VIF
X_train_vif = X_train_full.copy().reset_index(drop=True)
X_vif_reduced, _ = compute_vif(X_train_vif, thresh=5.0)
vif_features = X_vif_reduced.columns.tolist()

# 2. Univariate Cox Filter
uni_df = pd.concat([X_train_vif[vif_features],
                    pd.Series(y_t_train_full.values, name=duration_col),
                    pd.Series(y_e_train_full.values, name=event_col)], axis=1)
uni_keep = univariate_cox_filter(uni_df, duration_col, event_col, vif_features, p_threshold=0.20)

final_features = uni_keep if len(uni_keep) > 0 else vif_features[:10]
print(f"Final Features Selected ({len(final_features)}): {final_features}")

# Update Datasets with Final Features
X_train = X_train_full[final_features]
X_test = X_test[final_features]

# =============================================================================
# PART 4: MODEL TRAINING
# =============================================================================
print("\n--- 3. Training 4 Models ---")
models = {}

# --- A. CoxPH (Lifelines) ---
print("Fitting CoxPH...")
train_df_cox = pd.concat([X_train, y_t_train_full, y_e_train_full], axis=1)
cph = CoxPHFitter(penalizer=0.1)
cph.fit(train_df_cox, duration_col=duration_col, event_col=event_col)
models['CoxPH'] = (cph, 'lifelines', X_train)

# --- B. CoxNet (Scikit-Survival) ---
print("Fitting CoxNet (ElasticNet)...")
y_train_sk = Surv.from_arrays(event=y_e_train_full.astype(bool), time=y_t_train_full)
coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01)
coxnet.fit(X_train, y_train_sk)
models['CoxNet'] = (coxnet, 'sksurv', X_train)

# --- C. RSF (Random Survival Forest) ---
print("Fitting Random Survival Forest...")
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=SEED)
rsf.fit(X_train, y_train_sk)
models['RSF'] = (rsf, 'sksurv', X_train)

# --- D. DeepSurv (PyCox) ---
print("Fitting DeepSurv...")
# Split Train into Train/Val for Deep Learning
X_t_deep, X_v_deep, yt_t_deep, yt_v_deep, ye_t_deep, ye_v_deep = train_test_split(
    X_train, y_t_train_full, y_e_train_full, test_size=0.2, stratify=y_e_train_full, random_state=SEED
)
x_train_t = X_t_deep.values.astype('float32')
x_val_t = X_v_deep.values.astype('float32')
x_test_t = X_test.values.astype('float32')
y_train_t = (yt_t_deep.values.astype('float32'), ye_t_deep.values.astype('float32'))
y_val_t = (yt_v_deep.values.astype('float32'), ye_v_deep.values.astype('float32'))
val_data_t = tt.tuplefy(x_val_t, y_val_t)

net = tt.practical.MLPVanilla(in_features=x_train_t.shape[1], num_nodes=[32, 32], out_features=1, batch_norm=True,
                              dropout=0.1)
deepsurv = CoxPH(net, tt.optim.Adam)
lrfinder = deepsurv.lr_finder(x_train_t, y_train_t, batch_size=64, tolerance=10)
deepsurv.optimizer.set_lr(lrfinder.get_best_lr())
deepsurv.fit(x_train_t, y_train_t, batch_size=64, epochs=100, callbacks=[tt.callbacks.EarlyStopping()],
             val_data=val_data_t, verbose=False)
_ = deepsurv.compute_baseline_hazards()
models['DeepSurv'] = (deepsurv, 'pycox', x_train_t)

# =============================================================================
# PART 5: EVALUATION LOOP (Identical for ALL models)
# =============================================================================
print("\n--- 4. Evaluating All Models ---")
comparison_results = []
eval_time = y_t_test.median()  # Median follow-up time for Calibration/DCA

for name, (model, m_type, X_train_data) in models.items():
    print(f"\n>>> Evaluating {name}...")

    # Select correct input format
    X_input = x_test_t if name == 'DeepSurv' else X_test
    X_train_shap = x_train_t if name == 'DeepSurv' else X_train_data

    # 1. Predictions
    risk_scores = get_risk_scores(model, X_input, m_type)
    surv_probs = get_survival_at_t(model, X_input, eval_time, m_type)

    # 2. C-Index
    c_index = concordance_index(y_t_test, -risk_scores, y_e_test)

    # 3. Bootstrap C-Index
    boot_scores = []
    for _ in range(100):
        idx = np.random.choice(len(y_t_test), len(y_t_test), replace=True)
        if len(np.unique(y_e_test.iloc[idx])) < 2: continue
        boot_scores.append(concordance_index(y_t_test.iloc[idx], -risk_scores[idx], y_e_test.iloc[idx]))
    c_lower, c_upper = np.percentile(boot_scores, 2.5), np.percentile(boot_scores, 97.5)

    comparison_results.append({
        "Model": name, "C-Index": c_index, "95% CI Lower": c_lower, "95% CI Upper": c_upper
    })

    # 4. KM Curves (High vs Low Risk)
    median_risk = np.median(risk_scores)
    high_mask = risk_scores >= median_risk

    plt.figure(figsize=(6, 4))
    kmf = KaplanMeierFitter()
    kmf.fit(y_t_test[high_mask], y_e_test[high_mask], label="High Risk")
    ax = kmf.plot_survival_function()
    kmf.fit(y_t_test[~high_mask], y_e_test[~high_mask], label="Low Risk")
    kmf.plot_survival_function(ax=ax)
    lr_p = logrank_test(y_t_test[high_mask], y_t_test[~high_mask], y_e_test[high_mask], y_e_test[~high_mask]).p_value
    plt.title(f"{name} Risk Groups (p={lr_p:.4f})")
    plt.savefig(f"KM_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Calibration Plot
    risk_groups = pd.qcut(surv_probs, 3, labels=["High Risk", "Medium Risk", "Low Risk"])  # High Surv = Low Risk
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'r--', label="Perfect")
    obs, pred = [], []
    for group in ["High Risk", "Medium Risk", "Low Risk"]:
        mask = (risk_groups == group)
        if not any(mask): continue
        mean_pred = np.mean(surv_probs[mask])
        kmf_c = KaplanMeierFitter()
        kmf_c.fit(y_t_test[mask], y_e_test[mask])
        try:
            # Interpolate if time point missing
            idx = (np.abs(kmf_c.survival_function_.index - eval_time)).argmin()
            mean_obs = kmf_c.survival_function_.iloc[idx].values[0]

            # CI
            ci = kmf_c.confidence_interval_survival_function_
            ci_idx = (np.abs(ci.index - eval_time)).argmin()
            yerr = [[mean_obs - ci.iloc[ci_idx, 0]], [ci.iloc[ci_idx, 1] - mean_obs]]
            plt.errorbar(mean_pred, mean_obs, yerr=yerr, fmt='o', capsize=5, label=group)
            obs.append(mean_obs);
            pred.append(mean_pred)
        except:
            pass
    plt.plot(pred, obs, 'b-', alpha=0.5)
    plt.title(f"{name} Calibration (Median Time)")
    plt.xlabel("Predicted Survival");
    plt.ylabel("Observed Survival")
    plt.legend();
    plt.savefig(f"Calibration_{name}.png", dpi=300, bbox_inches='tight');
    plt.close()

    # 6. DCA (Decision Curve Analysis)
    pred_risk_dca = 1 - surv_probs
    thresholds = np.linspace(0.01, 0.9, 100)
    nb_model = calculate_net_benefit(y_e_test.values, pred_risk_dca, thresholds)
    nb_all = calculate_net_benefit(y_e_test.values, np.ones_like(y_e_test.values), thresholds)
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, nb_model, label=name, color='blue')
    plt.plot(thresholds, nb_all, label="Treat All", color='gray', linestyle='--')
    plt.axhline(y=0, color='black')
    plt.title(f"{name} DCA");
    plt.legend();
    plt.ylim(-0.05, 0.25)
    plt.savefig(f"DCA_{name}.png", dpi=300, bbox_inches='tight');
    plt.close()

    # 7. SHAP (Feature Importance)
    print(f"   Computing SHAP for {name}...")
    plt.figure()
    try:
        if name == 'CoxPH':
            # Linear Explainer (use params)
            explainer = shap.Explainer(model.predict_partial_hazard, X_train_shap)
            shap_values = explainer(X_test)
            shap.plots.beeswarm(shap_values, show=False)

        elif name == 'CoxNet':
            # Linear Explainer (compatible with sksurv linear models)
            explainer = shap.LinearExplainer(model, X_train_shap)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, show=False)

        elif name == 'RSF':
            # Kernel Explainer (robust fallback for sksurv trees)
            # Use small background summary (kmeans) for speed
            bg_summary = shap.kmeans(X_train_shap, 10)
            explainer = shap.KernelExplainer(model.predict, bg_summary)
            # Run on small subset of test for speed, or full if valid
            shap_values = explainer.shap_values(X_test.iloc[:50])  # Limit to 50 test samples for speed
            shap.summary_plot(shap_values, X_test.iloc[:50], show=False)

        elif name == 'DeepSurv':
            # Deep Explainer
            bg = torch.tensor(X_train_shap[:100])
            explainer = shap.DeepExplainer(model.net, bg)
            shap_values = explainer.shap_values(torch.tensor(X_input))
            if isinstance(shap_values, list): shap_values = shap_values[0]
            shap.summary_plot(shap_values, pd.DataFrame(X_input, columns=final_features), show=False)

        plt.title(f"{name} SHAP")
        plt.savefig(f"SHAP_{name}.png", bbox_inches='tight')
    except Exception as e:
        print(f"   SHAP Error for {name}: {e}")
    plt.close()

# =============================================================================
# PART 6: FINAL COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
df_res = pd.DataFrame(comparison_results).set_index("Model").sort_values("C-Index", ascending=False)
print(df_res)
df_res.to_csv("Final_Model_Comparison.csv")
print("\nProcessing Complete. All plots and CSVs saved.")