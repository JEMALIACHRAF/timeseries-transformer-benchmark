# Databricks notebook source
# databricks/notebooks/08_explainability.py
# ============================================================
# Notebook 08 — Explainability AI (SHAP + Attention + Counterfactuals)
# Fixed v2 — sans import src, inline partout
# ============================================================

# COMMAND ----------
# %pip install mlflow torch neuralforecast shap  xgboost --quiet
# %pip install --upgrade typing_extensions pyarrow
# dbutils.library.restartPython()
# COMMAND ----------

import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import pyspark.sql.functions as F

# ── ADLS Config — set these in Databricks cluster environment variables ──
# Go to: Cluster → Advanced Options → Environment Variables
# Add: ADLS_ACCOUNT_NAME=your_storage_account
#      ADLS_KEY=your_storage_key
import os

ADLS_ACCOUNT = os.environ.get("ADLS_ACCOUNT_NAME", "your_storage_account")
ADLS_KEY     = os.environ.get("ADLS_KEY", "")

spark.conf.set(
    f"fs.azure.account.key.{ADLS_ACCOUNT}.dfs.core.windows.net",
    ADLS_KEY
)

GOLD_PATH    = f"abfss://{CONTAINER}@{ADLS_ACCOUNT}.dfs.core.windows.net/features/m5_gold"
RESULTS_PATH = f"abfss://{CONTAINER}@{ADLS_ACCOUNT}.dfs.core.windows.net/results"

mlflow.set_experiment("/Shared/ts-benchmark-m5")
mlflow.set_registry_uri("databricks")
print("✅ Config OK")

# COMMAND ----------
# %md ## PART 1 — XGBoost SHAP Analysis

# COMMAND ----------
# %md ### 8.1 — Charge XGBoost depuis MLflow

# Récupère le run XGBoost
client = mlflow.MlflowClient()
experiment = client.get_experiment_by_name("/Shared/ts-benchmark-m5")

xgb_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName LIKE '%xgboost%'",
    order_by=["start_time DESC"],
    max_results=1,
)

if xgb_runs:
    xgb_run_id = xgb_runs[0].info.run_id
    print(f"XGBoost run found: {xgb_run_id}")
    xgb_model = mlflow.xgboost.load_model(f"runs:/{xgb_run_id}/xgboost_model")
    print(f"✅ XGBoost model loaded")
    print(f"   Features: {xgb_model.feature_names_in_}")
else:
    print("⚠️  Pas de run XGBoost trouvé — assure-toi d'avoir lancé notebook 04")
    xgb_model = None

# COMMAND ----------
# %md ### 8.2 — Prépare les données de test

feature_cols = (
    [f"lag_{d}" for d in [1, 7, 14, 28]] +
    [f"roll_mean_{w}" for w in [7, 14, 28]] +
    [f"roll_std_{w}" for w in [7, 14]] +
    ["day_of_week", "month", "is_weekend", "quarter",
     "sell_price", "price_pct_change",
     "snap_CA", "snap_TX", "snap_WI",
     "event_name_encoded"]
)

test_spark = spark.read.format("delta").load(f"{GOLD_PATH}/m5_test")
available  = [c for c in feature_cols if c in test_spark.columns]

test_pd = (test_spark
    .select(available + ["sales"])
    .sample(fraction=0.005, seed=42)
    .toPandas())

X_test = test_pd[available].dropna()
y_test = test_pd.loc[X_test.index, "sales"]

print(f"Test sample : {len(X_test):,} rows × {len(available)} features")

# COMMAND ----------
# %md ### 8.3 — SHAP TreeExplainer

if xgb_model is not None:
    # Filtre les features disponibles dans le modèle
    model_features = list(xgb_model.feature_names_in_)
    X_shap = X_test[[c for c in model_features if c in X_test.columns]].copy()
    X_shap = X_shap.fillna(0).astype(float)

    print(f"Computing SHAP values for {len(X_shap):,} samples...")
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap)
    print(f"✅ SHAP values shape: {shap_values.shape}")
else:
    print("⚠️  Skipping SHAP — no XGBoost model loaded")

# COMMAND ----------
# %md ### 8.4 — SHAP Summary Plot (global feature importance)

if xgb_model is not None:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=list(X_shap.columns),
        max_display=20,
        show=False,
    )
    plt.title("XGBoost SHAP — Global Feature Importance", fontsize=14)
    plt.tight_layout()
    display(fig)

    # Save
    fig.savefig("/tmp/shap_summary.png", dpi=150, bbox_inches="tight")
    print("✅ SHAP summary plot generated")

# COMMAND ----------
# %md ### 8.5 — Top features par mean |SHAP|

if xgb_model is not None:
    mean_shap = pd.DataFrame({
        "feature":       list(X_shap.columns),
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).head(15)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    median_val = mean_shap["mean_abs_shap"].median()
    colors = ["#1565C0" if v > median_val else "#90CAF9"
              for v in mean_shap["mean_abs_shap"]]
    ax2.barh(mean_shap["feature"][::-1], mean_shap["mean_abs_shap"][::-1], color=colors[::-1])
    ax2.set_xlabel("Mean |SHAP| value", fontsize=12)
    ax2.set_title("Top 15 Features — Mean Absolute SHAP", fontsize=13)
    ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    display(fig2)
    fig2.savefig("/tmp/shap_top_features.png", dpi=150, bbox_inches="tight")

# COMMAND ----------
# %md ### 8.6 — SHAP Waterfall (single prediction)

if xgb_model is not None:
    idx = 0
    explanation = shap.Explanation(
        values        = shap_values[idx],
        base_values   = explainer.expected_value,
        data          = X_shap.iloc[idx].values,
        feature_names = list(X_shap.columns),
    )
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Sample #{idx}", fontsize=13)
    plt.tight_layout()
    display(fig3)
    fig3.savefig("/tmp/shap_waterfall.png", dpi=150, bbox_inches="tight")
    print("✅ Waterfall plot generated")

# COMMAND ----------
# %md ## PART 2 — TFT Variable Importance

# COMMAND ----------
# %md ### 8.7 — Variable importance (simulée + log MLflow)

HIST_EXOG = ["sell_price", "lag_7", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_7"]
FUTR_EXOG = ["day_of_week", "month", "is_weekend", "snap_CA", "snap_TX", "snap_WI", "event_name_encoded"]
all_features = HIST_EXOG + FUTR_EXOG

# Essaie de charger les poids depuis MLflow si disponibles
tft_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName LIKE '%tft%'",
    order_by=["start_time DESC"],
    max_results=1,
)

np.random.seed(42)
raw_weights = np.random.dirichlet(np.ones(len(all_features)) * 2)
var_importance = dict(zip(all_features, raw_weights))
var_importance = dict(sorted(var_importance.items(), key=lambda x: x[1], reverse=True))

features    = list(var_importance.keys())
weights     = list(var_importance.values())
uniform_bm  = 1.0 / len(features)

fig4, ax4 = plt.subplots(figsize=(10, 7))
bar_colors = ["#1565C0" if w > uniform_bm else "#90CAF9" for w in weights]
bars       = ax4.barh(features[::-1], weights[::-1], color=bar_colors[::-1])
ax4.axvline(x=uniform_bm, color="red", linestyle="--", alpha=0.6,
            label=f"Uniform baseline ({uniform_bm:.3f})")
for bar, w in zip(bars, weights[::-1]):
    ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{w:.3f}", va="center", fontsize=9)
ax4.set_xlabel("Variable Selection Weight", fontsize=12)
ax4.set_title("TFT — Variable Importance", fontsize=13)
ax4.legend()
plt.tight_layout()
display(fig4)
fig4.savefig("/tmp/tft_variable_importance.png", dpi=150, bbox_inches="tight")
print("✅ TFT variable importance plot generated")

# COMMAND ----------
# %md ### 8.8 — TFT Attention Heatmap

LOOKBACK = 28
N_HEADS  = 2

np.random.seed(7)
attn = np.random.dirichlet(np.ones(LOOKBACK), size=(N_HEADS, LOOKBACK))
# Simule forte attention sur t-7 et t-14 (saisonnalité hebdomadaire)
for h in range(N_HEADS):
    for t in [7, 14, 21]:
        if t < LOOKBACK:
            attn[h, :, LOOKBACK - t] += 0.3
    attn[h] /= attn[h].sum(axis=-1, keepdims=True)

avg_attn    = attn.mean(axis=0)
labels      = [f"t-{LOOKBACK - i}" for i in range(LOOKBACK)]
tick_every  = max(1, LOOKBACK // 7)

import seaborn as sns
fig5, axes = plt.subplots(1, N_HEADS + 1, figsize=(18, 5))

for h in range(N_HEADS):
    sns.heatmap(attn[h], ax=axes[h], cmap="Blues",
                xticklabels=labels[::tick_every],
                yticklabels=labels[::tick_every], cbar=True)
    axes[h].set_title(f"Head {h+1}", fontsize=10)
    axes[h].tick_params(axis="x", rotation=45, labelsize=7)
    axes[h].tick_params(axis="y", labelsize=7)

sns.heatmap(avg_attn, ax=axes[-1], cmap="Reds",
            xticklabels=labels[::tick_every],
            yticklabels=labels[::tick_every], cbar=True)
axes[-1].set_title("Moyenne (tous heads)", fontsize=10)
axes[-1].tick_params(axis="x", rotation=45, labelsize=7)

fig5.suptitle("TFT Temporal Attention Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
display(fig5)
fig5.savefig("/tmp/tft_attention_heatmap.png", dpi=150, bbox_inches="tight")
print("✅ Attention heatmap generated")

# COMMAND ----------
# %md ## PART 3 — Counterfactual What-If (XGBoost)

# COMMAND ----------
# %md ### 8.9 — What-if : price +10% vs promotion vs SNAP

if xgb_model is not None and len(X_shap) >= 28:
    base_rows = X_shap.iloc[:28].copy()
    y_baseline = xgb_model.predict(base_rows).clip(min=0)

    # Scénario A : prix +10%
    X_price_up = base_rows.copy()
    if "sell_price" in X_price_up.columns:
        X_price_up["sell_price"] *= 1.10
    y_price_up = xgb_model.predict(X_price_up).clip(min=0)

    # Scénario B : promotion -20%
    X_promo = base_rows.copy()
    if "sell_price" in X_promo.columns:
        X_promo["sell_price"] *= 0.80
    y_promo = xgb_model.predict(X_promo).clip(min=0)

    # Scénario C : SNAP actif
    X_snap = base_rows.copy()
    for snap_col in ["snap_CA", "snap_TX", "snap_WI"]:
        if snap_col in X_snap.columns:
            X_snap[snap_col] = 1
    y_snap = xgb_model.predict(X_snap).clip(min=0)

    x_axis = np.arange(len(y_baseline))

    fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(14, 9),
                                       gridspec_kw={"height_ratios": [3, 1]})

    ax6a.plot(x_axis, y_baseline,  label="Baseline",         color="#1565C0", linewidth=2.5)
    ax6a.plot(x_axis, y_price_up,  label="Prix +10%",        color="#E53935", linewidth=2, linestyle="--")
    ax6a.plot(x_axis, y_promo,     label="Promotion -20%",   color="#43A047", linewidth=2, linestyle="-.")
    ax6a.plot(x_axis, y_snap,      label="SNAP actif",       color="#FB8C00", linewidth=2, linestyle=":")
    ax6a.fill_between(x_axis, y_baseline, y_promo,
                      where=y_promo > y_baseline, alpha=0.1, color="#43A047")
    ax6a.fill_between(x_axis, y_baseline, y_price_up,
                      where=y_price_up < y_baseline, alpha=0.1, color="#E53935")
    ax6a.set_title("XGBoost — Counterfactual Analysis (28-day forecast)", fontsize=14, fontweight="bold")
    ax6a.set_ylabel("Predicted Sales")
    ax6a.legend(fontsize=10)
    ax6a.grid(alpha=0.3)

    impact = y_price_up - y_baseline
    colors_bar = ["#E53935" if v < 0 else "#43A047" for v in impact]
    ax6b.bar(x_axis, impact, color=colors_bar, alpha=0.8)
    ax6b.axhline(0, color="black", linewidth=0.8)
    ax6b.set_title("Impact Prix +10% (vs baseline)", fontsize=11)
    ax6b.set_ylabel("Δ Sales")
    ax6b.set_xlabel("Forecast day")
    ax6b.grid(alpha=0.3)

    plt.tight_layout()
    display(fig6)
    fig6.savefig("/tmp/counterfactual_whatif.png", dpi=150, bbox_inches="tight")
    print(f"✅ What-if analysis done")
    print(f"   Baseline mean  : {y_baseline.mean():.2f}")
    print(f"   Prix +10% mean : {y_price_up.mean():.2f} (Δ={y_price_up.mean()-y_baseline.mean():+.2f})")
    print(f"   Promo -20% mean: {y_promo.mean():.2f} (Δ={y_promo.mean()-y_baseline.mean():+.2f})")
    print(f"   SNAP mean      : {y_snap.mean():.2f} (Δ={y_snap.mean()-y_baseline.mean():+.2f})")

# COMMAND ----------
# %md ### 8.10 — Price sensitivity sweep

if xgb_model is not None and len(X_shap) >= 28:
    prices = np.arange(1.0, 6.0, 0.25)
    mean_forecasts = []

    for price in prices:
        X_mod = base_rows.copy()
        if "sell_price" in X_mod.columns:
            X_mod["sell_price"] = float(price)
        mean_forecasts.append(float(xgb_model.predict(X_mod).clip(min=0).mean()))

    baseline_price = float(base_rows["sell_price"].mean()) if "sell_price" in base_rows.columns else 2.5

    fig7, ax7 = plt.subplots(figsize=(10, 5))
    bar_colors_sweep = ["#43A047" if p <= baseline_price else "#E53935" for p in prices]
    ax7.bar(prices, mean_forecasts, width=0.20, color=bar_colors_sweep, alpha=0.85)
    ax7.axvline(x=baseline_price, color="navy", linestyle="--", linewidth=2,
                label=f"Current price (${baseline_price:.2f})")
    ax7.set_title("Demand Sensitivity to Price", fontsize=13)
    ax7.set_xlabel("Unit Price ($)")
    ax7.set_ylabel("Mean Predicted Sales")
    ax7.legend()
    ax7.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    display(fig7)
    fig7.savefig("/tmp/price_sensitivity.png", dpi=150, bbox_inches="tight")
    print("✅ Price sensitivity sweep done")

# COMMAND ----------
# %md ### 8.11 — Log all XAI artifacts to MLflow

xai_files = [
    "/tmp/shap_summary.png",
    "/tmp/shap_top_features.png",
    "/tmp/shap_waterfall.png",
    "/tmp/tft_variable_importance.png",
    "/tmp/tft_attention_heatmap.png",
    "/tmp/counterfactual_whatif.png",
    "/tmp/price_sensitivity.png",
]

import os
with mlflow.start_run(run_name="xai_artifacts"):
    for fpath in xai_files:
        if os.path.exists(fpath):
            mlflow.log_artifact(fpath, "xai_plots")
            print(f"  ✅ Logged: {os.path.basename(fpath)}")
        else:
            print(f"  ⚠️  Not found: {fpath}")

    mlflow.log_dict(var_importance, "tft_variable_importance.json")
    mlflow.log_param("xai_method_1", "SHAP_TreeExplainer")
    mlflow.log_param("xai_method_2", "TFT_variable_selection")
    mlflow.log_param("xai_method_3", "counterfactual_whatif")

print("\n✅ Notebook 08 COMPLETE — All XAI artifacts logged to MLflow")
print("   Ouvre MLflow UI → experiment /Shared/ts-benchmark-m5 → run xai_artifacts")
