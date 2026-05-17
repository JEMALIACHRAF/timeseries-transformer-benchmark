# Databricks notebook source
# databricks/notebooks/07_benchmark_results.py
# ============================================================
# Notebook 07 — Benchmark Results & Visualization (fixed v2)
# ============================================================

# COMMAND ----------
# %pip install mlflow torch neuralforecast --quiet
# %pip install --upgrade typing_extensions pyarrow
# dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
print("✅ Config OK")

# COMMAND ----------
# %md ### 7.1 — Pull all MLflow runs

client     = mlflow.MlflowClient()
experiment = client.get_experiment_by_name("/Shared/ts-benchmark-m5")
print(f"Experiment ID : {experiment.experiment_id}")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["start_time DESC"],
)
print(f"Total runs found : {len(runs)}")

# COMMAND ----------
# %md ### 7.2 — Build leaderboard

records = []
for run in runs:
    m = run.data.metrics
    p = run.data.params

    # Cherche les métriques avec différents préfixes (val_ ou test_ ou sans préfixe)
    def get_metric(keys):
        for k in keys:
            if k in m:
                return round(m[k], 4)
        return None

    rmsse = get_metric(["val_rmsse", "test_rmsse", "rmsse"])
    rmse  = get_metric(["val_rmse",  "test_rmse",  "rmse"])
    mae   = get_metric(["val_mae",   "test_mae",   "mae"])
    smape = get_metric(["val_smape", "test_smape", "smape_pct", "smape"])

    if rmse is not None:  # ne garde que les runs avec métriques
        records.append({
            "run_name":     run.info.run_name,
            "run_id":       run.info.run_id[:8],
            "rmsse":        rmsse,
            "rmse":         rmse,
            "mae":          mae,
            "smape":        smape,
            "start_time":   pd.Timestamp(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
        })

leaderboard = pd.DataFrame(records)
if "rmsse" in leaderboard.columns and leaderboard["rmsse"].notna().any():
    leaderboard = leaderboard.sort_values("rmsse", na_position="last")
elif "rmse" in leaderboard.columns:
    leaderboard = leaderboard.sort_values("rmse", na_position="last")

print("\n" + "="*70)
print("  BENCHMARK LEADERBOARD — M5 Forecasting")
print("="*70)
print(leaderboard.to_string(index=False))
print("="*70)

# COMMAND ----------
# %md ### 7.3 — Bar chart RMSE

fig = go.Figure()
colors = {
    "tft":      "#1565C0",
    "patchtst": "#0288D1",
    "prophet":  "#E53935",
    "xgboost":  "#FB8C00",
}

for _, row in leaderboard.iterrows():
    if pd.notna(row.get("rmse")):
        model_key = row["run_name"].lower().split("_")[0]
        color = colors.get(model_key, "#888888")
        fig.add_trace(go.Bar(
            name=row["run_name"],
            x=[row["run_name"]],
            y=[row["rmse"]],
            marker_color=color,
            text=f"{row['rmse']:.4f}",
            textposition="outside",
        ))

fig.update_layout(
    title="Model Benchmark — RMSE (lower is better)",
    xaxis_title="Model",
    yaxis_title="RMSE",
    showlegend=False,
    template="plotly_white",
    height=450,
)
fig.show()

# COMMAND ----------
# %md ### 7.4 — Multi-metric comparison

metrics_cols = ["rmsse", "rmse", "mae", "smape"]
radar_df = leaderboard[[c for c in ["run_name"] + metrics_cols
                         if c in leaderboard.columns]].dropna()

if len(radar_df) >= 2:
    fig2 = go.Figure()
    theta = [c.upper() for c in metrics_cols if c in radar_df.columns]

    for _, row in radar_df.iterrows():
        vals = [row[c] for c in metrics_cols if c in radar_df.columns]
        # Normalise 0-1 inversé (plus grand = meilleur)
        max_vals = radar_df[[c for c in metrics_cols if c in radar_df.columns]].max()
        norm_vals = [1 - (row[c] / (max_vals[c] + 1e-8))
                     for c in metrics_cols if c in radar_df.columns]
        norm_vals.append(norm_vals[0])

        model_key = row["run_name"].lower().split("_")[0]
        fig2.add_trace(go.Scatterpolar(
            r=norm_vals,
            theta=theta + [theta[0]],
            fill="toself",
            name=row["run_name"],
            opacity=0.6,
            line_color=colors.get(model_key, "#888888"),
        ))

    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Normalized Metrics Comparison (outer = better)",
        template="plotly_white",
        height=500,
    )
    fig2.show()

# COMMAND ----------
# %md ### 7.5 — Forecast visualization (sample series)

model_preds = {}
for model_name in ["tft", "patchtst", "prophet"]:
    try:
        path = f"{RESULTS_PATH}/{model_name}_predictions"
        df   = spark.read.format("delta").load(path).toPandas()
        model_preds[model_name] = df
        print(f"✅ Loaded {model_name}: {len(df):,} rows")
    except Exception as e:
        print(f"⚠️  {model_name} not found: {e}")

if model_preds:
    # Prend un ID commun à toutes les prédictions
    all_ids = None
    for df in model_preds.values():
        ids = set(df["unique_id"].unique() if "unique_id" in df.columns
                  else (df["item_id"] + "_" + df["store_id"]).unique())
        all_ids = ids if all_ids is None else all_ids & ids

    if all_ids:
        sample_id = list(all_ids)[0]
        print(f"\nVisualizing series: {sample_id}")

        # Charge actuals
        val_spark   = spark.read.format("delta").load(f"{GOLD_PATH}/m5_val")
        val_actuals = (val_spark
            .withColumn("unique_id", F.concat(F.col("item_id"), F.lit("_"), F.col("store_id")))
            .filter(F.col("unique_id") == sample_id)
            .select("unique_id", F.to_date("date").alias("date"), F.col("sales"))
            .toPandas())

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=pd.to_datetime(val_actuals["date"]),
            y=val_actuals["sales"],
            name="Actual", line=dict(color="black", width=2),
        ))

        palette = ["#1565C0", "#E53935", "#43A047", "#FB8C00"]
        for i, (model_name, preds_df) in enumerate(model_preds.items()):
            if "unique_id" in preds_df.columns:
                series_preds = preds_df[preds_df["unique_id"] == sample_id]
            else:
                series_preds = preds_df[
                    (preds_df["item_id"] + "_" + preds_df["store_id"]) == sample_id
                ]

            pred_col = [c for c in series_preds.columns
                        if c not in ["unique_id", "ds", "item_id", "store_id",
                                     "yhat_lower", "yhat_upper", "date"]][0]

            fig3.add_trace(go.Scatter(
                x=pd.to_datetime(series_preds["ds"] if "ds" in series_preds.columns
                                 else series_preds["date"]),
                y=series_preds[pred_col].clip(lower=0),
                name=model_name.upper(),
                line=dict(color=palette[i % len(palette)], dash="dash", width=1.5),
            ))

        fig3.update_layout(
            title=f"Forecast Comparison — {sample_id}",
            xaxis_title="Date", yaxis_title="Sales",
            template="plotly_white", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig3.show()

# COMMAND ----------
# %md ### 7.6 — Save leaderboard

leaderboard_spark = spark.createDataFrame(leaderboard)
(leaderboard_spark.write.format("delta").mode("overwrite")
 .save(f"{RESULTS_PATH}/benchmark_leaderboard"))
print(f"✅ Leaderboard sauvegardé → {RESULTS_PATH}/benchmark_leaderboard")

leaderboard.to_csv("/tmp/benchmark_leaderboard.csv", index=False)
with mlflow.start_run(run_name="benchmark_summary"):
    mlflow.log_artifact("/tmp/benchmark_leaderboard.csv", "benchmark")
    mlflow.log_param("n_models", len(leaderboard))
    if "rmse" in leaderboard.columns and leaderboard["rmse"].notna().any():
        best = leaderboard.dropna(subset=["rmse"]).iloc[0]
        mlflow.log_param("best_model", best["run_name"])
        mlflow.log_metric("best_rmse", float(best["rmse"]))
        print(f"\n🏆 Best model : {best['run_name']} (RMSE={best['rmse']:.4f})")

print("✅ Notebook 07 COMPLETE")
