# Databricks notebook source
# databricks/notebooks/05_deep_learning_tft.py
# ============================================================
# Notebook 05 — TFT Training (CPU compatible, fully fixed v2)
# ============================================================

# COMMAND ----------
# %pip install mlflow torch --quiet
# %pip install --upgrade typing_extensions
# dbutils.library.restartPython()
# COMMAND ----------

import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
import torch
import inspect
import pyspark.sql.functions as F
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT

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

HORIZON = 28
FREQ    = "D"
SEED    = 42

mlflow.set_experiment("/Shared/ts-benchmark-m5")
mlflow.set_registry_uri("databricks")

print(f"GPU available : {torch.cuda.is_available()}")
print(f"Mode          : {'GPU' if torch.cuda.is_available() else 'CPU'}")

# COMMAND ----------
# %md ### Vérifie les vrais paramètres TFT

params = inspect.signature(TFT.__init__).parameters
print("TFT accepted params:")
for name, param in params.items():
    if name != "self":
        print(f"  {name}: {param.default}")

# COMMAND ----------
# %md ### 5.1 — Covariables

HIST_EXOG = ["sell_price", "lag_7", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_7"]
FUTR_EXOG = ["day_of_week", "month", "is_weekend", "snap_CA", "snap_TX", "snap_WI", "event_name_encoded"]
N_SERIES  = 1000
MIN_STEPS = HORIZON * 3 + 10


def load_and_clean(gold_path, split, n_series=200, seed=42):
    spark_df = spark.read.format("delta").load(f"{gold_path}/m5_{split}")
    all_cols = [c for c in ["item_id", "store_id", "date", "sales"] + HIST_EXOG + FUTR_EXOG
                if c in spark_df.columns]

    sample_ids = (spark_df.select("item_id", "store_id").distinct()
                .sample(fraction=0.1, seed=seed).limit(n_series))

    pdf = (spark_df.join(sample_ids, on=["item_id", "store_id"], how="inner")
           .select(all_cols).toPandas())

    pdf["unique_id"] = pdf["item_id"] + "_" + pdf["store_id"]
    pdf["ds"] = pd.to_datetime(pdf["date"])
    pdf["y"]  = pdf["sales"].clip(lower=0).fillna(0)

    for col in HIST_EXOG:
        if col in pdf.columns:
            pdf[col] = pdf.groupby("unique_id")[col].transform(
                lambda x: x.ffill().bfill().fillna(0))

    for col in FUTR_EXOG:
        if col in pdf.columns:
            pdf[col] = pdf[col].fillna(0).astype(int)

    pdf = pdf.drop(columns=[c for c in ["item_id", "store_id", "date", "sales"]
                             if c in pdf.columns])
    nulls = pdf[HIST_EXOG + FUTR_EXOG].isnull().sum().sum()
    print(f"  [{split}] {len(pdf):,} rows | {pdf['unique_id'].nunique()} series | NaN: {nulls}")
    return pdf


print("Loading data...")
train_df = load_and_clean(GOLD_PATH, "train", N_SERIES, SEED)
val_df   = load_and_clean(GOLD_PATH, "val",   N_SERIES, SEED)

# COMMAND ----------
# %md ### 5.2 — Filtre + combine train+val

train_val_df = pd.concat([train_df, val_df], ignore_index=True)
train_val_df = train_val_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

series_len = train_val_df.groupby("unique_id")["ds"].count()
valid_ids  = series_len[series_len >= MIN_STEPS].index.tolist()

train_val_filtered = train_val_df[train_val_df["unique_id"].isin(valid_ids)].copy()
val_size = int(val_df[val_df["unique_id"].isin(valid_ids)]
               .groupby("unique_id")["ds"].nunique().min())

print(f"Valid series   : {len(valid_ids)}")
print(f"Train+val rows : {len(train_val_filtered):,}")
print(f"Val size       : {val_size}")
print(f"Min series len : {train_val_filtered.groupby('unique_id')['ds'].count().min()}")

# COMMAND ----------
# %md ### 5.3 — TFT Config (paramètres validés pour cette version)

# Paramètres réels vérifiés via inspect:
# n_head, n_rnn_layers, start_padding_enabled
tft_config = dict(
    h                         = HORIZON,
    input_size                = HORIZON,

    hidden_size               = 64,
    n_head                    = 2,

    attn_dropout              = 0.1,
    dropout                   = 0.1,

    learning_rate             = 1e-3,

    batch_size                = 64,
    windows_batch_size        = 512,

    max_steps                 = 1000,
    val_check_steps           = 50,
    early_stop_patience_steps = 10,

    hist_exog_list            = [c for c in HIST_EXOG if c in train_val_filtered.columns],
    futr_exog_list            = [c for c in FUTR_EXOG if c in train_val_filtered.columns],

    accelerator               = "gpu" if torch.cuda.is_available() else "cpu",
    devices                   = 1,

    enable_progress_bar       = True,
    random_seed               = SEED,

    start_padding_enabled     = True,
)

print("TFT config OK:")
for k, v in tft_config.items():
    print(f"  {k}: {v}")

# COMMAND ----------
# %md ### 5.4 — Train TFT

print(f"\nStarting TFT — {len(valid_ids)} series, {tft_config['max_steps']} steps...")

with mlflow.start_run(run_name="tft_v1") as run:
    tft_run_id = run.info.run_id
    mlflow.log_params({k: str(v) for k, v in tft_config.items()})
    mlflow.log_param("n_series", len(valid_ids))
    mlflow.log_param("mode", "GPU" if torch.cuda.is_available() else "CPU")

    model = TFT(**tft_config)
    nf    = NeuralForecast(models=[model], freq=FREQ)
    nf.fit(df=train_val_filtered, val_size=val_size)

    print(f"✅ TFT training complete! Run ID: {tft_run_id}")

# COMMAND ----------
# %md ### 5.5 — Predict

# Dernière date observée par série
last_dates = (
    train_val_filtered
    .groupby("unique_id")["ds"]
    .max()
    .reset_index()
)

# Construction future dataframe
future_rows = []

for _, row in last_dates.iterrows():
    uid = row["unique_id"]
    last_date = row["ds"]

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=HORIZON,
        freq=FREQ
    )

    tmp = pd.DataFrame({
        "unique_id": uid,
        "ds": future_dates
    })

    future_rows.append(tmp)

futr_base = pd.concat(future_rows, ignore_index=True)

# Future exogenous features
futr_base["day_of_week"]        = futr_base["ds"].dt.dayofweek.astype(int)
futr_base["month"]              = futr_base["ds"].dt.month.astype(int)
futr_base["is_weekend"]         = (futr_base["ds"].dt.dayofweek >= 5).astype(int)

futr_base["snap_CA"]            = 0
futr_base["snap_TX"]            = 0
futr_base["snap_WI"]            = 0

futr_base["event_name_encoded"] = -1

print(f"Future DF shape: {futr_base.shape}")

# Predict
predictions = nf.predict(futr_df=futr_base)

# Fix anciennes versions NeuralForecast
if "unique_id" not in predictions.columns:
    predictions = predictions.reset_index()

predictions["TFT"] = predictions["TFT"].clip(lower=0)
predictions["ds"]  = pd.to_datetime(predictions["ds"])

print(predictions.columns)
print(predictions.head())

# COMMAND ----------
# %md ### 5.6 — Evaluate sur Val

model_ids   = predictions["unique_id"].unique().tolist()
pred_dates  = set(predictions["ds"].dt.date.unique())

val_spark   = spark.read.format("delta").load(f"{GOLD_PATH}/m5_val")
val_actuals = (val_spark
    .withColumn("unique_id", F.concat(F.col("item_id"), F.lit("_"), F.col("store_id")))
    .filter(F.col("unique_id").isin(model_ids))
    .select("unique_id", F.to_date("date").alias("ds"), F.col("sales").alias("y"))
    .toPandas())
val_actuals["ds"] = pd.to_datetime(val_actuals["ds"])

eval_df = predictions.merge(val_actuals, on=["unique_id", "ds"], how="inner")
print(f"Eval rows matched : {len(eval_df):,}")

if len(eval_df) == 0:
    print("⚠️  Aucun match — vérifie les dates")
    print(f"Pred dates  : {predictions['ds'].min()} → {predictions['ds'].max()}")
    print(f"Val  dates  : {val_actuals['ds'].min()} → {val_actuals['ds'].max()}")
else:
    pred_col = "TFT-median" if "TFT-median" in eval_df.columns else "TFT"
    y_true   = eval_df["y"].values
    y_pred   = eval_df[pred_col].clip(lower=0).values
    y_train  = train_val_filtered["y"].values

    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    smape = float(np.mean(np.abs(y_true - y_pred) /
                  ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8)) * 100)
    scale = float(np.mean(np.diff(y_train) ** 2)) + 1e-8
    rmsse = float(np.sqrt(np.mean((y_true - y_pred) ** 2) / scale))

    metrics = {"val_rmsse": rmsse, "val_rmse": rmse, "val_mae": mae, "val_smape": smape}

    print(f"\n📊 TFT Val Metrics:")
    for k, v in metrics.items():
        print(f"  {k.upper()} : {v:.4f}")

    with mlflow.start_run(run_id=tft_run_id):
        mlflow.log_metrics(metrics)
    print("✅ Métriques loggées dans MLflow")

# COMMAND ----------
# %md ### 5.7 — Save predictions + model

# Fix schema types
predictions["unique_id"] = predictions["unique_id"].astype(str)
predictions["TFT"]       = predictions["TFT"].astype(float)
predictions["ds"]        = pd.to_datetime(predictions["ds"])

# Spark DF
pred_spark = spark.createDataFrame(predictions)

# Save
(pred_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{RESULTS_PATH}/tft_predictions"))

print(f"✅ Prédictions sauvegardées → {RESULTS_PATH}/tft_predictions")

with mlflow.start_run(run_id=tft_run_id):
    mlflow.pytorch.log_model(nf.models[0], artifact_path="tft_model")
    model_uri = f"runs:/{tft_run_id}/tft_model"
    try:
        mlflow.register_model(model_uri=model_uri, name="tft-m5")
        print("✅ TFT model registered: tft-m5")
    except Exception as e:
        print(f"⚠️  Registration skipped: {e}")
        print(f"   URI: {model_uri}")

print(f"\n✅ Notebook 05 COMPLETE")
print(f"   Run ID : {tft_run_id}")
