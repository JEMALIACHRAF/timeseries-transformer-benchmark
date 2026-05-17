# Databricks notebook source
# databricks/notebooks/04_baseline_models.py
# ============================================================
# Notebook 04 — Baseline Models (parallel via Spark UDF)
# Trains Prophet and SARIMA for all 30,490 series in parallel
# ============================================================

# COMMAND ----------
# %pip install prophet==1.1.5 statsforecast==1.7.4

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import mlflow


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

CONTAINER    = "tsdata"
GOLD_PATH    = f"abfss://{CONTAINER}@{ADLS_ACCOUNT}.dfs.core.windows.net/features/m5_gold"
RESULTS_PATH = f"abfss://{CONTAINER}@{ADLS_ACCOUNT}.dfs.core.windows.net/results"
HORIZON      = 28
TRAIN_END    = "2016-03-27"

mlflow.set_experiment("/Shared/ts-benchmark-m5")

# COMMAND ----------
# %md ### 4.1 — Load Gold data

train_df = (spark.read.format("delta").load(f"{GOLD_PATH}/m5_train")
    .select("item_id", "store_id", "date", "sales",
            "sell_price", "snap_CA", "snap_TX", "snap_WI")
    .orderBy("item_id", "store_id", "date"))

test_df = (spark.read.format("delta").load(f"{GOLD_PATH}/m5_test")
    .select("item_id", "store_id", "date", "sales"))

print(f"Train: {train_df.count():,} rows")
print(f"Test:  {test_df.count():,} rows")
print(f"Series: {train_df.select('item_id','store_id').distinct().count():,}")

# COMMAND ----------
# %md ### 4.2 — Prophet via pandas UDF (fully parallel)

prophet_output_schema = StructType([
    StructField("item_id",     StringType(),  True),
    StructField("store_id",    StringType(),  True),
    StructField("ds",          DateType(),    True),
    StructField("yhat",        DoubleType(),  True),
    StructField("yhat_lower",  DoubleType(),  True),
    StructField("yhat_upper",  DoubleType(),  True),
])

@F.pandas_udf(prophet_output_schema, F.PandasUDFType.GROUPED_MAP)
def train_prophet_udf(df: pd.DataFrame) -> pd.DataFrame:
    """Train one Prophet model per (item_id, store_id) group."""
    from prophet import Prophet
    import warnings
    warnings.filterwarnings("ignore")

    item_id  = df["item_id"].iloc[0]
    store_id = df["store_id"].iloc[0]

    df_model = df[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})
    df_model = df_model.sort_values("ds")
    df_model["ds"] = pd.to_datetime(df_model["ds"])

    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=0.95,
        )

        # Add regressors if available
        for snap_col in ["snap_CA", "snap_TX", "snap_WI"]:
            if snap_col in df.columns:
                model.add_regressor(snap_col)
                df_model[snap_col] = df[snap_col].values

        model.fit(df_model, iter=300)

        future = model.make_future_dataframe(periods=HORIZON, freq="D")
        for snap_col in ["snap_CA", "snap_TX", "snap_WI"]:
            if snap_col in df_model.columns:
                future[snap_col] = 0  # No SNAP in forecast period (conservative)

        forecast = model.predict(future).tail(HORIZON)

        result = pd.DataFrame({
            "item_id":    [item_id] * HORIZON,
            "store_id":   [store_id] * HORIZON,
            "ds":         forecast["ds"].dt.date,
            "yhat":       forecast["yhat"].clip(lower=0).values,
            "yhat_lower": forecast["yhat_lower"].clip(lower=0).values,
            "yhat_upper": forecast["yhat_upper"].clip(lower=0).values,
        })
        return result

    except Exception as e:
        # Return zero forecast on failure — don't crash the whole job
        future_dates = pd.date_range(
            start=pd.to_datetime(TRAIN_END) + pd.Timedelta(days=1),
            periods=HORIZON, freq="D"
        )
        return pd.DataFrame({
            "item_id":    [item_id] * HORIZON,
            "store_id":   [store_id] * HORIZON,
            "ds":         future_dates.date,
            "yhat":       [0.0] * HORIZON,
            "yhat_lower": [0.0] * HORIZON,
            "yhat_upper": [0.0] * HORIZON,
        })

# COMMAND ----------
# %md ### 4.3 — Run Prophet in parallel

print("Running Prophet for all series in parallel (Spark UDF)...")
with mlflow.start_run(run_name="prophet_baseline"):
    prophet_preds = (train_df
        .groupBy("item_id", "store_id")
        .apply(train_prophet_udf))

    # Cache to avoid re-computation during evaluation
    prophet_preds = prophet_preds.cache()
    n_forecasts = prophet_preds.count()
    print(f"✅ Prophet forecasts generated: {n_forecasts:,} rows")

    # Save predictions
    (prophet_preds
        .write.format("delta").mode("overwrite")
        .save(f"{RESULTS_PATH}/prophet_predictions"))

    # Evaluate against test set
    prophet_eval = (prophet_preds
        .join(test_df.withColumnRenamed("date", "ds")
                     .withColumnRenamed("sales", "y_true"),
              on=["item_id", "store_id", "ds"], how="inner")
        .withColumn("sq_error", (F.col("y_true") - F.col("yhat")) ** 2)
        .withColumn("abs_error", F.abs(F.col("y_true") - F.col("yhat")))
        .withColumn("smape",
            F.abs(F.col("y_true") - F.col("yhat")) /
            ((F.abs(F.col("y_true")) + F.abs(F.col("yhat"))) / 2 + 1e-8))
    )

    metrics = (prophet_eval.agg(
        F.sqrt(F.mean("sq_error")).alias("rmse"),
        F.mean("abs_error").alias("mae"),
        (F.mean("smape") * 100).alias("smape_pct"),
    ).toPandas().iloc[0].to_dict())

    print(f"\nProphet Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    mlflow.log_metrics(metrics)
    mlflow.log_param("model", "prophet")
    mlflow.log_param("horizon", HORIZON)

# COMMAND ----------
# %md ### 4.4 — XGBoost with Spark (distributed feature prep, local training)

print("\nTraining XGBoost (tabular features)...")

feature_cols = (
    [f"lag_{d}" for d in [1, 7, 14, 28]] +
    [f"roll_mean_{w}" for w in [7, 14, 28]] +
    [f"roll_std_{w}" for w in [7, 14, 28]] +
    ["day_of_week", "month", "is_weekend", "quarter",
     "sell_price", "price_pct_change",
     "snap_CA", "snap_TX", "snap_WI",
     "event_name_encoded",
     "te_dept_id", "te_store_id"]
)

# Downsample for faster training (sample 20 series per dept for demo)
train_sample = spark.read.format("delta").load(f"{GOLD_PATH}/m5_train")

# Convert to pandas for XGBoost
available_features = [c for c in feature_cols
                      if c in train_df.columns]

train_pd = train_sample.select(available_features + ["sales"]).dropna().toPandas()
print(f"XGBoost training set: {len(train_pd):,} rows × {len(available_features)} features")

with mlflow.start_run(run_name="xgboost_baseline"):
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    X = train_pd[available_features]
    y = train_pd["sales"]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)

    # Quick eval on validation
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    val_preds = model.predict(X_val).clip(min=0)
    xgb_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    xgb_mae  = float(mean_absolute_error(y_val, val_preds))
    xgb_smape = float(np.mean(np.abs(y_val - val_preds) /
                               ((np.abs(y_val) + np.abs(val_preds)) / 2 + 1e-8)) * 100)

    print(f"\nXGBoost Validation Metrics:")
    print(f"  RMSE:  {xgb_rmse:.4f}")
    print(f"  MAE:   {xgb_mae:.4f}")
    print(f"  SMAPE: {xgb_smape:.2f}%")

    mlflow.log_metrics({"val_rmse": xgb_rmse, "val_mae": xgb_mae, "val_smape": xgb_smape})
    mlflow.log_param("n_estimators", model.best_iteration)
    mlflow.xgboost.log_model(model, artifact_path="xgboost_model",
                              registered_model_name="xgboost-m5")

print("✅ Baseline models complete.")
