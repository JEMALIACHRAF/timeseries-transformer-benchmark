# Databricks notebook source
# databricks/notebooks/02_feature_engineering.py
# ============================================================
# Notebook 02 — Distributed Feature Engineering (PySpark)
# Reads from Silver Delta layer, writes Gold feature layer
# ============================================================

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
import os
ADLS_ACCOUNT = os.environ.get("ADLS_ACCOUNT_NAME", "your_storage_account")
ADLS_KEY     = os.environ.get("ADLS_KEY", "")
spark.conf.set(
    f"fs.azure.account.key.{ADLS_ACCOUNT}.dfs.core.windows.net",
    ADLS_KEY
)
CONTAINER      = "tsdata"
SILVER_PATH    = f"abfss://{CONTAINER}@{ADLS_ACCOUNT}.dfs.core.windows.net/processed/m5_silver"
GOLD_PATH      = f"abfss://{CONTAINER}@{ADLS_ACCOUNT}.dfs.core.windows.net/features/m5_gold"

# COMMAND ----------
# %md ### 2.1 — Load Silver data

df = spark.read.format("delta").load(SILVER_PATH)
df = df.withColumn("date", F.to_date("date"))
df = df.withColumn("sales", F.col("sales").cast("double"))

print(f"Silver rows: {df.count():,}")
df.printSchema()

# COMMAND ----------
# %md ### 2.2 — Lag features

series_window = Window.partitionBy("item_id", "store_id").orderBy("date")

LAG_DAYS = [1, 7, 14, 28]
for lag in LAG_DAYS:
    df = df.withColumn(f"lag_{lag}", F.lag("sales", lag).over(series_window))

print(f"Added lag features: {[f'lag_{d}' for d in LAG_DAYS]}")

# COMMAND ----------
# %md ### 2.3 — Rolling statistics

WINDOWS = [7, 14, 28]
FUNCS   = {
    "mean": F.mean,
    "std":  F.stddev,
    "min":  F.min,
    "max":  F.max,
}

for w in WINDOWS:
    rolling_window = (
        Window.partitionBy("item_id", "store_id")
        .orderBy("date")
        .rowsBetween(-w, -1)
    )
    for fname, ffunc in FUNCS.items():
        df = df.withColumn(f"roll_{fname}_{w}", ffunc("sales").over(rolling_window))

print(f"Added rolling features: {len(WINDOWS) * len(FUNCS)} columns")

# COMMAND ----------
# %md ### 2.4 — Calendar features

df = (df
    .withColumn("day_of_week",  F.dayofweek("date"))
    .withColumn("day_of_month", F.dayofmonth("date"))
    .withColumn("week_of_year", F.weekofyear("date"))
    .withColumn("month",        F.month("date"))
    .withColumn("year",         F.year("date"))
    .withColumn("quarter",      F.quarter("date"))
    .withColumn("is_weekend",   (F.dayofweek("date").isin([1, 7])).cast("int"))
)

# COMMAND ----------
# %md ### 2.5 — Price features

df = (df
    .withColumn("price_lag_1",
        F.lag("sell_price", 1).over(series_window))
    .withColumn("price_change",
        F.col("sell_price") - F.col("price_lag_1"))
    .withColumn("price_pct_change",
        F.col("price_change") / (F.col("price_lag_1") + F.lit(1e-8)))
)

# Normalized price vs store average per week
store_week_window = Window.partitionBy("store_id", "wm_yr_wk")
df = df.withColumn(
    "price_norm_store",
    (F.col("sell_price") - F.mean("sell_price").over(store_week_window))
    / (F.stddev("sell_price").over(store_week_window) + F.lit(1e-8))
)

# COMMAND ----------
# %md ### 2.6 — Target encoding (smoothed, no leakage)

# Encode only on training data dates
TRAIN_END = "2016-03-27"
train_only = df.filter(F.col("date") <= TRAIN_END)

for col in ["dept_id", "cat_id", "store_id", "state_id"]:
    group_mean = (train_only
        .groupBy(col)
        .agg(F.mean("sales").alias(f"te_{col}"))
    )
    df = df.join(group_mean, on=col, how="left")

print("Added target encodings: te_dept_id, te_cat_id, te_store_id, te_state_id")

# COMMAND ----------
# %md ### 2.7 — Event encoding

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

indexer = StringIndexer(
    inputCol="event_name_1",
    outputCol="event_name_encoded",
    handleInvalid="keep"
)
df = indexer.fit(df).transform(df)

# COMMAND ----------
# %md ### 2.8 — Data validation

print("=== Missing values per feature ===")
feature_cols = (
    [f"lag_{d}" for d in [1,7,14,28]] +
    [f"roll_{f}_{w}" for w in [7,14,28] for f in ["mean","std","min","max"]] +
    ["day_of_week", "month", "is_weekend", "sell_price", "price_pct_change"]
)

null_counts = df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in feature_cols
])
null_counts.show(vertical=True)

# Drop rows with nulls in lag features (first N days per series)
df_clean = df.dropna(subset=[f"lag_{max([1,7,14,28])}"])
print(f"Rows before dropna: {df.count():,}")
print(f"Rows after  dropna: {df_clean.count():,}")

# COMMAND ----------
# %md ### 2.9 — Train / Val / Test splits + save to Gold layer

splits = {
    "train": ("2011-01-29", "2016-03-27"),
    "val":   ("2016-03-28", "2016-04-24"),
    "test":  ("2016-04-25", "2016-05-22"),
}

for split_name, (start, end) in splits.items():
    split_df = df_clean.filter(
        (F.col("date") >= start) & (F.col("date") <= end)
    )
    out_path = f"{GOLD_PATH}/m5_{split_name}"
    (split_df
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("dept_id")
        .save(out_path)
    )
    print(f"✅ Saved {split_name}: {split_df.count():,} rows → {out_path}")

# COMMAND ----------
# %md ### 2.10 — Final feature summary

print("=== Gold layer feature columns ===")
gold = spark.read.format("delta").load(f"{GOLD_PATH}/m5_train")
print(f"Rows: {gold.count():,} | Cols: {len(gold.columns)}")
print("Columns:", gold.columns)

gold.select(feature_cols).describe().show()
print("✅ Feature engineering complete.")
