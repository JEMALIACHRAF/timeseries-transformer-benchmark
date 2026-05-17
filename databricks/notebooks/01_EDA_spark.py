# Databricks notebook source
# databricks/notebooks/01_EDA_spark.py
# ============================================================
# Notebook 01 — Exploratory Data Analysis (PySpark)
# Run on: Azure Databricks cluster (Standard_DS3_v2 or higher)
# ============================================================

# COMMAND ----------
# %md
# ## 01 — Exploratory Data Analysis
# M5 Forecasting Dataset — 58M rows, 30,490 time series
# Tools: PySpark, Delta Lake, Plotly

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
# Azure Data Lake config (set in Databricks secrets or cluster env)
ADLS_ACCOUNT = os.environ.get("ADLS_ACCOUNT_NAME", "your_storage_account")
ADLS_KEY     = os.environ.get("ADLS_KEY", "")
spark.conf.set(
    f"fs.azure.account.key.{ADLS_ACCOUNT}.dfs.core.windows.net",
    ADLS_KEY
)

CONTAINER     = "tsdata"
RAW_PATH      = f"abfss://{CONTAINER}@{ADLS_ACCOUNT}.dfs.core.windows.net/raw"
PROCESSED_PATH = f"abfss://{CONTAINER}@{ADLS_ACCOUNT}.dfs.core.windows.net/processed"

print(f"ADLS path: {RAW_PATH}")

# COMMAND ----------
# %md ### 1.1 — Load raw M5 data from ADLS

sales_df = (spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(f"{RAW_PATH}/sales_train_evaluation.csv"))

calendar_df = (spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(f"{RAW_PATH}/calendar.csv"))

prices_df = (spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(f"{RAW_PATH}/sell_prices.csv"))

print(f"Sales shape:    {sales_df.count():,} rows × {len(sales_df.columns)} cols")
print(f"Calendar shape: {calendar_df.count()} rows × {len(calendar_df.columns)} cols")
print(f"Prices shape:   {prices_df.count():,} rows × {len(prices_df.columns)} cols")

# COMMAND ----------
# %md ### 1.2 — Melt sales (wide → long)

id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
d_cols  = [c for c in sales_df.columns if c.startswith("d_")]

# Stack d_1 ... d_1941 columns into rows
sales_long = sales_df.select(
    id_cols + d_cols
).unpivot(id_cols, d_cols, "d", "sales")

print(f"Long format shape: {sales_long.count():,} rows")
sales_long.show(5)

# COMMAND ----------
# %md ### 1.3 — Merge calendar + prices

sales_long = sales_long.join(
    calendar_df.select("d", "date", "wm_yr_wk", "weekday", "wday",
                        "month", "year", "event_name_1", "event_type_1",
                        "snap_CA", "snap_TX", "snap_WI"),
    on="d", how="left"
)
sales_long = sales_long.withColumn("date", F.to_date("date"))

sales_long = sales_long.join(
    prices_df,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)

print(f"Enriched shape: {sales_long.count():,} rows × {len(sales_long.columns)} cols")

# COMMAND ----------
# %md ### 1.4 — Basic statistics

print("=== Sales Distribution ===")
sales_long.select("sales").summary().show()

print("=== Zero sales rate by category ===")
(sales_long
    .withColumn("is_zero", (F.col("sales") == 0).cast("int"))
    .groupBy("cat_id")
    .agg(
        F.mean("is_zero").alias("zero_rate"),
        F.mean("sales").alias("avg_sales"),
        F.stddev("sales").alias("std_sales"),
        F.count("sales").alias("n_obs")
    )
    .orderBy("zero_rate", ascending=False)
    .show()
)

# COMMAND ----------
# %md ### 1.5 — Temporal trend analysis

daily_sales = (sales_long
    .groupBy("date")
    .agg(F.sum("sales").alias("total_sales"))
    .orderBy("date")
    .toPandas()
)

fig = px.line(
    daily_sales, x="date", y="total_sales",
    title="M5 — Total Daily Sales Across All Stores",
    labels={"total_sales": "Total Units Sold", "date": "Date"},
    template="plotly_white"
)
fig.update_traces(line_color="#1565C0", line_width=1.5)
fig.show()

# COMMAND ----------
# %md ### 1.6 — Seasonality patterns by department

dept_monthly = (sales_long
    .groupBy("dept_id", "month", "year")
    .agg(F.mean("sales").alias("avg_sales"))
    .orderBy("dept_id", "year", "month")
    .toPandas()
)
dept_monthly["period"] = dept_monthly["year"].astype(str) + "-" + dept_monthly["month"].astype(str).str.zfill(2)

fig = px.line(
    dept_monthly[dept_monthly["dept_id"].isin(dept_monthly["dept_id"].unique()[:5])],
    x="period", y="avg_sales", color="dept_id",
    title="Monthly Average Sales by Department",
    template="plotly_white"
)
fig.show()

# COMMAND ----------
# %md ### 1.7 — Day of week patterns

dow_sales = (sales_long
    .groupBy("wday", "dept_id")
    .agg(F.mean("sales").alias("avg_sales"))
    .orderBy("dept_id", "wday")
    .toPandas()
)

fig = px.bar(
    dow_sales[dow_sales["dept_id"].isin(["FOODS_1", "FOODS_2", "HOBBIES_1"])],
    x="wday", y="avg_sales", color="dept_id", barmode="group",
    title="Average Sales by Day of Week",
    labels={"wday": "Day (1=Mon, 7=Sun)", "avg_sales": "Avg Units Sold"},
    template="plotly_white"
)
fig.show()

# COMMAND ----------
# %md ### 1.8 — Price distribution and promotions

price_stats = (sales_long
    .filter(F.col("sell_price").isNotNull())
    .groupBy("dept_id")
    .agg(
        F.mean("sell_price").alias("avg_price"),
        F.stddev("sell_price").alias("std_price"),
        F.min("sell_price").alias("min_price"),
        F.max("sell_price").alias("max_price"),
        F.countDistinct("sell_price").alias("n_price_points")
    )
    .orderBy("dept_id")
)
price_stats.show()

# COMMAND ----------
# %md ### 1.9 — STL Decomposition (sample series)

from statsmodels.tsa.seasonal import STL

# Sample one series for STL
sample_series = (sales_long
    .filter((F.col("item_id") == "FOODS_3_090") & (F.col("store_id") == "CA_3"))
    .orderBy("date")
    .select("date", "sales")
    .toPandas()
    .set_index("date")
)

stl = STL(sample_series["sales"], period=7, robust=True)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
axes[0].plot(sample_series["sales"], color="#1565C0"); axes[0].set_title("Original")
axes[1].plot(result.trend, color="#E53935"); axes[1].set_title("Trend")
axes[2].plot(result.seasonal, color="#43A047"); axes[2].set_title("Seasonality (weekly)")
axes[3].plot(result.resid, color="#FB8C00"); axes[3].set_title("Residual")
plt.suptitle("STL Decomposition — FOODS_3_090_CA_3", fontsize=14, fontweight="bold")
plt.tight_layout()
display(fig)

# COMMAND ----------
# %md ### 1.10 — Save enriched data to Delta Lake (Silver layer)

(sales_long
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("dept_id")
    .save(f"{PROCESSED_PATH}/m5_silver/"))

print(f"✅ Silver layer saved to {PROCESSED_PATH}/m5_silver/")

# Optimize Delta table for read performance
spark.sql(f"OPTIMIZE delta.`{PROCESSED_PATH}/m5_silver/` ZORDER BY (item_id, store_id, date)")
print("✅ Delta table optimized.")
