# Databricks notebook source
# databricks/notebooks/03_data_validation.py
# ============================================================
# Notebook 03 — Data Quality Validation (Great Expectations)
# Validates Gold feature layer before model training
# ============================================================

# COMMAND ----------
# %pip install great-expectations==0.18.14

# COMMAND ----------

import great_expectations as gx
import pyspark.sql.functions as F
import pandas as pd

import os
# Azure Data Lake config (set in Databricks secrets or cluster env)
ADLS_ACCOUNT = os.environ.get("ADLS_ACCOUNT_NAME", "your_storage_account")
ADLS_KEY     = os.environ.get("ADLS_KEY", "")
spark.conf.set(
    f"fs.azure.account.key.{ADLS_ACCOUNT}.dfs.core.windows.net",
    ADLS_KEY
)
CONTAINER    = "tsdata"
GOLD_PATH    = f"abfss://{CONTAINER}@{ADLS_ACCOUNT}.dfs.core.windows.net/features/m5_gold"

# COMMAND ----------
# %md ### 3.1 — Load Gold train set

train_df = spark.read.format("delta").load(f"{GOLD_PATH}/m5_train")
train_pd  = train_df.sample(fraction=0.01, seed=42).toPandas()  # Sample for GE

print(f"Train rows (full):   {train_df.count():,}")
print(f"Train rows (sample): {len(train_pd):,}")

# COMMAND ----------
# %md ### 3.2 — Build Great Expectations suite

context = gx.get_context()
ds = context.sources.add_pandas("m5_gold_sample")
da = ds.add_dataframe_asset("train_sample")
batch_request = da.build_batch_request(dataframe=train_pd)
validator = context.get_validator(batch_request=batch_request)

# COMMAND ----------
# %md ### 3.3 — Define expectations

# ── Schema ────────────────────────────────────────────────────
validator.expect_table_columns_to_match_ordered_list(
    column_list=[
        "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "d", "date", "sales",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_14", "roll_mean_28",
        "roll_std_7", "sell_price", "is_weekend",
    ],
    result_format="BASIC",
)

# ── Sales ─────────────────────────────────────────────────────
validator.expect_column_values_to_not_be_null("sales")
validator.expect_column_values_to_be_between("sales", min_value=0, max_value=1000)
validator.expect_column_mean_to_be_between("sales", min_value=1.0, max_value=30.0)

# ── Lag features ──────────────────────────────────────────────
for lag in ["lag_7", "lag_14", "lag_28"]:
    validator.expect_column_values_to_not_be_null(lag)
    validator.expect_column_values_to_be_between(lag, min_value=0, max_value=1000)

# ── Rolling features ──────────────────────────────────────────
for col in ["roll_mean_7", "roll_mean_14", "roll_mean_28"]:
    validator.expect_column_values_to_not_be_null(col)
    validator.expect_column_values_to_be_between(col, min_value=0, max_value=500)

# ── Calendar ──────────────────────────────────────────────────
validator.expect_column_values_to_be_between("is_weekend", min_value=0, max_value=1)
validator.expect_column_values_to_be_between("day_of_week", min_value=1, max_value=7)
validator.expect_column_values_to_be_between("month", min_value=1, max_value=12)

# ── Price ─────────────────────────────────────────────────────
validator.expect_column_values_to_be_between("sell_price", min_value=0.01, max_value=200.0)

# ── Categorical ───────────────────────────────────────────────
validator.expect_column_values_to_be_in_set(
    "state_id", value_set=["CA", "TX", "WI"]
)
validator.expect_column_values_to_be_in_set(
    "cat_id", value_set=["FOODS", "HOBBIES", "HOUSEHOLD"]
)

# ── Date coverage ─────────────────────────────────────────────
validator.expect_column_values_to_be_between(
    "date",
    min_value="2011-01-29",
    max_value="2016-03-27",
)

# ── Uniqueness check ──────────────────────────────────────────
validator.expect_compound_columns_to_be_unique(
    column_list=["item_id", "store_id", "date"]
)

# COMMAND ----------
# %md ### 3.4 — Run validation and report

results = validator.validate()

print(f"\n{'='*50}")
print(f"  VALIDATION SUMMARY")
print(f"{'='*50}")
print(f"  Total expectations : {results['statistics']['evaluated_expectations']}")
print(f"  Successful         : {results['statistics']['successful_expectations']}")
print(f"  Failed             : {results['statistics']['unsuccessful_expectations']}")
print(f"  Success rate       : {results['statistics']['success_percent']:.1f}%")
print(f"  Overall success    : {results['success']}")
print(f"{'='*50}\n")

# Show failures in detail
failures = [r for r in results["results"] if not r["success"]]
if failures:
    print(f"\n⚠️  {len(failures)} expectations FAILED:\n")
    for f in failures:
        et = f["expectation_config"]["expectation_type"]
        col = f["expectation_config"]["kwargs"].get("column", "table")
        print(f"  ✗ [{col}] {et}")
        print(f"    Result: {f['result']}\n")
else:
    print("✅ All expectations passed — data quality validated!")

# COMMAND ----------
# %md ### 3.5 — Log validation results to MLflow

import mlflow

with mlflow.start_run(run_name="data_validation"):
    mlflow.log_metric("total_expectations",
                      results["statistics"]["evaluated_expectations"])
    mlflow.log_metric("passed_expectations",
                      results["statistics"]["successful_expectations"])
    mlflow.log_metric("failed_expectations",
                      results["statistics"]["unsuccessful_expectations"])
    mlflow.log_metric("success_rate_pct",
                      results["statistics"]["success_percent"])
    mlflow.log_param("dataset", "m5_gold_train")
    mlflow.log_param("validation_passed", results["success"])

    if not results["success"]:
        raise ValueError(
            f"Data validation FAILED — {len(failures)} expectations not met. "
            "Fix data quality issues before training."
        )

print("✅ Validation results logged to MLflow.")
