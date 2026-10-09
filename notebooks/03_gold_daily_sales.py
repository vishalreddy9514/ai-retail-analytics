# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Gold: daily sales metrics
# MAGIC Aggregates Silver into one row per calendar day (`gold_daily_sales`), runs
# MAGIC reconciliation checks, and exports a CSV for the local forecasting, LLM and dashboard stages.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from retail_analytics.config import GOLD_TABLE, Settings
from retail_analytics.pipeline import export_gold_csv, run_gold

settings = Settings.for_databricks()

# COMMAND ----------

print(f"Gold rows (days): {run_gold(spark, settings):,}")
gold = spark.table(settings.table_name(GOLD_TABLE))
display(gold)

# COMMAND ----------

# Download this file from Catalog > workspace > retail > raw_files > exports
# and save it locally as data/exports/gold_daily_sales.csv
print(export_gold_csv(gold, settings.export_dir))
