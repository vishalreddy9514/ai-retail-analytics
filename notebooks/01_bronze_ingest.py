# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Bronze: ingest the raw CSV
# MAGIC Loads `online_retail.csv` from the volume into `workspace.retail.bronze_online_retail`,
# MAGIC keeping every column as a string and adding `_source_file` and `_ingested_at`.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))  # import the project package from the Git folder

from retail_analytics.config import BRONZE_TABLE, Settings
from retail_analytics.pipeline import run_bronze

settings = Settings.for_databricks()

# COMMAND ----------

row_count = run_bronze(spark, settings)
print(f"Bronze rows: {row_count:,}")

# COMMAND ----------

display(spark.table(settings.table_name(BRONZE_TABLE)).limit(10))
