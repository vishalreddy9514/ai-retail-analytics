# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Silver: clean and validate
# MAGIC Types the Bronze columns, removes cancellations, non-positive quantities/prices,
# MAGIC non-product codes and duplicates, and runs data quality checks.
# MAGIC Rejected rows are kept in `quarantine_online_retail` with a `rejection_reason`.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from retail_analytics.config import SILVER_TABLE, Settings
from retail_analytics.pipeline import run_silver

settings = Settings.for_databricks()

# COMMAND ----------

stats = run_silver(spark, settings)  # raises DataQualityError if a check fails
print(f"Silver rows: {stats['silver_rows']:,}")
print("Quarantined rows by reason:")
for reason, count in stats["rejected"].items():
    print(f"  {reason}: {count:,}")

# COMMAND ----------

display(spark.table(settings.table_name(SILVER_TABLE)).limit(10))
