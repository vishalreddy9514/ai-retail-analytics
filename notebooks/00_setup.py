# Databricks notebook source
# MAGIC %md
# MAGIC # 00 Setup
# MAGIC Creates the `workspace.retail` schema and the `raw_files` volume that holds the CSV.
# MAGIC After running this, upload `online_retail.csv` to **Catalog > workspace > retail > raw_files**.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "retail"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw_files")
display(spark.sql(f"SHOW VOLUMES IN {CATALOG}.{SCHEMA}"))

# COMMAND ----------

# Run this after uploading: it should list online_retail.csv.
display(dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/raw_files"))
