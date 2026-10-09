"""Spark session and Delta read/write helpers that work on Databricks and locally."""
from pyspark.sql import DataFrame, SparkSession

from retail_analytics.config import Settings


def get_local_spark(app_name: str = "ai-retail-analytics") -> SparkSession:
    """Local SparkSession with Delta Lake. Downloads the Delta jars on first run."""
    from delta import configure_spark_with_delta_pip

    builder = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def write_delta(df: DataFrame, settings: Settings, table: str) -> None:
    """Full-refresh overwrite, so every run is reproducible and idempotent."""
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if settings.is_databricks:
        writer.saveAsTable(settings.table_name(table))
    else:
        writer.save(settings.table_path(table))


def read_delta(spark: SparkSession, settings: Settings, table: str) -> DataFrame:
    if settings.is_databricks:
        return spark.read.table(settings.table_name(table))
    return spark.read.format("delta").load(settings.table_path(table))
