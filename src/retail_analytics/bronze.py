"""Bronze: land the raw CSV as-is (every column a string) plus ingestion metadata."""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

SOURCE_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

# Strings only: typing and validation belong in Silver, so nothing is lost on load.
RAW_SCHEMA = StructType([StructField(name, StringType(), True) for name in SOURCE_COLUMNS])


def read_raw_csv(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read.option("header", "true")
        .option("multiLine", "true")  # some descriptions contain quoted commas
        .option("escape", '"')
        .schema(RAW_SCHEMA)
        .csv(path)
    )


def add_ingestion_metadata(raw_df: DataFrame) -> DataFrame:
    return raw_df.select(
        *SOURCE_COLUMNS,
        F.col("_metadata.file_path").alias("_source_file"),
        F.current_timestamp().alias("_ingested_at"),
    )


def build_bronze(spark: SparkSession, path: str) -> DataFrame:
    return add_ingestion_metadata(read_raw_csv(spark, path))
