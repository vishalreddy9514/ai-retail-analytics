"""Silver: typed, cleaned, de-duplicated sales lines. Rejected rows go to a quarantine table."""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# Stock codes that are fees, postage or manual adjustments rather than products.
NON_PRODUCT_CODES = [
    "POST", "DOT", "M", "D", "S", "B", "C2", "CRUK", "PADS",
    "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2",
]

BUSINESS_COLUMNS = [
    "invoice_no", "stock_code", "description", "quantity",
    "invoice_ts", "unit_price", "customer_id", "country",
]


def standardise(bronze_df: DataFrame) -> DataFrame:
    """Rename to snake_case and cast types. try_* functions return NULL instead of failing."""
    return bronze_df.select(
        F.trim("InvoiceNo").alias("invoice_no"),
        F.upper(F.trim("StockCode")).alias("stock_code"),
        F.trim("Description").alias("description"),
        F.expr("try_cast(trim(Quantity) AS INT)").alias("quantity"),
        F.expr("try_to_timestamp(trim(InvoiceDate), 'yyyy-MM-dd HH:mm:ss')").alias("invoice_ts"),
        F.expr("try_cast(trim(UnitPrice) AS DOUBLE)").alias("unit_price"),
        # Excel exports can turn IDs into floats ("17850.0"); keep the ID as a clean string.
        F.nullif(F.regexp_replace(F.trim("CustomerID"), r"\.0$", ""), F.lit("")).alias("customer_id"),
        F.trim("Country").alias("country"),
    )


def tag_rejections(df: DataFrame) -> DataFrame:
    """Add rejection_reason: NULL means the row is valid. The first failing rule wins."""
    reason = (
        F.when(F.col("invoice_no").isNull() | (F.col("invoice_no") == ""), "missing_invoice_no")
        .when(F.col("stock_code").isNull() | (F.col("stock_code") == ""), "missing_stock_code")
        .when(F.col("invoice_ts").isNull(), "invalid_invoice_date")
        .when(F.col("quantity").isNull(), "invalid_quantity")
        .when(F.col("unit_price").isNull(), "invalid_unit_price")
        .when(F.col("invoice_no").startswith("C"), "cancelled_invoice")
        .when(F.col("quantity") <= 0, "non_positive_quantity")
        .when(F.col("unit_price") <= 0, "non_positive_unit_price")
        .when(F.col("stock_code").isin(NON_PRODUCT_CODES), "non_product_code")
    )
    # Exact repeats of an otherwise valid line are flagged as duplicates (first copy kept).
    window = Window.partitionBy(*BUSINESS_COLUMNS).orderBy(F.lit(1))
    return (
        df.withColumn("rejection_reason", reason)
        .withColumn("_row_num", F.row_number().over(window))
        .withColumn(
            "rejection_reason",
            F.when(
                F.col("rejection_reason").isNull() & (F.col("_row_num") > 1), "duplicate"
            ).otherwise(F.col("rejection_reason")),
        )
        .drop("_row_num")
    )


def build_silver(bronze_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Return (silver, quarantine)."""
    tagged = tag_rejections(standardise(bronze_df))

    silver = (
        tagged.filter(F.col("rejection_reason").isNull())
        .drop("rejection_reason")
        .withColumn("invoice_date", F.to_date("invoice_ts"))
        .withColumn("line_total", F.round(F.col("quantity") * F.col("unit_price"), 2))
    )
    quarantine = tagged.filter(F.col("rejection_reason").isNotNull())
    return silver, quarantine
