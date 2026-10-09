"""Gold: one row per calendar day with sales metrics, ready for forecasting and the dashboard."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_gold_daily_sales(silver_df: DataFrame) -> DataFrame:
    daily = silver_df.groupBy(F.col("invoice_date").alias("sales_date")).agg(
        F.round(F.sum("line_total"), 2).alias("revenue"),
        F.countDistinct("invoice_no").alias("orders"),
        F.sum("quantity").cast("long").alias("units_sold"),
        F.countDistinct("customer_id").alias("unique_customers"),
        F.count(F.lit(1)).alias("line_items"),
    )

    # Fill every day between the first and last sale, so days without trading
    # (e.g. Saturdays, Christmas) appear as explicit zeros instead of missing rows.
    calendar = silver_df.agg(
        F.min("invoice_date").alias("start"), F.max("invoice_date").alias("end")
    ).select(F.explode(F.sequence("start", "end")).alias("sales_date"))

    metrics = ["revenue", "orders", "units_sold", "unique_customers", "line_items"]
    return (
        calendar.join(daily, "sales_date", "left")
        .fillna(0, subset=metrics)
        .withColumn("revenue", F.col("revenue").cast("double"))
        .withColumn(
            "avg_order_value",
            F.when(F.col("orders") > 0, F.round(F.col("revenue") / F.col("orders"), 2)),
        )
        .withColumn("day_of_week", F.date_format("sales_date", "EEE"))
        .withColumn("is_trading_day", F.col("orders") > 0)
        .select("sales_date", "day_of_week", "is_trading_day", *metrics, "avg_order_value")
        .orderBy("sales_date")
    )
