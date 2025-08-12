"""
Sample PySpark Script for Testing CodeAnalyzer
This script demonstrates common PySpark patterns and operations
that will be analyzed by the CodeAnalyzer module.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when, sum as spark_sum, count, avg
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Custom imports (simulated)
from utils.data_validator import validate_data_quality
from config.spark_config import get_spark_config


def create_spark_session(app_name: str = "DataProcessingApp") -> SparkSession:
    """
    Create and configure Spark session with optimized settings.

    Args:
        app_name: Name of the Spark application

    Returns:
        SparkSession: Configured Spark session
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    return spark


def load_customer_data(spark: SparkSession, data_path: str) -> DataFrame:
    """
    Load customer data from various sources.

    Args:
        spark: Active Spark session
        data_path: Path to the data source

    Returns:
        DataFrame: Loaded customer data
    """
    # Define schema
    schema = StructType([
        StructField("customer_id", StringType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("email", StringType(), True),
        StructField("registration_date", StringType(), True),
        StructField("total_spent", DoubleType(), True)
    ])

    # Load data
    df = spark.read \
        .option("header", "true") \
        .schema(schema) \
        .csv(data_path)

    return df


def load_transaction_data(spark: SparkSession, data_path: str) -> DataFrame:
    """
    Load transaction data with dynamic partitioning.

    Args:
        spark: Active Spark session
        data_path: Path to transaction data

    Returns:
        DataFrame: Loaded transaction data
    """
    df = spark.read \
        .option("multiline", "true") \
        .json(data_path)

    return df


def clean_customer_data(df: DataFrame) -> DataFrame:
    """
    Clean and validate customer data.

    Args:
        df: Raw customer DataFrame

    Returns:
        DataFrame: Cleaned customer data
    """
    # Remove duplicates and null values
    cleaned_df = df.dropDuplicates(["customer_id"]) \
        .filter(col("customer_id").isNotNull()) \
        .filter(col("email").isNotNull())

    # Add derived columns
    cleaned_df = cleaned_df.withColumn(
        "age_group",
        when(col("age") < 25, "Young")
        .when(col("age") < 65, "Adult")
        .otherwise("Senior")
    )

    # Validate data quality
    validated_df = validate_data_quality(cleaned_df)

    return validated_df


def enrich_customer_data(customer_df: DataFrame, transaction_df: DataFrame) -> DataFrame:
    """
    Enrich customer data with transaction analytics.

    Args:
        customer_df: Clean customer data
        transaction_df: Transaction data

    Returns:
        DataFrame: Enriched customer data with analytics
    """
    # Calculate transaction metrics per customer
    transaction_metrics = transaction_df.groupBy("customer_id") \
        .agg(
        count("transaction_id").alias("transaction_count"),
        spark_sum("amount").alias("total_transaction_amount"),
        avg("amount").alias("avg_transaction_amount")
    )

    # Join customer data with transaction metrics
    enriched_df = customer_df.join(
        transaction_metrics,
        on="customer_id",
        how="left"
    )

    # Add customer tier based on spending
    enriched_df = enriched_df.withColumn(
        "customer_tier",
        when(col("total_transaction_amount") > 10000, "Premium")
        .when(col("total_transaction_amount") > 5000, "Gold")
        .when(col("total_transaction_amount") > 1000, "Silver")
        .otherwise("Bronze")
    )

    return enriched_df


def generate_customer_insights(df: DataFrame) -> DataFrame:
    """
    Generate business insights from customer data.

    Args:
        df: Enriched customer data

    Returns:
        DataFrame: Customer insights summary
    """
    insights_df = df.groupBy("age_group", "customer_tier") \
        .agg(
        count("customer_id").alias("customer_count"),
        avg("total_spent").alias("avg_spending"),
        spark_sum("total_transaction_amount").alias("total_revenue")
    ) \
        .orderBy("age_group", "customer_tier")

    return insights_df


def save_results(df: DataFrame, output_path: str, format_type: str = "parquet") -> None:
    """
    Save processed data to specified location.

    Args:
        df: DataFrame to save
        output_path: Output file path
        format_type: Output format (parquet, csv, json)
    """
    writer = df.coalesce(1).write.mode("overwrite")

    if format_type.lower() == "parquet":
        writer.parquet(output_path)
    elif format_type.lower() == "csv":
        writer.option("header", "true").csv(output_path)
    elif format_type.lower() == "json":
        writer.json(output_path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def main():
    """
    Main data processing pipeline.
    Orchestrates the complete data processing workflow.
    """
    # Initialize Spark session
    spark = create_spark_session("CustomerAnalytics")

    try:
        # Load data
        customer_data = load_customer_data(spark, "/data/customers.csv")
        transaction_data = load_transaction_data(spark, "/data/transactions.json")

        # Process data
        clean_customers = clean_customer_data(customer_data)
        enriched_customers = enrich_customer_data(clean_customers, transaction_data)
        customer_insights = generate_customer_insights(enriched_customers)

        # Save results
        save_results(enriched_customers, "/output/enriched_customers", "parquet")
        save_results(customer_insights, "/output/customer_insights", "csv")

        # Show sample results
        print("=== Customer Insights Summary ===")
        customer_insights.show(20, truncate=False)

        print(f"Total customers processed: {enriched_customers.count()}")
        print("Data processing completed successfully!")

    except Exception as e:
        print(f"Error in data processing pipeline: {str(e)}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()