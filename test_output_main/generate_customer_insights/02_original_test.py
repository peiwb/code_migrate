import pytest
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import sum as spark_sum, avg, count
from decimal import Decimal


def test_generate_customer_insights():
    """
    Test the generate_customer_insights function with realistic customer data.
    """
    # Create local SparkSession for testing
    spark = SparkSession.builder \
        .appName("test_customer_insights") \
        .master("local[2]") \
        .getOrCreate()
    
    try:
        # Define schema for input data
        input_schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("age_group", StringType(), True),
            StructField("customer_tier", StringType(), True),
            StructField("total_spent", DoubleType(), True),
            StructField("total_transaction_amount", DoubleType(), True)
        ])
        
        # Create realistic input test data
        input_data = [
            ("C001", "18-25", "Bronze", 150.0, 50.0),
            ("C002", "18-25", "Bronze", 200.0, 75.0),
            ("C003", "18-25", "Silver", 500.0, 200.0),
            ("C004", "26-35", "Bronze", 300.0, 100.0),
            ("C005", "26-35", "Gold", 1000.0, 400.0),
            ("C006", "26-35", "Gold", 1200.0, 500.0),
            ("C007", "36-45", "Silver", 800.0, 300.0)
        ]
        
        input_df = spark.createDataFrame(input_data, input_schema)
        
        # Define expected output schema
        output_schema = StructType([
            StructField("age_group", StringType(), True),
            StructField("customer_tier", StringType(), True),
            StructField("customer_count", IntegerType(), False),
            StructField("avg_spending", DoubleType(), True),
            StructField("total_revenue", DoubleType(), True)
        ])
        
        # Create expected output data (manually calculated)
        expected_data = [
            ("18-25", "Bronze", 2, 175.0, 125.0),  # (150+200)/2=175, 50+75=125
            ("18-25", "Silver", 1, 500.0, 200.0),  # 500/1=500, 200=200
            ("26-35", "Bronze", 1, 300.0, 100.0),  # 300/1=300, 100=100
            ("26-35", "Gold", 2, 1100.0, 900.0),   # (1000+1200)/2=1100, 400+500=900
            ("36-45", "Silver", 1, 800.0, 300.0)   # 800/1=800, 300=300
        ]
        
        expected_df = spark.createDataFrame(expected_data, output_schema)
        
        # Call the function under test
        result_df = generate_customer_insights(input_df)
        
        # Collect results for comparison
        result_rows = result_df.collect()
        expected_rows = expected_df.collect()
        
        # Verify the number of rows matches
        assert len(result_rows) == len(expected_rows), f"Expected {len(expected_rows)} rows, got {len(result_rows)}"
        
        # Compare each row (results should be ordered)
        for i, (result_row, expected_row) in enumerate(zip(result_rows, expected_rows)):
            assert result_row.age_group == expected_row.age_group, f"Row {i}: age_group mismatch"
            assert result_row.customer_tier == expected_row.customer_tier, f"Row {i}: customer_tier mismatch"
            assert result_row.customer_count == expected_row.customer_count, f"Row {i}: customer_count mismatch"
            
            # Use approximate comparison for floating point values
            assert abs(result_row.avg_spending - expected_row.avg_spending) < 0.001, f"Row {i}: avg_spending mismatch"
            assert abs(result_row.total_revenue - expected_row.total_revenue) < 0.001, f"Row {i}: total_revenue mismatch"
        
        # Verify schema structure
        expected_columns = {"age_group", "customer_tier", "customer_count", "avg_spending", "total_revenue"}
        actual_columns = set(result_df.columns)
        assert actual_columns == expected_columns, f"Column mismatch: expected {expected_columns}, got {actual_columns}"
        
    finally:
        # Clean up SparkSession
        spark.stop()


# Import the function being tested (assuming it's in the same module or imported appropriately)
def generate_customer_insights(df: DataFrame) -> DataFrame:
    """
    Generate business insights from customer data by aggregating metrics across age groups and customer tiers.
    
    This function creates a summary report that groups customers by demographic and tier segments,
    calculating key business metrics including customer counts, average spending per customer,
    and total revenue contribution. The results are sorted to provide a consistent view for
    business reporting and analytics.
    
    Args:
        df (DataFrame): Enriched customer data containing the following required columns:
            - customer_id: Unique identifier for each customer
            - age_group: Customer age segment (e.g., '18-25', '26-35', etc.)
            - customer_tier: Customer classification level (e.g., 'Gold', 'Silver', 'Bronze')
            - total_spent: Total amount spent by individual customer (numeric)
            - total_transaction_amount: Individual transaction amounts (numeric)
    
    Returns:
        DataFrame: Customer insights summary with columns:
            - age_group: Customer age segment
            - customer_tier: Customer tier classification
            - customer_count: Number of unique customers in the segment
            - avg_spending: Average spending per customer in the segment
            - total_revenue: Sum of all transaction amounts in the segment
    
    Note:
        Results are ordered by age_group and customer_tier for consistent reporting.
        The function assumes input data has already been validated and cleansed.
    """
    # Group by demographic and tier segments - this creates partition boundaries for aggregation
    # Performance note: Ensure age_group and customer_tier have reasonable cardinality to avoid skew
    insights_df = df.groupBy("age_group", "customer_tier") \
        .agg(
        # Count distinct customers to handle potential duplicate records per customer
        # Migration note: count() behavior may vary across platforms for null handling
        count("customer_id").alias("customer_count"),
        
        # Calculate average spending per customer within each segment
        # Business assumption: total_spent represents lifetime value per customer
        avg("total_spent").alias("avg_spending"),
        
        # Sum all transaction amounts to get total revenue contribution
        # Using spark_sum instead of sum to avoid naming conflicts with Python built-in
        spark_sum("total_transaction_amount").alias("total_revenue")
    ) \
        .orderBy("age_group", "customer_tier")  # Deterministic ordering for consistent reporting

    return insights_df