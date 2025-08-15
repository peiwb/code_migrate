import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import sum as spark_sum, count, avg
from pyspark.sql import DataFrame


def test_generate_customer_insights():
    """
    Test the generate_customer_insights function with realistic customer data.
    """
    # Create local SparkSession for testing
    spark = SparkSession.builder \
        .appName("test_customer_insights") \
        .master("local[*]") \
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
        
        # Generate realistic input test data
        input_data = [
            ("C001", "18-25", "Bronze", 150.0, 50.0),
            ("C002", "18-25", "Bronze", 200.0, 75.0),
            ("C003", "18-25", "Silver", 300.0, 100.0),
            ("C004", "26-35", "Bronze", 250.0, 80.0),
            ("C005", "26-35", "Gold", 500.0, 200.0),
            ("C006", "26-35", "Gold", 600.0, 250.0)
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
        
        # Generate expected output data
        # Calculations:
        # 18-25, Bronze: 2 customers, avg_spending=(150+200)/2=175, total_revenue=50+75=125
        # 18-25, Silver: 1 customer, avg_spending=300, total_revenue=100
        # 26-35, Bronze: 1 customer, avg_spending=250, total_revenue=80
        # 26-35, Gold: 2 customers, avg_spending=(500+600)/2=550, total_revenue=200+250=450
        expected_data = [
            ("18-25", "Bronze", 2, 175.0, 125.0),
            ("18-25", "Silver", 1, 300.0, 100.0),
            ("26-35", "Bronze", 1, 250.0, 80.0),
            ("26-35", "Gold", 2, 550.0, 450.0)
        ]
        
        expected_df = spark.createDataFrame(expected_data, output_schema)
        
        # Call the function under test
        actual_df = generate_customer_insights(input_df)
        
        # Collect results for comparison
        actual_rows = actual_df.collect()
        expected_rows = expected_df.collect()
        
        # Assert same number of rows
        assert len(actual_rows) == len(expected_rows), f"Expected {len(expected_rows)} rows, got {len(actual_rows)}"
        
        # Assert each row matches expected values
        for actual_row, expected_row in zip(actual_rows, expected_rows):
            assert actual_row.age_group == expected_row.age_group
            assert actual_row.customer_tier == expected_row.customer_tier
            assert actual_row.customer_count == expected_row.customer_count
            assert abs(actual_row.avg_spending - expected_row.avg_spending) < 0.001  # Handle floating point precision
            assert abs(actual_row.total_revenue - expected_row.total_revenue) < 0.001  # Handle floating point precision
        
        # Verify the data is properly ordered
        age_groups = [row.age_group for row in actual_rows]
        customer_tiers = [row.customer_tier for row in actual_rows]
        
        # Check ordering within the results
        assert age_groups == ["18-25", "18-25", "26-35", "26-35"]
        assert customer_tiers == ["Bronze", "Silver", "Bronze", "Gold"]
        
    finally:
        # Clean up SparkSession
        spark.stop()