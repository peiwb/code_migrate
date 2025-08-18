import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, count, sum as spark_sum, avg, when
from pyspark.sql import DataFrame

def test_enrich_customer_data():
    """
    Test the enrich_customer_data function with comprehensive scenarios.
    """
    # Create local SparkSession for testing
    spark = SparkSession.builder \
        .appName("test_enrich_customer_data") \
        .master("local[2]") \
        .getOrCreate()
    
    try:
        # Create realistic input data for testing
        # Customer DataFrame with various customer profiles
        customer_schema = StructType([
            StructField("customer_id", StringType(), False),
            StructField("customer_name", StringType(), True),
            StructField("email", StringType(), True)
        ])
        
        customer_data = [
            ("C001", "John Doe", "john.doe@email.com"),
            ("C002", "Jane Smith", "jane.smith@email.com"),
            ("C003", "Bob Johnson", "bob.johnson@email.com"),
            ("C004", "Alice Brown", "alice.brown@email.com"),
            ("C005", "Charlie Wilson", "charlie.wilson@email.com")  # Customer with no transactions
        ]
        
        customer_df = spark.createDataFrame(customer_data, customer_schema)
        
        # Transaction DataFrame with varying spending patterns
        transaction_schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("customer_id", StringType(), False),
            StructField("amount", DoubleType(), False)
        ])
        
        transaction_data = [
            ("T001", "C001", 12000.0),  # Premium tier customer
            ("T002", "C001", 3000.0),
            ("T003", "C002", 6000.0),   # Gold tier customer
            ("T004", "C002", 2000.0),
            ("T005", "C003", 1500.0),   # Silver tier customer
            ("T006", "C004", 500.0),    # Bronze tier customer
            ("T007", "C004", 300.0)
            # C005 has no transactions - should get Bronze tier with null metrics
        ]
        
        transaction_df = spark.createDataFrame(transaction_data, transaction_schema)
        
        # Define expected output data
        expected_schema = StructType([
            StructField("customer_id", StringType(), False),
            StructField("customer_name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("transaction_count", IntegerType(), True),
            StructField("total_transaction_amount", DoubleType(), True),
            StructField("avg_transaction_amount", DoubleType(), True),
            StructField("customer_tier", StringType(), True)
        ])
        
        expected_data = [
            ("C001", "John Doe", "john.doe@email.com", 2, 15000.0, 7500.0, "Premium"),
            ("C002", "Jane Smith", "jane.smith@email.com", 2, 8000.0, 4000.0, "Gold"),
            ("C003", "Bob Johnson", "bob.johnson@email.com", 1, 1500.0, 1500.0, "Silver"),
            ("C004", "Alice Brown", "alice.brown@email.com", 2, 800.0, 400.0, "Bronze"),
            ("C005", "Charlie Wilson", "charlie.wilson@email.com", None, None, None, "Bronze")
        ]
        
        expected_df = spark.createDataFrame(expected_data, expected_schema)
        
        # Call the function under test
        actual_df = enrich_customer_data(customer_df, transaction_df)
        
        # Sort both DataFrames by customer_id for consistent comparison
        actual_sorted = actual_df.orderBy("customer_id").collect()
        expected_sorted = expected_df.orderBy("customer_id").collect()
        
        # Verify row count matches
        assert len(actual_sorted) == len(expected_sorted), \
            f"Row count mismatch: expected {len(expected_sorted)}, got {len(actual_sorted)}"
        
        # Verify each row matches expected output
        for i, (actual_row, expected_row) in enumerate(zip(actual_sorted, expected_sorted)):
            assert actual_row.customer_id == expected_row.customer_id, \
                f"Row {i}: customer_id mismatch"
            assert actual_row.customer_name == expected_row.customer_name, \
                f"Row {i}: customer_name mismatch"
            assert actual_row.email == expected_row.email, \
                f"Row {i}: email mismatch"
            assert actual_row.transaction_count == expected_row.transaction_count, \
                f"Row {i}: transaction_count mismatch"
            
            # Handle null values and floating point comparison
            if expected_row.total_transaction_amount is None:
                assert actual_row.total_transaction_amount is None, \
                    f"Row {i}: expected null total_transaction_amount"
            else:
                assert abs(actual_row.total_transaction_amount - expected_row.total_transaction_amount) < 0.01, \
                    f"Row {i}: total_transaction_amount mismatch"
            
            if expected_row.avg_transaction_amount is None:
                assert actual_row.avg_transaction_amount is None, \
                    f"Row {i}: expected null avg_transaction_amount"
            else:
                assert abs(actual_row.avg_transaction_amount - expected_row.avg_transaction_amount) < 0.01, \
                    f"Row {i}: avg_transaction_amount mismatch"
            
            assert actual_row.customer_tier == expected_row.customer_tier, \
                f"Row {i}: customer_tier mismatch"
        
        # Verify schema structure
        expected_columns = {"customer_id", "customer_name", "email", "transaction_count", 
                          "total_transaction_amount", "avg_transaction_amount", "customer_tier"}
        actual_columns = set(actual_df.columns)
        assert actual_columns == expected_columns, \
            f"Column mismatch: expected {expected_columns}, got {actual_columns}"
    
    finally:
        # Clean up SparkSession
        spark.stop()