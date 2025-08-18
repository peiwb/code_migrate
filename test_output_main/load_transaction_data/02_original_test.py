import pytest
import tempfile
import os
import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from pyspark.sql.functions import col


def test_load_transaction_data():
    """Test the load_transaction_data function with multiline JSON data."""
    # Create a local SparkSession for testing
    spark = SparkSession.builder \
        .appName("test_load_transaction_data") \
        .master("local[2]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()
    
    try:
        # Create temporary directory and file for test data
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file_path = os.path.join(temp_dir, "transactions.json")
            
            # Generate realistic multiline JSON transaction data
            test_transactions = [
                {
                    "transaction_id": "TXN001",
                    "customer_id": "CUST123",
                    "amount": 150.75,
                    "currency": "USD",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "details": {
                        "merchant": "Amazon",
                        "category": "Electronics"
                    }
                },
                {
                    "transaction_id": "TXN002",
                    "customer_id": "CUST456",
                    "amount": 89.99,
                    "currency": "USD",
                    "timestamp": "2024-01-15T11:45:00Z",
                    "details": {
                        "merchant": "Starbucks",
                        "category": "Food & Beverage"
                    }
                }
            ]
            
            # Write multiline JSON to file (each record on multiple lines)
            with open(test_file_path, 'w') as f:
                f.write('[\n')
                for i, transaction in enumerate(test_transactions):
                    if i > 0:
                        f.write(',\n')
                    f.write(json.dumps(transaction, indent=2))
                f.write('\n]')
            
            # Call the function under test
            result_df = load_transaction_data(spark, test_file_path)
            
            # Create expected DataFrame for comparison
            expected_data = [
                ("TXN001", "CUST123", 150.75, "USD", "2024-01-15T10:30:00Z", 
                 {"merchant": "Amazon", "category": "Electronics"}),
                ("TXN002", "CUST456", 89.99, "USD", "2024-01-15T11:45:00Z", 
                 {"merchant": "Starbucks", "category": "Food & Beverage"})
            ]
            
            # Verify the function returns a DataFrame
            assert isinstance(result_df, DataFrame), "Function should return a PySpark DataFrame"
            
            # Verify the number of rows
            actual_count = result_df.count()
            expected_count = len(test_transactions)
            assert actual_count == expected_count, f"Expected {expected_count} rows, got {actual_count}"
            
            # Verify schema contains expected columns
            actual_columns = set(result_df.columns)
            expected_columns = {"transaction_id", "customer_id", "amount", "currency", "timestamp", "details"}
            assert expected_columns.issubset(actual_columns), f"Missing expected columns. Got: {actual_columns}"
            
            # Verify specific data values
            result_list = result_df.collect()
            
            # Sort both lists by transaction_id for consistent comparison
            result_list = sorted(result_list, key=lambda x: x.transaction_id)
            
            # Verify first transaction
            first_row = result_list[0]
            assert first_row.transaction_id == "TXN001"
            assert first_row.customer_id == "CUST123"
            assert first_row.amount == 150.75
            assert first_row.currency == "USD"
            assert first_row.details.merchant == "Amazon"
            
            # Verify second transaction
            second_row = result_list[1]
            assert second_row.transaction_id == "TXN002"
            assert second_row.customer_id == "CUST456"
            assert second_row.amount == 89.99
            assert second_row.details.category == "Food & Beverage"
            
    finally:
        # Clean up Spark session
        spark.stop()