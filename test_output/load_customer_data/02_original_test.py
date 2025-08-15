import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import tempfile
import os


def test_load_customer_data():
    """Test load_customer_data function with realistic CSV data."""
    # Create local SparkSession for testing
    spark = SparkSession.builder \
        .appName("test_load_customer_data") \
        .master("local[*]") \
        .getOrCreate()
    
    try:
        # Create temporary CSV file with realistic test data
        test_csv_content = """customer_id,name,age,email,registration_date,total_spent
C001,John Smith,25,john.smith@email.com,2023-01-15,150.75
C002,Jane Doe,32,jane.doe@email.com,2023-02-20,299.99
C003,Bob Johnson,45,bob.johnson@email.com,2023-01-10,75.50
C004,Alice Brown,28,,2023-03-05,0.0"""
        
        # Write test data to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write(test_csv_content)
            temp_csv_path = temp_file.name
        
        try:
            # Call function under test
            result_df = load_customer_data(spark, temp_csv_path)
            
            # Create expected DataFrame with same schema
            expected_schema = StructType([
                StructField("customer_id", StringType(), True),
                StructField("name", StringType(), True),
                StructField("age", IntegerType(), True), 
                StructField("email", StringType(), True),
                StructField("registration_date", StringType(), True),
                StructField("total_spent", DoubleType(), True)
            ])
            
            expected_data = [
                ("C001", "John Smith", 25, "john.smith@email.com", "2023-01-15", 150.75),
                ("C002", "Jane Doe", 32, "jane.doe@email.com", "2023-02-20", 299.99),
                ("C003", "Bob Johnson", 45, "bob.johnson@email.com", "2023-01-10", 75.50),
                ("C004", "Alice Brown", 28, None, "2023-03-05", 0.0)
            ]
            
            expected_df = spark.createDataFrame(expected_data, expected_schema)
            
            # Collect data for comparison
            result_rows = result_df.collect()
            expected_rows = expected_df.collect()
            
            # Assert schema matches exactly
            assert result_df.schema == expected_df.schema, "Schema mismatch"
            
            # Assert row count matches
            assert len(result_rows) == len(expected_rows), f"Row count mismatch: got {len(result_rows)}, expected {len(expected_rows)}"
            
            # Assert data content matches (sort by customer_id for consistent comparison)
            result_sorted = sorted(result_rows, key=lambda x: x.customer_id)
            expected_sorted = sorted(expected_rows, key=lambda x: x.customer_id)
            
            for i, (result_row, expected_row) in enumerate(zip(result_sorted, expected_sorted)):
                assert result_row == expected_row, f"Row {i} mismatch: got {result_row}, expected {expected_row}"
                
        finally:
            # Clean up temporary file
            os.unlink(temp_csv_path)
            
    finally:
        # Stop SparkSession
        spark.stop()