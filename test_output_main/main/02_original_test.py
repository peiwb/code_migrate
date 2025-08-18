import pytest
from unittest.mock import Mock, patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import sys
from io import StringIO


def test_main_successful_pipeline():
    """Test the main function with successful pipeline execution."""
    # Create local Spark session for testing
    spark = SparkSession.builder \
        .appName("TestCustomerAnalytics") \
        .master("local[2]") \
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
        .getOrCreate()
    
    try:
        # Define schemas for test data
        customer_schema = StructType([
            StructField("customer_id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("age", IntegerType(), True)
        ])
        
        transaction_schema = StructType([
            StructField("customer_id", IntegerType(), True),
            StructField("amount", DoubleType(), True),
            StructField("transaction_date", StringType(), True)
        ])
        
        insights_schema = StructType([
            StructField("customer_id", IntegerType(), True),
            StructField("total_spent", DoubleType(), True),
            StructField("transaction_count", IntegerType(), True),
            StructField("customer_segment", StringType(), True)
        ])
        
        # Create realistic input data
        customer_data = spark.createDataFrame([
            (1, "John Doe", "john@example.com", 30),
            (2, "Jane Smith", "jane@example.com", 25),
            (3, "Bob Johnson", "bob@example.com", 35)
        ], schema=customer_schema)
        
        transaction_data = spark.createDataFrame([
            (1, 100.0, "2023-01-01"),
            (1, 150.0, "2023-01-15"),
            (2, 200.0, "2023-01-10"),
            (3, 75.0, "2023-01-05")
        ], schema=transaction_schema)
        
        # Create expected output data
        expected_enriched_customers = spark.createDataFrame([
            (1, "John Doe", "john@example.com", 30, 250.0, 2),
            (2, "Jane Smith", "jane@example.com", 25, 200.0, 1),
            (3, "Bob Johnson", "bob@example.com", 35, 75.0, 1)
        ], schema=StructType([
            StructField("customer_id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("total_spent", DoubleType(), True),
            StructField("transaction_count", IntegerType(), True)
        ]))
        
        expected_insights = spark.createDataFrame([
            (1, 250.0, 2, "High Value"),
            (2, 200.0, 1, "Medium Value"),
            (3, 75.0, 1, "Low Value")
        ], schema=insights_schema)
        
        # Mock all the external functions
        with patch('__main__.create_spark_session', return_value=spark) as mock_create_spark, \
             patch('__main__.load_customer_data', return_value=customer_data) as mock_load_customers, \
             patch('__main__.load_transaction_data', return_value=transaction_data) as mock_load_transactions, \
             patch('__main__.clean_customer_data', return_value=customer_data) as mock_clean, \
             patch('__main__.enrich_customer_data', return_value=expected_enriched_customers) as mock_enrich, \
             patch('__main__.generate_customer_insights', return_value=expected_insights) as mock_insights, \
             patch('__main__.save_results') as mock_save:
            
            # Capture stdout to test print statements
            captured_output = StringIO()
            with patch('sys.stdout', captured_output):
                # Import and call the main function
                from __main__ import main
                main()
            
            # Verify all functions were called with correct parameters
            mock_create_spark.assert_called_once_with("CustomerAnalytics")
            mock_load_customers.assert_called_once_with(spark, "/data/customers.csv")
            mock_load_transactions.assert_called_once_with(spark, "/data/transactions.json")
            mock_clean.assert_called_once_with(customer_data)
            mock_enrich.assert_called_once_with(customer_data, transaction_data)
            mock_insights.assert_called_once_with(expected_enriched_customers)
            
            # Verify save_results was called twice with correct parameters
            assert mock_save.call_count == 2
            mock_save.assert_any_call(expected_enriched_customers, "/output/enriched_customers", "parquet")
            mock_save.assert_any_call(expected_insights, "/output/customer_insights", "csv")
            
            # Verify console output contains expected messages
            output = captured_output.getvalue()
            assert "=== Customer Insights Summary ===" in output
            assert "Total customers processed: 3" in output
            assert "Data processing completed successfully!" in output
            
    finally:
        spark.stop()


def test_main_with_exception_handling():
    """Test the main function handles exceptions properly."""
    spark = SparkSession.builder \
        .appName("TestCustomerAnalyticsError") \
        .master("local[2]") \
        .getOrCreate()
    
    try:
        # Mock functions to simulate an error
        with patch('__main__.create_spark_session', return_value=spark), \
             patch('__main__.load_customer_data', side_effect=Exception("File not found")), \
             patch('sys.stdout', StringIO()) as captured_output:
            
            # Test that exception is properly handled and re-raised
            with pytest.raises(Exception, match="File not found"):
                from __main__ import main
                main()
            
            # Verify error message was printed
            output = captured_output.getvalue()
            assert "Error in data processing pipeline: File not found" in output
            
    finally:
        spark.stop()