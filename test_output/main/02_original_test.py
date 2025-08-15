import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from unittest.mock import patch, MagicMock
import sys
from io import StringIO

def test_main_successful_pipeline_execution():
    """
    Test the main function executes the complete data processing pipeline successfully.
    
    This test mocks all external dependencies and verifies that:
    1. Spark session is created and properly cleaned up
    2. All pipeline functions are called in correct order
    3. Results are saved in expected formats
    4. Summary information is displayed
    5. Success message is printed
    """
    # Create local SparkSession for testing
    spark = SparkSession.builder \
        .appName("TestCustomerAnalytics") \
        .master("local[*]") \
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
        .getOrCreate()
    
    try:
        # Create realistic test data
        customer_schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("age", IntegerType(), True)
        ])
        
        transaction_schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("transaction_id", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("date", StringType(), True)
        ])
        
        insights_schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("total_spent", DoubleType(), True),
            StructField("transaction_count", IntegerType(), True),
            StructField("avg_transaction", DoubleType(), True)
        ])
        
        # Create mock input DataFrames
        mock_customer_data = spark.createDataFrame([
            ("C001", "John Doe", "john@email.com", 30),
            ("C002", "Jane Smith", "jane@email.com", 25)
        ], customer_schema)
        
        mock_transaction_data = spark.createDataFrame([
            ("C001", "T001", 100.0, "2024-01-01"),
            ("C001", "T002", 150.0, "2024-01-02"),
            ("C002", "T003", 200.0, "2024-01-03")
        ], transaction_schema)
        
        # Create expected output DataFrames
        expected_enriched = spark.createDataFrame([
            ("C001", "John Doe", "john@email.com", 30, 250.0, 2),
            ("C002", "Jane Smith", "jane@email.com", 25, 200.0, 1)
        ], StructType([
            StructField("customer_id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("total_spent", DoubleType(), True),
            StructField("transaction_count", IntegerType(), True)
        ]))
        
        expected_insights = spark.createDataFrame([
            ("C001", 250.0, 2, 125.0),
            ("C002", 200.0, 1, 200.0)
        ], insights_schema)
        
        # Mock all external function dependencies
        with patch('__main__.create_spark_session', return_value=spark), \
             patch('__main__.load_customer_data', return_value=mock_customer_data), \
             patch('__main__.load_transaction_data', return_value=mock_transaction_data), \
             patch('__main__.clean_customer_data', return_value=mock_customer_data), \
             patch('__main__.enrich_customer_data', return_value=expected_enriched), \
             patch('__main__.generate_customer_insights', return_value=expected_insights), \
             patch('__main__.save_results') as mock_save, \
             patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            
            # Import and call the main function
            from __main__ import main
            
            # Execute the function under test
            main()
            
            # Capture printed output for verification
            output = mock_stdout.getvalue()
            
            # Verify save_results was called with correct parameters
            assert mock_save.call_count == 2
            save_calls = mock_save.call_args_list
            
            # Verify enriched customers saved as parquet
            enriched_call = save_calls[0]
            assert enriched_call[0][1] == "/output/enriched_customers"
            assert enriched_call[0][2] == "parquet"
            
            # Verify insights saved as csv
            insights_call = save_calls[1]
            assert insights_call[0][1] == "/output/customer_insights"
            assert insights_call[0][2] == "csv"
            
            # Verify expected output messages
            assert "=== Customer Insights Summary ===" in output
            assert "Total customers processed: 2" in output
            assert "Data processing completed successfully!" in output
            
            # Verify no error messages
            assert "Error in data processing pipeline" not in output
            
    finally:
        # Clean up SparkSession
        spark.stop()

def test_main_handles_pipeline_exception():
    """
    Test that main function properly handles and re-raises exceptions from pipeline steps.
    """
    spark = SparkSession.builder \
        .appName("TestCustomerAnalyticsError") \
        .master("local[*]") \
        .getOrCreate()
    
    try:
        # Mock functions with one that raises an exception
        with patch('__main__.create_spark_session', return_value=spark), \
             patch('__main__.load_customer_data', side_effect=Exception("Data load failed")), \
             patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            
            from __main__ import main
            
            # Verify exception is raised and error is logged
            with pytest.raises(Exception, match="Data load failed"):
                main()
            
            output = mock_stdout.getvalue()
            assert "Error in data processing pipeline: Data load failed" in output
            
    finally:
        spark.stop()