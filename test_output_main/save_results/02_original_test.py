import pytest
import tempfile
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import os


def test_save_results():
    """Test save_results function with multiple formats and edge cases."""
    # Create local SparkSession for testing
    spark = SparkSession.builder \
        .appName("test_save_results") \
        .master("local[2]") \
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
        .getOrCreate()
    
    try:
        # Define schema for consistent testing
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("category", StringType(), True)
        ])
        
        # Create realistic input DataFrame
        input_data = [
            (1, "Alice", "Engineer"),
            (2, "Bob", "Manager"),
            (3, "Charlie", "Analyst")
        ]
        input_df = spark.createDataFrame(input_data, schema)
        
        # Create temporary directory for test outputs
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test 1: Parquet format (default)
            parquet_path = os.path.join(temp_dir, "test_parquet")
            save_results(input_df, parquet_path)
            
            # Verify parquet output
            saved_parquet_df = spark.read.parquet(parquet_path)
            assert saved_parquet_df.count() == 3
            assert set(saved_parquet_df.columns) == {"id", "name", "category"}
            
            # Verify data integrity
            parquet_data = saved_parquet_df.collect()
            expected_data = input_df.collect()
            assert len(parquet_data) == len(expected_data)
            
            # Test 2: CSV format with explicit parameter
            csv_path = os.path.join(temp_dir, "test_csv")
            save_results(input_df, csv_path, "csv")
            
            # Verify CSV output (schema inference needed due to CSV limitations)
            saved_csv_df = spark.read.option("header", "true").csv(csv_path)
            assert saved_csv_df.count() == 3
            assert set(saved_csv_df.columns) == {"id", "name", "category"}
            
            # Test 3: JSON format
            json_path = os.path.join(temp_dir, "test_json")
            save_results(input_df, json_path, "JSON")  # Test case insensitivity
            
            # Verify JSON output
            saved_json_df = spark.read.json(json_path)
            assert saved_json_df.count() == 3
            assert set(saved_json_df.columns) == {"id", "name", "category"}
            
            # Test 4: Case insensitivity
            parquet_path_2 = os.path.join(temp_dir, "test_parquet_case")
            save_results(input_df, parquet_path_2, "PARQUET")
            saved_df_case = spark.read.parquet(parquet_path_2)
            assert saved_df_case.count() == 3
            
            # Test 5: Unsupported format should raise ValueError
            with pytest.raises(ValueError, match="Unsupported format: xml"):
                save_results(input_df, os.path.join(temp_dir, "test_error"), "xml")
            
            # Test 6: Empty DataFrame handling
            empty_df = spark.createDataFrame([], schema)
            empty_path = os.path.join(temp_dir, "test_empty")
            save_results(empty_df, empty_path, "parquet")
            
            saved_empty_df = spark.read.parquet(empty_path)
            assert saved_empty_df.count() == 0
            assert set(saved_empty_df.columns) == {"id", "name", "category"}
            
        finally:
            # Cleanup temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    finally:
        # Stop SparkSession
        spark.stop()