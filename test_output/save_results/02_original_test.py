import pytest
import tempfile
import shutil
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


def test_save_results():
    """Test the save_results function with different formats and scenarios."""
    # Create local SparkSession for testing
    spark = SparkSession.builder \
        .appName("test_save_results") \
        .master("local[2]") \
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
        .getOrCreate()
    
    try:
        # Define schema for test data
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("department", StringType(), True)
        ])
        
        # Create realistic input test data
        test_data = [
            (1, "Alice Johnson", "Engineering"),
            (2, "Bob Smith", "Marketing"), 
            (3, "Carol Davis", "Sales"),
            (4, "David Wilson", "Engineering")
        ]
        
        input_df = spark.createDataFrame(test_data, schema)
        
        # Create temporary directory for test outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Test 1: Save as Parquet (default format)
            parquet_path = os.path.join(temp_dir, "test_parquet")
            save_results(input_df, parquet_path)
            
            # Verify parquet file was created and data is correct
            assert os.path.exists(parquet_path)
            parquet_result = spark.read.parquet(parquet_path)
            assert parquet_result.count() == 4
            assert parquet_result.columns == ["id", "name", "department"]
            
            # Verify data integrity by comparing sorted results
            expected_data_sorted = sorted(test_data)
            actual_data_sorted = sorted(parquet_result.collect())
            assert actual_data_sorted == [tuple(row) for row in expected_data_sorted]
            
            # Test 2: Save as CSV with explicit format specification
            csv_path = os.path.join(temp_dir, "test_csv")
            save_results(input_df, csv_path, "csv")
            
            # Verify CSV file was created with headers
            assert os.path.exists(csv_path)
            csv_result = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)
            assert csv_result.count() == 4
            assert csv_result.columns == ["id", "name", "department"]
            
            # Test 3: Save as JSON
            json_path = os.path.join(temp_dir, "test_json")
            save_results(input_df, json_path, "json")
            
            # Verify JSON file was created and data is correct
            assert os.path.exists(json_path)
            json_result = spark.read.json(json_path)
            assert json_result.count() == 4
            assert set(json_result.columns) == {"id", "name", "department"}
            
            # Test 4: Case insensitive format handling
            parquet_upper_path = os.path.join(temp_dir, "test_parquet_upper")
            save_results(input_df, parquet_upper_path, "PARQUET")
            assert os.path.exists(parquet_upper_path)
            
            # Test 5: Error handling for unsupported format
            with pytest.raises(ValueError, match="Unsupported format: xml"):
                save_results(input_df, os.path.join(temp_dir, "test_error"), "xml")
            
            # Test 6: Verify overwrite behavior
            # First write
            overwrite_path = os.path.join(temp_dir, "test_overwrite")
            save_results(input_df, overwrite_path)
            first_write_files = os.listdir(overwrite_path)
            
            # Create different data for second write
            new_test_data = [(5, "Eve Brown", "HR")]
            new_input_df = spark.createDataFrame(new_test_data, schema)
            
            # Second write should overwrite
            save_results(new_input_df, overwrite_path)
            overwrite_result = spark.read.parquet(overwrite_path)
            assert overwrite_result.count() == 1
            assert overwrite_result.collect()[0]["name"] == "Eve Brown"
            
    finally:
        # Clean up SparkSession
        spark.stop()