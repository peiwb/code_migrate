```python
import pytest
import tempfile
import shutil
import os
from snowflake.snowpark import Session, DataFrame
from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType


def test_save_results():
    """Test the save_results function with different formats and scenarios."""
    # TODO: [MANUAL MIGRATION REQUIRED] - Create Snowpark Session for testing
    # Snowpark requires connection parameters to Snowflake instance
    session = None  # Need to configure with actual Snowflake connection parameters
    
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
        
        input_df = session.create_dataframe(test_data, schema)
        
        # Create temporary directory for test outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Test 1: Save as Parquet (default format)
            parquet_path = os.path.join(temp_dir, "test_parquet")
            save_results(input_df, parquet_path)
            
            # TODO: [MANUAL MIGRATION REQUIRED] - Snowpark doesn't support local file operations
            # Verify parquet file was created and data is correct
            assert os.path.exists(parquet_path)
            parquet_result = session.read.parquet(parquet_path)
            assert parquet_result.count() == 4
            assert parquet_result.columns == ["id", "name", "department"]
            
            # Verify data integrity by comparing sorted results
            expected_data_sorted = sorted(test_data)
            actual_data_sorted = sorted(parquet_result.collect())
            assert actual_data_sorted == [tuple(row) for row in expected_data_sorted]
            
            # Test 2: Save as CSV with explicit format specification
            csv_path = os.path.join(temp_dir, "test_csv")
            save_results(input_df, csv_path, "csv")
            
            # TODO: [MANUAL MIGRATION REQUIRED] - Snowpark CSV reading with options
            assert os.path.exists(csv_path)
            csv_result = session.read.option("header", "true").option("inferSchema", "true").csv(csv_path)
            assert csv_result.count() == 4
            assert csv_result.columns == ["id", "name", "department"]
            
            # Test 3: Save as JSON
            json_path = os.path.join(temp_dir, "test_json")
            save_results(input_df, json_path, "json")
            
            # TODO: [MANUAL MIGRATION REQUIRED] - Snowpark JSON reading
            assert os.path.exists(json_path)
            json_result = session.read.json(json_path)
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
            new_input_df = session.create_dataframe(new_test_data, schema)
            
            # Second write should overwrite
            save_results(new_input_df, overwrite_path)
            overwrite_result = session.read.parquet(overwrite_path)
            assert overwrite_result.count() == 1
            assert overwrite_result.collect()[0]["name"] == "Eve Brown"
            
    finally:
        # TODO: [MANUAL MIGRATION REQUIRED] - Snowpark Session cleanup
        if session:
            session.close()
```