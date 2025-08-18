```python
import pytest
from snowflake.snowpark import Session
from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType, DoubleType
import tempfile
import os


def test_load_customer_data():
    """Test the load_customer_data function with realistic customer data."""
    # Create Snowpark session for testing
    # TODO: [MANUAL MIGRATION REQUIRED] - Configure Snowpark session with appropriate connection parameters
    session = Session.builder.configs({
        "account": "test_account",
        "user": "test_user", 
        "password": "test_password",
        "warehouse": "test_warehouse",
        "database": "test_database",
        "schema": "test_schema"
    }).create()
    
    try:
        # Create temporary CSV file with realistic test data
        test_data_content = """customer_id,name,age,email,registration_date,total_spent
CUST001,John Smith,25,john.smith@email.com,2023-01-15,1250.75
CUST002,Jane Doe,32,jane.doe@email.com,2023-02-20,2100.50
CUST003,Bob Johnson,45,bob.johnson@email.com,2023-03-10,750.25
CUST004,Alice Brown,28,alice.brown@email.com,2023-04-05,3200.00
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write(test_data_content)
            temp_file_path = temp_file.name
        
        # Define expected schema for validation
        expected_schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("email", StringType(), True),
            StructField("registration_date", StringType(), True),
            StructField("total_spent", DoubleType(), True)
        ])
        
        # Create expected DataFrame for comparison
        expected_data = [
            ("CUST001", "John Smith", 25, "john.smith@email.com", "2023-01-15", 1250.75),
            ("CUST002", "Jane Doe", 32, "jane.doe@email.com", "2023-02-20", 2100.50),
            ("CUST003", "Bob Johnson", 45, "bob.johnson@email.com", "2023-03-10", 750.25),
            ("CUST004", "Alice Brown", 28, "alice.brown@email.com", "2023-04-05", 3200.00)
        ]
        expected_df = session.create_dataframe(expected_data, expected_schema)
        
        # Call the function under test
        result_df = load_customer_data(session, temp_file_path)
        
        # Assert schema matches exactly
        assert result_df.schema == expected_schema, f"Schema mismatch: {result_df.schema} != {expected_schema}"
        
        # Assert row count matches
        assert result_df.count() == expected_df.count(), f"Row count mismatch: {result_df.count()} != {expected_df.count()}"
        
        # Convert to sorted lists for reliable comparison
        result_data = sorted(result_df.collect())
        expected_data_rows = sorted(expected_df.collect())
        
        # Assert all data matches exactly
        assert result_data == expected_data_rows, f"Data mismatch: {result_data} != {expected_data_rows}"
        
        # Assert specific column types are correctly enforced
        # TODO: [MANUAL MIGRATION REQUIRED] - Verify dtypes property equivalent in Snowpark
        result_dtypes = {field.name: str(field.datatype) for field in result_df.schema.fields}
        assert "IntegerType" in result_dtypes["age"], "Age column should be integer type"
        assert "DoubleType" in result_dtypes["total_spent"], "Total spent column should be double type"
        assert "StringType" in result_dtypes["customer_id"], "Customer ID column should be string type"
        
    finally:
        # Clean up temporary file and Snowpark session
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        session.close()
```