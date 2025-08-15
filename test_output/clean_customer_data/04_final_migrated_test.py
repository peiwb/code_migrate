```python
import pytest
from snowflake.snowpark import Session
from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType
from snowflake.snowpark.functions import col
from unittest.mock import patch


def test_clean_customer_data():
    """
    Test the clean_customer_data function with various data scenarios.
    """
    # TODO: [MANUAL MIGRATION REQUIRED] - Create Snowpark Session for testing (requires Snowflake connection parameters)
    session = Session.builder.configs({
        "account": "test_account",
        "user": "test_user",
        "password": "test_password",
        "role": "test_role",
        "warehouse": "test_warehouse",
        "database": "test_database",
        "schema": "test_schema"
    }).create()
    
    try:
        # Define schema for test data
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("email", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("name", StringType(), True)
        ])
        
        # Generate realistic input data with various test cases
        input_data = [
            ("C001", "john@example.com", 22, "John Doe"),      # Young customer
            ("C002", "jane@example.com", 35, "Jane Smith"),    # Adult customer  
            ("C003", "bob@example.com", 70, "Bob Johnson"),    # Senior customer
            ("C001", "john@example.com", 22, "John Doe"),      # Duplicate customer_id
            (None, "invalid@example.com", 30, "Invalid User"), # Null customer_id
            ("C004", None, 40, "No Email User"),               # Null email
            ("C005", "alice@example.com", None, "Alice Brown"), # Null age
        ]
        
        input_df = session.create_dataframe(input_data, schema)
        
        # Define expected output data after cleaning
        expected_data = [
            ("C001", "john@example.com", 22, "John Doe", "Young"),
            ("C002", "jane@example.com", 35, "Jane Smith", "Adult"),  
            ("C003", "bob@example.com", 70, "Bob Johnson", "Senior"),
            ("C005", "alice@example.com", None, "Alice Brown", "Senior"),  # Null age becomes Senior
        ]
        
        expected_schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("email", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("age_group", StringType(), True)
        ])
        
        expected_df = session.create_dataframe(expected_data, expected_schema)
        
        # Mock the validate_data_quality function to return input unchanged
        with patch('__main__.validate_data_quality', side_effect=lambda x: x):
            # Call the function under test
            actual_df = clean_customer_data(input_df)
            
            # Collect results for comparison (sorted by customer_id for consistent comparison)
            actual_result = actual_df.order_by("customer_id").collect()
            expected_result = expected_df.order_by("customer_id").collect()
            
            # Assert that the number of rows matches
            assert len(actual_result) == len(expected_result), \
                f"Expected {len(expected_result)} rows, but got {len(actual_result)}"
            
            # Assert that each row matches expected values
            for actual_row, expected_row in zip(actual_result, expected_result):
                assert actual_row == expected_row, \
                    f"Row mismatch: expected {expected_row}, got {actual_row}"
            
            # Verify that duplicates were removed
            customer_ids = [row.customer_id for row in actual_result]
            assert len(customer_ids) == len(set(customer_ids)), \
                "Duplicate customer_ids found in result"
            
            # Verify that no null customer_ids or emails remain
            null_customer_ids = actual_df.filter(col("customer_id").is_null()).count()
            null_emails = actual_df.filter(col("email").is_null()).count()
            
            assert null_customer_ids == 0, "Found null customer_ids in cleaned data"
            assert null_emails == 0, "Found null emails in cleaned data"
            
            # Verify age_group column was added and has correct values
            age_groups = {row.age_group for row in actual_result}
            expected_age_groups = {"Young", "Adult", "Senior"}
            assert age_groups.issubset(expected_age_groups), \
                f"Invalid age groups found: {age_groups - expected_age_groups}"
    
    finally:
        # Clean up Snowpark Session
        session.close()
```