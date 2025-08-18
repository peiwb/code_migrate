import pytest
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, when
from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType
from unittest.mock import patch


def test_clean_customer_data():
    """
    Test the clean_customer_data function with various data scenarios.
    """
    # Create Snowpark Session for testing using environment variables or test configuration
    import os
    session = Session.builder.configs({
        "account": os.getenv("SNOWFLAKE_TEST_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_TEST_USER"),
        "password": os.getenv("SNOWFLAKE_TEST_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_TEST_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_TEST_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_TEST_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_TEST_SCHEMA")
    }).create()

    try:
        # Define schema for consistent testing
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("email", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("name", StringType(), True)
        ])

        # Generate realistic input data with various edge cases
        input_data = [
            ("C001", "john@email.com", 22, "John Doe"),        # Young
            ("C002", "jane@email.com", 35, "Jane Smith"),      # Adult
            ("C003", "bob@email.com", 70, "Bob Johnson"),      # Senior
            ("C001", "john2@email.com", 23, "John Duplicate"), # Duplicate customer_id
            (None, "null@email.com", 30, "Null ID"),          # Null customer_id
            ("C004", None, 25, "Null Email"),                 # Null email
            ("C005", "senior@email.com", 65, "Senior Boundary"), # Boundary case for Senior
            ("C006", "young@email.com", 24, "Young Boundary")   # Boundary case for Young
        ]

        input_df = session.create_dataframe(input_data, schema)

        # Define expected output data after cleaning
        expected_data = [
            ("C001", "john@email.com", 22, "John Doe", "Young"),
            ("C002", "jane@email.com", 35, "Jane Smith", "Adult"),
            ("C003", "bob@email.com", 70, "Bob Johnson", "Senior"),
            ("C005", "senior@email.com", 65, "Senior Boundary", "Senior"),
            ("C006", "young@email.com", 24, "Young Boundary", "Young")
        ]

        expected_schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("email", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("age_group", StringType(), True)
        ])

        expected_df = session.create_dataframe(expected_data, expected_schema)

        # Mock the validate_data_quality function to return input unchanged for testing
        with patch('__main__.validate_data_quality', side_effect=lambda x: x):
            # Call the function under test
            actual_df = clean_customer_data(input_df)

            # Collect results for comparison (sort by customer_id for consistent comparison)
            actual_result = actual_df.order_by("customer_id").collect()
            expected_result = expected_df.order_by("customer_id").collect()

            # Assert that the results match exactly
            assert len(actual_result) == len(expected_result), f"Row count mismatch: expected {len(expected_result)}, got {len(actual_result)}"

            for actual_row, expected_row in zip(actual_result, expected_result):
                assert actual_row == expected_row, f"Row mismatch: expected {expected_row}, got {actual_row}"

            # Additional assertions for specific business logic
            age_groups = [row['age_group'] for row in actual_result]
            assert "Young" in age_groups, "Should contain Young age group"
            assert "Adult" in age_groups, "Should contain Adult age group"
            assert "Senior" in age_groups, "Should contain Senior age group"

            # Verify no duplicates remain
            customer_ids = [row['customer_id'] for row in actual_result]
            assert len(customer_ids) == len(set(customer_ids)), "Should not contain duplicate customer_ids"

            # Verify no null values in required fields
            for row in actual_result:
                assert row['customer_id'] is not None, "customer_id should not be null"
                assert row['email'] is not None, "email should not be null"

    finally:
        # Clean up Snowpark Session
        session.close()