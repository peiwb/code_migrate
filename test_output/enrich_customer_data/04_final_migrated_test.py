```python
import pytest
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, count, sum as snowpark_sum, avg, when
from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType, DoubleType

def test_enrich_customer_data():
    """
    Test the enrich_customer_data function with comprehensive scenarios.
    """
    # Create Snowpark session for testing
    session = Session.builder.configs({
        "account": "your_account",
        "user": "your_user", 
        "password": "your_password",
        "role": "your_role",
        "warehouse": "your_warehouse",
        "database": "your_database",
        "schema": "your_schema"
    }).create()
    
    try:
        # Define schemas for consistent data types
        customer_schema = StructType([
            StructField("customer_id", StringType(), False),
            StructField("customer_name", StringType(), True),
            StructField("email", StringType(), True)
        ])
        
        transaction_schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("customer_id", StringType(), False),
            StructField("amount", DoubleType(), True)
        ])
        
        # Create realistic input data
        customer_data = [
            ("C001", "John Doe", "john@example.com"),
            ("C002", "Jane Smith", "jane@example.com"), 
            ("C003", "Bob Johnson", "bob@example.com"),
            ("C004", "Alice Brown", "alice@example.com")  # Customer with no transactions
        ]
        
        transaction_data = [
            ("T001", "C001", 15000.0),  # Premium tier customer
            ("T002", "C001", 2000.0),
            ("T003", "C002", 7500.0),   # Gold tier customer  
            ("T004", "C002", 1000.0),
            ("T005", "C003", 800.0),    # Silver tier customer
            ("T006", "C003", 500.0),
            ("T007", "C003", 200.0)
            # C004 has no transactions - will test Bronze tier default
        ]
        
        customer_df = session.create_dataframe(customer_data, customer_schema)
        transaction_df = session.create_dataframe(transaction_data, transaction_schema)
        
        # Define expected output data
        expected_schema = StructType([
            StructField("customer_id", StringType(), False),
            StructField("customer_name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("transaction_count", IntegerType(), True),
            StructField("total_transaction_amount", DoubleType(), True),
            StructField("avg_transaction_amount", DoubleType(), True),
            StructField("customer_tier", StringType(), True)
        ])
        
        expected_data = [
            ("C001", "John Doe", "john@example.com", 2, 17000.0, 8500.0, "Premium"),
            ("C002", "Jane Smith", "jane@example.com", 2, 8500.0, 4250.0, "Gold"),
            ("C003", "Bob Johnson", "bob@example.com", 3, 1500.0, 500.0, "Silver"),
            ("C004", "Alice Brown", "alice@example.com", None, None, None, "Bronze")
        ]
        
        expected_df = session.create_dataframe(expected_data, expected_schema)
        
        # Call the function under test
        actual_df = enrich_customer_data(customer_df, transaction_df)
        
        # Sort both DataFrames by customer_id for consistent comparison
        actual_df_sorted = actual_df.order_by("customer_id")
        expected_df_sorted = expected_df.order_by("customer_id")
        
        # TODO: [MANUAL MIGRATION REQUIRED] - assertDataFrameEqual not available in Snowpark
        # Manual comparison needed
        actual_rows = actual_df_sorted.collect()
        expected_rows = expected_df_sorted.collect()
        
        assert len(actual_rows) == len(expected_rows)
        for actual, expected in zip(actual_rows, expected_rows):
            assert actual == expected
        
        # Additional verification: Check specific business logic
        result_list = actual_df.collect()
        
        # Verify Premium tier assignment
        premium_customer = [row for row in result_list if row["CUSTOMER_ID"] == "C001"][0]
        assert premium_customer["CUSTOMER_TIER"] == "Premium"
        assert premium_customer["TOTAL_TRANSACTION_AMOUNT"] > 10000
        
        # Verify Gold tier assignment  
        gold_customer = [row for row in result_list if row["CUSTOMER_ID"] == "C002"][0]
        assert gold_customer["CUSTOMER_TIER"] == "Gold"
        assert 5000 < gold_customer["TOTAL_TRANSACTION_AMOUNT"] <= 10000
        
        # Verify Silver tier assignment
        silver_customer = [row for row in result_list if row["CUSTOMER_ID"] == "C003"][0]
        assert silver_customer["CUSTOMER_TIER"] == "Silver"
        assert 1000 < silver_customer["TOTAL_TRANSACTION_AMOUNT"] <= 5000
        
        # Verify Bronze tier for customer with no transactions
        bronze_customer = [row for row in result_list if row["CUSTOMER_ID"] == "C004"][0]
        assert bronze_customer["CUSTOMER_TIER"] == "Bronze"
        assert bronze_customer["TOTAL_TRANSACTION_AMOUNT"] is None
        
    finally:
        # Clean up Snowpark session
        session.close()
```