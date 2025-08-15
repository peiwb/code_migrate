import pytest
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import tempfile
import os
import json


def test_load_transaction_data():
    """Test the load_transaction_data function with multiline JSON data."""
    # Create a local SparkSession for testing
    spark = SparkSession.builder \
        .appName("test_load_transaction_data") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    try:
        # Generate realistic multiline JSON test data
        test_transactions = [
            {
                "transaction_id": "TXN001",
                "customer_id": "CUST123",
                "amount": 150.75,
                "currency": "USD",
                "items": [
                    {"product": "Laptop", "quantity": 1},
                    {"product": "Mouse", "quantity": 2}
                ]
            },
            {
                "transaction_id": "TXN002",
                "customer_id": "CUST456",
                "amount": 89.50,
                "currency": "USD",
                "items": [
                    {"product": "Book", "quantity": 3}
                ]
            }
        ]
        
        # Create temporary file with multiline JSON data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            for transaction in test_transactions:
                # Write each JSON object with pretty printing (multiline)
                json.dump(transaction, temp_file, indent=2)
                temp_file.write('\n')
            temp_file_path = temp_file.name
        
        # Call the function under test
        result_df = load_transaction_data(spark, temp_file_path)
        
        # Collect results for assertion
        result_data = result_df.collect()
        
        # Verify the function loaded data correctly
        assert len(result_data) == 2, f"Expected 2 records, got {len(result_data)}"
        
        # Convert to list of dictionaries for easier comparison
        result_dict_list = [row.asDict() for row in result_data]
        
        # Sort both lists by transaction_id for consistent comparison
        result_dict_list.sort(key=lambda x: x['transaction_id'])
        expected_sorted = sorted(test_transactions, key=lambda x: x['transaction_id'])
        
        # Verify specific field values
        assert result_dict_list[0]['transaction_id'] == 'TXN001'
        assert result_dict_list[0]['customer_id'] == 'CUST123'
        assert result_dict_list[0]['amount'] == 150.75
        assert result_dict_list[0]['currency'] == 'USD'
        
        assert result_dict_list[1]['transaction_id'] == 'TXN002'
        assert result_dict_list[1]['customer_id'] == 'CUST456'
        assert result_dict_list[1]['amount'] == 89.50
        
        # Verify nested array structure is preserved
        assert len(result_dict_list[0]['items']) == 2
        assert result_dict_list[0]['items'][0]['product'] == 'Laptop'
        assert result_dict_list[0]['items'][0]['quantity'] == 1
        
        assert len(result_dict_list[1]['items']) == 1
        assert result_dict_list[1]['items'][0]['product'] == 'Book'
        assert result_dict_list[1]['items'][0]['quantity'] == 3
        
        # Verify DataFrame schema includes expected columns
        expected_columns = {'transaction_id', 'customer_id', 'amount', 'currency', 'items'}
        actual_columns = set(result_df.columns)
        assert expected_columns.issubset(actual_columns), f"Missing columns: {expected_columns - actual_columns}"
        
    finally:
        # Clean up temporary file
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        
        # Stop SparkSession
        spark.stop()