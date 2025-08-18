```python
def load_transaction_data(session, data_path: str):
    """Load transaction data from JSON files with support for multiline JSON records.
    
    This function reads JSON transaction data files using Snowpark's DataFrameReader.
    It specifically handles multiline JSON format which is common for complex
    transaction records that span multiple lines for readability.
    
    Args:
        session: Active Snowpark session instance used for reading data.
            Must be properly initialized with appropriate configurations.
        data_path (str): File system path to the JSON transaction data.
            Can be a single file path, directory path, or glob pattern.
            Supports stage locations and other Snowflake-compatible
            file systems.
    
    Returns:
        DataFrame: Snowpark DataFrame containing the loaded transaction data.
            Schema is inferred automatically from the JSON structure.
            Column names and types depend on the source JSON format.
    
    Note:
        - The function handles JSON records that span multiple lines
        - Schema inference happens during the read operation, requiring a full
          scan of the data which can be expensive for large datasets
        - Consider providing an explicit schema for production workloads
        - No validation or error handling is performed on the input path
    """
    # TODO: [MANUAL MIGRATION REQUIRED] - Snowpark's JSON reading capabilities may differ from PySpark's multiline option
    # The exact equivalent of multiline=true parsing needs to be verified for Snowpark
    df = session.read \
        .option("multiline", "true") \
        .json(data_path)
    
    return df
```