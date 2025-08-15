```python
def load_transaction_data(session, data_path: str):
    """Load transaction data from JSON files with multiline support.
    
    This function reads JSON transaction data files that may contain multiline JSON records.
    The multiline option is crucial for parsing JSON files where individual records span
    multiple lines, which is common in transaction data exports from various systems.
    
    Performance Considerations:
        - The multiline option requires reading the entire file into memory before parsing,
          which can be a bottleneck for very large files
        - For better performance with large datasets, consider preprocessing data to single-line JSON
        - Dynamic partitioning mentioned in summary may refer to automatic partition discovery
          if the data_path contains partitioned directory structure
    
    Args:
        session: Active Snowpark session used for reading data. Must be properly
            configured with appropriate memory settings for the expected data size.
        data_path (str): File system path to the transaction data. Can be a single file,
            directory containing JSON files, or a glob pattern. Supports various file systems
            accessible to Snowflake.
    
    Returns:
        DataFrame: Snowpark DataFrame containing the loaded transaction data with schema
            automatically inferred from the JSON structure. Column names and types will
            match the JSON field names and value types.
    
    Raises:
        Exception: If the specified path does not exist or contains no readable files
        Exception: If the data_path is not accessible
        Exception: If JSON files contain invalid JSON syntax
    """
    # TODO: [MANUAL MIGRATION REQUIRED] - Snowpark does not have direct equivalent for reading JSON files with multiline option
    # Snowflake typically requires data to be staged first, then loaded using COPY INTO or similar commands
    # Consider using session.read.option() or file format options if available
    df = session.read \
        .option("multiline", "true") \
        .json(data_path)

    return df
```