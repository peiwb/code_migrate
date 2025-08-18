```python
def save_results(df: DataFrame, output_path: str, format_type: str = "parquet") -> None:
    """Save processed DataFrame to specified location with format validation.
    
    This function provides a unified interface for saving DataFrames in multiple formats.
    It uses coalesce(1) to optimize for small to medium datasets by reducing the number
    of output files, which is particularly beneficial for downstream systems that prefer
    fewer files over parallelism.
    
    Args:
        df: The Snowpark DataFrame to save. Must be a valid DataFrame with at least
            one column. Performance note: Function assumes DataFrame fits comfortably
            in a single partition after coalesce(1).
        output_path: Target directory path for saving the DataFrame. Path will be
            created if it doesn't exist. For cloud storage (S3, ADLS, GCS), ensure
            proper credentials are configured in Snowflake session.
        format_type: Output format specification. Supported values are 'parquet',
            'csv', and 'json' (case-insensitive). Defaults to 'parquet' for optimal
            performance and schema preservation.
    
    Returns:
        None: Function performs side effects (file I/O) and returns nothing.
    
    Raises:
        ValueError: If format_type is not one of the supported formats.
        
    Note:
        - Uses overwrite mode by default - existing data at output_path will be replaced
        - coalesce(1) reduces parallelism but creates single output file per format
        - CSV format includes headers for better interoperability
    """
    # TODO: [MANUAL MIGRATION REQUIRED] - coalesce(1) equivalent in Snowpark
    coalesced_df = df.coalesce(1)
    
    # Format selection with case-insensitive matching for user convenience
    if format_type.lower() == "parquet":
        # Parquet: Best for performance, schema preservation, and compression
        # TODO: [MANUAL MIGRATION REQUIRED] - Snowpark DataFrame write to parquet with overwrite mode
        coalesced_df.write.mode("overwrite").parquet(output_path)
    elif format_type.lower() == "csv":
        # CSV: Include headers for external tool compatibility (Excel, etc.)
        # TODO: [MANUAL MIGRATION REQUIRED] - Snowpark DataFrame write to CSV with headers and overwrite mode
        coalesced_df.write.mode("overwrite").option("header", "true").csv(output_path)
    elif format_type.lower() == "json":
        # JSON: Each row becomes a separate JSON object (newline-delimited JSON)
        # TODO: [MANUAL MIGRATION REQUIRED] - Snowpark DataFrame write to JSON with overwrite mode
        coalesced_df.write.mode("overwrite").json(output_path)
    else:
        # Fail fast on unsupported formats to prevent silent failures
        raise ValueError(f"Unsupported format: {format_type}")
```