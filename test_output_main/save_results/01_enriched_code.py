def save_results(df: DataFrame, output_path: str, format_type: str = "parquet") -> None:
    """Save processed DataFrame to specified location with format validation.
    
    This function provides a unified interface for saving DataFrames in multiple formats.
    It uses coalesce(1) to optimize for small to medium datasets by reducing the number
    of output files, which is particularly beneficial for downstream systems that prefer
    fewer files over parallelism.
    
    Args:
        df: The PySpark DataFrame to save. Must be a valid DataFrame with at least
            one column. Performance note: Function assumes DataFrame fits comfortably
            in a single partition after coalesce(1).
        output_path: Target directory path for saving the DataFrame. Path will be
            created if it doesn't exist. For cloud storage (S3, ADLS, GCS), ensure
            proper credentials are configured in SparkSession.
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
        - Migration consideration: coalesce(1) behavior may need adjustment based on
          target platform's file handling capabilities
    """
    # Reduce partitions to 1 for consolidated output - trade parallelism for fewer files
    # This is optimal for small-medium datasets but may cause memory issues for large data
    writer = df.coalesce(1).write.mode("overwrite")

    # Format selection with case-insensitive matching for user convenience
    if format_type.lower() == "parquet":
        # Parquet: Best for performance, schema preservation, and compression
        writer.parquet(output_path)
    elif format_type.lower() == "csv":
        # CSV: Include headers for external tool compatibility (Excel, etc.)
        writer.option("header", "true").csv(output_path)
    elif format_type.lower() == "json":
        # JSON: Each row becomes a separate JSON object (newline-delimited JSON)
        writer.json(output_path)
    else:
        # Fail fast on unsupported formats to prevent silent failures
        raise ValueError(f"Unsupported format: {format_type}")