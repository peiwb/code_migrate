def load_transaction_data(spark: SparkSession, data_path: str) -> DataFrame:
    """Load transaction data from JSON files with support for multiline JSON records.
    
    This function reads JSON transaction data files using Spark's DataFrameReader.
    It specifically handles multiline JSON format which is common for complex
    transaction records that span multiple lines for readability.
    
    Args:
        spark (SparkSession): Active Spark session instance used for reading data.
            Must be properly initialized with appropriate configurations.
        data_path (str): File system path to the JSON transaction data.
            Can be a single file path, directory path, or glob pattern.
            Supports HDFS, S3, local file system, and other Hadoop-compatible
            file systems.
    
    Returns:
        DataFrame: PySpark DataFrame containing the loaded transaction data.
            Schema is inferred automatically from the JSON structure.
            Column names and types depend on the source JSON format.
    
    Note:
        - The function uses multiline=true option which is essential for JSON
          records that span multiple lines but may impact performance on large files
        - Schema inference happens during the read operation, requiring a full
          scan of the data which can be expensive for large datasets
        - Consider providing an explicit schema for production workloads
        - No validation or error handling is performed on the input path
    """
    # Enable multiline JSON parsing - critical for transaction records that
    # may contain nested structures formatted across multiple lines
    # Performance note: multiline parsing requires reading entire file into memory
    # per partition, which may cause memory issues with very large files
    df = spark.read \
        .option("multiline", "true") \
        .json(data_path)
    
    # Return DataFrame with auto-inferred schema
    # Migration note: Schema inference behavior may vary across platforms
    # Consider capturing and preserving the inferred schema for consistency
    return df