def load_transaction_data(spark: SparkSession, data_path: str) -> DataFrame:
    """Load transaction data from JSON files with multiline support.
    
    This function reads JSON transaction data files that may contain multiline JSON records.
    The multiline option is crucial for parsing JSON files where individual records span
    multiple lines, which is common in transaction data exports from various systems.
    
    Performance Considerations:
        - The multiline option requires Spark to read the entire file into memory on a single
          executor before parsing, which can be a bottleneck for very large files
        - For better performance with large datasets, consider preprocessing data to single-line JSON
        - Dynamic partitioning mentioned in summary may refer to automatic partition discovery
          if the data_path contains partitioned directory structure
    
    Args:
        spark (SparkSession): Active Spark session used for reading data. Must be properly
            configured with appropriate memory and executor settings for the expected data size.
        data_path (str): File system path to the transaction data. Can be a single file,
            directory containing JSON files, or a glob pattern. Supports local filesystem,
            HDFS, S3, and other Hadoop-compatible file systems.
    
    Returns:
        DataFrame: PySpark DataFrame containing the loaded transaction data with schema
            automatically inferred from the JSON structure. Column names and types will
            match the JSON field names and value types.
    
    Raises:
        AnalysisException: If the specified path does not exist or contains no readable files
        FileNotFoundException: If the data_path is not accessible
        ParseException: If JSON files contain invalid JSON syntax
    """
    # Enable multiline JSON parsing - critical for transaction data that often contains
    # nested structures or arrays that span multiple lines in the source files
    # Note: This setting forces single-threaded parsing per file, impacting performance
    df = spark.read \
        .option("multiline", "true") \
        .json(data_path)  # Schema inference happens automatically on first read

    return df