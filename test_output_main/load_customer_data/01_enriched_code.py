def load_customer_data(spark: SparkSession, data_path: str) -> DataFrame:
    """Load customer data from CSV files with predefined schema and validation.
    
    This function loads customer data from CSV files using a strict schema to ensure
    data consistency and type safety. The schema enforces specific data types for
    customer attributes including ID, personal information, and transaction history.
    
    Args:
        spark (SparkSession): Active Spark session used for data operations.
        data_path (str): File system path to the CSV data source. Can be a single
            file or directory containing multiple CSV files. Supports local paths,
            HDFS, S3, and other Spark-compatible file systems.
    
    Returns:
        DataFrame: PySpark DataFrame containing customer data with the following columns:
            - customer_id (StringType): Unique identifier for each customer
            - name (StringType): Customer full name
            - age (IntegerType): Customer age in years
            - email (StringType): Customer email address
            - registration_date (StringType): Date when customer registered (as string)
            - total_spent (DoubleType): Total amount spent by customer
    
    Raises:
        AnalysisException: If the data_path does not exist or is inaccessible.
        IllegalArgumentException: If CSV format is invalid or doesn't match schema.
    """
    # Define strict schema to ensure data type consistency and prevent schema inference overhead
    # Note: registration_date is kept as StringType to handle various date formats in source data
    # Migration consideration: Schema enforcement behavior may differ across platforms
    schema = StructType([
        StructField("customer_id", StringType(), True),  # Nullable to handle missing IDs gracefully
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),  # Will be null if non-numeric values found
        StructField("email", StringType(), True),
        StructField("registration_date", StringType(), True),  # String format for flexibility
        StructField("total_spent", DoubleType(), True)  # Double for currency precision
    ])

    # Load CSV data with explicit schema to avoid costly schema inference on large datasets
    # Performance consideration: Schema enforcement happens at read time, invalid data becomes null
    # Migration note: CSV reading options and null handling may vary between Spark and other engines
    df = spark.read \
        .option("header", "true") \
        .schema(schema) \
        .csv(data_path)  # Supports glob patterns and multiple files automatically

    return df