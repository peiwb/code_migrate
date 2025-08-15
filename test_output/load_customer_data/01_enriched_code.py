def load_customer_data(spark: SparkSession, data_path: str) -> DataFrame:
    """Load customer data from CSV files with predefined schema.
    
    This function loads customer data from CSV sources using a strict schema definition
    to ensure data type consistency and prevent schema inference overhead. The function
    assumes CSV files have headers and handles nullable fields appropriately.
    
    Args:
        spark (SparkSession): Active Spark session for data operations
        data_path (str): File path or directory path to CSV data source(s).
                        Supports local paths, HDFS, S3, and other Spark-compatible filesystems
    
    Returns:
        DataFrame: PySpark DataFrame containing customer data with the following columns:
            - customer_id (StringType): Unique customer identifier (nullable)
            - name (StringType): Customer full name (nullable)
            - age (IntegerType): Customer age in years (nullable)
            - email (StringType): Customer email address (nullable)
            - registration_date (StringType): Date when customer registered (nullable)
            - total_spent (DoubleType): Total amount spent by customer (nullable)
    
    Note:
        - All fields are nullable (True) to handle incomplete data gracefully
        - Registration date is kept as StringType; downstream processing should handle date parsing
        - Schema enforcement prevents automatic type inference, improving performance for large files
    """
    # Define strict schema to avoid costly schema inference on large CSV files
    # All fields nullable=True to handle real-world data quality issues
    schema = StructType([
        StructField("customer_id", StringType(), True),
        StructField("name", StringType(), True), 
        StructField("age", IntegerType(), True),
        StructField("email", StringType(), True),
        StructField("registration_date", StringType(), True),  # Keep as string for flexible date format handling
        StructField("total_spent", DoubleType(), True)
    ])

    # Load CSV with header=true assumption - ensure source files have header row
    # Schema enforcement will cause failure if CSV structure doesn't match
    df = spark.read \
        .option("header", "true") \
        .schema(schema) \
        .csv(data_path)  # Supports glob patterns and directory paths

    return df