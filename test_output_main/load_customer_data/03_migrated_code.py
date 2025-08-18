```python
def load_customer_data(session, data_path: str):
    """Load customer data from CSV files with predefined schema and validation.
    
    This function loads customer data from CSV files using a strict schema to ensure
    data consistency and type safety. The schema enforces specific data types for
    customer attributes including ID, personal information, and transaction history.
    
    Args:
        session: Active Snowpark session used for data operations.
        data_path (str): File system path to the CSV data source. Can be a single
            file or directory containing multiple CSV files. Supports local paths,
            HDFS, S3, and other Spark-compatible file systems.
    
    Returns:
        DataFrame: Snowpark DataFrame containing customer data with the following columns:
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
    from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType, DoubleType
    
    schema = StructType([
        StructField("customer_id", StringType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("email", StringType(), True),
        StructField("registration_date", StringType(), True),
        StructField("total_spent", DoubleType(), True)
    ])

    # TODO: [MANUAL MIGRATION REQUIRED] - Snowpark CSV reading with schema enforcement may differ from PySpark
    df = session.read \
        .option("FIELD_OPTIONALLY_ENCLOSED_BY", '"') \
        .option("SKIP_HEADER", 1) \
        .schema(schema) \
        .csv(data_path)

    return df
```