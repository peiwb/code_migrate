```python
def load_customer_data(session, data_path: str):
    """Load customer data from CSV files with predefined schema.
    
    This function loads customer data from CSV sources using a strict schema definition
    to ensure data type consistency and prevent schema inference overhead. The function
    assumes CSV files have headers and handles nullable fields appropriately.
    
    Args:
        session: Active Snowpark session for data operations
        data_path (str): File path or directory path to CSV data source(s).
                        Supports stage paths and other Snowflake-compatible sources
    
    Returns:
        DataFrame: Snowpark DataFrame containing customer data with the following columns:
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
    from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType, DoubleType
    
    schema = StructType([
        StructField("customer_id", StringType(), True),
        StructField("name", StringType(), True), 
        StructField("age", IntegerType(), True),
        StructField("email", StringType(), True),
        StructField("registration_date", StringType(), True),
        StructField("total_spent", DoubleType(), True)
    ])

    df = session.read \
        .option("FIELD_DELIMITER", ",") \
        .option("SKIP_HEADER", 1) \
        .schema(schema) \
        .csv(data_path)

    return df
```