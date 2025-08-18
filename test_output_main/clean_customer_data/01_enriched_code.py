def clean_customer_data(df: DataFrame) -> DataFrame:
    """
    Clean and validate customer data by removing duplicates, filtering null values, and adding derived columns.
    
    This function performs essential data quality operations on customer data including:
    - Deduplication based on customer_id (keeps first occurrence)
    - Filtering out records with null customer_id or email
    - Adding age group categorization
    - Additional data quality validation
    
    Args:
        df (DataFrame): Raw customer DataFrame containing at minimum the columns:
            - customer_id: Unique identifier for customers
            - email: Customer email address
            - age: Customer age as numeric value
            
    Returns:
        DataFrame: Cleaned and validated customer data with additional derived columns:
            - All original columns (after cleaning)
            - age_group: Categorical age grouping (Young/Adult/Senior)
            
    Note:
        This function depends on validate_data_quality() function which must be available
        in the current scope. The function assumes age is a numeric column.
    """
    # Remove duplicates based on customer_id - keeps first occurrence due to PySpark's default behavior
    # This is critical for downstream analytics to avoid double-counting customers
    # Performance note: dropDuplicates triggers a shuffle operation, consider partitioning by customer_id if data is large
    cleaned_df = df.dropDuplicates(["customer_id"]) \
        .filter(col("customer_id").isNotNull()) \
        .filter(col("email").isNotNull())
    
    # Add business-driven age segmentation for marketing and analytics purposes
    # Age thresholds: <25 (Young), 25-64 (Adult), 65+ (Senior)
    # Note: when() evaluates conditions sequentially, so order matters for performance
    cleaned_df = cleaned_df.withColumn(
        "age_group",
        when(col("age") < 25, "Young")
        .when(col("age") < 65, "Adult")
        .otherwise("Senior")  # Handles age >= 65 and null ages
    )
    
    # Apply additional data quality validation - this is a custom function dependency
    # Migration note: validate_data_quality() implementation needs to be provided separately
    validated_df = validate_data_quality(cleaned_df)
    
    return validated_df