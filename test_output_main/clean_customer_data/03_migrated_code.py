```python
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
    cleaned_df = df.drop_duplicates(["customer_id"]) \
        .filter(col("customer_id").is_not_null()) \
        .filter(col("email").is_not_null())
    
    cleaned_df = cleaned_df.with_column(
        "age_group",
        when(col("age") < 25, "Young")
        .when(col("age") < 65, "Adult")
        .otherwise("Senior")
    )
    
    validated_df = validate_data_quality(cleaned_df)
    
    return validated_df
```