```python
def clean_customer_data(df: DataFrame) -> DataFrame:
    """
    Clean and validate customer data by removing duplicates, filtering null values,
    and enriching with derived columns.
    
    This function performs essential data quality operations including deduplication
    based on customer_id, null value filtering for critical fields, and age group
    categorization. The function assumes that customer_id and email are mandatory
    fields and that age values are numeric.
    
    Args:
        df (DataFrame): Raw customer DataFrame containing at minimum the columns:
            - customer_id: Unique identifier for each customer (required, non-null)
            - email: Customer email address (required, non-null)
            - age: Customer age in years (numeric, used for age group derivation)
    
    Returns:
        DataFrame: Cleaned and enriched customer data with the following transformations:
            - Duplicates removed based on customer_id (keeps first occurrence)
            - Records with null customer_id or email filtered out
            - New 'age_group' column added with categorical values:
              'Young' (<25), 'Adult' (25-64), 'Senior' (>=65)
            - Additional validation applied via validate_data_quality function
    
    Note:
        This function depends on an external validate_data_quality function that
        must be available in the execution context. The function uses Snowpark's
        lazy evaluation, so actual computation occurs only when an action is triggered.
    """
    from snowflake.snowpark.functions import col, when
    
    cleaned_df = df.drop_duplicates(["customer_id"]) \
        .filter(col("customer_id").is_not_null()) \
        .filter(col("email").is_not_null())
    
    cleaned_df = cleaned_df.with_column(
        "age_group",
        when(col("age") < 25, "Young")
        .when(col("age") < 65, "Adult")
        .otherwise("Senior")
    )
    
    # TODO: [MANUAL MIGRATION REQUIRED] - validate_data_quality function needs to be migrated separately
    validated_df = validate_data_quality(cleaned_df)
    
    return validated_df
```