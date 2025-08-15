```python
def enrich_customer_data(customer_df: DataFrame, transaction_df: DataFrame) -> DataFrame:
    """
    Enrich customer data with transaction analytics and tiering information.
    
    This function performs aggregation of transaction data per customer and joins it with
    customer data to provide comprehensive customer analytics. It calculates transaction
    metrics and assigns customer tiers based on total spending patterns.
    
    Args:
        customer_df (DataFrame): Clean customer data containing at minimum a 'customer_id' column.
                                Must be pre-validated and deduplicated to ensure one record per customer.
        transaction_df (DataFrame): Transaction data containing 'customer_id', 'transaction_id',
                                   and 'amount' columns. Amount should be numeric and in consistent currency.
    
    Returns:
        DataFrame: Enriched customer data with the following additional columns:
                  - transaction_count: Total number of transactions per customer
                  - total_transaction_amount: Sum of all transaction amounts per customer
                  - avg_transaction_amount: Average transaction amount per customer
                  - customer_tier: Categorical tier (Premium/Gold/Silver/Bronze) based on total spending
    
    Note:
        - Customers with no transactions will have null values for transaction metrics
        - Customer tier will be 'Bronze' for customers with null transaction amounts
        - The function assumes transaction amounts are positive values
    """
    from snowflake.snowpark.functions import count, sum as snowpark_sum, avg, when, col
    
    transaction_metrics = transaction_df.group_by("customer_id") \
        .agg(
        count("transaction_id").alias("transaction_count"),
        snowpark_sum("amount").alias("total_transaction_amount"),
        avg("amount").alias("avg_transaction_amount")
    )

    enriched_df = customer_df.join(
        transaction_metrics,
        on="customer_id",
        how="left"
    )

    enriched_df = enriched_df.with_column(
        "customer_tier",
        when(col("total_transaction_amount") > 10000, "Premium")
        .when(col("total_transaction_amount") > 5000, "Gold")
        .when(col("total_transaction_amount") > 1000, "Silver")
        .otherwise("Bronze")
    )

    return enriched_df
```