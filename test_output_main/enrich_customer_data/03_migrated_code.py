```python
def enrich_customer_data(customer_df: DataFrame, transaction_df: DataFrame) -> DataFrame:
    """
    Enrich customer data with transaction analytics and categorize customers into tiers.
    
    This function aggregates transaction data per customer to calculate key metrics,
    then joins this information with customer master data. It also applies business
    rules to categorize customers into spending tiers based on total transaction amounts.
    
    Args:
        customer_df (DataFrame): Clean customer master data containing customer_id and
            other customer attributes. Must contain 'customer_id' column as the join key.
        transaction_df (DataFrame): Transaction data containing individual transaction
            records. Must contain columns: 'customer_id', 'transaction_id', and 'amount'.
            The 'amount' column should be numeric for aggregation operations.
    
    Returns:
        DataFrame: Enriched customer data with the following additional columns:
            - transaction_count: Total number of transactions per customer
            - total_transaction_amount: Sum of all transaction amounts per customer
            - avg_transaction_amount: Average transaction amount per customer
            - customer_tier: Categorical tier (Premium/Gold/Silver/Bronze) based on spending
    
    Note:
        Customers without any transactions will have null values for transaction metrics
        due to the left join, and will be assigned 'Bronze' tier by the otherwise clause.
    """
    from snowflake.snowpark.functions import count, sum as snowpark_sum, avg, when, col
    
    # Calculate transaction metrics per customer
    transaction_metrics = transaction_df.group_by("customer_id") \
        .agg(
        count("transaction_id").alias("transaction_count"),
        snowpark_sum("amount").alias("total_transaction_amount"),
        avg("amount").alias("avg_transaction_amount")
    )

    # Join customer data with transaction metrics
    enriched_df = customer_df.join(
        transaction_metrics,
        on="customer_id",
        how="left"
    )

    # Add customer tier based on spending
    enriched_df = enriched_df.with_column(
        "customer_tier",
        when(col("total_transaction_amount") > 10000, "Premium")
        .when(col("total_transaction_amount") > 5000, "Gold")
        .when(col("total_transaction_amount") > 1000, "Silver")
        .otherwise("Bronze")
    )

    return enriched_df
```