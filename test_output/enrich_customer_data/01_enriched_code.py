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
    # Calculate transaction metrics per customer using groupBy aggregation
    # Performance note: This triggers a shuffle operation - consider partitioning by customer_id if dataset is large
    transaction_metrics = transaction_df.groupBy("customer_id") \
        .agg(
        count("transaction_id").alias("transaction_count"),  # Count of non-null transaction_ids
        spark_sum("amount").alias("total_transaction_amount"),  # Sum aggregation handles nulls by ignoring them
        avg("amount").alias("avg_transaction_amount")  # Average calculation excludes null values
    )

    # Left join preserves all customers, even those without transactions
    # Migration note: Ensure target platform handles null propagation consistently in joins
    enriched_df = customer_df.join(
        transaction_metrics,
        on="customer_id",
        how="left"  # Preserves customers with zero transactions (metrics will be null)
    )

    # Business logic: Tier assignment based on total spending thresholds
    # Note: Customers with null total_transaction_amount default to 'Bronze' via otherwise() clause
    enriched_df = enriched_df.withColumn(
        "customer_tier",
        when(col("total_transaction_amount") > 10000, "Premium")  # Top tier: >$10k total spend
        .when(col("total_transaction_amount") > 5000, "Gold")     # Mid-high tier: >$5k total spend
        .when(col("total_transaction_amount") > 1000, "Silver")   # Mid tier: >$1k total spend
        .otherwise("Bronze")  # Default tier for low spenders and null values
    )

    return enriched_df