def generate_customer_insights(df: DataFrame) -> DataFrame:
    """
    Generate business insights from customer data by aggregating metrics across age groups and customer tiers.
    
    This function performs a comprehensive analysis of customer behavior by grouping customers
    based on their age demographics and service tier levels, then calculating key business
    metrics including customer distribution, average spending patterns, and total revenue
    contribution per segment.
    
    Args:
        df (DataFrame): Enriched customer data containing the following required columns:
            - customer_id: Unique identifier for each customer
            - age_group: Categorical age segmentation (e.g., '18-25', '26-35', etc.)
            - customer_tier: Service tier classification (e.g., 'Bronze', 'Silver', 'Gold')
            - total_spent: Individual customer's total spending amount
            - total_transaction_amount: Transaction-level amount for revenue aggregation
    
    Returns:
        DataFrame: Customer insights summary with columns:
            - age_group: Age demographic segment
            - customer_tier: Customer service tier
            - customer_count: Number of unique customers in each segment
            - avg_spending: Average spending per customer in the segment
            - total_revenue: Sum of all transaction amounts for the segment
    
    Note:
        Results are ordered by age_group and customer_tier for consistent reporting.
        The function assumes data has been pre-processed and enriched with derived columns.
    """
    # Group by demographic and tier dimensions for business segment analysis
    # This creates a two-dimensional customer segmentation matrix
    insights_df = df.groupBy("age_group", "customer_tier") \
        .agg(
        # Count distinct customers to measure segment size - critical for market sizing
        count("customer_id").alias("customer_count"),
        # Calculate average individual spending - key metric for customer value assessment
        avg("total_spent").alias("avg_spending"),
        # Sum transaction amounts for total revenue attribution per segment
        # Note: spark_sum is used instead of sum to avoid naming conflicts with Python built-in
        spark_sum("total_transaction_amount").alias("total_revenue")
    ) \
        .orderBy("age_group", "customer_tier")  # Deterministic ordering for consistent reporting

    return insights_df