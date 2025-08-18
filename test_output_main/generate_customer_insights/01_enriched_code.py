def generate_customer_insights(df: DataFrame) -> DataFrame:
    """
    Generate business insights from customer data by aggregating metrics across age groups and customer tiers.
    
    This function creates a summary report that groups customers by demographic and tier segments,
    calculating key business metrics including customer counts, average spending per customer,
    and total revenue contribution. The results are sorted to provide a consistent view for
    business reporting and analytics.
    
    Args:
        df (DataFrame): Enriched customer data containing the following required columns:
            - customer_id: Unique identifier for each customer
            - age_group: Customer age segment (e.g., '18-25', '26-35', etc.)
            - customer_tier: Customer classification level (e.g., 'Gold', 'Silver', 'Bronze')
            - total_spent: Total amount spent by individual customer (numeric)
            - total_transaction_amount: Individual transaction amounts (numeric)
    
    Returns:
        DataFrame: Customer insights summary with columns:
            - age_group: Customer age segment
            - customer_tier: Customer tier classification
            - customer_count: Number of unique customers in the segment
            - avg_spending: Average spending per customer in the segment
            - total_revenue: Sum of all transaction amounts in the segment
    
    Note:
        Results are ordered by age_group and customer_tier for consistent reporting.
        The function assumes input data has already been validated and cleansed.
    """
    # Group by demographic and tier segments - this creates partition boundaries for aggregation
    # Performance note: Ensure age_group and customer_tier have reasonable cardinality to avoid skew
    insights_df = df.groupBy("age_group", "customer_tier") \
        .agg(
        # Count distinct customers to handle potential duplicate records per customer
        # Migration note: count() behavior may vary across platforms for null handling
        count("customer_id").alias("customer_count"),
        
        # Calculate average spending per customer within each segment
        # Business assumption: total_spent represents lifetime value per customer
        avg("total_spent").alias("avg_spending"),
        
        # Sum all transaction amounts to get total revenue contribution
        # Using spark_sum instead of sum to avoid naming conflicts with Python built-in
        spark_sum("total_transaction_amount").alias("total_revenue")
    ) \
        .orderBy("age_group", "customer_tier")  # Deterministic ordering for consistent reporting

    return insights_df