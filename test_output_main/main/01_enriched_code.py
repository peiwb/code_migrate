def main():
    """Main data processing pipeline for customer analytics.
    
    Orchestrates a complete ETL workflow that processes customer and transaction data
    to generate enriched customer profiles and business insights. This function serves
    as the entry point for the entire data processing pipeline.
    
    The pipeline performs the following operations:
    1. Loads customer data from CSV and transaction data from JSON
    2. Cleans and validates customer data
    3. Enriches customer profiles with transaction metrics
    4. Generates analytical insights for business reporting
    5. Saves results in both parquet (for analytics) and CSV (for reporting)
    
    Args:
        None
        
    Returns:
        None: This function performs side effects (data processing and file output)
        but does not return any values.
        
    Raises:
        Exception: Re-raises any exceptions that occur during the pipeline execution
        after logging the error message.
        
    Note:
        - Requires external functions: create_spark_session, load_customer_data,
          load_transaction_data, clean_customer_data, enrich_customer_data,
          generate_customer_insights, save_results
        - Hard-coded file paths may need parameterization for different environments
        - Uses eager evaluation (.count(), .show()) which triggers Spark actions
    """
    # Initialize Spark session with descriptive app name for cluster monitoring
    # AI Migration Note: Spark session management varies significantly across platforms
    spark = create_spark_session("CustomerAnalytics")

    try:
        # Load data from different formats - demonstrates Spark's format flexibility
        # AI Migration Note: File path handling and format support varies by platform
        customer_data = load_customer_data(spark, "/data/customers.csv")
        transaction_data = load_transaction_data(spark, "/data/transactions.json")

        # Sequential processing pipeline - each step depends on the previous
        # AI Migration Note: Lazy evaluation means no computation happens until action
        clean_customers = clean_customer_data(customer_data)
        enriched_customers = enrich_customer_data(clean_customers, transaction_data)
        customer_insights = generate_customer_insights(enriched_customers)

        # Save in different formats optimized for different use cases
        # Parquet: columnar format optimized for analytics queries
        # CSV: human-readable format for business users and reporting tools
        save_results(enriched_customers, "/output/enriched_customers", "parquet")
        save_results(customer_insights, "/output/customer_insights", "csv")

        # Display results - triggers Spark action, pulls data to driver
        # AI Migration Note: .show() and .count() are expensive operations
        print("=== Customer Insights Summary ===")
        customer_insights.show(20, truncate=False)

        # Count triggers full dataset scan - consider caching if used multiple times
        print(f"Total customers processed: {enriched_customers.count()}")
        print("Data processing completed successfully!")

    except Exception as e:
        # Log error before re-raising for upstream error handling
        print(f"Error in data processing pipeline: {str(e)}")
        raise
    finally:
        # Critical: Always stop Spark session to release cluster resources
        # AI Migration Note: Resource cleanup patterns vary significantly by platform
        spark.stop()