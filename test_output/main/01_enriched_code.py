def main():
    """
    Main data processing pipeline for customer analytics.
    
    Orchestrates a complete ETL workflow that loads customer and transaction data,
    performs data cleaning and enrichment operations, generates business insights,
    and saves results to different output formats. This function serves as the
    primary entry point for the customer analytics batch processing job.
    
    The pipeline follows these stages:
    1. Initialize Spark session with appropriate configuration
    2. Load raw customer data (CSV) and transaction data (JSON)
    3. Clean and validate customer data for quality issues
    4. Enrich customer records with aggregated transaction metrics
    5. Generate analytical insights and KPIs
    6. Persist results in optimized formats (Parquet for data, CSV for reports)
    7. Display summary statistics for monitoring
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        Exception: Re-raises any exception that occurs during the pipeline execution
                  after logging the error message. Common failures include:
                  - File not found errors for input data paths
                  - Schema validation errors during data processing
                  - Memory/resource exhaustion during large data operations
                  - Permission errors when writing to output directories
    
    Note:
        This function manages the Spark session lifecycle and ensures proper
        resource cleanup via try-finally block. The function expects external
        helper functions to be available in the same module scope.
    """
    # Initialize Spark session with customer analytics job name
    # AI Migration Note: Spark session creation may need platform-specific configs
    # for memory management, serialization, and cluster connectivity
    spark = create_spark_session("CustomerAnalytics")

    try:
        # Load raw data from external sources
        # AI Migration Note: File path assumptions - these are absolute paths that
        # may need environment-specific configuration or parameter injection
        customer_data = load_customer_data(spark, "/data/customers.csv")
        transaction_data = load_transaction_data(spark, "/data/transactions.json")

        # Data quality pipeline - handle missing values, duplicates, invalid formats
        # AI Migration Note: Cleaning logic may contain business rules specific to
        # customer data structure and quality requirements
        clean_customers = clean_customer_data(customer_data)
        
        # Business logic enrichment - join customer data with aggregated transactions
        # AI Migration Note: This likely involves complex joins and aggregations that
        # may require different optimization strategies on other platforms
        enriched_customers = enrich_customer_data(clean_customers, transaction_data)
        
        # Analytics layer - compute KPIs, segments, and business metrics
        # AI Migration Note: May contain domain-specific calculations and window functions
        customer_insights = generate_customer_insights(enriched_customers)

        # Persist results in different formats for downstream consumption
        # AI Migration Note: Parquet is chosen for analytical workloads (columnar, compressed)
        # CSV for business user accessibility - consider partitioning strategy
        save_results(enriched_customers, "/output/enriched_customers", "parquet")
        save_results(customer_insights, "/output/customer_insights", "csv")

        # Display monitoring information for job validation
        # AI Migration Note: show() is Spark-specific - may need platform equivalent
        print("=== Customer Insights Summary ===")
        customer_insights.show(20, truncate=False)

        # Collect metrics for monitoring and alerting
        # AI Migration Note: count() triggers action and may be expensive on large datasets
        print(f"Total customers processed: {enriched_customers.count()}")
        print("Data processing completed successfully!")

    except Exception as e:
        # Log and re-raise for upstream error handling and monitoring systems
        print(f"Error in data processing pipeline: {str(e)}")
        raise
    finally:
        # Critical: Ensure Spark resources are always released
        # AI Migration Note: Platform-specific cleanup may be required
        spark.stop()