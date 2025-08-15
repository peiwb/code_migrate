```python
def main():
    """
    Main data processing pipeline for customer analytics.
    
    Orchestrates a complete ETL workflow that loads customer and transaction data,
    performs data cleaning and enrichment operations, generates business insights,
    and saves results to different output formats. This function serves as the
    primary entry point for the customer analytics batch processing job.
    
    The pipeline follows these stages:
    1. Initialize Snowpark session with appropriate configuration
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
        This function manages the Snowpark session lifecycle and ensures proper
        resource cleanup via try-finally block. The function expects external
        helper functions to be available in the same module scope.
    """
    # TODO: [MANUAL MIGRATION REQUIRED] - Create Snowpark session with appropriate connection parameters
    session = create_snowpark_session("CustomerAnalytics")

    try:
        # TODO: [MANUAL MIGRATION REQUIRED] - File paths need to be configured for Snowflake stages or external locations
        customer_data = load_customer_data(session, "/data/customers.csv")
        transaction_data = load_transaction_data(session, "/data/transactions.json")

        clean_customers = clean_customer_data(customer_data)
        
        enriched_customers = enrich_customer_data(clean_customers, transaction_data)
        
        customer_insights = generate_customer_insights(enriched_customers)

        # TODO: [MANUAL MIGRATION REQUIRED] - Output paths need to be configured for Snowflake stages or tables
        save_results(enriched_customers, "/output/enriched_customers", "parquet")
        save_results(customer_insights, "/output/customer_insights", "csv")

        print("=== Customer Insights Summary ===")
        customer_insights.show(20, truncate=False)

        print(f"Total customers processed: {enriched_customers.count()}")
        print("Data processing completed successfully!")

    except Exception as e:
        print(f"Error in data processing pipeline: {str(e)}")
        raise
    finally:
        session.close()
```