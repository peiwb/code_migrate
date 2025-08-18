```python
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
        - Requires external functions: create_snowpark_session, load_customer_data,
          load_transaction_data, clean_customer_data, enrich_customer_data,
          generate_customer_insights, save_results
        - Hard-coded file paths may need parameterization for different environments
        - Uses eager evaluation (.count(), .show()) which triggers Snowpark actions
    """
    # TODO: [MANUAL MIGRATION REQUIRED] - create_spark_session equivalent for Snowpark
    session = create_snowpark_session("CustomerAnalytics")

    try:
        # TODO: [MANUAL MIGRATION REQUIRED] - File path handling and format support for Snowpark
        customer_data = load_customer_data(session, "/data/customers.csv")
        transaction_data = load_transaction_data(session, "/data/transactions.json")

        clean_customers = clean_customer_data(customer_data)
        enriched_customers = enrich_customer_data(clean_customers, transaction_data)
        customer_insights = generate_customer_insights(enriched_customers)

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
        # TODO: [MANUAL MIGRATION REQUIRED] - Resource cleanup patterns for Snowpark session
        session.close()
```