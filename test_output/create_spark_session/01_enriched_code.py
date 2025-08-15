def create_spark_session(app_name: str = "DataProcessingApp") -> SparkSession:
    """Create and configure Spark session with optimized settings for data processing workloads.
    
    This function initializes a SparkSession with Adaptive Query Execution (AQE) enabled,
    which provides runtime optimizations including dynamic partition coalescing and 
    join strategy optimization. The configuration is optimized for typical ETL workloads
    with moderate data volumes.
    
    Args:
        app_name (str, optional): Name of the Spark application that will appear in 
            Spark UI and logs. Defaults to "DataProcessingApp".
    
    Returns:
        SparkSession: Configured Spark session instance with AQE optimizations enabled.
            The session uses getOrCreate() pattern, so it will reuse existing session
            if one already exists with the same configuration.
    
    Note:
        This function enables Spark 3.0+ Adaptive Query Execution features that may
        not be available in earlier Spark versions. Migration consideration: ensure
        target platform supports AQE or remove these configurations.
    """
    # Use builder pattern to construct SparkSession with chained configurations
    # getOrCreate() ensures singleton pattern - reuses existing session if available
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # AQE (Adaptive Query Execution) optimizations explanation:
    # - adaptive.enabled: Enables runtime query re-optimization based on actual data statistics
    # - coalescePartitions.enabled: Automatically reduces number of partitions after 
    #   shuffle operations to avoid small partition overhead
    # These settings improve performance for most analytical workloads but may need
    # tuning for streaming or very large dataset scenarios
    
    return spark