def create_spark_session(app_name: str = "DataProcessingApp") -> SparkSession:
    """
    Create and configure Spark session with optimized settings for adaptive query execution.
    
    This function creates a SparkSession with adaptive query execution enabled to optimize
    performance by dynamically adjusting query plans at runtime. The adaptive coalesce
    feature helps reduce the number of output partitions when data size is smaller than
    expected, improving performance for downstream operations.
    
    Args:
        app_name: Name of the Spark application that will appear in the Spark UI.
                 Defaults to "DataProcessingApp".
    
    Returns:
        SparkSession: A configured SparkSession instance with adaptive query execution
                     enabled. If a session already exists, it will return the existing
                     session (getOrCreate() behavior).
    
    Note:
        - Uses getOrCreate() pattern to reuse existing sessions in the same JVM
        - Adaptive query execution requires Spark 3.0+ for full functionality
        - These configurations are session-wide and affect all DataFrames/SQL queries
    """
    # Use builder pattern to construct SparkSession with configuration
    # Note: getOrCreate() returns existing session if one exists, ignoring new configs
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Return configured session - important for migration: session is singleton per JVM
    return spark