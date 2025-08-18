import pytest
from pyspark.sql import SparkSession
from unittest.mock import patch, MagicMock


def test_create_spark_session():
    """
    Test the create_spark_session function to ensure it creates a properly configured SparkSession.
    """
    # Clean up any existing SparkSession to ensure test isolation
    if SparkSession._instantiatedSession is not None:
        SparkSession._instantiatedSession.stop()
        SparkSession._instantiatedSession = None
    
    # Test with default app name
    spark_session = create_spark_session()
    
    # Verify SparkSession was created
    assert spark_session is not None
    assert isinstance(spark_session, SparkSession)
    
    # Verify app name (default)
    assert spark_session.sparkContext.appName == "DataProcessingApp"
    
    # Verify adaptive query execution configurations
    spark_conf = spark_session.sparkContext.getConf()
    assert spark_conf.get("spark.sql.adaptive.enabled") == "true"
    assert spark_conf.get("spark.sql.adaptive.coalescePartitions.enabled") == "true"
    
    # Clean up
    spark_session.stop()
    SparkSession._instantiatedSession = None
    
    # Test with custom app name
    custom_app_name = "TestCustomApp"
    spark_session_custom = create_spark_session(app_name=custom_app_name)
    
    # Verify custom app name
    assert spark_session_custom.sparkContext.appName == custom_app_name
    
    # Verify configurations are still applied
    spark_conf_custom = spark_session_custom.sparkContext.getConf()
    assert spark_conf_custom.get("spark.sql.adaptive.enabled") == "true"
    assert spark_conf_custom.get("spark.sql.adaptive.coalescePartitions.enabled") == "true"
    
    # Test getOrCreate behavior - should return same session
    spark_session_reuse = create_spark_session(app_name="DifferentName")
    assert spark_session_reuse is spark_session_custom
    
    # Clean up
    spark_session_custom.stop()
    SparkSession._instantiatedSession = None


def test_create_spark_session_functional_verification():
    """
    Functional test to verify the SparkSession works correctly with DataFrame operations.
    """
    # Clean up any existing SparkSession
    if SparkSession._instantiatedSession is not None:
        SparkSession._instantiatedSession.stop()
        SparkSession._instantiatedSession = None
    
    # Create SparkSession using the function
    spark = create_spark_session(app_name="FunctionalTest")
    
    try:
        # Create test DataFrame to verify session functionality
        test_data = [(1, "Alice", 25), (2, "Bob", 30), (3, "Charlie", 35)]
        columns = ["id", "name", "age"]
        
        df = spark.createDataFrame(test_data, columns)
        
        # Verify DataFrame operations work
        assert df.count() == 3
        assert df.columns == columns
        
        # Test a simple transformation to ensure adaptive execution context works
        filtered_df = df.filter(df.age > 28)
        result = filtered_df.collect()
        
        # Expected: Bob (30) and Charlie (35)
        assert len(result) == 2
        assert result[0]["name"] in ["Bob", "Charlie"]
        assert result[1]["name"] in ["Bob", "Charlie"]
        
    finally:
        # Clean up
        spark.stop()
        SparkSession._instantiatedSession = None