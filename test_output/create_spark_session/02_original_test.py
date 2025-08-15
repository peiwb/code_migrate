import pytest
from pyspark.sql import SparkSession
from pyspark import SparkConf


def test_create_spark_session():
    """Test create_spark_session function with various scenarios."""
    
    # Test 1: Default app name
    spark1 = create_spark_session()
    
    # Verify SparkSession is created successfully
    assert spark1 is not None
    assert isinstance(spark1, SparkSession)
    
    # Verify default app name is set correctly
    assert spark1.conf.get("spark.app.name") == "DataProcessingApp"
    
    # Verify AQE configurations are applied
    assert spark1.conf.get("spark.sql.adaptive.enabled") == "true"
    assert spark1.conf.get("spark.sql.adaptive.coalescePartitions.enabled") == "true"
    
    # Test 2: Custom app name
    custom_app_name = "TestCustomApp"
    spark2 = create_spark_session(custom_app_name)
    
    assert spark2 is not None
    assert isinstance(spark2, SparkSession)
    assert spark2.conf.get("spark.app.name") == custom_app_name
    
    # Test 3: Verify getOrCreate() behavior - should return same session
    spark3 = create_spark_session(custom_app_name)
    assert spark3 is spark2  # Should be the same instance due to getOrCreate()
    
    # Test 4: Functional test - verify session can perform basic operations
    test_data = [(1, "Alice", 25), (2, "Bob", 30), (3, "Charlie", 35)]
    expected_data = [(1, "Alice", 25), (2, "Bob", 30), (3, "Charlie", 35)]
    
    # Create DataFrame to test session functionality
    df = spark1.createDataFrame(test_data, ["id", "name", "age"])
    
    # Verify DataFrame creation and basic operations work
    assert df.count() == 3
    collected_data = df.collect()
    actual_data = [(row.id, row.name, row.age) for row in collected_data]
    assert actual_data == expected_data
    
    # Test 5: Verify session can execute SQL queries
    df.createOrReplaceTempView("test_table")
    sql_result = spark1.sql("SELECT COUNT(*) as count FROM test_table").collect()[0]
    assert sql_result['count'] == 3
    
    # Clean up: Stop the session
    spark1.stop()
    
    # Test 6: Verify session restart with different configuration
    spark4 = create_spark_session("FreshSession")
    assert spark4.conf.get("spark.app.name") == "FreshSession"
    spark4.stop()