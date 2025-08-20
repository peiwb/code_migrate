#!/usr/bin/env python3
"""
Test file for PySpark data analysis operations
This file contains common PySpark patterns that would need migration
from PySpark to Snowflake/SageMaker
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, isnan, isnull, count, mean, stddev, min, max, sum as spark_sum,
    percentile_approx, corr, coalesce, lit, round as spark_round, abs as spark_abs,
    upper, lower, regexp_replace, split, size, explode, collect_list, collect_set,
    row_number, rank, dense_rank, lag, lead, first, last, ntile,
    year, month, dayofmonth, date_add, date_sub, datediff, current_timestamp,
    unix_timestamp, from_unixtime, to_date, desc, asc, concat, concat_ws,
    trim, ltrim, rtrim, length, substring, regexp_extract, array_contains,
    map_keys, map_values, struct, array, create_map, expr, date_format, hour,
    countDistinct, variance, skewness, kurtosis
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType,
    DoubleType, BooleanType, TimestampType, DateType, ArrayType, MapType
)
from pyspark.sql.window import Window
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, MinMaxScaler, Normalizer,
    StringIndexer, OneHotEncoder, Bucketizer, QuantileDiscretizer,
    PCA, ChiSqSelector, UnivariateFeatureSelector, VarianceThresholdSelector
)
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
import random
import numpy as np


def create_spark_session():
    """Create and configure Spark session"""
    spark = SparkSession.builder \
        .appName("PySpark Data Analysis Test") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    print(f"Spark Version: {spark.version}")
    print(f"Spark Config: {dict(spark.sparkContext.getConf().getAll())}")

    return spark


def create_sample_dataset(spark):
    """Create sample datasets for testing"""
    print("=== CREATING SAMPLE DATASETS ===")

    # Customer data
    customer_data = []
    random.seed(42)
    np.random.seed(42)

    regions = ['North', 'South', 'East', 'West']
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']

    for i in range(1, 10001):  # 10K records
        customer_data.append({
            'customer_id': i,
            'age': int(np.random.normal(40, 15)),
            'income': float(np.random.normal(60000, 20000)),
            'credit_score': int(np.random.normal(700, 100)),
            'years_employed': float(np.random.exponential(5)),
            'region': random.choice(regions),
            'education': random.choice(education_levels),
            'marital_status': random.choice(marital_statuses),
            'has_mortgage': random.choice([True, False]),
            'num_accounts': int(np.random.poisson(4)),
            'registration_date': f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            'last_login': f"2024-{random.randint(1, 8):02d}-{random.randint(1, 28):02d}"
        })

    # Transaction data
    transaction_data = []
    for i in range(1, 50001):  # 50K transactions
        customer_id = random.randint(1, 10000)
        transaction_data.append({
            'transaction_id': i,
            'customer_id': customer_id,
            'transaction_date': f"2024-{random.randint(1, 8):02d}-{random.randint(1, 28):02d}",
            'amount': float(np.random.lognormal(4, 1.5)),
            'category': random.choice(['Food', 'Entertainment', 'Shopping', 'Travel', 'Bills']),
            'merchant': f"Merchant_{random.randint(1, 1000)}",
            'is_online': random.choice([True, False]),
            'is_fraud': random.choice([True, False]) if random.random() < 0.05 else False
        })

    # Product data
    product_data = []
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
    for i in range(1, 1001):  # 1K products
        product_data.append({
            'product_id': i,
            'product_name': f"Product_{i}",
            'category': random.choice(categories),
            'price': float(np.random.uniform(10, 1000)),
            'rating': round(random.uniform(1, 5), 1),
            'reviews_count': random.randint(0, 1000),
            'in_stock': random.choice([True, False]),
            'launch_date': f"202{random.randint(0, 4)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        })

    # Define schemas
    customer_schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("age", IntegerType(), True),
        StructField("income", DoubleType(), True),
        StructField("credit_score", IntegerType(), True),
        StructField("years_employed", DoubleType(), True),
        StructField("region", StringType(), True),
        StructField("education", StringType(), True),
        StructField("marital_status", StringType(), True),
        StructField("has_mortgage", BooleanType(), True),
        StructField("num_accounts", IntegerType(), True),
        StructField("registration_date", StringType(), True),
        StructField("last_login", StringType(), True)
    ])

    transaction_schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("customer_id", IntegerType(), True),
        StructField("transaction_date", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("category", StringType(), True),
        StructField("merchant", StringType(), True),
        StructField("is_online", BooleanType(), True),
        StructField("is_fraud", BooleanType(), True)
    ])

    product_schema = StructType([
        StructField("product_id", IntegerType(), True),
        StructField("product_name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("rating", DoubleType(), True),
        StructField("reviews_count", IntegerType(), True),
        StructField("in_stock", BooleanType(), True),
        StructField("launch_date", StringType(), True)
    ])

    # Create DataFrames
    customers_df = spark.createDataFrame(customer_data, schema=customer_schema)
    transactions_df = spark.createDataFrame(transaction_data, schema=transaction_schema)
    products_df = spark.createDataFrame(product_data, schema=product_schema)

    # Convert date strings to actual dates
    customers_df = customers_df.withColumn("registration_date", to_date(col("registration_date"))) \
        .withColumn("last_login", to_date(col("last_login")))

    transactions_df = transactions_df.withColumn("transaction_date", to_date(col("transaction_date")))
    products_df = products_df.withColumn("launch_date", to_date(col("launch_date")))

    # Introduce some null values for testing
    customers_df = customers_df.withColumn(
        "income",
        when(col("customer_id") % 100 == 0, None).otherwise(col("income"))
    ).withColumn(
        "credit_score",
        when(col("customer_id") % 150 == 0, None).otherwise(col("credit_score"))
    )

    print(f"Created customers_df: {customers_df.count()} rows")
    print(f"Created transactions_df: {transactions_df.count()} rows")
    print(f"Created products_df: {products_df.count()} rows")

    return customers_df, transactions_df, products_df


def basic_dataframe_operations(customers_df, transactions_df, products_df):
    """Demonstrate basic DataFrame operations"""
    print("\n=== BASIC DATAFRAME OPERATIONS ===")

    # Schema and basic info
    print("Customer DataFrame Schema:")
    customers_df.printSchema()

    print(f"Number of partitions - customers: {customers_df.rdd.getNumPartitions()}")
    print(f"Number of partitions - transactions: {transactions_df.rdd.getNumPartitions()}")

    # Select and filter operations
    print("\nSelect and Filter Operations:")
    high_income_customers = customers_df.select("customer_id", "age", "income", "region") \
        .filter(col("income") > 80000) \
        .orderBy(desc("income"))

    print(f"High income customers: {high_income_customers.count()}")
    high_income_customers.show(10)

    # Complex filtering
    complex_filter = customers_df.filter(
        (col("age").between(25, 45)) &
        (col("education").isin(["Bachelor", "Master"])) &
        (col("has_mortgage") == True)
    )

    print(f"Customers with complex filter: {complex_filter.count()}")

    # Column operations
    customers_enhanced = customers_df.withColumn(
        "age_group",
        when(col("age") < 25, "Young")
        .when((col("age") >= 25) & (col("age") < 45), "Adult")
        .when((col("age") >= 45) & (col("age") < 65), "Middle-aged")
        .otherwise("Senior")
    ).withColumn(
        "income_bracket",
        when(col("income") < 40000, "Low")
        .when((col("income") >= 40000) & (col("income") < 80000), "Medium")
        .otherwise("High")
    ).withColumn(
        "days_since_registration",
        datediff(current_timestamp(), col("registration_date"))
    )

    print("\nCustomers with derived columns:")
    customers_enhanced.select("customer_id", "age", "age_group", "income", "income_bracket").show(10)

    return customers_enhanced


def aggregations_and_grouping(customers_df, transactions_df):
    """Demonstrate aggregation and grouping operations"""
    print("\n=== AGGREGATIONS AND GROUPING ===")

    # Basic aggregations
    print("Basic Statistics:")
    customers_df.agg(
        count("*").alias("total_customers"),
        mean("age").alias("avg_age"),
        stddev("income").alias("income_stddev"),
        min("credit_score").alias("min_credit_score"),
        max("credit_score").alias("max_credit_score"),
        percentile_approx("income", 0.5).alias("median_income")
    ).show()

    # Group by operations
    print("\nGrouping by Region:")
    region_stats = customers_df.groupBy("region").agg(
        count("*").alias("customer_count"),
        mean("age").alias("avg_age"),
        mean("income").alias("avg_income"),
        stddev("income").alias("income_std"),
        percentile_approx("credit_score", 0.5).alias("median_credit_score")
    ).orderBy(desc("customer_count"))

    region_stats.show()

    # Multiple grouping columns
    print("\nGrouping by Region and Education:")
    region_education_stats = customers_df.groupBy("region", "education").agg(
        count("*").alias("count"),
        mean("income").alias("avg_income")
    ).orderBy("region", "education")

    region_education_stats.show(20)

    # Transaction aggregations
    print("\nTransaction Analysis:")
    transaction_stats = transactions_df.groupBy("category").agg(
        count("*").alias("transaction_count"),
        spark_sum("amount").alias("total_amount"),
        mean("amount").alias("avg_amount"),
        percentile_approx("amount", [0.25, 0.5, 0.75]).alias("amount_quartiles")
    ).orderBy(desc("total_amount"))

    transaction_stats.show()

    # Monthly transaction trends
    monthly_trends = transactions_df.withColumn("month", month("transaction_date")) \
        .groupBy("month").agg(
        count("*").alias("transaction_count"),
        spark_sum("amount").alias("total_amount"),
        mean("amount").alias("avg_amount")
    ).orderBy("month")

    print("\nMonthly Transaction Trends:")
    monthly_trends.show()


def window_functions(customers_df, transactions_df):
    """Demonstrate window functions"""
    print("\n=== WINDOW FUNCTIONS ===")

    # Define window specifications
    region_window = Window.partitionBy("region").orderBy(desc("income"))
    age_window = Window.partitionBy("age_group").orderBy(desc("income"))

    # Add age groups first
    customers_with_groups = customers_df.withColumn(
        "age_group",
        when(col("age") < 30, "Young")
        .when((col("age") >= 30) & (col("age") < 50), "Adult")
        .otherwise("Senior")
    )

    # Ranking functions
    customers_ranked = customers_with_groups.withColumn(
        "income_rank_in_region", row_number().over(region_window)
    ).withColumn(
        "income_dense_rank_in_region", dense_rank().over(region_window)
    ).withColumn(
        "income_percentile", ntile(10).over(region_window)
    )

    print("Top earners by region:")
    customers_ranked.filter(col("income_rank_in_region") <= 3) \
        .select("customer_id", "region", "income", "income_rank_in_region") \
        .orderBy("region", "income_rank_in_region") \
        .show()

    # Analytical functions
    transaction_window = Window.partitionBy("customer_id").orderBy("transaction_date")

    transactions_with_analytics = transactions_df.withColumn(
        "prev_transaction_amount", lag("amount", 1).over(transaction_window)
    ).withColumn(
        "next_transaction_amount", lead("amount", 1).over(transaction_window)
    ).withColumn(
        "running_total", spark_sum("amount").over(
            Window.partitionBy("customer_id").orderBy("transaction_date")
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
    ).withColumn(
        "transaction_number", row_number().over(transaction_window)
    )

    print("\nTransaction analytics with window functions:")
    transactions_with_analytics.filter(col("customer_id") == 1) \
        .select("customer_id", "transaction_date", "amount",
                "prev_transaction_amount", "running_total", "transaction_number") \
        .show()


def joins_and_unions(customers_df, transactions_df, products_df):
    """Demonstrate various join and union operations"""
    print("\n=== JOINS AND UNIONS ===")

    # Inner join - customers with transactions
    customer_transactions = customers_df.alias("c").join(
        transactions_df.alias("t"),
        col("c.customer_id") == col("t.customer_id"),
        "inner"
    )

    print(f"Customer-Transaction inner join: {customer_transactions.count()} records")

    # Left join - all customers with their transaction summary
    transaction_summary = transactions_df.groupBy("customer_id").agg(
        count("*").alias("transaction_count"),
        spark_sum("amount").alias("total_spent"),
        mean("amount").alias("avg_transaction_amount"),
        max("transaction_date").alias("last_transaction_date")
    )

    customers_with_transactions = customers_df.join(
        transaction_summary,
        "customer_id",
        "left"
    ).fillna(0, ["transaction_count", "total_spent", "avg_transaction_amount"])

    print("Customers with transaction summary:")
    customers_with_transactions.select(
        "customer_id", "region", "transaction_count", "total_spent"
    ).orderBy(desc("total_spent")).show(10)

    # Self join - find customers in same region with similar income
    customers_alias1 = customers_df.alias("c1")
    customers_alias2 = customers_df.alias("c2")

    similar_income_customers = customers_alias1.join(
        customers_alias2,
        (col("c1.region") == col("c2.region")) &
        (col("c1.customer_id") != col("c2.customer_id")) &
        (spark_abs(col("c1.income") - col("c2.income")) < 5000),
        "inner"
    ).select(
        col("c1.customer_id").alias("customer1"),
        col("c2.customer_id").alias("customer2"),
        col("c1.region").alias("region"),
        col("c1.income").alias("income1"),
        col("c2.income").alias("income2")
    ).limit(20)

    print("Customers with similar income in same region:")
    similar_income_customers.show()

    # Union operations
    high_value_customers = customers_df.filter(col("income") > 90000).select("customer_id", "region")
    frequent_buyers = transaction_summary.filter(col("transaction_count") > 20) \
        .join(customers_df, "customer_id") \
        .select("customer_id", "region")

    important_customers = high_value_customers.union(frequent_buyers).distinct()
    print(f"Important customers (high income or frequent buyers): {important_customers.count()}")


def string_operations_and_regex(customers_df, transactions_df):
    """Demonstrate string operations and regex"""
    print("\n=== STRING OPERATIONS AND REGEX ===")

    # String manipulation
    customers_strings = customers_df.withColumn(
        "region_upper", upper("region")
    ).withColumn(
        "education_length", length("education")
    ).withColumn(
        "initials", concat(
            substring("region", 1, 1),
            lit("-"),
            substring("education", 1, 1)
        )
    ).withColumn(
        "customer_info", concat_ws(
            " | ",
            col("customer_id").cast("string"),
            col("region"),
            col("education")
        )
    )

    print("String operations examples:")
    customers_strings.select("customer_id", "region", "region_upper", "initials", "customer_info").show(10)

    # Regex operations
    merchants_with_patterns = transactions_df.withColumn(
        "merchant_number", regexp_extract("merchant", r"Merchant_(\d+)", 1)
    ).withColumn(
        "merchant_category",
        when(regexp_extract("merchant", r"Merchant_(\d+)", 1).cast("int") < 100, "Small")
        .when(regexp_extract("merchant", r"Merchant_(\d+)", 1).cast("int") < 500, "Medium")
        .otherwise("Large")
    ).withColumn(
        "clean_merchant", regexp_replace("merchant", r"_", " ")
    )

    print("Regex operations on merchant data:")
    merchants_with_patterns.select("merchant", "merchant_number", "merchant_category", "clean_merchant") \
        .distinct().orderBy("merchant_number").show(15)


def advanced_transformations(customers_df, transactions_df):
    """Demonstrate advanced transformations"""
    print("\n=== ADVANCED TRANSFORMATIONS ===")

    # Pivot operations
    pivot_data = transactions_df.groupBy("customer_id").pivot("category").agg(
        spark_sum("amount").alias("total"),
        count("*").alias("count")
    )

    print("Pivot table - customer spending by category:")
    pivot_data.show(10, truncate=False)

    # Complex case when statements
    customer_segments = customers_df.withColumn(
        "customer_segment",
        when((col("income") > 80000) & (col("credit_score") > 750), "Premium")
        .when((col("income") > 60000) & (col("credit_score") > 700), "Gold")
        .when((col("income") > 40000) & (col("credit_score") > 650), "Silver")
        .otherwise("Bronze")
    ).withColumn(
        "risk_score",
        when(col("credit_score") > 750, 1)
        .when(col("credit_score") > 700, 2)
        .when(col("credit_score") > 650, 3)
        .when(col("credit_score") > 600, 4)
        .otherwise(5)
    )

    print("Customer segmentation:")
    customer_segments.groupBy("customer_segment").agg(
        count("*").alias("count"),
        mean("income").alias("avg_income"),
        mean("credit_score").alias("avg_credit_score")
    ).show()

    # Array and map operations
    transaction_arrays = transactions_df.groupBy("customer_id").agg(
        collect_list("category").alias("categories"),
        collect_set("category").alias("unique_categories"),
        collect_list("amount").alias("amounts")
    ).withColumn(
        "category_count", size("unique_categories")
    ).withColumn(
        "prefers_online", array_contains("categories", "Online")
    )

    print("Array operations on transaction data:")
    transaction_arrays.select("customer_id", "unique_categories", "category_count").show(10, truncate=False)


def ml_feature_engineering(customers_df, transactions_df):
    """Demonstrate ML feature engineering operations"""
    print("\n=== ML FEATURE ENGINEERING ===")

    # Prepare customer features
    customer_features = customers_df.fillna({
        "income": customers_df.agg(mean("income")).collect()[0][0],
        "credit_score": customers_df.agg(mean("credit_score")).collect()[0][0]
    })

    # String indexing
    region_indexer = StringIndexer(inputCol="region", outputCol="region_indexed")
    education_indexer = StringIndexer(inputCol="education", outputCol="education_indexed")
    marital_indexer = StringIndexer(inputCol="marital_status", outputCol="marital_indexed")

    # One-hot encoding
    region_encoder = OneHotEncoder(inputCol="region_indexed", outputCol="region_encoded")
    education_encoder = OneHotEncoder(inputCol="education_indexed", outputCol="education_encoded")

    # Bucketing/Binning
    age_bucketizer = Bucketizer(
        splits=[0, 25, 35, 45, 55, 65, float('inf')],
        inputCol="age",
        outputCol="age_bucket"
    )

    income_discretizer = QuantileDiscretizer(
        numBuckets=5,
        inputCol="income",
        outputCol="income_quintile"
    )

    # Assemble features - need to handle encoded columns properly
    # Note: In real implementation, you'd get the actual column names after encoding
    numeric_cols = ["age", "income", "credit_score", "years_employed", "num_accounts"]

    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")

    # Scaling
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                            withStd=True, withMean=True)

    # Create pipeline - simplified to avoid encoding column name issues
    pipeline = Pipeline(stages=[
        region_indexer, education_indexer, marital_indexer,
        age_bucketizer, income_discretizer,
        assembler, scaler
    ])

    # Fit and transform
    pipeline_model = pipeline.fit(customer_features)
    feature_df = pipeline_model.transform(customer_features)

    print("Feature engineering pipeline completed")
    feature_df.select("customer_id", "age_bucket", "income_quintile", "region_indexed").show(10)

    # PCA for dimensionality reduction
    pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")  # Reduced to 3 components
    pca_model = pca.fit(feature_df)
    pca_df = pca_model.transform(feature_df)

    print("PCA transformation completed")
    print(f"Explained variance: {pca_model.explainedVariance}")

    return feature_df


def statistical_operations(customers_df, transactions_df):
    """Demonstrate statistical operations"""
    print("\n=== STATISTICAL OPERATIONS ===")

    # Correlation analysis
    numeric_cols = ["age", "income", "credit_score", "years_employed"]

    # Fill null values for correlation
    customers_clean = customers_df.fillna({
        "income": customers_df.agg(mean("income")).collect()[0][0],
        "credit_score": customers_df.agg(mean("credit_score")).collect()[0][0]
    })

    # Pearson correlation
    print("Correlation matrix:")
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i <= j:
                correlation = customers_clean.stat.corr(col1, col2)
                print(f"{col1} - {col2}: {correlation:.3f}")

    # Cross tabulation
    print("\nCross tabulation - Region vs Education:")
    cross_tab = customers_df.stat.crosstab("region", "education")
    cross_tab.show()

    # Quantile calculations
    print("\nQuantile analysis:")
    quantiles = customers_clean.stat.approxQuantile("income", [0.25, 0.5, 0.75, 0.9], 0.05)
    print(f"Income quantiles (25%, 50%, 75%, 90%): {quantiles}")

    # Frequent items
    print("\nFrequent items analysis:")
    frequent_categories = transactions_df.stat.freqItems(["category"], 0.3)
    frequent_categories.show()


def performance_optimizations(customers_df, transactions_df):
    """Demonstrate performance optimization techniques"""
    print("\n=== PERFORMANCE OPTIMIZATIONS ===")

    # Caching
    customers_df.cache()
    transactions_df.cache()

    print(f"Customers cached: {customers_df.is_cached}")
    print(f"Transactions cached: {transactions_df.is_cached}")

    # Repartitioning
    print(f"Original partitions - customers: {customers_df.rdd.getNumPartitions()}")
    customers_repartitioned = customers_df.repartition(8, "region")
    print(f"After repartitioning: {customers_repartitioned.rdd.getNumPartitions()}")

    # Coalescing
    transactions_coalesced = transactions_df.coalesce(4)
    print(f"After coalescing transactions: {transactions_coalesced.rdd.getNumPartitions()}")

    # Broadcast join hint (simulated)
    print("\nBroadcast join simulation completed")

    # Bucketing (conceptual - would be used when writing to storage)
    print("Bucketing strategy identified for customer_id")


def data_quality_checks(customers_df, transactions_df):
    """Perform data quality checks"""
    print("\n=== DATA QUALITY CHECKS ===")

    # Null value analysis
    print("Null value analysis for customers:")
    for col_name in customers_df.columns:
        null_count = customers_df.filter(col(col_name).isNull()).count()
        total_count = customers_df.count()
        null_percentage = (null_count / total_count) * 100
        print(f"{col_name}: {null_count} nulls ({null_percentage:.2f}%)")

    # Duplicate analysis
    duplicate_customers = customers_df.groupBy("customer_id").count().filter(col("count") > 1)
    print(f"\nDuplicate customer IDs: {duplicate_customers.count()}")

    # Data range validation
    print("\nData range validation:")
    age_issues = customers_df.filter((col("age") < 18) | (col("age") > 100)).count()
    income_issues = customers_df.filter(col("income") < 0).count()
    credit_issues = customers_df.filter((col("credit_score") < 300) | (col("credit_score") > 850)).count()

    print(f"Age issues: {age_issues}")
    print(f"Income issues: {income_issues}")
    print(f"Credit score issues: {credit_issues}")

    # Referential integrity
    orphan_transactions = transactions_df.join(customers_df, "customer_id", "left_anti")
    print(f"Orphan transactions (no matching customer): {orphan_transactions.count()}")


def main():
    """Main function to run all PySpark operations"""
    print("Starting PySpark data analysis test...")

    # Create Spark session
    spark = create_spark_session()

    try:
        # Create sample datasets
        customers_df, transactions_df, products_df = create_sample_dataset(spark)

        # Run various operations
        customers_enhanced = basic_dataframe_operations(customers_df, transactions_df, products_df)

        aggregations_and_grouping(customers_df, transactions_df)

        window_functions(customers_df, transactions_df)

        joins_and_unions(customers_df, transactions_df, products_df)

        string_operations_and_regex(customers_df, transactions_df)

        advanced_transformations(customers_df, transactions_df)

        feature_df = ml_feature_engineering(customers_df, transactions_df)

        statistical_operations(customers_df, transactions_df)

        performance_optimizations(customers_df, transactions_df)

        data_quality_checks(customers_df, transactions_df)

        # Additional advanced operations
        complex_analytics(customers_df, transactions_df, products_df)

        time_series_operations(transactions_df)

        # Summary
        print("\n" + "=" * 60)
        print("PYSPARK ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Customers processed: {customers_df.count():,}")
        print(f"Transactions processed: {transactions_df.count():,}")
        print(f"Products in catalog: {products_df.count():,}")
        print("All PySpark operations completed successfully!")
        print("Ready for migration testing to Snowflake/SageMaker")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Stop Spark session
        spark.stop()
        print("Spark session stopped.")


def complex_analytics(customers_df, transactions_df, products_df):
    """Demonstrate complex analytical operations"""
    print("\n=== COMPLEX ANALYTICS ===")

    # Customer Lifetime Value calculation
    customer_ltv = transactions_df.groupBy("customer_id").agg(
        spark_sum("amount").alias("total_spent"),
        count("*").alias("transaction_count"),
        datediff(max("transaction_date"), min("transaction_date")).alias("customer_lifetime_days"),
        mean("amount").alias("avg_transaction_value")
    ).withColumn(
        "transactions_per_day",
        col("transaction_count") / (col("customer_lifetime_days") + 1)
    ).withColumn(
        "estimated_annual_value",
        col("transactions_per_day") * col("avg_transaction_value") * 365
    )

    print("Customer Lifetime Value Analysis:")
    customer_ltv.orderBy(desc("estimated_annual_value")).show(10)

    # Cohort analysis preparation
    customers_with_registration = customers_df.withColumn(
        "registration_month",
        concat(year("registration_date"), lit("-"),
               when(month("registration_date") < 10, concat(lit("0"), month("registration_date")))
               .otherwise(month("registration_date")))
    )

    transactions_with_month = transactions_df.withColumn(
        "transaction_month",
        concat(year("transaction_date"), lit("-"),
               when(month("transaction_date") < 10, concat(lit("0"), month("transaction_date")))
               .otherwise(month("transaction_date")))
    )

    # RFM Analysis (Recency, Frequency, Monetary)
    current_date = "2024-08-20"  # Simulate current date

    rfm_analysis = transactions_df.groupBy("customer_id").agg(
        datediff(lit(current_date), max("transaction_date")).alias("recency"),
        count("*").alias("frequency"),
        spark_sum("amount").alias("monetary")
    ).withColumn(
        "recency_score",
        when(col("recency") <= 30, 5)
        .when(col("recency") <= 60, 4)
        .when(col("recency") <= 90, 3)
        .when(col("recency") <= 180, 2)
        .otherwise(1)
    ).withColumn(
        "frequency_score",
        when(col("frequency") >= 20, 5)
        .when(col("frequency") >= 15, 4)
        .when(col("frequency") >= 10, 3)
        .when(col("frequency") >= 5, 2)
        .otherwise(1)
    ).withColumn(
        "monetary_score",
        when(col("monetary") >= 5000, 5)
        .when(col("monetary") >= 2000, 4)
        .when(col("monetary") >= 1000, 3)
        .when(col("monetary") >= 500, 2)
        .otherwise(1)
    ).withColumn(
        "rfm_score",
        concat(col("recency_score"), col("frequency_score"), col("monetary_score"))
    )

    print("RFM Analysis:")
    rfm_analysis.select("customer_id", "recency", "frequency", "monetary", "rfm_score") \
        .orderBy(desc("monetary")).show(15)

    # Market basket analysis preparation
    market_basket = transactions_df.groupBy("customer_id", "transaction_date").agg(
        collect_list("category").alias("basket")
    ).withColumn(
        "basket_size", size("basket")
    ).filter(col("basket_size") > 1)

    print("Market Basket Analysis (sample baskets):")
    market_basket.show(10, truncate=False)

    # Seasonal analysis
    seasonal_analysis = transactions_df.withColumn("month", month("transaction_date")) \
        .groupBy("month", "category").agg(
        spark_sum("amount").alias("total_sales"),
        count("*").alias("transaction_count"),
        mean("amount").alias("avg_transaction_value")
    ).withColumn(
        "month_category",
        concat(col("month"), lit("-"), col("category"))
    )

    print("Seasonal Analysis by Category:")
    seasonal_analysis.orderBy("month", desc("total_sales")).show(20)

    # Fraud detection features
    fraud_features = transactions_df.withColumn(
        "is_weekend",
        when(date_format(col("transaction_date"), "E").isin(["Sat", "Sun"]), 1).otherwise(0)
    ).withColumn(
        "day_of_week_num",
        date_format(col("transaction_date"), "u").cast("int")  # 1=Monday, 7=Sunday
    ).withColumn(
        "amount_zscore",
        expr("(amount - avg(amount) over()) / stddev(amount) over()")
    )

    # Customer transaction patterns
    customer_patterns = transactions_df.withColumn(
        "day_of_week", date_format("transaction_date", "E")
    ).withColumn(
        "is_high_value", when(col("amount") > 1000, 1).otherwise(0)
    ).groupBy("customer_id").agg(
        count("*").alias("total_transactions"),
        spark_sum("is_high_value").alias("high_value_transactions"),
        collect_set("day_of_week").alias("active_days"),
        collect_set("category").alias("preferred_categories"),
        stddev("amount").alias("spending_volatility")
    ).withColumn(
        "high_value_ratio", col("high_value_transactions") / col("total_transactions")
    ).withColumn(
        "active_days_count", size("active_days")
    ).withColumn(
        "category_diversity", size("preferred_categories")
    )

    print("Customer Transaction Patterns:")
    customer_patterns.select("customer_id", "total_transactions", "high_value_ratio",
                             "active_days_count", "category_diversity", "spending_volatility") \
        .orderBy(desc("spending_volatility")).show(15)

    # Product performance analysis
    product_performance = transactions_df.join(products_df,
                                               transactions_df.category == products_df.category, "inner") \
        .groupBy("product_id", "product_name", products_df.category, "price").agg(
        count("*").alias("times_in_transactions"),
        spark_sum("amount").alias("revenue_attributed")
    ).withColumn(
        "revenue_per_transaction", col("revenue_attributed") / col("times_in_transactions")
    )

    print("Product Performance Analysis:")
    product_performance.orderBy(desc("revenue_attributed")).show(15)


def time_series_operations(transactions_df):
    """Demonstrate time series operations"""
    print("\n=== TIME SERIES OPERATIONS ===")

    # Daily aggregations
    daily_metrics = transactions_df.groupBy("transaction_date").agg(
        count("*").alias("daily_transactions"),
        spark_sum("amount").alias("daily_revenue"),
        mean("amount").alias("avg_daily_transaction"),
        countDistinct("customer_id").alias("unique_customers")
    ).orderBy("transaction_date")

    # Moving averages using window functions
    days_window = Window.orderBy("transaction_date").rowsBetween(-6, 0)  # 7-day window

    daily_with_ma = daily_metrics.withColumn(
        "revenue_7day_ma", mean("daily_revenue").over(days_window)
    ).withColumn(
        "transactions_7day_ma", mean("daily_transactions").over(days_window)
    ).withColumn(
        "revenue_growth",
        expr("""
        (daily_revenue - lag(daily_revenue, 1) over (order by transaction_date)) /
        lag(daily_revenue, 1) over (order by transaction_date) * 100
        """)
    )

    print("Daily metrics with moving averages:")
    daily_with_ma.show(15)

    # Cumulative metrics
    cumulative_metrics = daily_metrics.withColumn(
        "cumulative_revenue",
        spark_sum("daily_revenue").over(Window.orderBy("transaction_date")
                                        .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    ).withColumn(
        "cumulative_transactions",
        spark_sum("daily_transactions").over(Window.orderBy("transaction_date")
                                             .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    )

    print("Cumulative metrics:")
    cumulative_metrics.show(10)


if __name__ == "__main__":
    main()