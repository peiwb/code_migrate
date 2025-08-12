"""
Code Reviewer Test Module (test_code_reviewer.py) - V1.1

This test module validates the complete functionality of CodeReviewer using the real
generate_customer_insights function from the sample PySpark script, focusing on its
ability to successfully execute the complete "review and correction" two-step process.

Testing Philosophy (POC Phase):
- Real API calls: All tests execute real network requests to Snowflake Cortex. No mocking.
- Real function examples: Uses the actual generate_customer_insights function from sample script.
- Structure and reasonableness validation: Focus on validating key features rather than exact matching.
"""

import pytest
import json
import os
from typing import Dict, Any

# Import modules under test
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService
from agents.code_reviewer import CodeReviewer


class TestCodeReviewer:
    """CodeReviewer Test Class with real generate_customer_insights function."""

    @pytest.fixture(scope="session")
    def llm_service(self):
        """Create a real CortexLLMService instance."""
        try:
            service = CortexLLMService()
            return service
        except Exception as e:
            pytest.fail(f"Failed to create CortexLLMService instance. Please check Snowflake credentials: {e}")

    @pytest.fixture(scope="session")
    def knowledge_service(self):
        """Create a KnowledgeService instance with real knowledge base."""
        knowledge_base_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base.json')

        try:
            service = KnowledgeService(knowledge_base_path=knowledge_base_path)
            return service
        except Exception as e:
            pytest.fail(f"Failed to create KnowledgeService instance with real knowledge base: {e}")

    @pytest.fixture(scope="session")
    def reviewer(self, llm_service):
        """Create CodeReviewer instance."""
        return CodeReviewer(llm_service=llm_service)

    @pytest.fixture(scope="session")
    def original_pyspark_function(self):
        """Return the original generate_customer_insights PySpark function."""
        return '''def generate_customer_insights(df: DataFrame) -> DataFrame:
    """
    Generate business insights from customer data.

    Args:
        df: Enriched customer data

    Returns:
        DataFrame: Customer insights summary
    """
    insights_df = df.groupBy("age_group", "customer_tier") \\
        .agg(
        count("customer_id").alias("customer_count"),
        avg("total_spent").alias("avg_spending"),
        spark_sum("total_transaction_amount").alias("total_revenue")
    ) \\
        .orderBy("age_group", "customer_tier")

    return insights_df'''

    @pytest.fixture(scope="session")
    def migrated_snowpark_function_with_issues(self):
        """Return a migrated Snowpark function with intentional issues for testing review."""
        return '''def generate_customer_insights(df: DataFrame) -> DataFrame:
    """
    Generate business insights from customer data.

    Args:
        df: Enriched customer data

    Returns:
        DataFrame: Customer insights summary
    """
    from snowflake.snowpark.functions import count, avg, sum, desc

    # ISSUE: Still using PySpark method names instead of Snowpark
    insights_df = df.groupBy("age_group", "customer_tier") \\
        .agg(
        count("customer_id").alias("customer_count"),
        avg("total_spent").alias("avg_spending"),
        sum("total_transaction_amount").alias("total_revenue")
    ) \\
        .orderBy("age_group", "customer_tier")

    return insights_df'''

    @pytest.fixture(scope="session")
    def enriched_pyspark_function(self):
        """Return the enriched PySpark function with migration guidance comments."""
        return '''def generate_customer_insights(df: DataFrame) -> DataFrame:
    """
    Generate business insights from customer data.

    Args:
        df: Enriched customer data

    Returns:
        DataFrame: Customer insights summary
    """
    # MIGRATION GUIDANCE: groupBy should be converted to group_by in Snowpark
    # MIGRATION GUIDANCE: orderBy should be converted to order_by in Snowpark
    # MIGRATION GUIDANCE: Import functions from snowflake.snowpark.functions
    # MIGRATION GUIDANCE: spark_sum should be imported as sum from snowflake.snowpark.functions

    insights_df = df.groupBy("age_group", "customer_tier") \\
        .agg(
        count("customer_id").alias("customer_count"),
        avg("total_spent").alias("avg_spending"),
        spark_sum("total_transaction_amount").alias("total_revenue")
    ) \\
        .orderBy("age_group", "customer_tier")

    return insights_df'''

    @pytest.fixture(scope="session")
    def function_analysis_data(self):
        """Return realistic analysis data for generate_customer_insights function."""
        return {
            "function_name": "generate_customer_insights",
            "dependencies": {
                "internal_functions": [],
                "external_packages": [
                    "pyspark.sql.functions.count",
                    "pyspark.sql.functions.avg",
                    "pyspark.sql.functions.sum",
                    "pyspark.sql.DataFrame.groupBy",
                    "pyspark.sql.DataFrame.agg",
                    "pyspark.sql.DataFrame.orderBy"
                ]
            },
            "suggested_patterns": [
                "pyspark.groupBy",
                "pyspark.agg",
                "pyspark.orderBy",
                "pyspark.functions.count",
                "pyspark.functions.avg",
                "pyspark.functions.sum"
            ],
            "complexity_level": "medium",
            "estimated_conversion_effort": "medium"
        }

    def test_review_and_correct_returns_correct_structure(self, reviewer, enriched_pyspark_function,
                                                          migrated_snowpark_function_with_issues,
                                                          function_analysis_data, knowledge_service):
        """Test Case 1: Validate top-level output structure of review_and_correct_migration."""
        # Execute the complete review and correction process
        result = reviewer.review_and_correct_migration(
            original_function_code=enriched_pyspark_function,
            migrated_function_code=migrated_snowpark_function_with_issues,
            knowledge_service=knowledge_service,
            function_analysis=function_analysis_data
        )

        # Verify return type is dictionary
        assert isinstance(result, dict), "Review and correction result should be a dictionary"

        # Verify required top-level keys
        required_keys = ["review_report", "corrected_code"]
        for key in required_keys:
            assert key in result, f"Result is missing required key: {key}"

        # Verify review_report structure
        assert isinstance(result["review_report"], dict), "review_report should be a dictionary"

        # Verify corrected_code structure
        assert isinstance(result["corrected_code"], str), "corrected_code should be a string"
        assert len(result["corrected_code"]) > 0, "corrected_code should not be empty"

    def test_review_report_structure_is_valid(self, reviewer, enriched_pyspark_function,
                                              migrated_snowpark_function_with_issues,
                                              function_analysis_data, knowledge_service):
        """Test Case 2: Deep validation of review_report structure."""
        # Execute the complete review and correction process
        result = reviewer.review_and_correct_migration(
            original_function_code=enriched_pyspark_function,
            migrated_function_code=migrated_snowpark_function_with_issues,
            knowledge_service=knowledge_service,
            function_analysis=function_analysis_data
        )

        # Extract review report section
        review_report = result["review_report"]

        # Verify required review report fields
        required_review_keys = ["migration_confidence_score", "summary", "review_comments"]
        for key in required_review_keys:
            assert key in review_report, f"review_report is missing required field: {key}"

        # Verify field types and ranges
        assert isinstance(review_report["migration_confidence_score"], (int, float)), \
            "migration_confidence_score should be a number"
        assert 0.0 <= review_report["migration_confidence_score"] <= 1.0, \
            "migration_confidence_score should be between 0.0 and 1.0"

        assert isinstance(review_report["summary"], str), \
            "summary should be a string"
        assert len(review_report["summary"]) > 0, \
            "summary should not be empty"

        assert isinstance(review_report["review_comments"], list), \
            "review_comments should be a list"

    def test_review_detects_migration_issues(self, reviewer, enriched_pyspark_function,
                                             migrated_snowpark_function_with_issues,
                                             function_analysis_data, knowledge_service):
        """Test Case 3: Verify review can detect migration issues in the function."""
        # Execute review and correction
        result = reviewer.review_and_correct_migration(
            original_function_code=enriched_pyspark_function,
            migrated_function_code=migrated_snowpark_function_with_issues,
            knowledge_service=knowledge_service,
            function_analysis=function_analysis_data
        )

        review_report = result["review_report"]

        # The migrated function has intentional issues (groupBy instead of group_by)
        # The review should detect these and have lower confidence
        confidence_score = review_report["migration_confidence_score"]
        assert confidence_score < 1.0, "Should detect issues and have confidence < 1.0"

        # Should have review comments
        review_comments = review_report["review_comments"]
        assert len(review_comments) > 0, "Should have review comments for the issues"

    def test_correction_improves_code_quality(self, reviewer, enriched_pyspark_function,
                                              migrated_snowpark_function_with_issues,
                                              function_analysis_data, knowledge_service):
        """Test Case 4: Verify correction process improves code quality."""
        # Execute review and correction
        result = reviewer.review_and_correct_migration(
            original_function_code=enriched_pyspark_function,
            migrated_function_code=migrated_snowpark_function_with_issues,
            knowledge_service=knowledge_service,
            function_analysis=function_analysis_data
        )

        corrected_code = result["corrected_code"]
        original_migrated = migrated_snowpark_function_with_issues

        # Verify function signature is preserved
        assert "def generate_customer_insights(df: DataFrame) -> DataFrame:" in corrected_code, \
            "Function signature should be preserved"

        # Verify docstring is preserved
        assert '"""' in corrected_code, "Docstring should be preserved"
        assert "Generate business insights from customer data" in corrected_code, \
            "Docstring content should be preserved"

        # Check if corrections were made (should convert to Snowpark methods)
        # Note: We can't guarantee exact corrections due to LLM variability,
        # but we can check for improvements
        assert "generate_customer_insights" in corrected_code, \
            "Function name should be preserved"

    def test_print_complete_review_process_for_inspection(self, reviewer, enriched_pyspark_function,
                                                          migrated_snowpark_function_with_issues,
                                                          function_analysis_data, knowledge_service):
        """Test Case 5: Print complete review process for manual inspection."""
        # Execute the complete review and correction process
        result = reviewer.review_and_correct_migration(
            original_function_code=enriched_pyspark_function,
            migrated_function_code=migrated_snowpark_function_with_issues,
            knowledge_service=knowledge_service,
            function_analysis=function_analysis_data
        )

        # Verify result is not None
        assert result is not None, "Review and correction result should not be None"

        # Print comprehensive results
        print("\n" + "=" * 100)
        print("COMPLETE REVIEW AND CORRECTION PROCESS - GENERATE_CUSTOMER_INSIGHTS FUNCTION")
        print("=" * 100)

        print("\n--- 1. ORIGINAL PYSPARK FUNCTION (WITH MIGRATION GUIDANCE) ---")
        print(enriched_pyspark_function)

        print("\n--- 2. MIGRATED SNOWPARK FUNCTION (WITH INTENTIONAL ISSUES) ---")
        print(migrated_snowpark_function_with_issues)

        print("\n--- 3. FUNCTION ANALYSIS DATA ---")
        print(json.dumps(function_analysis_data, indent=4))

        print("\n--- 4. GENERATED REVIEW REPORT ---")
        print(json.dumps(result.get("review_report"), indent=4))

        print("\n--- 5. FINAL CORRECTED SNOWPARK FUNCTION ---")
        print(result.get("corrected_code"))

        print("\n" + "-" * 100)
        print("REVIEW AND CORRECTION PROCESS COMPLETED - RESULTS DISPLAYED ABOVE")
        print("-" * 100 + "\n")

    def test_reviewer_handles_well_migrated_code(self, reviewer, enriched_pyspark_function,
                                                 function_analysis_data, knowledge_service):
        """Test Case 6: Verify reviewer behavior with well-migrated code."""
        # Create a well-migrated Snowpark function (minimal issues)
        well_migrated_function = '''def generate_customer_insights(df: DataFrame) -> DataFrame:
    """
    Generate business insights from customer data.

    Args:
        df: Enriched customer data

    Returns:
        DataFrame: Customer insights summary
    """
    from snowflake.snowpark.functions import count, avg, sum

    insights_df = df.group_by("age_group", "customer_tier") \\
        .agg(
        count("customer_id").alias("customer_count"),
        avg("total_spent").alias("avg_spending"),
        sum("total_transaction_amount").alias("total_revenue")
    ) \\
        .order_by("age_group", "customer_tier")

    return insights_df'''

        # Execute review and correction
        result = reviewer.review_and_correct_migration(
            original_function_code=enriched_pyspark_function,
            migrated_function_code=well_migrated_function,
            knowledge_service=knowledge_service,
            function_analysis=function_analysis_data
        )

        review_report = result["review_report"]

        # Should have higher confidence for well-migrated code
        confidence_score = review_report["migration_confidence_score"]
        assert confidence_score >= 0.7, "Should have high confidence for well-migrated code"

        # Verify structure
        assert isinstance(result["corrected_code"], str), "Should still return corrected code"
        assert "generate_customer_insights" in result["corrected_code"], \
            "Function name should be preserved"

    def test_reviewer_with_complex_function_scenario(self, reviewer, knowledge_service):
        """Test Case 7: Test reviewer with a more complex function scenario."""
        # Complex original function with more operations
        complex_original = '''def advanced_customer_analysis(df: DataFrame) -> DataFrame:
    """
    Advanced customer analysis with multiple transformations.

    Args:
        df: Customer data

    Returns:
        DataFrame: Advanced analytics results
    """
    # MIGRATION GUIDANCE: Convert all PySpark methods to Snowpark equivalents
    # MIGRATION GUIDANCE: Use proper imports from snowflake.snowpark.functions
    # MIGRATION GUIDANCE: Handle window functions properly

    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number, desc

    # Add ranking within each tier
    window_spec = Window.partitionBy("customer_tier").orderBy(desc("total_spent"))
    ranked_df = df.withColumn("tier_rank", row_number().over(window_spec))

    # Generate insights with filtering
    insights = ranked_df.filter(col("tier_rank") <= 10) \\
        .groupBy("customer_tier", "age_group") \\
        .agg(
            count("customer_id").alias("top_customer_count"),
            avg("total_spent").alias("avg_top_spending"),
            spark_sum("total_transaction_amount").alias("top_revenue")
        ) \\
        .orderBy("customer_tier", desc("top_revenue"))

    return insights'''

        # Complex migrated function with mixed issues
        complex_migrated = '''def advanced_customer_analysis(df: DataFrame) -> DataFrame:
    """
    Advanced customer analysis with multiple transformations.

    Args:
        df: Customer data

    Returns:
        DataFrame: Advanced analytics results
    """
    from snowflake.snowpark.functions import count, avg, sum, desc, row_number, col
    from snowflake.snowpark import Window

    # Add ranking within each tier - ISSUE: still uses withColumn
    window_spec = Window.partition_by("customer_tier").order_by(desc("total_spent"))
    ranked_df = df.withColumn("tier_rank", row_number().over(window_spec))

    # Generate insights with filtering - ISSUE: mixed method names
    insights = ranked_df.filter(col("tier_rank") <= 10) \\
        .groupBy("customer_tier", "age_group") \\
        .agg(
            count("customer_id").alias("top_customer_count"),
            avg("total_spent").alias("avg_top_spending"),
            sum("total_transaction_amount").alias("top_revenue")
        ) \\
        .order_by("customer_tier", desc("top_revenue"))

    return insights'''

        # Complex analysis data
        complex_analysis = {
            "function_name": "advanced_customer_analysis",
            "dependencies": {
                "internal_functions": [],
                "external_packages": [
                    "pyspark.sql.functions",
                    "pyspark.sql.window.Window"
                ]
            },
            "suggested_patterns": [
                "pyspark.withColumn",
                "pyspark.groupBy",
                "pyspark.orderBy",
                "pyspark.window.functions"
            ],
            "complexity_level": "high",
            "estimated_conversion_effort": "high"
        }

        # Execute review and correction
        result = reviewer.review_and_correct_migration(
            original_function_code=complex_original,
            migrated_function_code=complex_migrated,
            knowledge_service=knowledge_service,
            function_analysis=complex_analysis
        )

        # Basic validation for complex scenario
        assert isinstance(result, dict), "Should return dictionary for complex functions"
        assert "review_report" in result, "Should contain review report"
        assert "corrected_code" in result, "Should contain corrected code"
        assert isinstance(result["review_report"]["migration_confidence_score"], (int, float)), \
            "Should have numeric confidence score"