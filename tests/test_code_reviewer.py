"""
Code Reviewer Test Module (test_code_reviewer.py) - V2.0

This test module validates the complete functionality of CodeReviewer V2.0 using the
"Auditor + Surgeon" two-phase approach with structured JSON schema validation.

Key Changes from V1.1:
- Updated to test new JSON schema with 'overall_assessment' and 'findings'
- Validates PERFECT vs NEEDS_REFINEMENT status handling
- Tests structured findings with faulty_code_snippet and suggested_correction
- Focuses on the two-phase workflow validation

Testing Philosophy (POC Phase):
- Real API calls: All tests execute real network requests to Snowflake Cortex. No mocking.
- Real function examples: Uses actual generate_customer_insights function from sample script.
- Structure and workflow validation: Focus on validating Auditor-Surgeon pattern execution.
"""

import pytest
import json
import os
from typing import Dict, Any

# Import modules under test
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService
from agents.code_reviewer import CodeReviewer


class TestCodeReviewerV2:
    """CodeReviewer V2.0 Test Class with Auditor-Surgeon pattern validation."""

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
    from snowflake.snowpark.functions import count, avg, sum

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
    def well_migrated_snowpark_function(self):
        """Return a well-migrated Snowpark function for testing PERFECT status."""
        return '''def generate_customer_insights(df: DataFrame) -> DataFrame:
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

    def test_review_and_correct_returns_correct_structure_v2(self, reviewer, enriched_pyspark_function,
                                                             migrated_snowpark_function_with_issues,
                                                             function_analysis_data, knowledge_service):
        """Test Case 1: Validate V2.0 output structure of review_and_correct_migration."""
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

    def test_v2_review_report_schema_validation(self, reviewer, enriched_pyspark_function,
                                                migrated_snowpark_function_with_issues,
                                                function_analysis_data, knowledge_service):
        """Test Case 2: Deep validation of V2.0 JSON schema structure."""
        # Execute the complete review and correction process
        result = reviewer.review_and_correct_migration(
            original_function_code=enriched_pyspark_function,
            migrated_function_code=migrated_snowpark_function_with_issues,
            knowledge_service=knowledge_service,
            function_analysis=function_analysis_data
        )

        # Extract review report section
        review_report = result["review_report"]

        # Verify V2.0 required fields
        assert "overall_assessment" in review_report, "Missing 'overall_assessment' field"
        assert "findings" in review_report, "Missing 'findings' field"

        # Validate overall_assessment structure
        overall_assessment = review_report["overall_assessment"]
        assert isinstance(overall_assessment, dict), "overall_assessment should be a dictionary"
        assert "status" in overall_assessment, "Missing 'status' in overall_assessment"
        assert "summary" in overall_assessment, "Missing 'summary' in overall_assessment"

        # Validate status enum values
        status = overall_assessment["status"]
        assert status in ["PERFECT", "NEEDS_REFINEMENT"], f"Invalid status: {status}"

        # Validate summary
        assert isinstance(overall_assessment["summary"], str), "summary should be a string"
        assert len(overall_assessment["summary"]) > 0, "summary should not be empty"

        # Validate findings structure
        findings = review_report["findings"]
        assert isinstance(findings, list), "findings should be a list"

        # If status is NEEDS_REFINEMENT, findings should not be empty
        if status == "NEEDS_REFINEMENT":
            assert len(findings) > 0, "findings should not be empty when status is NEEDS_REFINEMENT"

            # Validate each finding structure
            for finding in findings:
                assert isinstance(finding, dict), "Each finding should be a dictionary"

                required_finding_keys = ["category", "faulty_code_snippet", "issue_description", "suggested_correction"]
                for key in required_finding_keys:
                    assert key in finding, f"Missing key '{key}' in finding"

                # Validate category enum
                assert finding["category"] in ["API_MISUSE", "LOGIC_DIVERGENCE", "BEST_PRACTICE_VIOLATION", "STYLE_ISSUE"], \
                    f"Invalid category: {finding['category']}"

                # Validate string fields are not empty
                for key in ["faulty_code_snippet", "issue_description", "suggested_correction"]:
                    assert isinstance(finding[key], str), f"{key} should be a string"
                    assert len(finding[key]) > 0, f"{key} should not be empty"

        # If status is PERFECT, findings should be empty
        elif status == "PERFECT":
            assert len(findings) == 0, "findings should be empty when status is PERFECT"

    def test_auditor_detects_issues_v2(self, reviewer, enriched_pyspark_function,
                                       migrated_snowpark_function_with_issues,
                                       function_analysis_data, knowledge_service):
        """Test Case 3: Verify V2.0 auditor detects migration issues."""
        # Execute review and correction
        result = reviewer.review_and_correct_migration(
            original_function_code=enriched_pyspark_function,
            migrated_function_code=migrated_snowpark_function_with_issues,
            knowledge_service=knowledge_service,
            function_analysis=function_analysis_data
        )

        review_report = result["review_report"]
        overall_assessment = review_report["overall_assessment"]

        # The migrated function has intentional issues (groupBy instead of group_by)
        # The auditor should detect these and mark as NEEDS_REFINEMENT
        assert overall_assessment["status"] == "NEEDS_REFINEMENT", \
            "Should detect issues and mark as NEEDS_REFINEMENT"

        # Should have findings with specific issues
        findings = review_report["findings"]
        assert len(findings) > 0, "Should have findings for the detected issues"

        # Verify findings contain actionable corrections
        for finding in findings:
            assert len(finding["faulty_code_snippet"]) > 0, \
                "faulty_code_snippet should contain actual problematic code"
            assert len(finding["suggested_correction"]) > 0, \
                "suggested_correction should contain replacement code"

    def test_surgeon_applies_corrections_v2(self, reviewer, enriched_pyspark_function,
                                            migrated_snowpark_function_with_issues,
                                            function_analysis_data, knowledge_service):
        """Test Case 4: Verify V2.0 surgeon applies corrections properly."""
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

        # Verify corrected_code is different from original (corrections were applied)
        assert corrected_code != original_migrated, \
            "Corrected code should be different from original migrated code"

        # Basic validation that it's still Python code
        assert "def " in corrected_code, "Should still contain function definition"
        assert "return " in corrected_code, "Should still contain return statement"

    def test_perfect_code_workflow_v2(self, reviewer, enriched_pyspark_function,
                                      well_migrated_snowpark_function,
                                      function_analysis_data, knowledge_service):
        """Test Case 5: Verify V2.0 workflow with well-migrated code (PERFECT status)."""
        # Execute review and correction with well-migrated code
        result = reviewer.review_and_correct_migration(
            original_function_code=enriched_pyspark_function,
            migrated_function_code=well_migrated_snowpark_function,
            knowledge_service=knowledge_service,
            function_analysis=function_analysis_data
        )

        review_report = result["review_report"]
        overall_assessment = review_report["overall_assessment"]

        # Should recognize good code as PERFECT (or at least high quality)
        # Note: Due to LLM variability, we allow for either PERFECT or high-quality NEEDS_REFINEMENT
        status = overall_assessment["status"]
        assert status in ["PERFECT", "NEEDS_REFINEMENT"], f"Status should be valid: {status}"

        # If PERFECT, findings should be empty
        if status == "PERFECT":
            assert len(review_report["findings"]) == 0, \
                "Findings should be empty for PERFECT status"

            # Corrected code should be identical to original when PERFECT
            assert result["corrected_code"] == well_migrated_snowpark_function, \
                "Corrected code should be unchanged for PERFECT status"

        # Verify structure regardless of status
        assert isinstance(result["corrected_code"], str), "Should return corrected code"
        assert "generate_customer_insights" in result["corrected_code"], \
            "Function name should be preserved"

    def test_complete_v2_workflow_inspection(self, reviewer, enriched_pyspark_function,
                                             migrated_snowpark_function_with_issues,
                                             function_analysis_data, knowledge_service):
        """Test Case 6: Print complete V2.0 Auditor-Surgeon workflow for inspection."""
        # Execute the complete review and correction process
        result = reviewer.review_and_correct_migration(
            original_function_code=enriched_pyspark_function,
            migrated_function_code=migrated_snowpark_function_with_issues,
            knowledge_service=knowledge_service,
            function_analysis=function_analysis_data
        )

        # Verify result is not None
        assert result is not None, "Review and correction result should not be None"

        # Print comprehensive V2.0 results
        print("\n" + "=" * 100)
        print("CODE REVIEWER V2.0 - AUDITOR + SURGEON WORKFLOW INSPECTION")
        print("=" * 100)

        print("\n--- PHASE 0: INPUT DATA ---")
        print("Original PySpark Function (with guidance):")
        print(enriched_pyspark_function)
        print("\nMigrated Snowpark Function (with issues):")
        print(migrated_snowpark_function_with_issues)
        print("\nFunction Analysis Data:")
        print(json.dumps(function_analysis_data, indent=2))

        print("\n--- PHASE 1: AUDITOR RESULTS ---")
        review_report = result.get("review_report")
        print("Overall Assessment:")
        print(json.dumps(review_report.get("overall_assessment"), indent=2))
        print("\nFindings:")
        print(json.dumps(review_report.get("findings"), indent=2))

        print("\n--- PHASE 2: SURGEON RESULTS ---")
        print("Final Corrected Code:")
        print(result.get("corrected_code"))

        print("\n--- WORKFLOW SUMMARY ---")
        status = review_report.get("overall_assessment", {}).get("status")
        findings_count = len(review_report.get("findings", []))
        print(f"Audit Status: {status}")
        print(f"Issues Found: {findings_count}")
        print(f"Surgeon Applied: {'Yes' if status == 'NEEDS_REFINEMENT' else 'No (Perfect Code)'}")

        print("\n" + "-" * 100)
        print("V2.0 AUDITOR-SURGEON WORKFLOW COMPLETED")
        print("-" * 100 + "\n")

    def test_complex_function_v2_workflow(self, reviewer, knowledge_service):
        """Test Case 7: Test V2.0 workflow with complex function scenario."""
        # Complex original function
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
    from pyspark.sql.functions import row_number, desc, col
    
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

        # Complex migrated function with multiple issues
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
    
    # ISSUE: still uses withColumn instead of with_column
    window_spec = Window.partition_by("customer_tier").order_by(desc("total_spent"))
    ranked_df = df.withColumn("tier_rank", row_number().over(window_spec))
    
    # ISSUE: mixed method names (groupBy vs group_by, orderBy vs order_by)
    insights = ranked_df.filter(col("tier_rank") <= 10) \\
        .groupBy("customer_tier", "age_group") \\
        .agg(
            count("customer_id").alias("top_customer_count"),
            avg("total_spent").alias("avg_top_spending"),
            sum("total_transaction_amount").alias("top_revenue")
        ) \\
        .orderBy("customer_tier", desc("top_revenue"))
        
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

        # Validate V2.0 structure for complex scenario
        assert isinstance(result, dict), "Should return dictionary for complex functions"
        assert "review_report" in result, "Should contain review report"
        assert "corrected_code" in result, "Should contain corrected code"

        # Validate V2.0 schema compliance
        review_report = result["review_report"]
        assert "overall_assessment" in review_report, "Should have overall_assessment"
        assert "findings" in review_report, "Should have findings"

        overall_assessment = review_report["overall_assessment"]
        assert overall_assessment["status"] in ["PERFECT", "NEEDS_REFINEMENT"], \
            "Should have valid status"

        # Complex functions with issues should likely need refinement
        if overall_assessment["status"] == "NEEDS_REFINEMENT":
            findings = review_report["findings"]
            assert len(findings) > 0, "Should have findings for complex function issues"

            # Validate findings structure for complex scenario
            for finding in findings:
                assert finding["category"] in ["API_MISUSE", "LOGIC_DIVERGENCE", "BEST_PRACTICE_VIOLATION", "STYLE_ISSUE"], \
                    f"Invalid category in complex scenario: {finding['category']}"