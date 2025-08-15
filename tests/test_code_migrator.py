"""
Code Migrator Test Module (test_code_migrator.py) - V3.0

This test module validates the complete functionality of CodeMigrator's "triage doctor"
approach when interacting with real LLM services and knowledge services, focusing on
testing all three migration strategies: PROCEED_AND_MIGRATE, PROCEED_AND_ANNOTATE,
and ABORT_AND_FLAG.

Testing Philosophy (POC Phase):
- Real API calls: All tests execute real network requests. No mocking.
- Structure and reasonableness validation: Focus on validating key features of the
  returned code strings rather than exact text matching.
- Triage testing: Validate the migration feasibility checking logic.
"""

import pytest
import json
import os
from typing import Dict, Any

# Import modules under test
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService
from agents.code_analyzer import CodeAnalyzer
from agents.code_enricher import CodeEnricher
from agents.code_migrator import CodeMigrator, MigrationError


class TestCodeMigrator:
    """CodeMigrator Test Class with real service integration and triage testing."""

    @pytest.fixture(scope="session")
    def llm_service(self):
        """Create a real CortexLLMService instance."""
        try:
            service = CortexLLMService()
            return service
        except Exception as e:
            pytest.fail(f"Failed to create CortexLLMService instance: {e}")

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
    def code_analyzer(self, llm_service):
        """Create CodeAnalyzer instance."""
        return CodeAnalyzer(llm_service=llm_service)

    @pytest.fixture(scope="session")
    def code_enricher(self, llm_service):
        """Create CodeEnricher instance."""
        return CodeEnricher(llm_service=llm_service)

    @pytest.fixture(scope="session")
    def code_migrator(self, llm_service):
        """Create CodeMigrator instance."""
        return CodeMigrator(llm_service=llm_service)

    @pytest.fixture(scope="session")
    def target_function_code(self):
        """Return the target function to test migration."""
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
    def complete_script_content(self):
        """Return the complete sample script content."""
        script_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'sample_spark_script.py')
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback to embedded content if file not found
            return '''
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when, sum as spark_sum, count, avg

def generate_customer_insights(df: DataFrame) -> DataFrame:
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

    return insights_df
'''

    def test_triage_proceed_and_migrate(self, code_migrator, knowledge_service):
        """Test PROCEED_AND_MIGRATE scenario with safe dependencies."""

        safe_function = '''def safe_aggregation(df):
    """Safe function with only PySpark dependencies."""
    result = df.groupBy("category").agg(count("id").alias("total"))
    return result'''

        # Analysis with only safe dependencies
        safe_analysis = {
            "function_name": "safe_aggregation",
            "dependencies": {
                "internal_functions": [],
                "external_packages": ["from pyspark.sql.functions import count", "from pyspark.sql import DataFrame"]
            },
            "suggested_patterns": ["pyspark.groupBy", "pyspark.agg"]
        }

        migrated_code = code_migrator.migrate_function(
            source_code=safe_function,
            function_analysis=safe_analysis,
            knowledge_service=knowledge_service
        )

        # Should be normally migrated without warnings
        assert isinstance(migrated_code, str)
        assert "def safe_aggregation" in migrated_code
        assert "MIGRATION TOOL WARNING" not in migrated_code
        assert "MIGRATION SKIPPED" not in migrated_code
        assert "group_by" in migrated_code

        print("\n=== PROCEED_AND_MIGRATE Test Results ===")
        print(f"Original: {safe_function}")
        print(f"Migrated: {migrated_code}")

    def test_triage_proceed_and_annotate(self, code_migrator, knowledge_service):
        """Test PROCEED_AND_ANNOTATE scenario with custom module dependencies."""

        custom_module_function = '''def function_with_custom_modules(df):
    """Function with custom module dependencies."""
    from utils.data_validator import validate_schema
    from libs.transformation import apply_business_rules
    
    validated_df = validate_schema(df)
    result = apply_business_rules(validated_df)
    return result.groupBy("category").count()'''

        # Analysis with custom module dependencies
        custom_analysis = {
            "function_name": "function_with_custom_modules",
            "dependencies": {
                "internal_functions": [],
                "external_packages": [
                    "from utils.data_validator import validate_schema",
                    "from libs.transformation import apply_business_rules",
                    "from pyspark.sql.functions import count"
                ]
            },
            "suggested_patterns": ["pyspark.groupBy"]
        }

        migrated_code = code_migrator.migrate_function(
            source_code=custom_module_function,
            function_analysis=custom_analysis,
            knowledge_service=knowledge_service
        )

        # Should be migrated with warnings
        assert isinstance(migrated_code, str)
        assert "def function_with_custom_modules" in migrated_code
        assert "MIGRATION TOOL WARNING" in migrated_code
        assert "utils.data_validator" in migrated_code
        assert "libs.transformation" in migrated_code
        assert "MIGRATION SKIPPED" not in migrated_code

        print("\n=== PROCEED_AND_ANNOTATE Test Results ===")
        print(f"Original: {custom_module_function}")
        print(f"Migrated: {migrated_code}")

    def test_triage_abort_and_flag(self, code_migrator, knowledge_service):
        """Test ABORT_AND_FLAG scenario with fatal dependencies."""

        fatal_dependency_function = '''def function_with_fatal_deps(df):
    """Function with non-migratable dependencies."""
    import requests
    from config.spark_config import get_cluster_config
    
    config = get_cluster_config()
    response = requests.get("https://api.example.com/data")
    
    return df.groupBy("category").count()'''

        # Analysis with fatal dependencies
        fatal_analysis = {
            "function_name": "function_with_fatal_deps",
            "dependencies": {
                "internal_functions": [],
                "external_packages": [
                    "import requests",
                    "from config.spark_config import get_cluster_config",
                    "from pyspark.sql.functions import count"
                ]
            },
            "suggested_patterns": ["pyspark.groupBy"]
        }

        migrated_code = code_migrator.migrate_function(
            source_code=fatal_dependency_function,
            function_analysis=fatal_analysis,
            knowledge_service=knowledge_service
        )

        # Should be aborted with manual intervention stub
        assert isinstance(migrated_code, str)
        assert "MIGRATION SKIPPED: MANUAL INTERVENTION REQUIRED" in migrated_code
        assert "REASON:" in migrated_code
        assert fatal_dependency_function in migrated_code
        assert "def function_with_fatal_deps" not in migrated_code.split("'''")[0]  # Function should be in commented section

        print("\n=== ABORT_AND_FLAG Test Results ===")
        print(f"Original: {fatal_dependency_function}")
        print(f"Stub: {migrated_code}")

    def test_complete_migration_pipeline(self, code_analyzer, code_enricher, code_migrator,
                                         knowledge_service, complete_script_content, target_function_code):
        """
        Test the complete migration pipeline for generate_customer_insights function.

        This test simulates the real workflow:
        1. Analyze the complete script
        2. Extract and enrich the target function
        3. Migrate the enriched function to Snowpark
        """
        # Step 1: Analyze the complete script
        print("\n=== Step 1: Analyzing complete script ===")
        analysis_result = code_analyzer.analyze_script(
            script_content=complete_script_content,
            source_file_name="sample_spark_script.py"
        )

        assert "function_analysis" in analysis_result
        assert isinstance(analysis_result["function_analysis"], list)

        # Find the target function in analysis results
        target_function_analysis = None
        for func_analysis in analysis_result["function_analysis"]:
            if func_analysis.get("function_name") == "generate_customer_insights":
                target_function_analysis = func_analysis
                break

        assert target_function_analysis is not None, "Target function should be found in analysis"
        print(f"Found target function analysis: {target_function_analysis}")

        # Step 2: Enrich the target function (only use enriched_code, not test)
        print("\n=== Step 2: Enriching target function ===")
        enrichment_result = code_enricher.enrich_function(target_function_code)

        assert isinstance(enrichment_result, dict)
        assert "enriched_code" in enrichment_result

        enriched_code = enrichment_result["enriched_code"]
        print(f"Enriched code length: {len(enriched_code)} characters")

        # Step 3: Migrate only the enriched function code (not test)
        print("\n=== Step 3: Migrating enriched function ===")
        migrated_code = code_migrator.migrate_function(
            source_code=enriched_code,
            function_analysis=target_function_analysis,
            knowledge_service=knowledge_service
        )

        # Validate migration results
        assert isinstance(migrated_code, str), "Migrated code should be a string"
        assert len(migrated_code) > 0, "Migrated code should not be empty"

        # Check migration outcome based on dependencies
        if "MIGRATION SKIPPED" in migrated_code:
            print("Function was flagged for manual intervention")
            assert "MANUAL INTERVENTION REQUIRED" in migrated_code
        else:
            assert "def generate_customer_insights" in migrated_code, "Function signature should be preserved"

            # Check for Snowpark conversions if migration proceeded
            if "MIGRATION TOOL WARNING" not in migrated_code:
                assert "group_by" in migrated_code, "groupBy should be converted to group_by"
                assert "order_by" in migrated_code, "orderBy should be converted to order_by"

        print("\n" + "=" * 80)
        print("COMPLETE MIGRATION PIPELINE RESULTS")
        print("=" * 80)

        print("\n--- STEP 1: ANALYSIS RESULTS ---")
        print(f"Function Analysis for '{target_function_analysis['function_name']}':")
        print(f"  Dependencies: {target_function_analysis.get('dependencies', {})}")
        print(f"  Suggested Patterns: {target_function_analysis.get('suggested_patterns', [])}")

        print("\n--- STEP 2: ORIGINAL FUNCTION ---")
        print(target_function_code)

        print("\n--- STEP 3: ENRICHED FUNCTION ---")
        print(enriched_code)

        print("\n--- STEP 4: MIGRATED SNOWPARK FUNCTION ---")
        print(migrated_code)

        print("\n" + "=" * 80)
        print("MIGRATION PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)

    def test_migration_preserves_structure(self, code_enricher, code_migrator,
                                           knowledge_service, target_function_code):
        """Test that migration preserves function structure and comments."""

        # Create mock analysis for the target function with safe dependencies
        mock_analysis = {
            "function_name": "generate_customer_insights",
            "dependencies": {
                "internal_functions": [],
                "external_packages": ["from pyspark.sql.functions import count, avg, sum"]
            },
            "suggested_patterns": ["pyspark.groupBy", "pyspark.agg", "pyspark.orderBy"]
        }

        # Enrich the function (only use enriched_code)
        enrichment_result = code_enricher.enrich_function(target_function_code)
        enriched_code = enrichment_result["enriched_code"]

        # Migrate the enriched function
        migrated_code = code_migrator.migrate_function(
            source_code=enriched_code,
            function_analysis=mock_analysis,
            knowledge_service=knowledge_service
        )

        # Check if migration was successful or flagged
        if "MIGRATION SKIPPED" not in migrated_code:
            # Verify structure preservation for successful migrations
            assert "def generate_customer_insights" in migrated_code, \
                "Function signature should be preserved"
            assert '"""' in migrated_code, "Docstring should be preserved"
            assert "Args:" in migrated_code, "Docstring Args section should be preserved"
            assert "Returns:" in migrated_code, "Docstring Returns section should be preserved"

        # Print results for inspection
        print("\n" + "=" * 60)
        print("STRUCTURE PRESERVATION TEST RESULTS")
        print("=" * 60)
        print("\n--- ENRICHED CODE ---")
        print(enriched_code)
        print("\n--- MIGRATED CODE ---")
        print(migrated_code)
        print("=" * 60)

    def test_migration_handles_complex_aggregations(self, code_migrator, knowledge_service):
        """Test migration of complex aggregation patterns."""

        # Function with complex aggregations and safe dependencies
        complex_agg_function = '''def complex_aggregation(df):
    """Complex aggregation with multiple functions."""
    # Group and aggregate with multiple functions
    result = df.groupBy("category") \\
               .agg(
                   count("id").alias("total_count"),
                   avg("amount").alias("avg_amount"),
                   spark_sum("revenue").alias("total_revenue")
               ) \\
               .orderBy("total_revenue")
    return result'''

        mock_analysis = {
            "function_name": "complex_aggregation",
            "dependencies": {
                "internal_functions": [],
                "external_packages": ["from pyspark.sql.functions import count, avg, sum as spark_sum"]
            },
            "suggested_patterns": ["pyspark.groupBy", "pyspark.agg", "pyspark.orderBy",
                                   "pyspark.functions.count", "pyspark.functions.avg", "pyspark.functions.sum"]
        }

        # Migrate the function
        migrated_code = code_migrator.migrate_function(
            source_code=complex_agg_function,
            function_analysis=mock_analysis,
            knowledge_service=knowledge_service
        )

        # Verify migration result
        assert isinstance(migrated_code, str), "Should return valid migrated code"

        if "MIGRATION SKIPPED" not in migrated_code:
            assert "def complex_aggregation" in migrated_code, "Function name should be preserved"
            assert "group_by" in migrated_code, "groupBy should be converted"
            assert "order_by" in migrated_code, "orderBy should be converted"

    def test_migration_error_handling(self, code_migrator, knowledge_service):
        """Test error handling in migration process."""

        # Test with malformed function analysis structure
        malformed_analysis = {
            "function_name": "test_function",
            # Missing 'dependencies' key
            "suggested_patterns": []
        }

        with pytest.raises(MigrationError):
            code_migrator.migrate_function(
                source_code="def test(): pass",
                function_analysis=malformed_analysis,
                knowledge_service=knowledge_service
            )

    def test_check_migration_feasibility_direct(self, code_migrator):
        """Test the _check_migration_feasibility method directly."""

        # Test safe dependencies
        safe_deps = ["from pyspark.sql.functions import count", "import pyspark.sql as sql"]
        result = code_migrator._check_migration_feasibility(safe_deps)
        assert result['action'] == 'PROCEED_AND_MIGRATE'
        assert result['details'] == []

        # Test custom module dependencies
        custom_deps = ["from utils.helper import process", "from pyspark.sql import DataFrame"]
        result = code_migrator._check_migration_feasibility(custom_deps)
        assert result['action'] == 'PROCEED_AND_ANNOTATE'
        assert "from utils.helper import process" in result['details']

        # Test fatal dependencies
        fatal_deps = ["import requests", "from pyspark.sql import DataFrame"]
        result = code_migrator._check_migration_feasibility(fatal_deps)
        assert result['action'] == 'ABORT_AND_FLAG'
        assert "import requests" in result['details']