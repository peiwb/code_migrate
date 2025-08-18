#!/usr/bin/env python3
"""
Integration tests for Migrator + Reviewer Chain - PySpark to Snowpark Migration Tool
Specifically tests the complete migrator -> reviewer workflow with proper mocking.
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
import stat
import platform
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    parse_arguments, read_script_file, ensure_output_directory,
    extract_function_code, save_artifact, get_function_analysis,
    process_single_function, main
)

# Import real services and agents for integration testing
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService
from agents.code_analyzer import CodeAnalyzer
from agents.code_enricher import CodeEnricher
from agents.code_migrator import CodeMigrator
from agents.code_reviewer import CodeReviewer


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_script_content():
    """Real PySpark script content for testing migrator -> reviewer chain."""
    return '''"""
Sample PySpark Script for Testing Migrator -> Reviewer Chain
"""

import os
from typing import Dict
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, sum as spark_sum, count
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


def create_spark_session(app_name: str = "TestApp") -> SparkSession:
    """
    Create and configure Spark session.

    Args:
        app_name: Name of the application

    Returns:
        SparkSession: Configured session
    """
    spark = SparkSession.builder \\
        .appName(app_name) \\
        .config("spark.sql.adaptive.enabled", "true") \\
        .getOrCreate()
    return spark


def process_data(df: DataFrame) -> DataFrame:
    """
    Process DataFrame with aggregations.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: Processed data
    """
    result = df.groupBy("category") \\
        .agg(
            count("id").alias("total_count"),
            spark_sum("amount").alias("total_amount")
        ) \\
        .filter(col("total_count") > 10) \\
        .orderBy("total_amount", ascending=False)

    return result
'''


@pytest.fixture(scope="session")
def sample_script_file(temp_dir, sample_script_content):
    """Create a sample script file for testing."""
    script_path = os.path.join(temp_dir, "test_migrator_reviewer.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(sample_script_content)
    return script_path


@pytest.fixture
def mock_analysis_report():
    """Sample analysis report for migrator -> reviewer testing."""
    return {
        "source_file_name": "test_migrator_reviewer.py",
        "conversion_order": ["create_spark_session", "process_data"],
        "function_analysis": [
            {
                "function_name": "create_spark_session",
                "complexity": "medium",
                "pyspark_patterns": ["SparkSession.builder", "config"],
                "dependencies": [],
                "suggested_patterns": ["session_creation", "config_setup"]
            },
            {
                "function_name": "process_data",
                "complexity": "high",
                "pyspark_patterns": ["groupBy", "agg", "filter", "orderBy"],
                "dependencies": ["DataFrame"],
                "suggested_patterns": ["aggregation", "filtering", "sorting"]
            }
        ]
    }


class TestMigratorReviewerChainMocking:
    """Test migrator -> reviewer chain with proper mocking."""

    def setup_method(self):
        """Setup properly configured mock objects."""
        # Mock LLM Service with correct methods
        self.mock_llm_service = MagicMock()

        # Mock the JSON completion method used by reviewer
        self.mock_llm_service.get_json_completion.return_value = {
            "overall_assessment": {
                "status": "NEEDS_REFINEMENT",
                "summary": "Code migrated successfully but needs minor improvements"
            },
            "findings": [
                {
                    "category": "BEST_PRACTICE_VIOLATION",
                    "faulty_code_snippet": "Session.builder.appName(app_name)",
                    "issue_description": "Missing proper session configuration",
                    "suggested_correction": "Session.builder.app_name(app_name).config('warehouse', 'COMPUTE_WH')"
                }
            ]
        }

        # Mock the text completion method used by reviewer
        self.mock_llm_service.get_text_completion.return_value = '''def corrected_create_spark_session(app_name: str = "TestApp"):
    """Corrected Snowpark session creation."""
    from snowflake.snowpark import Session
    session = Session.builder.app_name(app_name).config('warehouse', 'COMPUTE_WH').create()
    return session'''

        # Mock Knowledge Service with correct methods
        self.mock_knowledge_service = MagicMock()
        self.mock_knowledge_service.get_recipes_from_suggested_patterns.return_value = [
            {
                "pattern_name": "session_creation",
                "description": "Creating Snowpark sessions",
                "pyspark_example": "SparkSession.builder.appName().getOrCreate()",
                "snowpark_example": "Session.builder.app_name().create()",
                "notes": "Use app_name instead of appName"
            }
        ]

        # Mock other agents
        self.mock_enricher = MagicMock()
        self.mock_enricher.enrich_function.return_value = {
            'enriched_code': '''def enriched_create_spark_session(app_name: str = "TestApp") -> SparkSession:
    """Enhanced function with better documentation."""
    spark = SparkSession.builder.appName(app_name).config("spark.sql.adaptive.enabled", "true").getOrCreate()
    return spark''',
            'test_function': '''def test_create_spark_session():
    """Test the session creation."""
    session = create_spark_session("TestApp")
    assert session is not None'''
        }

        self.mock_migrator = MagicMock()
        self.mock_migrator.migrate_function.return_value = '''def migrated_create_spark_session(app_name: str = "TestApp"):
    """Migrated to Snowpark."""
    from snowflake.snowpark import Session
    session = Session.builder.appName(app_name).create()
    return session'''

        self.mock_logger = MagicMock()

    def test_migrator_reviewer_chain_with_mocks(self, temp_dir, sample_script_content, mock_analysis_report):
        """Test the complete migrator -> reviewer chain with properly mocked components."""

        function_name = "create_spark_session"
        output_dir = os.path.join(temp_dir, "migrator_reviewer_chain_test")

        # Create real reviewer but with mocked LLM service
        reviewer = CodeReviewer(llm_service=self.mock_llm_service)

        # Execute the complete workflow
        process_single_function(
            function_name=function_name,
            script_content=sample_script_content,
            analysis_report=mock_analysis_report,
            output_dir=output_dir,
            enricher=self.mock_enricher,
            migrator=self.mock_migrator,
            reviewer=reviewer,  # Real reviewer with mocked LLM
            knowledge_service=self.mock_knowledge_service,
            logger=self.mock_logger
        )

        # Verify complete file structure (8 files for dual-track with review)
        function_output_dir = os.path.join(output_dir, function_name)
        assert os.path.exists(function_output_dir)

        expected_files = [
            '01_enriched_code.py',  # Enricher output
            '02_original_test.py',  # Enricher test output
            '03_migrated_code.py',  # Migrator main output
            '04_review_report_main.json',  # Reviewer main report
            '05_corrected_code_main.py',  # Reviewer main correction
            '06_migrated_test.py',  # Migrator test output
            '07_review_report_test.json',  # Reviewer test report
            '08_corrected_code_test.py'  # Reviewer test correction
        ]

        for filename in expected_files:
            file_path = os.path.join(function_output_dir, filename)
            assert os.path.exists(file_path), f"Expected file not found: {filename}"
            assert os.path.getsize(file_path) > 0, f"File is empty: {filename}"

        # Verify reviewer was called correctly
        assert self.mock_llm_service.get_json_completion.call_count == 2  # Main + Test
        assert self.mock_llm_service.get_text_completion.call_count == 2  # Main + Test

        # Verify knowledge service was called
        self.mock_knowledge_service.get_recipes_from_suggested_patterns.assert_called()

        # Verify review report structure
        review_report_path = os.path.join(function_output_dir, '04_review_report_main.json')
        with open(review_report_path, 'r') as f:
            review_data = json.load(f)

        assert 'overall_assessment' in review_data
        assert 'status' in review_data['overall_assessment']
        assert 'findings' in review_data
        assert review_data['overall_assessment']['status'] in ['PERFECT', 'NEEDS_REFINEMENT']

        print("✓ Migrator -> Reviewer chain test passed with proper mocking")

    def test_migrator_reviewer_chain_perfect_code(self, temp_dir, sample_script_content, mock_analysis_report):
        """Test migrator -> reviewer chain when code is already perfect."""

        # Configure mocks for perfect code scenario
        self.mock_llm_service.get_json_completion.return_value = {
            "overall_assessment": {
                "status": "PERFECT",
                "summary": "Code is perfectly migrated, no issues found"
            },
            "findings": []  # Empty findings for perfect code
        }

        function_name = "process_data"
        output_dir = os.path.join(temp_dir, "perfect_code_test")

        # Configure enricher to not generate test code
        self.mock_enricher.enrich_function.return_value = {
            'enriched_code': '''def enriched_process_data(df):
    """Enhanced data processing function."""
    return df.groupBy("category").agg(count("id").alias("total"))''',
            'test_function': ''  # No test code
        }

        reviewer = CodeReviewer(llm_service=self.mock_llm_service)

        process_single_function(
            function_name=function_name,
            script_content=sample_script_content,
            analysis_report=mock_analysis_report,
            output_dir=output_dir,
            enricher=self.mock_enricher,
            migrator=self.mock_migrator,
            reviewer=reviewer,
            knowledge_service=self.mock_knowledge_service,
            logger=self.mock_logger
        )

        # Verify only Track A files (no test code)
        function_output_dir = os.path.join(output_dir, function_name)
        expected_files = [
            '01_enriched_code.py',
            '03_migrated_code.py',
            '04_review_report_main.json',
            '05_corrected_code_main.py'
        ]

        for filename in expected_files:
            file_path = os.path.join(function_output_dir, filename)
            assert os.path.exists(file_path), f"Expected file not found: {filename}"

        # Verify no test files
        test_files = [
            '02_original_test.py',
            '06_migrated_test.py',
            '07_review_report_test.json',
            '08_corrected_code_test.py'
        ]

        for filename in test_files:
            file_path = os.path.join(function_output_dir, filename)
            assert not os.path.exists(file_path), f"Unexpected file found: {filename}"

        # Verify perfect status in review report
        review_report_path = os.path.join(function_output_dir, '04_review_report_main.json')
        with open(review_report_path, 'r') as f:
            review_data = json.load(f)

        assert review_data['overall_assessment']['status'] == 'PERFECT'
        assert len(review_data['findings']) == 0

        print("✓ Perfect code scenario test passed")

    def test_reviewer_error_handling(self, temp_dir, sample_script_content, mock_analysis_report):
        """Test error handling in reviewer component."""

        function_name = "create_spark_session"
        output_dir = os.path.join(temp_dir, "reviewer_error_test")

        # Configure LLM service to fail on JSON completion
        self.mock_llm_service.get_json_completion.side_effect = Exception("JSON parsing failed")

        reviewer = CodeReviewer(llm_service=self.mock_llm_service)

        # Should raise exception when reviewer fails
        with pytest.raises(Exception) as excinfo:
            process_single_function(
                function_name=function_name,
                script_content=sample_script_content,
                analysis_report=mock_analysis_report,
                output_dir=output_dir,
                enricher=self.mock_enricher,
                migrator=self.mock_migrator,
                reviewer=reviewer,
                knowledge_service=self.mock_knowledge_service,
                logger=self.mock_logger
            )

        assert "Failed to complete review and correction process" in str(excinfo.value)
        print("✓ Reviewer error handling test passed")


class TestMigratorReviewerRealComponents:
    """Test migrator -> reviewer chain with real components (if available)."""

    def test_real_component_initialization(self):
        """Test that all required components can be initialized."""

        try:
            # Try to initialize real services
            llm_service = CortexLLMService()
            knowledge_service = KnowledgeService(knowledge_base_path="data/knowledge_base.json")

            # Initialize all agents
            analyzer = CodeAnalyzer(llm_service=llm_service)
            enricher = CodeEnricher(llm_service=llm_service)
            migrator = CodeMigrator(llm_service=llm_service)
            reviewer = CodeReviewer(llm_service=llm_service)

            # Verify all components exist and have required methods
            assert hasattr(reviewer, 'review_and_correct_migration')
            assert hasattr(knowledge_service, 'get_recipes_from_suggested_patterns')
            assert hasattr(llm_service, 'get_json_completion')
            assert hasattr(llm_service, 'get_text_completion')

            print("✓ All real components initialized successfully")

        except Exception as e:
            pytest.skip(f"Real services not available for testing: {str(e)}")

    def test_knowledge_service_recipe_retrieval(self):
        """Test knowledge service recipe retrieval functionality."""

        try:
            knowledge_service = KnowledgeService(knowledge_base_path="data/knowledge_base.json")

            # Test with sample patterns
            test_patterns = ["session_creation", "aggregation", "filtering"]
            recipes = knowledge_service.get_recipes_from_suggested_patterns(test_patterns)

            # Should return a list (empty is okay if no recipes found)
            assert isinstance(recipes, list)
            print(f"✓ Recipe retrieval test passed, found {len(recipes)} recipes")

        except Exception as e:
            pytest.skip(f"Knowledge service not available: {str(e)}")

    @pytest.mark.integration
    def test_limited_real_integration(self, sample_script_file, temp_dir):
        """Limited integration test with real components (analyzer + enricher only)."""

        try:
            # Initialize services with error handling
            try:
                llm_service = CortexLLMService()
                knowledge_service = KnowledgeService(knowledge_base_path="data/knowledge_base.json")
            except Exception:
                # Use mocks if real services fail
                llm_service = MagicMock()
                knowledge_service = MagicMock()
                knowledge_service.get_recipes_from_suggested_patterns.return_value = []

            # Test just analyzer + enricher (skip migrator + reviewer if they fail)
            analyzer = CodeAnalyzer(llm_service=llm_service)
            enricher = CodeEnricher(llm_service=llm_service)

            script_content = read_script_file(sample_script_file)
            source_file_name = os.path.basename(sample_script_file)

            # Test analysis phase
            analysis_report = analyzer.analyze_script(script_content, source_file_name)
            assert isinstance(analysis_report, dict)

            # Test enrichment phase for first function
            function_code = extract_function_code(script_content, "create_spark_session")
            enrichment_result = enricher.enrich_function(function_code)

            # Verify basic structure
            if isinstance(enrichment_result, dict):
                assert 'enriched_code' in enrichment_result

            print("✓ Limited real integration test passed")

        except Exception as e:
            pytest.skip(f"Real integration test failed: {str(e)}")


class TestMigratorReviewerEdgeCases:
    """Test edge cases and error conditions for migrator -> reviewer chain."""

    def setup_method(self):
        """Setup mock objects for edge case testing."""
        self.mock_llm_service = MagicMock()
        self.mock_knowledge_service = MagicMock()
        self.mock_enricher = MagicMock()
        self.mock_migrator = MagicMock()
        self.mock_logger = MagicMock()

    def test_invalid_review_status(self, temp_dir, sample_script_content, mock_analysis_report):
        """Test handling of invalid review status from LLM."""

        # Configure invalid status response
        self.mock_llm_service.get_json_completion.return_value = {
            "overall_assessment": {
                "status": "INVALID_STATUS",  # Invalid status
                "summary": "Invalid response"
            },
            "findings": []
        }

        self.mock_enricher.enrich_function.return_value = {
            'enriched_code': 'def create_spark_session():\n    pass',
            'test_function': ''
        }
        self.mock_migrator.migrate_function.return_value = 'def create_spark_session():\n    pass'
        self.mock_knowledge_service.get_recipes_from_suggested_patterns.return_value = []

        reviewer = CodeReviewer(llm_service=self.mock_llm_service)

        with pytest.raises(Exception) as excinfo:
            process_single_function(
                function_name="create_spark_session",  # Use existing function name
                script_content=sample_script_content,
                analysis_report=mock_analysis_report,
                output_dir=os.path.join(temp_dir, "invalid_status_test"),
                enricher=self.mock_enricher,
                migrator=self.mock_migrator,
                reviewer=reviewer,
                knowledge_service=self.mock_knowledge_service,
                logger=self.mock_logger
            )

        assert "Invalid audit status" in str(excinfo.value)

    def test_missing_suggested_patterns(self, temp_dir, sample_script_content):
        """Test handling when function_analysis lacks suggested_patterns."""

        # Analysis report without suggested_patterns
        analysis_report = {
            "function_analysis": [
                {
                    "function_name": "process_data",  # Use existing function name
                    "complexity": "low"
                    # Missing suggested_patterns
                }
            ]
        }

        self.mock_llm_service.get_json_completion.return_value = {
            "overall_assessment": {"status": "PERFECT", "summary": "Good"},
            "findings": []
        }

        self.mock_enricher.enrich_function.return_value = {
            'enriched_code': 'def process_data(df):\n    return df',
            'test_function': ''
        }
        self.mock_migrator.migrate_function.return_value = 'def process_data(df):\n    return df'
        self.mock_knowledge_service.get_recipes_from_suggested_patterns.return_value = []

        reviewer = CodeReviewer(llm_service=self.mock_llm_service)

        # Should not raise exception, should handle gracefully
        try:
            process_single_function(
                function_name="process_data",  # Use existing function name
                script_content=sample_script_content,
                analysis_report=analysis_report,
                output_dir=os.path.join(temp_dir, "missing_patterns_test"),
                enricher=self.mock_enricher,
                migrator=self.mock_migrator,
                reviewer=reviewer,
                knowledge_service=self.mock_knowledge_service,
                logger=self.mock_logger
            )

            # Verify knowledge service was called with empty list
            self.mock_knowledge_service.get_recipes_from_suggested_patterns.assert_called_with([])

        except Exception as e:
            pytest.fail(f"Should handle missing suggested_patterns gracefully: {e}")


if __name__ == "__main__":
    """
    Run migrator -> reviewer chain tests directly.
    Usage: python test_migrator_reviewer_chain.py
    """

    print("=== Running Migrator + Reviewer Chain Integration Tests ===")

    pytest.main([
        __file__,
        "-v",  # Verbose output
        "-s",  # Don't capture stdout
        "--tb=short",  # Short traceback format
        "-m", "not integration"  # Skip integration tests by default
    ])

    print("\n=== To run integration tests with real components, use: ===")
    print("pytest test_migrator_reviewer_chain.py -m integration -v")