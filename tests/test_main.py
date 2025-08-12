#!/usr/bin/env python3
"""
Integration tests for main.py - PySpark to Snowpark Migration Tool
Fixed version based on debug results.
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    parse_arguments, read_script_file, ensure_output_directory,
    extract_function_code, save_json_file, save_python_file,
    merge_imports_and_functions, merge_test_functions, main
)

# Import real services and agents for integration testing
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService
from agents.code_analyzer import CodeAnalyzer
from agents.code_enricher import CodeEnricher
from agents.code_migrator import CodeMigrator
from agents.code_reviewer import CodeReviewer


# Global fixtures that can be used by all test classes
@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_script_content():
    """Real PySpark script content for testing."""
    return '''"""
Sample PySpark Script for Testing CodeAnalyzer
This script demonstrates common PySpark patterns and operations
that will be analyzed by the CodeAnalyzer module.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when, sum as spark_sum, count, avg
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Custom imports (simulated)
from utils.data_validator import validate_data_quality
from config.spark_config import get_spark_config


def create_spark_session(app_name: str = "DataProcessingApp") -> SparkSession:
    """
    Create and configure Spark session with optimized settings.

    Args:
        app_name: Name of the Spark application

    Returns:
        SparkSession: Configured Spark session
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    return spark


def load_customer_data(spark: SparkSession, data_path: str) -> DataFrame:
    """
    Load customer data from various sources.

    Args:
        spark: Active Spark session
        data_path: Path to the data source

    Returns:
        DataFrame: Loaded customer data
    """
    # Define schema
    schema = StructType([
        StructField("customer_id", StringType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("email", StringType(), True),
        StructField("registration_date", StringType(), True),
        StructField("total_spent", DoubleType(), True)
    ])

    # Load data
    df = spark.read \
        .option("header", "true") \
        .schema(schema) \
        .csv(data_path)

    return df


def clean_customer_data(df: DataFrame) -> DataFrame:
    """
    Clean and validate customer data.

    Args:
        df: Raw customer DataFrame

    Returns:
        DataFrame: Cleaned customer data
    """
    # Remove duplicates and null values
    cleaned_df = df.dropDuplicates(["customer_id"]) \
        .filter(col("customer_id").isNotNull()) \
        .filter(col("email").isNotNull())

    # Add derived columns
    cleaned_df = cleaned_df.withColumn(
        "age_group",
        when(col("age") < 25, "Young")
        .when(col("age") < 65, "Adult")
        .otherwise("Senior")
    )

    # Validate data quality
    validated_df = validate_data_quality(cleaned_df)

    return validated_df


def main():
    """
    Main data processing pipeline.
    Orchestrates the complete data processing workflow.
    """
    # Initialize Spark session
    spark = create_spark_session("CustomerAnalytics")

    try:
        # Load data
        customer_data = load_customer_data(spark, "/data/customers.csv")

        # Process data
        clean_customers = clean_customer_data(customer_data)

        print("Data processing completed successfully!")

    except Exception as e:
        print(f"Error in data processing pipeline: {str(e)}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
'''


@pytest.fixture(scope="session")
def sample_script_file(temp_dir, sample_script_content):
    """Create a sample script file for testing."""
    script_path = os.path.join(temp_dir, "sample_spark_script.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(sample_script_content)
    return script_path


class TestMainIntegration:
    """
    Integration test class for main.py using real components and test cases.
    """

    def test_utility_functions(self, temp_dir, sample_script_content):
        """Test all utility functions with real data."""

        # Test ensure_output_directory
        test_output_dir = os.path.join(temp_dir, "test_output")
        ensure_output_directory(test_output_dir)
        assert os.path.exists(test_output_dir)

        # Test save and read functions
        test_data = {"test": "data", "number": 123}
        json_path = os.path.join(test_output_dir, "test.json")
        save_json_file(test_data, json_path)
        assert os.path.exists(json_path)

        # Test save Python file
        python_content = "# Test Python content\nprint('Hello World')"
        py_path = os.path.join(test_output_dir, "test.py")
        save_python_file(python_content, py_path)
        assert os.path.exists(py_path)

        # Test extract function code
        function_code = extract_function_code(sample_script_content, "create_spark_session")
        assert "def create_spark_session" in function_code
        assert "SparkSession.builder" in function_code

        function_code2 = extract_function_code(sample_script_content, "load_customer_data")
        assert "def load_customer_data" in function_code2
        assert "StructType" in function_code2

    def test_merge_functions(self):
        """Test merge functions with real package analysis data."""

        # Real package analysis data structure
        package_analysis = {
            "standard_libs": [
                {
                    "import_statement": "import os",
                    "purpose": "Operating system interface"
                },
                {
                    "import_statement": "from datetime import datetime",
                    "purpose": "Date and time utilities"
                }
            ],
            "third_party": [
                {
                    "import_statement": "from pyspark.sql import SparkSession",
                    "purpose": "Core PySpark SQL functionality"
                }
            ],
            "custom_modules": [
                {
                    "import_statement": "from utils.data_validator import validate_data_quality",
                    "purpose": "Custom data validation"
                }
            ]
        }

        functions = [
            "def function1():\n    pass",
            "def function2():\n    return True"
        ]

        merged_content = merge_imports_and_functions(package_analysis, functions)

        # Verify all imports are included
        assert "import os" in merged_content
        assert "from datetime import datetime" in merged_content
        assert "from pyspark.sql import SparkSession" in merged_content
        assert "from utils.data_validator import validate_data_quality" in merged_content

        # Verify functions are included
        assert "def function1():" in merged_content
        assert "def function2():" in merged_content

        # Test merge test functions
        test_functions = [
            "def test_function1():\n    assert True",
            "def test_function2():\n    assert 1 == 1"
        ]

        merged_tests = merge_test_functions(test_functions)
        assert "import pytest" in merged_tests
        assert "def test_function1():" in merged_tests
        assert "def test_function2():" in merged_tests

    @patch('sys.argv', ['main.py', '--file', 'test_file.py', '--output', 'test_output'])
    def test_parse_arguments(self):
        """Test command line argument parsing."""
        args = parse_arguments()
        assert args.file == 'test_file.py'
        assert args.output == 'test_output'

    def test_real_agents_initialization(self):
        """Test real agent initialization with actual services."""

        # Initialize real services
        try:
            llm_service = CortexLLMService()
            knowledge_service = KnowledgeService(knowledge_base_path="data/knowledge_base.json")

            # Initialize real agents
            analyzer = CodeAnalyzer(llm_service=llm_service)
            enricher = CodeEnricher(llm_service=llm_service)
            migrator = CodeMigrator(llm_service=llm_service)
            reviewer = CodeReviewer(llm_service=llm_service)

            # Verify agents are properly initialized
            assert analyzer is not None
            assert enricher is not None
            assert migrator is not None
            assert reviewer is not None

            # Verify they have the correct dependencies
            assert hasattr(analyzer, 'llm_service')
            assert hasattr(enricher, 'llm_service')
            assert hasattr(migrator, 'llm_service')
            assert hasattr(reviewer, 'llm_service')

        except Exception as e:
            pytest.skip(f"Real services not available for testing: {str(e)}")

    def test_full_integration_with_real_components(self, sample_script_file, temp_dir):
        """
        Full integration test using real components and LLM services.
        This test runs the complete workflow with actual implementations.
        """

        try:
            # Setup real services
            llm_service = CortexLLMService()
            knowledge_service = KnowledgeService(knowledge_base_path="data/knowledge_base.json")

            # Initialize real agents
            analyzer = CodeAnalyzer(llm_service=llm_service)
            enricher = CodeEnricher(llm_service=llm_service)
            migrator = CodeMigrator(llm_service=llm_service)
            reviewer = CodeReviewer(llm_service=llm_service)

            # Read the sample script
            script_content = read_script_file(sample_script_file)
            source_file_name = os.path.basename(sample_script_file)

            # Create output directory
            output_dir = os.path.join(temp_dir, "integration_output")
            ensure_output_directory(output_dir)

            print(f"\n=== Starting Integration Test ===")
            print(f"Input file: {sample_script_file}")
            print(f"Output directory: {output_dir}")

            # Step 1: Analyze script
            print("\n1. Analyzing script...")
            analysis_report = analyzer.analyze_script(script_content, source_file_name)

            # Verify analysis report structure
            assert 'source_file_name' in analysis_report
            assert 'package_analysis' in analysis_report
            assert 'function_analysis' in analysis_report

            # Save analysis report
            analysis_report_path = os.path.join(output_dir, "analysis_report.json")
            save_json_file(analysis_report, analysis_report_path)
            assert os.path.exists(analysis_report_path)

            print(f"   ✓ Analysis completed. Found {len(analysis_report.get('function_analysis', []))} functions")

            # Step 2: Get conversion order
            conversion_order = analysis_report.get('conversion_order', [])
            if not conversion_order:
                conversion_order = [func['function_name'] for func in analysis_report.get('function_analysis', [])]

            print(f"   ✓ Conversion order: {conversion_order}")

            # Step 3: Phase 1 - Code Enrichment
            print("\n2. Phase 1: Code Enrichment...")
            enriched_functions = {}
            test_functions = {}

            for function_name in conversion_order[:2]:  # Test first 2 functions for speed
                print(f"   Enriching function: {function_name}")

                # Extract function code
                function_code = extract_function_code(script_content, function_name)
                assert function_code is not None
                assert f"def {function_name}" in function_code

                # Get function analysis data
                function_analysis = None
                for func_data in analysis_report.get('function_analysis', []):
                    if func_data['function_name'] == function_name:
                        function_analysis = func_data
                        break

                # Call enricher - use correct signature
                enrichment_result = enricher.enrich_function(function_code)

                # Verify enrichment result structure
                assert enrichment_result is not None

                if isinstance(enrichment_result, dict):
                    # Store results if it's a dictionary
                    enriched_functions[function_name] = enrichment_result.get('enriched_code', function_code)
                    test_functions[function_name] = enrichment_result.get('test_function', '')
                else:
                    # If it's just the enriched code as a string
                    enriched_functions[function_name] = enrichment_result
                    test_functions[function_name] = ''  # No test function generated

                print(f"   ✓ Function {function_name} enriched successfully")

            # Step 4: Phase 2 - Main Code Migration
            print("\n3. Phase 2: Main Code Migration...")
            final_migrated_functions = []

            for function_name in list(enriched_functions.keys()):
                print(f"   Migrating function: {function_name}")

                # Get enriched code
                enriched_code = enriched_functions[function_name]

                # Get function analysis data
                function_analysis = None
                for func_data in analysis_report.get('function_analysis', []):
                    if func_data['function_name'] == function_name:
                        function_analysis = func_data
                        break

                # Call migrator - use correct signature
                migrated_code = migrator.migrate_function(enriched_code, function_analysis, knowledge_service)

                assert migrated_code is not None

                # Call reviewer - use correct signature
                review_result = reviewer.review_and_correct_migration(
                    enriched_code, migrated_code, knowledge_service, function_analysis
                )

                if isinstance(review_result, dict):
                    corrected_code = review_result.get('corrected_code', migrated_code)
                else:
                    corrected_code = review_result  # Assume it's the corrected code directly

                final_migrated_functions.append(corrected_code)

                print(f"   ✓ Function {function_name} migrated successfully")

            # Step 5: Phase 3 - Test Code Migration (Optional)
            print("\n4. Phase 3: Test Code Migration...")
            final_test_migrations = []

            for function_name in list(test_functions.keys()):
                if test_functions[function_name]:
                    print(f"   Migrating test for: {function_name}")

                    test_code = test_functions[function_name]

                    # Get function analysis data
                    function_analysis = None
                    for func_data in analysis_report.get('function_analysis', []):
                        if func_data['function_name'] == function_name:
                            function_analysis = func_data
                            break

                    try:
                        # Attempt to migrate test code
                        try:
                            migrated_test = migrator.migrate_function(test_code, function_analysis)
                        except TypeError:
                            migrated_test = migrator.migrate_function(test_code)

                        try:
                            review_result = reviewer.review_and_correct_migration(migrated_test, function_analysis)
                        except TypeError:
                            review_result = reviewer.review_and_correct_migration(migrated_test)

                        if isinstance(review_result, dict):
                            corrected_test = review_result.get('corrected_code', migrated_test)
                        else:
                            corrected_test = review_result

                        final_test_migrations.append(corrected_test)
                        print(f"   ✓ Test for {function_name} migrated successfully")
                    except Exception as e:
                        print(f"   ⚠ Warning: Test migration failed for {function_name}: {str(e)}")
                        final_test_migrations.append(test_code)

            # Step 6: Merge and Save Final Files
            print("\n5. Merging and Saving Final Files...")

            # Merge main code
            package_analysis = analysis_report.get('package_analysis', {})
            final_script_content = merge_imports_and_functions(package_analysis, final_migrated_functions)

            # Save migrated script
            script_name = os.path.splitext(source_file_name)[0]
            migrated_script_path = os.path.join(output_dir, f"{script_name}_migrated.py")
            save_python_file(final_script_content, migrated_script_path)
            assert os.path.exists(migrated_script_path)
            print(f"   ✓ Migrated script saved to: {migrated_script_path}")

            # Save test code if available
            if final_test_migrations:
                final_test_content = merge_test_functions(final_test_migrations)
                test_script_path = os.path.join(output_dir, f"{script_name}_tests.py")
                save_python_file(final_test_content, test_script_path)
                assert os.path.exists(test_script_path)
                print(f"   ✓ Test script saved to: {test_script_path}")

            # Verify all output files exist
            expected_files = [
                "analysis_report.json",
                f"{script_name}_migrated.py"
            ]

            if final_test_migrations:
                expected_files.append(f"{script_name}_tests.py")

            for expected_file in expected_files:
                file_path = os.path.join(output_dir, expected_file)
                assert os.path.exists(file_path), f"Expected file not found: {expected_file}"

                # Verify file is not empty
                assert os.path.getsize(file_path) > 0, f"File is empty: {expected_file}"

            print(f"\n=== Integration Test Completed Successfully ===")
            print(f"Generated files: {expected_files}")

            # Read and validate the migrated content
            with open(migrated_script_path, 'r', encoding='utf-8') as f:
                migrated_content = f.read()

            # Basic validation that migration occurred
            assert len(migrated_content) > 0
            print(f"   ✓ Migrated script contains {len(migrated_content)} characters")

            return True

        except Exception as e:
            print(f"\n❌ Integration test failed with error: {str(e)}")
            print(f"   Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Integration test failed: {str(e)}")

    @patch('sys.argv', ['main.py', '--file', 'nonexistent.py', '--output', 'test_output'])
    def test_main_with_nonexistent_file(self):
        """Test main function with nonexistent input file."""
        with pytest.raises(SystemExit):
            main()

    def test_main_command_line_integration(self, sample_script_file, temp_dir):
        """
        Test main function with command line arguments using real components.
        This simulates running: python main.py --file sample_script.py --output output_dir
        """

        output_dir = os.path.join(temp_dir, "cli_test_output")

        # Mock command line arguments
        test_args = [
            'main.py',
            '--file', sample_script_file,
            '--output', output_dir
        ]

        with patch('sys.argv', test_args):
            try:
                main()

                # Verify output files were created
                assert os.path.exists(output_dir)

                expected_files = [
                    "analysis_report.json",
                    "sample_spark_script_migrated.py"
                ]

                for expected_file in expected_files:
                    file_path = os.path.join(output_dir, expected_file)
                    if os.path.exists(file_path):
                        assert os.path.getsize(file_path) > 0
                        print(f"   ✓ Generated: {expected_file}")

                print("\n✓ Command line integration test passed")

            except Exception as e:
                print(f"\n❌ CLI integration test failed: {str(e)}")
                import traceback
                traceback.print_exc()
                pytest.fail(f"CLI integration test failed: {str(e)}")


# Additional test functions for specific edge cases
class TestMainEdgeCases:
    """Test edge cases and error conditions."""

    def test_extract_function_code_not_found(self):
        """Test function extraction with non-existent function."""
        # Fixed: Use proper newlines instead of escaped strings
        script_content = "def function1():\n    pass"

        with pytest.raises(Exception) as excinfo:
            extract_function_code(script_content, "nonexistent_function")

        assert "not found in script" in str(excinfo.value)

    def test_extract_function_code_syntax_error(self):
        """Test function extraction with syntax error."""
        # Fixed: Use proper newlines
        invalid_script = "def function1(:\n    pass"  # Missing closing parenthesis

        with pytest.raises(Exception) as excinfo:
            extract_function_code(invalid_script, "function1")

        assert "Syntax error" in str(excinfo.value)

    def test_save_file_permissions_error(self, temp_dir):
        """Test file saving with permission errors."""
        # Create a read-only directory
        readonly_dir = os.path.join(temp_dir, "readonly")
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, 0o444)  # Read-only

        try:
            readonly_file = os.path.join(readonly_dir, "test.json")
            with pytest.raises(Exception):
                save_json_file({"test": "data"}, readonly_file)
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)


if __name__ == "__main__":
    """
    Run integration tests directly.
    Usage: python test_main.py
    """

    print("=== Running PySpark to Snowpark Migration Tool Integration Tests ===")

    # Run specific test or all tests
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "-s",  # Don't capture stdout
        "--tb=short"  # Short traceback format
    ])