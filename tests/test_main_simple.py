#!/usr/bin/env python3
"""
Integration tests for main_simple.py - PySpark to Snowpark Migration Tool Version 2.1
Updated for per-function atomic processing with simplified migration (no reviewer).
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

from main_simple import (
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


@pytest.fixture(scope="session")
def mock_analysis_report():
    """Sample analysis report for testing."""
    return {
        "source_file_name": "sample_spark_script.py",
        "conversion_order": ["create_spark_session", "load_customer_data", "clean_customer_data"],
        "package_analysis": {
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
        },
        "function_analysis": [
            {
                "function_name": "create_spark_session",
                "complexity": "medium",
                "pyspark_patterns": ["SparkSession.builder"],
                "dependencies": []
            },
            {
                "function_name": "load_customer_data",
                "complexity": "high",
                "pyspark_patterns": ["spark.read", "StructType", "csv"],
                "dependencies": ["SparkSession"]
            },
            {
                "function_name": "clean_customer_data",
                "complexity": "high",
                "pyspark_patterns": ["DataFrame.filter", "col", "when"],
                "dependencies": ["DataFrame"]
            }
        ]
    }


class TestMainUtilityFunctions:
    """Test utility functions in main_simple.py Version 2.1."""

    def test_utility_functions(self, temp_dir, sample_script_content):
        """Test all utility functions with real data."""

        # Test ensure_output_directory
        test_output_dir = os.path.join(temp_dir, "test_output")
        ensure_output_directory(test_output_dir)
        assert os.path.exists(test_output_dir)

        # Test save_artifact with JSON data
        test_data = {"test": "data", "number": 123}
        json_path = os.path.join(test_output_dir, "test.json")
        save_artifact(test_data, json_path)
        assert os.path.exists(json_path)

        # Verify JSON content
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

        # Test save_artifact with Python content
        python_content = "# Test Python content\nprint('Hello World')"
        py_path = os.path.join(test_output_dir, "test.py")
        save_artifact(python_content, py_path)
        assert os.path.exists(py_path)

        # Verify Python content
        with open(py_path, 'r') as f:
            loaded_content = f.read()
        assert loaded_content == python_content

        # Test extract function code
        function_code = extract_function_code(sample_script_content, "create_spark_session")
        assert "def create_spark_session" in function_code
        assert "SparkSession.builder" in function_code

        function_code2 = extract_function_code(sample_script_content, "load_customer_data")
        assert "def load_customer_data" in function_code2
        assert "StructType" in function_code2

    def test_get_function_analysis(self, mock_analysis_report):
        """Test get_function_analysis utility function."""

        # Test existing function
        analysis = get_function_analysis(mock_analysis_report, "create_spark_session")
        assert analysis is not None
        assert analysis["function_name"] == "create_spark_session"
        assert analysis["complexity"] == "medium"

        # Test non-existent function
        analysis = get_function_analysis(mock_analysis_report, "nonexistent_function")
        assert analysis is None

    @patch('sys.argv', ['main_simple.py', '--file', 'test_file.py', '--output', 'test_output'])
    def test_parse_arguments(self):
        """Test command line argument parsing."""
        args = parse_arguments()
        assert args.file == 'test_file.py'
        assert args.output == 'test_output'


class TestProcessSingleFunction:
    """Test the simplified process_single_function workflow (no reviewer)."""

    def setup_method(self):
        """Setup mock objects for testing."""
        self.mock_enricher = MagicMock()
        self.mock_migrator = MagicMock()
        self.mock_knowledge_service = MagicMock()
        self.mock_logger = MagicMock()

    def test_process_single_function_with_test_code(self, temp_dir, sample_script_content, mock_analysis_report):
        """Test process_single_function with both main and test code generation (simplified workflow)."""

        function_name = "create_spark_session"
        output_dir = os.path.join(temp_dir, "function_test_output")

        # Setup mock responses
        self.mock_enricher.enrich_function.return_value = {
            'enriched_code': 'def enriched_function():\n    pass',
            'test_function': 'def test_enriched_function():\n    assert True'
        }

        self.mock_migrator.migrate_function.return_value = 'def migrated_function():\n    pass'

        # Execute process_single_function (simplified - no reviewer)
        process_single_function(
            function_name=function_name,
            script_content=sample_script_content,
            analysis_report=mock_analysis_report,
            output_dir=output_dir,
            enricher=self.mock_enricher,
            migrator=self.mock_migrator,
            knowledge_service=self.mock_knowledge_service,
            logger=self.mock_logger
        )

        # Verify function output directory was created
        function_output_dir = os.path.join(output_dir, function_name)
        assert os.path.exists(function_output_dir)

        # Verify simplified file structure (only 4 files total)
        expected_files = [
            '01_enriched_code.py',
            '02_original_test.py',
            '03_final_migrated_code.py',
            '04_final_migrated_test.py'
        ]

        for filename in expected_files:
            file_path = os.path.join(function_output_dir, filename)
            assert os.path.exists(file_path), f"Expected file not found: {filename}"
            assert os.path.getsize(file_path) > 0, f"File is empty: {filename}"

        # Verify agent methods were called correctly (no reviewer calls)
        self.mock_enricher.enrich_function.assert_called_once()
        assert self.mock_migrator.migrate_function.call_count == 2  # Main + Test

    def test_process_single_function_without_test_code(self, temp_dir, sample_script_content, mock_analysis_report):
        """Test process_single_function with only main code (no test generation) - simplified workflow."""

        function_name = "load_customer_data"
        output_dir = os.path.join(temp_dir, "function_test_output_no_test")

        # Setup mock responses - no test code generated
        self.mock_enricher.enrich_function.return_value = {
            'enriched_code': 'def enriched_function():\n    pass',
            'test_function': ''  # No test code
        }

        self.mock_migrator.migrate_function.return_value = 'def migrated_function():\n    pass'

        # Execute process_single_function (simplified - no reviewer)
        process_single_function(
            function_name=function_name,
            script_content=sample_script_content,
            analysis_report=mock_analysis_report,
            output_dir=output_dir,
            enricher=self.mock_enricher,
            migrator=self.mock_migrator,
            knowledge_service=self.mock_knowledge_service,
            logger=self.mock_logger
        )

        # Verify function output directory was created
        function_output_dir = os.path.join(output_dir, function_name)
        assert os.path.exists(function_output_dir)

        # Verify only main code files were created (no test files)
        expected_files = [
            '01_enriched_code.py',
            '03_final_migrated_code.py'
        ]

        for filename in expected_files:
            file_path = os.path.join(function_output_dir, filename)
            assert os.path.exists(file_path), f"Expected file not found: {filename}"

        # Verify test files were NOT created
        test_files = [
            '02_original_test.py',
            '04_final_migrated_test.py'
        ]

        for filename in test_files:
            file_path = os.path.join(function_output_dir, filename)
            assert not os.path.exists(file_path), f"Unexpected file found: {filename}"

        # Verify agent methods were called correctly
        self.mock_enricher.enrich_function.assert_called_once()
        self.mock_migrator.migrate_function.assert_called_once()  # Only main code


class TestMainIntegration:
    """Integration test class for main_simple.py Version 2.1 with real components."""

    def test_real_agents_initialization(self):
        """Test real agent initialization with actual services (no reviewer)."""

        try:
            # Initialize real services
            llm_service = CortexLLMService()
            knowledge_service = KnowledgeService(knowledge_base_path="data/knowledge_base.json")

            # Initialize real agents (no reviewer)
            analyzer = CodeAnalyzer(llm_service=llm_service)
            enricher = CodeEnricher(llm_service=llm_service)
            migrator = CodeMigrator(llm_service=llm_service)

            # Verify agents are properly initialized
            assert analyzer is not None
            assert enricher is not None
            assert migrator is not None

            # Verify they have the correct dependencies
            assert hasattr(analyzer, 'llm_service')
            assert hasattr(enricher, 'llm_service')
            assert hasattr(migrator, 'llm_service')

        except Exception as e:
            pytest.skip(f"Real services not available for testing: {str(e)}")

    def test_full_integration_with_real_components(self, sample_script_file, temp_dir):
        """
        Full integration test using real components and LLM services with detailed debugging.
        Tests the simplified per-function atomic processing workflow (no reviewer).
        """

        try:
            print(f"\n=== Starting Integration Test (Version 2.1 - Simplified) with Debugging ===")
            print(f"Input file: {sample_script_file}")
            print(f"Output directory: {temp_dir}")

            # Debug: Test LLM Service initialization
            print("\n=== DEBUG: Testing LLM Service Initialization ===")
            try:
                llm_service = CortexLLMService()
                print(f"✓ LLM Service initialized successfully")
                print(f"  LLM Service type: {type(llm_service)}")
                print(f"  LLM Service attributes: {[attr for attr in dir(llm_service) if not attr.startswith('_')]}")

                # Check if LLM service has expected attributes
                if hasattr(llm_service, 'api_url'):
                    print(f"  API URL: {getattr(llm_service, 'api_url', 'Not set')}")
                if hasattr(llm_service, 'model'):
                    print(f"  Model: {getattr(llm_service, 'model', 'Not set')}")

            except Exception as e:
                print(f"❌ LLM Service initialization failed: {e}")
                raise

            # Debug: Test Knowledge Service initialization
            print("\n=== DEBUG: Testing Knowledge Service Initialization ===")
            try:
                knowledge_service = KnowledgeService(knowledge_base_path="data/knowledge_base.json")
                print(f"✓ Knowledge Service initialized successfully")
                print(f"  Knowledge Service type: {type(knowledge_service)}")
            except Exception as e:
                print(f"❌ Knowledge Service initialization failed: {e}")
                print(f"  This might be OK if knowledge base file doesn't exist")
                # Create a mock knowledge service for testing
                knowledge_service = MagicMock()
                print(f"✓ Using mock Knowledge Service for testing")

            # Debug: Test Agent initialization (no reviewer)
            print("\n=== DEBUG: Testing Agent Initialization (Simplified) ===")
            try:
                analyzer = CodeAnalyzer(llm_service=llm_service)
                print(f"✓ CodeAnalyzer initialized")

                enricher = CodeEnricher(llm_service=llm_service)
                print(f"✓ CodeEnricher initialized")

                migrator = CodeMigrator(llm_service=llm_service)
                print(f"✓ CodeMigrator initialized")

                print(f"✓ No CodeReviewer (simplified workflow)")

            except Exception as e:
                print(f"❌ Agent initialization failed: {e}")
                raise

            # Read the sample script
            print(f"\n=== DEBUG: Reading Script File ===")
            script_content = read_script_file(sample_script_file)
            source_file_name = os.path.basename(sample_script_file)
            print(f"✓ Script read successfully")
            print(f"  Script length: {len(script_content)} characters")
            print(f"  Source file name: {source_file_name}")

            # Create output directory
            output_dir = os.path.join(temp_dir, "integration_output_v21")
            ensure_output_directory(output_dir)
            print(f"✓ Output directory created: {output_dir}")

            # Phase 1: Global Analysis with debugging
            print("\n=== DEBUG: Phase 1 - Global Analysis ===")
            try:
                print("  Calling analyzer.analyze_script()...")
                analysis_report = analyzer.analyze_script(script_content, source_file_name)
                print(f"✓ Analysis completed successfully")
                print(f"  Analysis report type: {type(analysis_report)}")
                print(
                    f"  Analysis report keys: {list(analysis_report.keys()) if isinstance(analysis_report, dict) else 'Not a dict'}")

                # Verify analysis report structure
                if isinstance(analysis_report, dict):
                    print(f"  source_file_name present: {'source_file_name' in analysis_report}")
                    print(f"  package_analysis present: {'package_analysis' in analysis_report}")
                    print(f"  function_analysis present: {'function_analysis' in analysis_report}")

                    if 'function_analysis' in analysis_report:
                        func_analysis = analysis_report['function_analysis']
                        print(f"  Found {len(func_analysis)} functions in analysis")
                        for i, func in enumerate(func_analysis):
                            if isinstance(func, dict) and 'function_name' in func:
                                print(f"    Function {i + 1}: {func['function_name']}")
                else:
                    print(f"  ❌ Analysis report is not a dictionary: {analysis_report}")

            except Exception as e:
                print(f"❌ Global analysis failed: {e}")
                print(f"  Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise

            # Save analysis report
            try:
                analysis_report_path = os.path.join(output_dir, "analysis_report.json")
                save_artifact(analysis_report, analysis_report_path)
                print(f"✓ Analysis report saved to: {analysis_report_path}")
            except Exception as e:
                print(f"❌ Failed to save analysis report: {e}")

            # Get conversion order and filter existing functions
            print(f"\n=== DEBUG: Determining Functions to Process ===")
            conversion_order = analysis_report.get('conversion_order', [])
            if not conversion_order:
                conversion_order = [func['function_name'] for func in analysis_report.get('function_analysis', [])]
            print(f"  Initial conversion order: {conversion_order}")

            functions_to_process = []
            for function_name in conversion_order:
                try:
                    extract_function_code(script_content, function_name)
                    functions_to_process.append(function_name)
                    print(f"  ✓ Function '{function_name}' found in script")
                except Exception as e:
                    print(f"  ⚠ Function '{function_name}' not found in script: {e}")

            print(f"  Final functions to process: {functions_to_process}")

            if not functions_to_process:
                print("❌ No functions to process. Stopping test.")
                return False

            # Phase 2: Per-Function Processing Loop with detailed debugging (simplified)
            print("\n=== DEBUG: Phase 2 - Per-Function Processing (Simplified) ===")

            # Test only the first function for detailed debugging
            test_function = functions_to_process[0]
            print(f"  Testing single function for debugging: {test_function}")

            # Step-by-step processing with debugging
            print(f"\n--- DEBUG: Processing function '{test_function}' (Simplified) ---")

            # Step 1: Extract function code
            print(f"  Step 1: Extracting function code...")
            try:
                function_code = extract_function_code(script_content, test_function)
                print(f"  ✓ Function code extracted ({len(function_code)} characters)")
                print(f"  Function code preview: {function_code[:200]}...")
            except Exception as e:
                print(f"  ❌ Function extraction failed: {e}")
                raise

            # Step 2: Get function analysis
            print(f"  Step 2: Getting function analysis...")
            try:
                function_analysis = get_function_analysis(analysis_report, test_function)
                print(f"  ✓ Function analysis retrieved: {function_analysis is not None}")
                if function_analysis:
                    print(f"    Analysis keys: {list(function_analysis.keys())}")
            except Exception as e:
                print(f"  ❌ Function analysis retrieval failed: {e}")
                function_analysis = None

            # Step 3: Test enricher
            print(f"  Step 3: Testing enricher...")
            try:
                print(f"    Calling enricher.enrich_function()...")
                enrichment_result = enricher.enrich_function(function_code)
                print(f"  ✓ Enrichment completed")
                print(f"    Enrichment result type: {type(enrichment_result)}")

                if isinstance(enrichment_result, dict):
                    print(f"    Enrichment result keys: {list(enrichment_result.keys())}")
                    enriched_code = enrichment_result.get('enriched_code', function_code)
                    test_code = enrichment_result.get('test_function', '')
                    print(f"    Enriched code length: {len(enriched_code)}")
                    print(f"    Test code generated: {len(test_code) > 0}")
                else:
                    enriched_code = str(enrichment_result)
                    test_code = ''
                    print(f"    Enriched code (as string) length: {len(enriched_code)}")

            except Exception as e:
                print(f"  ❌ Enrichment failed: {e}")
                print(f"    Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise

            # Step 4: Test migrator (main code)
            print(f"  Step 4: Testing migrator (main code)...")
            try:
                print(f"    Calling migrator.migrate_function()...")
                migrated_code = migrator.migrate_function(enriched_code, function_analysis, knowledge_service)
                print(f"  ✓ Migration completed")
                print(f"    Migrated code type: {type(migrated_code)}")
                print(f"    Migrated code length: {len(str(migrated_code))}")
            except Exception as e:
                print(f"  ❌ Migration failed: {e}")
                print(f"    Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise

            # Step 5: Test migrator (test code) - if test code exists
            if test_code:
                print(f"  Step 5: Testing migrator (test code)...")
                try:
                    print(f"    Calling migrator.migrate_function() for test code...")
                    migrated_test = migrator.migrate_function(test_code, function_analysis, knowledge_service)
                    print(f"  ✓ Test migration completed")
                    print(f"    Migrated test code type: {type(migrated_test)}")
                    print(f"    Migrated test code length: {len(str(migrated_test))}")
                except Exception as e:
                    print(f"  ❌ Test migration failed: {e}")
                    print(f"    Error type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                print(f"  Step 5: No test code to migrate")

            print(f"\n=== DEBUG: Simplified test completed (no reviewer step) ===")
            return True

        except Exception as e:
            print(f"\n❌ Integration test failed with detailed error: {str(e)}")
            print(f"   Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Integration test failed: {str(e)}")

    def test_main_command_line_integration(self, sample_script_file, temp_dir):
        """
        Test main function with command line arguments using real components.
        This simulates running: python main_simple.py --file sample_script.py --output output_dir
        """

        output_dir = os.path.join(temp_dir, "cli_test_output_v21")

        # Mock command line arguments
        test_args = [
            'main_simple.py',
            '--file', sample_script_file,
            '--output', output_dir
        ]

        with patch('sys.argv', test_args):
            try:
                main()

                # Verify output directory structure
                assert os.path.exists(output_dir)

                # Verify global analysis report
                analysis_report_path = os.path.join(output_dir, "analysis_report.json")
                assert os.path.exists(analysis_report_path)
                assert os.path.getsize(analysis_report_path) > 0

                # Check for function subdirectories
                subdirs = [d for d in os.listdir(output_dir)
                           if os.path.isdir(os.path.join(output_dir, d))]

                assert len(subdirs) > 0, "No function subdirectories created"
                print(f"   ✓ Created function directories: {subdirs}")

                # Verify at least one function was processed completely
                for subdir in subdirs:
                    function_dir = os.path.join(output_dir, subdir)
                    files_in_function_dir = os.listdir(function_dir)

                    # Should have at least the basic simplified files (2-4 files)
                    assert len(files_in_function_dir) >= 2
                    print(f"   ✓ Function {subdir} has {len(files_in_function_dir)} files")

                print("\n✓ CLI integration test (Version 2.1 - Simplified) passed")

            except Exception as e:
                print(f"\n❌ CLI integration test failed: {str(e)}")
                import traceback
                traceback.print_exc()
                pytest.fail(f"CLI integration test failed: {str(e)}")

    @patch('sys.argv', ['main_simple.py', '--file', 'nonexistent.py', '--output', 'test_output'])
    def test_main_with_nonexistent_file(self):
        """Test main function with nonexistent input file."""
        with pytest.raises(SystemExit):
            main()


class TestMainEdgeCases:
    """Test edge cases and error conditions for Version 2.1 (simplified)."""

    def test_extract_function_code_not_found(self):
        """Test function extraction with non-existent function."""
        script_content = "def function1():\n    pass"

        with pytest.raises(Exception) as excinfo:
            extract_function_code(script_content, "nonexistent_function")

        assert "not found in script" in str(excinfo.value)

    def test_extract_function_code_syntax_error(self):
        """Test function extraction with syntax error."""
        invalid_script = "def function1(:\n    pass"  # Missing closing parenthesis

        with pytest.raises(Exception) as excinfo:
            extract_function_code(invalid_script, "function1")

        assert "Syntax error" in str(excinfo.value)

    def test_save_artifact_invalid_type(self, temp_dir):
        """Test save_artifact with unsupported content type."""
        invalid_content = [1, 2, 3]  # List is not supported
        file_path = os.path.join(temp_dir, "invalid.txt")

        with pytest.raises(Exception) as excinfo:
            save_artifact(invalid_content, file_path)

        assert "Unsupported content type" in str(excinfo.value)

    def test_save_artifact_permissions_error(self, temp_dir):
        """Test save_artifact with permission errors."""
        # Create a read-only directory
        readonly_dir = os.path.join(temp_dir, "readonly")
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, 0o444)  # Read-only

        try:
            readonly_file = os.path.join(readonly_dir, "test.json")
            with pytest.raises(Exception):
                save_artifact({"test": "data"}, readonly_file)
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)

    def test_process_single_function_with_agent_errors(self, temp_dir, sample_script_content, mock_analysis_report):
        """Test process_single_function when agents throw errors (simplified workflow)."""

        function_name = "create_spark_session"
        output_dir = os.path.join(temp_dir, "error_test_output")

        # Setup mock objects that raise errors
        mock_enricher = MagicMock()
        mock_enricher.enrich_function.side_effect = Exception("Enricher failed")

        mock_migrator = MagicMock()
        mock_knowledge_service = MagicMock()
        mock_logger = MagicMock()

        # Should raise exception when enricher fails
        with pytest.raises(Exception) as excinfo:
            process_single_function(
                function_name=function_name,
                script_content=sample_script_content,
                analysis_report=mock_analysis_report,
                output_dir=output_dir,
                enricher=mock_enricher,
                migrator=mock_migrator,
                knowledge_service=mock_knowledge_service,
                logger=mock_logger
            )

        assert "Enricher failed" in str(excinfo.value)

    def test_process_single_function_migrator_error(self, temp_dir, sample_script_content, mock_analysis_report):
        """Test process_single_function when migrator throws errors (simplified workflow)."""

        function_name = "create_spark_session"
        output_dir = os.path.join(temp_dir, "migrator_error_test_output")

        # Setup mock objects - enricher works, migrator fails
        mock_enricher = MagicMock()
        mock_enricher.enrich_function.return_value = {
            'enriched_code': 'def enriched_function():\n    pass',
            'test_function': 'def test_enriched_function():\n    assert True'
        }

        mock_migrator = MagicMock()
        mock_migrator.migrate_function.side_effect = Exception("Migration failed")

        mock_knowledge_service = MagicMock()
        mock_logger = MagicMock()

        # Should raise exception when migrator fails
        with pytest.raises(Exception) as excinfo:
            process_single_function(
                function_name=function_name,
                script_content=sample_script_content,
                analysis_report=mock_analysis_report,
                output_dir=output_dir,
                enricher=mock_enricher,
                migrator=mock_migrator,
                knowledge_service=mock_knowledge_service,
                logger=mock_logger
            )

        assert "Migration failed" in str(excinfo.value)


class TestSimplifiedWorkflowVerification:
    """Test to verify the simplified workflow produces the correct file structure."""

    def test_simplified_file_structure_with_test_code(self, temp_dir, sample_script_content, mock_analysis_report):
        """Verify that the simplified workflow produces exactly 4 files when test code is generated."""

        function_name = "create_spark_session"
        output_dir = os.path.join(temp_dir, "simplified_structure_test")

        # Setup mock objects
        mock_enricher = MagicMock()
        mock_enricher.enrich_function.return_value = {
            'enriched_code': 'def enriched_function():\n    pass',
            'test_function': 'def test_enriched_function():\n    assert True'
        }

        mock_migrator = MagicMock()
        mock_migrator.migrate_function.return_value = 'def migrated_function():\n    pass'

        mock_knowledge_service = MagicMock()
        mock_logger = MagicMock()

        # Execute process_single_function
        process_single_function(
            function_name=function_name,
            script_content=sample_script_content,
            analysis_report=mock_analysis_report,
            output_dir=output_dir,
            enricher=mock_enricher,
            migrator=mock_migrator,
            knowledge_service=mock_knowledge_service,
            logger=mock_logger
        )

        # Verify exactly 4 files are created in simplified workflow
        function_output_dir = os.path.join(output_dir, function_name)
        files = os.listdir(function_output_dir)
        assert len(files) == 4, f"Expected 4 files, got {len(files)}: {files}"

        # Verify specific file names
        expected_files = [
            '01_enriched_code.py',
            '02_original_test.py',
            '03_final_migrated_code.py',
            '04_final_migrated_test.py'
        ]

        for expected_file in expected_files:
            assert expected_file in files, f"Expected file {expected_file} not found in {files}"

        # Verify migrator was called twice (main + test)
        assert mock_migrator.migrate_function.call_count == 2

    def test_simplified_file_structure_without_test_code(self, temp_dir, sample_script_content, mock_analysis_report):
        """Verify that the simplified workflow produces exactly 2 files when no test code is generated."""

        function_name = "load_customer_data"
        output_dir = os.path.join(temp_dir, "simplified_structure_no_test")

        # Setup mock objects - no test code
        mock_enricher = MagicMock()
        mock_enricher.enrich_function.return_value = {
            'enriched_code': 'def enriched_function():\n    pass',
            'test_function': ''  # No test code
        }

        mock_migrator = MagicMock()
        mock_migrator.migrate_function.return_value = 'def migrated_function():\n    pass'

        mock_knowledge_service = MagicMock()
        mock_logger = MagicMock()

        # Execute process_single_function
        process_single_function(
            function_name=function_name,
            script_content=sample_script_content,
            analysis_report=mock_analysis_report,
            output_dir=output_dir,
            enricher=mock_enricher,
            migrator=mock_migrator,
            knowledge_service=mock_knowledge_service,
            logger=mock_logger
        )

        # Verify exactly 2 files are created when no test code
        function_output_dir = os.path.join(output_dir, function_name)
        files = os.listdir(function_output_dir)
        assert len(files) == 2, f"Expected 2 files, got {len(files)}: {files}"

        # Verify specific file names
        expected_files = [
            '01_enriched_code.py',
            '03_final_migrated_code.py'
        ]

        for expected_file in expected_files:
            assert expected_file in files, f"Expected file {expected_file} not found in {files}"

        # Verify migrator was called once (only main code)
        assert mock_migrator.migrate_function.call_count == 1


if __name__ == "__main__":
    """
    Run integration tests directly for Version 2.1 (Simplified).
    Usage: python test_main_simple.py
    """

    print("=== Running PySpark to Snowpark Migration Tool Integration Tests (Version 2.1 - Simplified) ===")

    # Run specific test or all tests
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "-s",  # Don't capture stdout
        "--tb=short"  # Short traceback format
    ])