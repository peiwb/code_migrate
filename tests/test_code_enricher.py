"""
Code Enricher Test Module (test_code_enricher.py) - V1.0

This test module validates the complete functionality of CodeEnricher when interacting
with real LLM services, focusing on its ability to receive a function code string and
return a properly structured dictionary containing "enriched code" and "generated test function".

The main focus of this test is to validate the effectiveness of the Prompt templates
designed for CodeEnricher.

Testing Philosophy (POC Phase):
- Real API calls: All tests execute real network requests to Claude API. No mocking.
- Structure and reasonableness validation: We acknowledge that LLM outputs are not 100%
  deterministic. Therefore, our assertions focus on validating the correctness of data
  structure and whether content contains key features, rather than exact text matching.
"""

import pytest
import json
import os
from typing import Dict, Any

# Import modules under test
from services.llm_service import CortexLLMService, LLMServiceError
from agents.code_enricher import CodeEnricher, EnrichmentError


class TestCodeEnricher:
    """CodeEnricher Test Class

    Contains all test cases for CodeEnricher, validating its performance
    in real LLM environments.
    """

    @pytest.fixture(scope="session")
    def llm_service(self):
        """Create a real CortexLLMService instance

        Scope: session - created only once for the entire test session

        Returns:
            CortexLLMService: Configured LLM service instance

        Raises:
            Exception: If Claude API credentials configuration is invalid
        """
        try:
            service = CortexLLMService()
            return service
        except Exception as e:
            pytest.fail(f"Failed to create CortexLLMService instance. Please check Claude API credentials: {e}")

    @pytest.fixture(scope="session")
    def enricher(self, llm_service):
        """Create CodeEnricher instance

        Args:
            llm_service: CortexLLMService instance

        Returns:
            CodeEnricher: Configured code enricher instance
        """
        return CodeEnricher(llm_service=llm_service)

    @pytest.fixture(scope="session")
    def sample_function_code(self):
        """Return a simple, typical PySpark function string as test input

        Returns:
            str: A function string containing simple PySpark logic for testing
        """
        return """
def process_data(df):
    from pyspark.sql import functions as F
    df_filtered = df.filter(df['age'] > 30)
    df_with_status = df_filtered.withColumn("status", F.lit("processed"))
    return df_with_status
"""

    @pytest.fixture(scope="session")
    def real_spark_script(self):
        """Load real PySpark code from examples/sample_spark_script.py

        Returns:
            str: Content of the sample spark script file
        """
        script_path = os.path.join("examples", "sample_spark_script.py")
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            pytest.skip(f"Sample script not found at {script_path}")
        except Exception as e:
            pytest.fail(f"Failed to read sample script: {e}")

    def test_enrich_function_returns_correct_structure(self, enricher, sample_function_code):
        """Test Case 1: Validate the output structure of enrich_function method

        Verifies that the enrich_function method returns a dictionary that conforms
        to our designed structure.

        Args:
            enricher: CodeEnricher instance
            sample_function_code: Sample function code string
        """
        # Execute enrichment
        result = enricher.enrich_function(function_code=sample_function_code)

        # Verify return type is dictionary
        assert isinstance(result, dict), "Enrichment result should be a dictionary"

        # Verify required top-level keys
        required_keys = ["enriched_code", "test_function"]

        for key in required_keys:
            assert key in result, f"Enrichment result is missing required key: {key}"

        # Verify enriched_code field
        assert isinstance(result["enriched_code"], str), "enriched_code should be a string"
        assert len(result["enriched_code"]) > 0, "enriched_code should not be empty"

        # Verify test_function field
        assert isinstance(result["test_function"], str), "test_function should be a string"
        assert len(result["test_function"]) > 0, "test_function should not be empty"

    def test_enrich_function_content_is_reasonable(self, enricher, sample_function_code):
        """Test Case 2: Deep validation of returned code string content

        Validates that the returned code strings contain expected key features
        and characteristics that indicate proper enrichment.

        Args:
            enricher: CodeEnricher instance
            sample_function_code: Sample function code string
        """
        # Execute enrichment
        result = enricher.enrich_function(function_code=sample_function_code)

        # Extract enriched code and test function
        enriched_code = result["enriched_code"]
        test_function = result["test_function"]

        # Assertions for enriched_code - more flexible matching
        assert "def process_data" in enriched_code, "Function definition should be preserved"

        # Check for documentation (flexible - could be docstring or comments)
        documentation_indicators = ['"""', "'''", "#"]
        has_documentation = any(indicator in enriched_code for indicator in documentation_indicators)
        assert has_documentation, "Should contain some form of documentation (docstring or comments)"

        # Assertions for test_function - more flexible
        assert "def test" in test_function or "def Test" in test_function, "Should be a test function"

        # Check for Spark-related content (flexible)
        spark_indicators = ["spark", "Spark", "pyspark", "DataFrame", "createDataFrame"]
        has_spark_content = any(indicator in test_function for indicator in spark_indicators)
        assert has_spark_content, "Test function should contain Spark-related content"

        # Should have some form of assertion
        assertion_indicators = ["assert", "assertEqual", "expect", "should"]
        has_assertions = any(indicator in test_function for indicator in assertion_indicators)
        assert has_assertions, "Test function should contain assertion statements"

    def test_print_full_enrichment_result_for_inspection(self, enricher, sample_function_code):
        """Test Case 3: Print complete enrichment results for manual inspection

        [For debugging and manual inspection] Executes the complete enrichment
        process and prints the final output dictionary in a clear format for
        developers to carefully examine the quality of LLM-generated documentation
        and test cases.

        Args:
            enricher: CodeEnricher instance
            sample_function_code: Sample function code string
        """
        # Execute enrichment
        result = enricher.enrich_function(function_code=sample_function_code)

        # Verify result is not None
        assert result is not None, "Enrichment result should not be None"

        # Print results in clear format
        print("\n" + "="*80)
        print("COMPLETE ENRICHMENT RESULT - MANUAL INSPECTION OUTPUT")
        print("="*80)

        print("\n--- Enriched Code ---")
        print(result.get("enriched_code", "N/A"))

        print("\n--- Generated Test Function ---")
        print(result.get("test_function", "N/A"))

        print("\n--- JSON Result ---")
        print(json.dumps(result, indent=2))

        print("\n" + "-"*80)
        print("ENRICHMENT RESULT PRINTING COMPLETED")
        print("-"*80 + "\n")

    def test_enrich_real_spark_script(self, enricher, real_spark_script):
        """Test Case 4: Test enrichment with real PySpark script from examples

        Uses the actual sample_spark_script.py file to test the enrichment process
        with real-world code and prints the complete JSON results.

        Args:
            enricher: CodeEnricher instance
            real_spark_script: Content of examples/sample_spark_script.py
        """
        # Execute enrichment on the real script
        result = enricher.enrich_function(function_code=real_spark_script)

        # Verify basic structure
        assert isinstance(result, dict), "Real script enrichment should return dictionary"
        assert "enriched_code" in result, "Real script result should contain enriched_code"
        assert "test_function" in result, "Real script result should contain test_function"

        # Verify non-empty content
        assert len(result["enriched_code"]) > 0, "Enriched real script code should not be empty"
        assert len(result["test_function"]) > 0, "Real script test function should not be empty"

        # Print complete results for inspection
        print("\n" + "="*100)
        print("REAL SPARK SCRIPT ENRICHMENT RESULTS")
        print("="*100)

        print("\n--- ORIGINAL SCRIPT ---")
        print(real_spark_script)

        print("\n--- ENRICHED CODE WITH DOCUMENTATION ---")
        print(result["enriched_code"])

        print("\n--- GENERATED UNIT TEST ---")
        print(result["test_function"])

        print("\n--- COMPLETE JSON RESULT ---")
        print(json.dumps(result, indent=2))

        print("\n" + "="*100)
        print("REAL SPARK SCRIPT ENRICHMENT COMPLETED")
        print("="*100 + "\n")

    def test_enricher_handles_different_function_types(self, enricher):
        """Additional Test Case: Verify handling of different function types

        Validates CodeEnricher's ability to handle various types of PySpark functions
        with different complexity levels.

        Args:
            enricher: CodeEnricher instance
        """
        # More complex function example
        complex_function = """
def advanced_data_processing(spark_df, threshold=100):
    from pyspark.sql.functions import col, when, sum as spark_sum
    
    result_df = spark_df.filter(col("value") > threshold) \
                       .groupBy("category") \
                       .agg(spark_sum("amount").alias("total_amount")) \
                       .withColumn("status", when(col("total_amount") > 1000, "high").otherwise("low"))
    
    return result_df
"""

        # Execute enrichment
        result = enricher.enrich_function(function_code=complex_function)

        # Basic validation
        assert isinstance(result, dict), "Should return dictionary for complex functions"
        assert "enriched_code" in result, "Should contain enriched_code for complex functions"
        assert "test_function" in result, "Should contain test_function for complex functions"

        # Verify content contains function name
        enriched_code = result["enriched_code"]
        test_function = result["test_function"]

        assert "advanced_data_processing" in enriched_code, "Should preserve original function name"

        # More flexible test function name checking
        test_indicators = ["test", "Test", "advanced_data_processing"]
        has_test_indicator = any(indicator in test_function for indicator in test_indicators)
        assert has_test_indicator, "Should generate appropriate test function"

    def test_enricher_handles_simple_function_gracefully(self, enricher):
        """Additional Test Case: Verify handling of very simple functions

        Validates that CodeEnricher can handle minimal function code gracefully.

        Args:
            enricher: CodeEnricher instance
        """
        # Very simple function
        simple_function = """
def get_count(df):
    return df.count()
"""

        # LLM should handle this gracefully - don't expect exceptions
        result = enricher.enrich_function(function_code=simple_function)

        # Verify basic structure even for simple functions
        assert isinstance(result, dict), "Should return dictionary for simple functions"
        assert "enriched_code" in result, "Should contain enriched_code field"
        assert "test_function" in result, "Should contain test_function field"

        # Verify non-empty content
        assert len(result["enriched_code"]) > 0, "Enriched code should not be empty"
        assert len(result["test_function"]) > 0, "Test function should not be empty"

    def test_enricher_preserves_function_logic(self, enricher, sample_function_code):
        """Additional Test Case: Verify that core function logic is preserved

        Ensures that the enrichment process maintains the original function's
        core logic and structure.

        Args:
            enricher: CodeEnricher instance
            sample_function_code: Sample function code string
        """
        # Execute enrichment
        result = enricher.enrich_function(function_code=sample_function_code)

        enriched_code = result["enriched_code"]

        # Verify core logic elements are preserved (more flexible matching)
        critical_elements = [
            "process_data",  # Function name
            "filter",        # Core operation
            "withColumn",    # Core operation
            "return"         # Return statement
        ]

        for element in critical_elements:
            assert element in enriched_code, f"Critical element '{element}' should be preserved in enriched code"

        # Verify some business logic is preserved (flexible)
        business_logic_indicators = ["age", "30", "status", "processed"]
        preserved_logic = sum(1 for indicator in business_logic_indicators if indicator in enriched_code)
        assert preserved_logic >= 2, "Should preserve most of the original business logic"

    def test_error_handling_with_edge_cases(self, enricher):
        """Additional Test Case: Verify behavior with edge cases

        Tests that CodeEnricher handles edge cases appropriately without crashing.

        Args:
            enricher: CodeEnricher instance
        """
        edge_cases = [
            "",  # Empty string
            "# Just a comment",  # Only comment
            "def incomplete_func(",  # Incomplete function
            "this is not python code",  # Invalid Python
            "def f(): pass",  # Minimal function
        ]

        for i, edge_case in enumerate(edge_cases):
            try:
                result = enricher.enrich_function(function_code=edge_case)

                # If it succeeds, verify it returns proper structure
                assert isinstance(result, dict), f"Edge case {i+1} should return dict if successful"
                assert "enriched_code" in result, f"Edge case {i+1} should have enriched_code key"
                assert "test_function" in result, f"Edge case {i+1} should have test_function key"

                print(f"Edge case {i+1} handled successfully")

            except (EnrichmentError, LLMServiceError) as e:
                # These exceptions are acceptable for edge cases
                print(f"Edge case {i+1} properly handled with exception: {type(e).__name__}")

            except Exception as e:
                # Unexpected exceptions should be investigated
                pytest.fail(f"Edge case {i+1} caused unexpected exception: {type(e).__name__}: {e}")

    def test_enricher_service_integration(self, enricher):
        """Additional Test Case: Verify integration with LLM service

        Tests the integration between CodeEnricher and CortexLLMService.

        Args:
            enricher: CodeEnricher instance
        """
        # Verify the enricher has a properly initialized LLM service
        assert enricher.llm_service is not None, "Enricher should have LLM service"
        assert hasattr(enricher.llm_service, 'get_json_completion'), "LLM service should have get_json_completion method"

        # Verify schema is properly defined
        assert hasattr(enricher, '_enrichment_schema'), "Enricher should have enrichment schema"
        assert isinstance(enricher._enrichment_schema, dict), "Schema should be a dictionary"

        # Verify required schema fields
        schema = enricher._enrichment_schema
        assert "properties" in schema, "Schema should have properties"
        assert "enriched_code" in schema["properties"], "Schema should define enriched_code"
        assert "test_function" in schema["properties"], "Schema should define test_function"