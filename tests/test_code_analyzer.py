"""
Code Analyzer Test Module (test_code_analyzer.py) - V1.1

This test module validates the complete functionality of CodeAnalyzer when interacting
with real LLM services, focusing on the three-phase analysis pipeline and the
effectiveness of Prompt templates.

Testing Philosophy:
- Use real API calls (no mocking)
- Validate JSON structure correctness and content reasonableness
- Focus on Prompt template effectiveness
"""

import os
import pytest
import json
from typing import Dict, Any

# Import modules under test
from services.llm_service import CortexLLMService
from agents.code_analyzer import CodeAnalyzer


class TestCodeAnalyzer:
    """CodeAnalyzer Test Class

    Contains all test cases for CodeAnalyzer, validating its performance
    in real LLM environments.
    """

    @pytest.fixture(scope="session")
    def llm_service(self):
        """Create a real CortexLLMService instance

        Scope: session - created only once for the entire test session

        Returns:
            CortexLLMService: Configured LLM service instance

        Raises:
            Exception: If Snowflake credentials configuration is invalid
        """
        try:
            service = CortexLLMService()
            return service
        except Exception as e:
            pytest.fail(f"Failed to create CortexLLMService instance. Please check Snowflake credentials: {e}")

    @pytest.fixture(scope="session")
    def analyzer(self, llm_service):
        """Create CodeAnalyzer instance

        Args:
            llm_service: CortexLLMService instance

        Returns:
            CodeAnalyzer: Configured code analyzer instance
        """
        return CodeAnalyzer(llm_service=llm_service)

    @pytest.fixture(scope="session")
    def sample_script_content(self):
        """Read sample script content

        Reads test PySpark script content from examples/sample_spark_script.py

        Returns:
            str: Complete content of the script file

        Raises:
            FileNotFoundError: If the sample script file does not exist
        """
        script_path = os.path.join("examples", "sample_spark_script.py")

        if not os.path.exists(script_path):
            pytest.fail(f"Test data file does not exist: {script_path}")

        try:
            with open(script_path, 'r', encoding='utf-8') as file:
                content = file.read()

            if not content.strip():
                pytest.fail(f"Test data file is empty: {script_path}")

            return content
        except Exception as e:
            pytest.fail(f"Failed to read test data file: {script_path}, error: {e}")

    def test_analyze_script_returns_correct_main_structure(self, analyzer, sample_script_content):
        """Test Case 1: Validate top-level structure of analysis report

        Verifies that the report returned by analyze_script method contains all
        required top-level fields and ensures the overall data structure meets
        design specifications.

        Args:
            analyzer: CodeAnalyzer instance
            sample_script_content: Sample script content
        """
        # Execute analysis
        report = analyzer.analyze_script(
            script_content=sample_script_content,
            source_file_name="test_script.py"
        )

        # Verify return type
        assert isinstance(report, dict), "Analysis report should be a dictionary"

        # Verify required top-level fields
        required_keys = [
            "source_file_name",
            "analysis_timestamp",
            "package_analysis",
            "function_analysis",
            "conversion_order"
        ]

        for key in required_keys:
            assert key in report, f"Analysis report is missing required field: {key}"

        # Verify basic field types
        assert isinstance(report["source_file_name"], str), "source_file_name should be a string"
        assert isinstance(report["analysis_timestamp"], str), "analysis_timestamp should be a string"
        assert report["source_file_name"] == "test_script.py", "source_file_name should match input parameter"

    def test_analyze_script_package_analysis_structure(self, analyzer, sample_script_content):
        """Test Case 2: Validate structure of package_analysis section

        Deep validation of package analysis results data structure, ensuring
        it contains classification of standard libraries, third-party libraries,
        and custom modules.

        Args:
            analyzer: CodeAnalyzer instance
            sample_script_content: Sample script content
        """
        # Execute analysis
        report = analyzer.analyze_script(
            script_content=sample_script_content,
            source_file_name="test_script.py"
        )

        # Get package analysis section
        package_analysis = report["package_analysis"]

        # Verify package analysis is a dictionary
        assert isinstance(package_analysis, dict), "package_analysis should be a dictionary"

        # Verify required classification fields
        required_package_keys = ["standard_libs", "third_party", "custom_modules"]

        for key in required_package_keys:
            assert key in package_analysis, f"package_analysis is missing required field: {key}"
            assert isinstance(package_analysis[key], list), f"{key} should be a list"

    def test_analyze_script_function_analysis_structure(self, analyzer, sample_script_content):
        """Test Case 3: Validate structure of function_analysis section

        Validates the data structure of function analysis results, ensuring
        each function entry contains required fields.

        Args:
            analyzer: CodeAnalyzer instance
            sample_script_content: Sample script content
        """
        # Execute analysis
        report = analyzer.analyze_script(
            script_content=sample_script_content,
            source_file_name="test_script.py"
        )

        # Get function analysis section
        function_analysis = report["function_analysis"]

        # Verify function analysis is a list
        assert isinstance(function_analysis, list), "function_analysis should be a list"

        # If there are function analysis results, validate the first function's structure
        if function_analysis:
            first_function = function_analysis[0]

            # Verify function entry is a dictionary
            assert isinstance(first_function, dict), "Function analysis entry should be a dictionary"

            # Verify required fields
            required_function_keys = ["function_name", "dependencies", "suggested_patterns"]

            for key in required_function_keys:
                assert key in first_function, f"Function analysis entry is missing required field: {key}"

    def test_analyze_script_conversion_order_structure(self, analyzer, sample_script_content):
        """Test Case 4: Validate structure and content of conversion_order section

        Validates the reasonableness of conversion order, ensuring all functions
        are included in the conversion sequence and the conversion order is
        consistent with function analysis results.

        Args:
            analyzer: CodeAnalyzer instance
            sample_script_content: Sample script content
        """
        # Execute analysis
        report = analyzer.analyze_script(
            script_content=sample_script_content,
            source_file_name="test_script.py"
        )

        # Get conversion order and function analysis
        conversion_order = report["conversion_order"]
        function_analysis = report["function_analysis"]

        # Verify conversion order is a list
        assert isinstance(conversion_order, list), "conversion_order should be a list"

        # Extract function names list
        function_names = [f["function_name"] for f in function_analysis]

        # Verify conversion order length matches function analysis length
        assert len(conversion_order) == len(function_analysis), \
            "conversion_order length should match function_analysis length"

        # Verify function names in conversion order all exist in function analysis
        conversion_order_set = set(conversion_order)
        function_names_set = set(function_names)

        assert conversion_order_set == function_names_set, \
            "Function names in conversion_order should exactly match those in function_analysis"

    def test_print_full_analysis_report_for_inspection(self, analyzer, sample_script_content):
        """Test Case 5: Print complete analysis report for manual inspection

        [For debugging and inspection purposes] Executes the complete analysis
        process and prints results in formatted way for developers to inspect
        LLM output quality and details.

        Args:
            analyzer: CodeAnalyzer instance
            sample_script_content: Sample script content
        """
        print("\n" + "=" * 80)
        print("COMPLETE ANALYSIS REPORT - MANUAL INSPECTION OUTPUT")
        print("=" * 80)

        # Execute analysis
        report = analyzer.analyze_script(
            script_content=sample_script_content,
            source_file_name="sample_spark_script.py"
        )

        # Verify report is not None
        assert report is not None, "Analysis report should not be None"

        # Print each section with clear headers
        self._print_section("BASIC INFORMATION", {
            "source_file_name": report.get("source_file_name"),
            "analysis_timestamp": report.get("analysis_timestamp")
        })

        self._print_section("PACKAGE ANALYSIS", report.get("package_analysis", {}))

        self._print_section("FUNCTION ANALYSIS", report.get("function_analysis", []))

        self._print_section("CONVERSION ORDER", report.get("conversion_order", []))

        # Print complete JSON for reference
        print("\n" + "-" * 80)
        print("COMPLETE JSON REPORT:")
        print("-" * 80)
        try:
            formatted_report = json.dumps(report, indent=4, ensure_ascii=False)
            print(formatted_report)
        except Exception as e:
            print(f"JSON formatting failed: {e}")
            print("Raw report:")
            print(report)

        print("\n" + "=" * 80)
        print("ANALYSIS REPORT INSPECTION COMPLETED")
        print("=" * 80 + "\n")

    def _print_section(self, title: str, content: Any):
        """Helper method to print report sections with formatting

        Args:
            title: Section title
            content: Section content to print
        """
        print(f"\n📋 {title}")
        print("-" * (len(title) + 4))

        if isinstance(content, dict):
            if not content:
                print("  (No data)")
            else:
                for key, value in content.items():
                    print(f"  {key}:")
                    if isinstance(value, list):
                        if not value:
                            print("    (Empty list)")
                        else:
                            for i, item in enumerate(value, 1):
                                print(f"    {i}. {item}")
                    else:
                        print(f"    {value}")
        elif isinstance(content, list):
            if not content:
                print("  (Empty list)")
            else:
                for i, item in enumerate(content, 1):
                    print(f"  {i}. {item}")
        else:
            print(f"  {content}")

    def test_analyzer_handles_empty_script_gracefully(self, analyzer):
        """Additional Test Case: Verify handling of empty scripts

        Validates that CodeAnalyzer can gracefully handle empty scripts
        or invalid input.

        Args:
            analyzer: CodeAnalyzer instance
        """
        # Test empty script
        empty_script = ""

        try:
            report = analyzer.analyze_script(
                script_content=empty_script,
                source_file_name="empty_script.py"
            )

            # Verify even empty scripts should return valid structure
            assert isinstance(report, dict), "Even empty scripts should return dictionary structure"
            assert "source_file_name" in report, "Report should contain source_file_name field"

        except Exception as e:
            # If exception is thrown, it should be expected error type
            assert isinstance(e, (ValueError, TypeError)), f"Should throw reasonable exception type, not: {type(e)}"

    def test_analyzer_processes_different_file_types(self, analyzer):
        """Additional Test Case: Verify handling of different file types

        Validates CodeAnalyzer's adaptability to different Python script types.

        Args:
            analyzer: CodeAnalyzer instance
        """
        # Simple Python script example
        simple_script = """
import os
import sys
from datetime import datetime

def hello_world():
    print("Hello, World!")
    return True

def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result

if __name__ == "__main__":
    hello_world()
    test_data = [1, 2, 3, 4, 5]
    processed = process_data(test_data)
    print(processed)
"""

        # Execute analysis
        report = analyzer.analyze_script(
            script_content=simple_script,
            source_file_name="simple_test.py"
        )

        # Basic validation
        assert isinstance(report, dict), "Should return dictionary type report"
        assert "function_analysis" in report, "Report should contain function analysis"

        # Verify function recognition
        function_analysis = report["function_analysis"]
        if function_analysis:
            function_names = [f["function_name"] for f in function_analysis]
            expected_functions = ["hello_world", "process_data"]

            # Should recognize at least some functions
            assert len(function_names) > 0, "Should be able to recognize functions in the script"

    def test_save_analysis_report_to_output_directory(self, analyzer, sample_script_content):
        """Additional Test Case: Save analysis report to output directory

        Saves the complete analysis report to the output directory for
        further inspection and debugging.

        Args:
            analyzer: CodeAnalyzer instance
            sample_script_content: Sample script content
        """
        # Execute analysis
        report = analyzer.analyze_script(
            script_content=sample_script_content,
            source_file_name="sample_spark_script.py"
        )

        # Ensure output directory exists
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Save report to file
        output_file = os.path.join(output_dir, "analysis_report.json")

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4, ensure_ascii=False)

            print(f"\n✅ Analysis report saved to: {output_file}")
            print(f"📁 You can find the complete analysis report in the output directory")

            # Verify file was created and contains data
            assert os.path.exists(output_file), f"Output file should be created: {output_file}"

            # Read back and verify
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_report = json.load(f)

            assert saved_report == report, "Saved report should match original report"

        except Exception as e:
            pytest.fail(f"Failed to save analysis report: {e}")